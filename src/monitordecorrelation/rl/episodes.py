"""Multi-turn episode driver: sample a turn → hand it to the env → append its observation → repeat.

Single-turn envs sample one completion per prompt (``rl/rollout.py``). Tool-loop envs (the terminal
env) need an agent loop: the env executes the turn's command and produces the next user message,
which is rendered and appended to the SAME token sequence the policy just produced, and the policy
continues. This module is that loop, generic over any env implementing the ``MultiTurnEnv`` protocol
(``start`` / ``step`` / ``finish`` — see ``envs/base.py``).

Concurrency: **every episode runs free**
--------------------------------------
Each episode gets its own driver thread and advances as fast as its own turns resolve — it never
waits for any other episode. An episode that finishes its command in 20 ms issues its next
generation immediately, while a sibling is still blocked on a 30-second command timeout or a long
generation. (The earlier driver advanced the whole batch in lockstep — sample every episode's turn,
wait for all of them, run every env step, wait for all of those, then start the next turn — so the
per-turn cost of a batch was the cost of its slowest episode, ~max_turns times over.)

What that means for the pieces:

- **generations** — ``sampling_client.sample`` schedules onto tinker's background event loop and
  returns a ``concurrent.futures.Future``, so calling it from many threads (and awaiting from many
  threads) is safe and genuinely parallel. EVERY request is a single sample — never one
  ``num_samples=G`` request per group: tinker applies one seed to a whole request, so a seeded
  8-sample request returns ~1.4 distinct sequences (measured, base Qwen3-8B), collapsing the GRPO
  group. Turn 0 is one request per episode, issued up front for the whole batch; each episode waits
  only for its own (never for its siblings'); every later turn is one call issued by that episode.
- **env steps** (command execution) — run in the episode's own thread, uncapped by the driver:
  every episode whose turn is ready runs its command at once. The only bound is the env's own — the
  terminal env takes a cross-process ``globalsem.code_exec_slot`` per command (half the box's
  cores, shared by every run on the box).
- **monitors** — ``on_rollout(index, rollout)`` fires the moment an episode is graded and flattened,
  from that episode's thread, so a caller (``rl/train.py``) can start its judge API calls for a
  finished episode while the rest of the batch is still generating.

The returned list keeps the original order (prompt-major, ``num_samples`` consecutive per prompt)
regardless of who finished first, so GRPO grouping downstream is unchanged.

What comes out is an ordinary ``Rollout`` with two extras in ``meta``:

- ``transitions``: ``[{"ob": [tokens], "ac": [tokens], "logprobs": [...]}, …]``. Each ``ob`` is a
  strict prefix-extension of the previous ``ob + ac`` (we append the sampled tokens verbatim, then
  whatever framing follows), so tinker-cookbook's ``trajectory_to_data`` folds the whole episode into
  ONE datum with observation tokens masked and every action token carrying the episode's advantage.
  ``rl/grpo.py`` reads this field. There is one transition per sampling call — normally one per turn,
  two when the thinking budget had to force an answer (below).
- ``episode``: the env's grading record (``finish()``'s meta) — ``score()`` is then a pure function
  of the rollout, so eval/train scoring code paths stay unchanged.

Thinking budget (``think_budget``): thinking models can spend the whole per-turn token budget inside
``<think>`` and never act — on a hard grid Qwen3-8B does exactly that, every episode ends truncated,
every reward is 0 and GRPO has nothing to learn from. With a budget, a turn is sampled with
``max_tokens=think_budget``; if it is cut off mid-thought, the renderer's budget-forcing suffix
("…I have to give the solution now.</think>") is appended as *observation* tokens and the answer is
sampled with ``answer_tokens`` more. This is Qwen3's documented thinking-budget mechanism and the
tinker-side equivalent of rg_obfuscation's ``max_thinking_tokens`` logit processor. Without a budget
a turn is a single call with ``max_tokens``.

Seeding: every sampling call gets its own seed, a pure function of its **position** —
``derive_sample_seed(seed, group, sample, call)``: the prompt's index in the batch, the episode's
index within that prompt's GRPO group, and the call's index within the episode (0 = turn 0) —
never of the order calls happen to be issued in. So the seeds are identical whatever the thread
interleaving, two episodes that reached an identical context still get different continuations
(which would otherwise silently collapse a GRPO group's advantage variance), and a run stays
reproducible from ``cfg.seed`` alone (the RL loop derives ``seed`` from the run seed, the phase and
the RL step — see ``rl/train.py``).
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import tinker

from monitordecorrelation.types import Prompt, Rollout


def derive_sample_seed(*parts: int | str) -> int:
    """A sampling seed that is a pure function of ``parts`` (a hash, so any change to any part gives
    an unrelated seed): e.g. ``(run seed, "train"|"eval", RL step)`` for a batch, then
    ``(batch seed, group, sample, call)`` for one sampling call. Stable across processes (not
    Python's randomized ``hash``), and in tinker's seed range [0, 2**31 - 1)."""
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


@dataclass
class _Episode:
    index: int                           # position in the returned list (prompt-major, group-consecutive)
    prompt: Prompt
    state: Any
    ob: list[int]                        # the observation tokens the NEXT sampling call starts from
    transitions: list[dict] = field(default_factory=list)
    done: bool = False
    stop_reason: str = ""
    turn_tokens: list[int] = field(default_factory=list)   # this turn's tokens (all segments)
    n_turns: int = 0
    n_forced: int = 0                                      # turns whose thinking was budget-forced
    n_truncated: int = 0                                   # turns whose FINAL segment hit max_tokens
    n_calls: int = 0                                       # sampling calls after turn 0 (issued by the episode)


def _only_sequence(response, what: str):
    """The one sequence of a single-sample response (every request here asks for exactly one)."""
    seqs = list(response.sequences)
    if len(seqs) != 1:
        raise RuntimeError(f"{what}: asked for 1 sample, got {len(seqs)}")
    return seqs[0]


def run_episodes(
    sampling_client,
    renderer,
    env,
    prompts: list[Prompt],
    *,
    num_samples: int = 1,
    max_tokens: int | None = None,
    temperature: float = 1.0,
    seed: int | None = None,
    max_turns: int | None = None,
    think_budget: int | None = None,
    answer_tokens: int | None = None,
    episode_workers: int | None = None,
    on_rollout: Callable[[int, Rollout], None] | None = None,
) -> list[Rollout]:
    """Run ``num_samples`` episodes per prompt (the GRPO group, kept consecutive in the output) and
    return one ``Rollout`` per episode. Every episode runs in its own thread and advances
    independently (see the module docstring). ``max_turns`` defaults to ``env.max_turns``. With
    ``think_budget`` set, each turn's thinking is capped at that many tokens (then force-closed) and
    the answer gets ``answer_tokens``; otherwise a turn is one call of ``max_tokens``. Those two
    modes use DISJOINT arguments, so exactly one set must be given: ``think_budget`` + ``answer_tokens``,
    or ``max_tokens`` alone. Passing the unused one is an error rather than a number that silently
    does nothing.

    ``episode_workers`` caps how many episodes are in flight; the default — one thread per episode —
    is what makes the batch fully decoupled. ``on_rollout(index, rollout)`` is called from the
    episode's own thread as soon as that episode is finished and flattened, so callers can start
    per-rollout work (monitor API calls) without waiting for the batch; ``index`` is the rollout's
    position in the returned list.
    """
    max_turns = max_turns or getattr(env, "max_turns", None)
    if not max_turns or max_turns < 1:
        raise ValueError("run_episodes needs max_turns >= 1 (from the env or the argument)")
    if think_budget is not None and think_budget < 1:
        raise ValueError("think_budget must be >= 1 (or None for no budget)")
    if think_budget is None:
        if max_tokens is None:
            raise ValueError("no think_budget: a turn is one call, so max_tokens must be given")
        if answer_tokens is not None:
            raise ValueError(
                f"no think_budget, so no answer is ever forced and answer_tokens={answer_tokens} would "
                "be unused — pass think_budget too, or drop answer_tokens"
            )
    else:
        if answer_tokens is None:
            raise ValueError(f"think_budget={think_budget} forces the answer, so answer_tokens must be given")
        if max_tokens is not None:
            raise ValueError(
                f"think_budget={think_budget} sizes the thinking call and answer_tokens the answer, so "
                f"max_tokens={max_tokens} would be unused — drop it"
            )
    first_call_tokens = think_budget if think_budget else max_tokens
    n_episodes = len(prompts) * num_samples

    def params(position: tuple[int, int, int], n_tokens: int) -> tinker.SamplingParams:
        """``position`` = (group, sample, call) of the call — its seed, when the batch is seeded."""
        return tinker.SamplingParams(
            max_tokens=n_tokens, temperature=temperature,
            seed=None if seed is None else derive_sample_seed(seed, *position),
            **({"stop": renderer.stop_tokens} if getattr(renderer, "stop_tokens", None) else {}),
        )

    def sample_one(ep: _Episode, n_tokens: int):
        """One continuation call for ``ep`` from its current observation, seeded by its position."""
        ep.n_calls += 1  # turn 0 is call 0, so the episode's own calls are 1, 2, …
        group, k = divmod(ep.index, num_samples)
        return sampling_client.sample(tinker.ModelInput.from_ints(ep.ob), 1,
                                      params((group, k, ep.n_calls), n_tokens))

    def record(ep: _Episode, seq) -> None:
        """Record one sampled segment as a transition from the episode's current observation."""
        tokens = list(seq.tokens)
        if seq.logprobs is None:
            raise RuntimeError("sampler returned no logprobs — GRPO needs sampling logprobs")
        ep.transitions.append({"ob": list(ep.ob), "ac": tokens, "logprobs": list(seq.logprobs)})
        ep.turn_tokens.extend(tokens)

    def finish_turn(ep: _Episode, seq) -> tuple[str, str, bool]:
        """Absorb the LAST segment of a turn: record it, then parse the whole turn for the env."""
        record(ep, seq)
        ep.stop_reason = str(seq.stop_reason)
        truncated = ep.stop_reason != "stop"
        cot, text, raw = renderer.parse(ep.turn_tokens)
        if truncated and not cot and "<think>" in raw and "</think>" not in raw:
            cot, text = text, ""  # cut off mid-thought: what parse() called "answer" is the thinking
        ep.n_turns += 1
        ep.n_truncated += int(truncated)
        return cot, text, truncated

    def step_env(ep: _Episode, cot: str, text: str, truncated: bool) -> bool:
        """Hand the turn to the env and extend the observation; returns False when the episode ends."""
        obs, done = env.step(ep.state, cot, text, truncated=truncated)
        ep.turn_tokens = []
        if done or obs is None:
            ep.done = True
            return False
        ac = ep.transitions[-1]["ac"]
        ep.ob = ep.ob + ac + list(renderer.continuation_tokens(obs, ended_cleanly=not truncated))
        return True

    def drive(ep: _Episode, first_seq) -> Rollout:
        """The whole life of ONE episode: turn → env step → turn → … → finish. Runs in its own
        thread, touching nothing another episode owns."""
        seq = first_seq
        for turn in range(max_turns):
            if turn:
                seq = _only_sequence(sample_one(ep, first_call_tokens).result(), f"episode {ep.index}")
            tokens = list(seq.tokens)
            if think_budget and str(seq.stop_reason) != "stop" and renderer.in_open_think(tokens):
                # Cut off inside <think>: close it with the renderer's forcing suffix (appended as
                # OBSERVATION tokens, masked in training) and sample the answer with a fresh budget.
                record(ep, seq)
                forced = list(renderer.force_answer_tokens())
                ep.ob = ep.ob + tokens + forced
                ep.turn_tokens.extend(forced)
                ep.n_forced += 1
                seq = _only_sequence(sample_one(ep, answer_tokens).result(), f"episode {ep.index}")
            cot, text, truncated = finish_turn(ep, seq)
            if not step_env(ep, cot, text, truncated):
                break
        view = env.finish(ep.state)
        return Rollout(
            prompt=ep.prompt,
            cot=view.cot,
            output=view.output,
            token_ids=[t for tr in ep.transitions for t in tr["ac"]],
            logprobs=[lp for tr in ep.transitions for lp in tr["logprobs"]],
            meta={
                "stop_reason": ep.stop_reason,
                "n_turns": ep.n_turns,
                "n_forced_answers": ep.n_forced,
                # Token accounting (one sampling call per transition; the final ob+ac IS the single
                # training datum, because every ob is a prefix-extension of the previous one).
                "n_sampling_calls": len(ep.transitions),
                "input_tokens": sum(len(tr["ob"]) for tr in ep.transitions),
                "output_tokens": sum(len(tr["ac"]) for tr in ep.transitions),
                "train_tokens": (len(ep.transitions[-1]["ob"]) + len(ep.transitions[-1]["ac"])
                                 if ep.transitions else 0),
                "n_truncated_turns": ep.n_truncated,
                "transitions": ep.transitions,
                "episode": view.meta,
            },
        )

    # -- turn 0: issued up front so the whole batch is in flight at once — one single-sample request
    # PER EPISODE (index i*num_samples + k), seeded by its position (i, k, call 0).
    turn0 = []
    for i, p in enumerate(prompts):
        mi = renderer.model_input(p.text)
        turn0 += [sampling_client.sample(mi, 1, params((i, k, 0), first_call_tokens))
                  for k in range(num_samples)]
    prompt_tokens = [list(renderer.prompt_tokens(p.text)) for p in prompts]

    out: list[Rollout | None] = [None] * n_episodes

    def run_one(index: int) -> None:
        """Entry point of one episode's thread: wait for its own turn-0 request, then drive it to
        completion and publish it."""
        p_i = index // num_samples
        first = _only_sequence(turn0[index].result(), f"episode {index}, turn 0")
        ep = _Episode(index=index, prompt=prompts[p_i], state=env.start(prompts[p_i]),
                      ob=list(prompt_tokens[p_i]))
        rollout = drive(ep, first)
        out[index] = rollout
        if on_rollout is not None:
            on_rollout(index, rollout)

    if n_episodes:
        workers = n_episodes if episode_workers is None else max(1, min(episode_workers, n_episodes))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="episode") as ex:
            # Collected rather than raised on the spot so every already-running episode finishes (and
            # closes its terminal session) before the first failure is re-raised. Failures are still
            # fatal — a broken episode is never quietly dropped from the batch.
            for fut in [ex.submit(run_one, i) for i in range(n_episodes)]:
                fut.result()

    missing = [i for i, r in enumerate(out) if r is None]
    if missing:
        raise RuntimeError(f"episodes {missing[:10]} produced no rollout")
    return [r for r in out if r is not None]
