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
  threads) is safe and genuinely parallel. Turn 0 is still ONE call per prompt (it returns the whole
  ``num_samples`` GRPO group); every later turn is one call per episode, issued by that episode.
- **env steps** (command execution) — run in the episode's own thread, bounded by a semaphore of
  ``step_workers`` permits so the box never sees more than that many concurrent commands. A
  semaphore is a resource cap, not a barrier: an episode waits for a free slot, never for its peers
  to reach the same turn.
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

Seeding: every sampling call gets its own seed, derived from a **position-addressed** slot — the
prompt index for a turn-0 call, then ``(episode index, call index within that episode)`` — rather
than from the order calls happen to be issued in. So the seeds are identical whatever the thread
interleaving, two episodes that reached an identical context still get different continuations
(which would otherwise silently collapse a GRPO group's advantage variance), and a run stays
reproducible from ``cfg.seed`` alone.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import tinker

from monitordecorrelation.types import Prompt, Rollout


def derive_sample_seed(base_seed: int, call_index: int) -> int:
    """A unique, reproducible seed per sampling call: same run seed → same sequence of call seeds, but
    each call differs (so GRPO groups vary across steps instead of collapsing to one group every step).
    Pure + deterministic so it's unit-testable without tinker."""
    return (base_seed * 1_000_003 + call_index) % (2**31 - 1)


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
    n_calls: int = 0                                       # sampling calls this episode issued itself


class _Gathered:
    """Presents ``n`` single-sample futures as one future whose ``.result().sequences`` is their
    concatenation — so a seeded turn-0 group (one request per sample) looks like an n-sample call."""

    def __init__(self, futures: list) -> None:
        self._futures = futures

    def result(self) -> Any:
        seqs = [seq for f in self._futures for seq in f.result().sequences]
        return type("_R", (), {"sequences": seqs})()


class _SharedFuture:
    """One API future consumed by the whole GRPO group.

    Turn 0 is a single ``sample(..., num_samples=G)`` call whose G sequences seed G episodes, each of
    which then runs in its own thread. This memoizes the result (and any exception) behind a lock so
    every one of those threads can wait on it independently, whatever kind of future the sampler
    returned — tinker hands back a ``concurrent.futures.Future``, the offline tests hand back a stub.
    """

    def __init__(self, future: Any) -> None:
        self._future = future
        self._lock = threading.Lock()
        self._done = False
        self._value: Any = None
        self._exc: BaseException | None = None

    def result(self) -> Any:
        with self._lock:
            if not self._done:
                try:
                    self._value = self._future.result()
                except BaseException as e:  # noqa: BLE001 — re-raised to every waiter below
                    self._exc = e
                self._done = True
        if self._exc is not None:
            raise self._exc
        return self._value


def run_episodes(
    sampling_client,
    renderer,
    env,
    prompts: list[Prompt],
    *,
    num_samples: int = 1,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    seed: int | None = None,
    max_turns: int | None = None,
    think_budget: int | None = None,
    answer_tokens: int = 512,
    step_workers: int = 16,
    episode_workers: int | None = None,
    on_rollout: Callable[[int, Rollout], None] | None = None,
) -> list[Rollout]:
    """Run ``num_samples`` episodes per prompt (the GRPO group, kept consecutive in the output) and
    return one ``Rollout`` per episode. Every episode runs in its own thread and advances
    independently (see the module docstring). ``max_turns`` defaults to ``env.max_turns``. With
    ``think_budget`` set, each turn's thinking is capped at that many tokens (then force-closed) and
    the answer gets ``answer_tokens``; otherwise a turn is one call of ``max_tokens``.

    ``step_workers`` caps how many env steps (command executions) run at once. ``episode_workers``
    caps how many episodes are in flight; the default — one thread per episode — is what makes the
    batch fully decoupled, and is what you want unless the env's steps are expensive in a way
    ``step_workers`` doesn't already bound. ``on_rollout(index, rollout)`` is called from the
    episode's own thread as soon as that episode is finished and flattened, so callers can start
    per-rollout work (monitor API calls) without waiting for the batch; ``index`` is the rollout's
    position in the returned list.
    """
    max_turns = max_turns or getattr(env, "max_turns", None)
    if not max_turns or max_turns < 1:
        raise ValueError("run_episodes needs max_turns >= 1 (from the env or the argument)")
    if think_budget is not None and think_budget < 1:
        raise ValueError("think_budget must be >= 1 (or None for no budget)")
    base_seed = 0 if seed is None else seed
    first_call_tokens = think_budget if think_budget else max_tokens
    n_episodes = len(prompts) * num_samples
    # Seed slots are addressed by POSITION, never by issue order, so threading can't move them:
    # slots [0, len(prompts)*num_samples) are the per-(prompt, sample) turn-0 calls, then episode e
    # owns the contiguous block [turn0_slots + e*per_ep, … + per_ep). An episode issues at most one call per turn after
    # turn 0, plus one forced-answer call per turn — hence 2*max_turns, which is a strict bound.
    per_ep_calls = 2 * max_turns
    turn0_slots = len(prompts) * num_samples  # one slot per (prompt, sample) turn-0 request
    step_sem = threading.Semaphore(max(1, step_workers))

    def params(slot: int, n_tokens: int) -> tinker.SamplingParams:
        return tinker.SamplingParams(
            max_tokens=n_tokens, temperature=temperature,
            seed=None if seed is None else derive_sample_seed(base_seed, slot),
            **({"stop": renderer.stop_tokens} if getattr(renderer, "stop_tokens", None) else {}),
        )

    def sample_one(ep: _Episode, n_tokens: int):
        """One continuation call for ``ep`` from its current observation, on its own seed slot."""
        if ep.n_calls >= per_ep_calls:
            raise RuntimeError(
                f"episode {ep.index} issued more than {per_ep_calls} sampling calls "
                f"(max_turns={max_turns}) — the per-episode seed block would overflow"
            )
        slot = turn0_slots + ep.index * per_ep_calls + ep.n_calls
        ep.n_calls += 1
        return sampling_client.sample(tinker.ModelInput.from_ints(ep.ob), 1, params(slot, n_tokens))

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
        """Hand the turn to the env and extend the observation; returns False when the episode ends.

        The env step is the out-of-process part (running the policy's command), so it is the one
        thing held under a semaphore — ``step_workers`` concurrent commands across the whole batch.
        """
        with step_sem:
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
                seq = sample_one(ep, first_call_tokens).result().sequences[0]
            tokens = list(seq.tokens)
            if think_budget and str(seq.stop_reason) != "stop" and renderer.in_open_think(tokens):
                # Cut off inside <think>: close it with the renderer's forcing suffix (appended as
                # OBSERVATION tokens, masked in training) and sample the answer with a fresh budget.
                record(ep, seq)
                forced = list(renderer.force_answer_tokens())
                ep.ob = ep.ob + tokens + forced
                ep.turn_tokens.extend(forced)
                ep.n_forced += 1
                seq = sample_one(ep, answer_tokens).result().sequences[0]
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

    # -- turn 0: issued up front so the whole batch is in flight at once. Seeded → one single-sample
    # request PER EPISODE with its own slot seed (a seeded n-sample request collapses the group to
    # ~1 distinct sequence — tinker seeds the whole request; see rollout.sample_rollouts). Unseeded →
    # one n-sample request per prompt. Either way group_futures[i].result().sequences is prompt i's
    # group of ``num_samples`` sequences.
    def _turn0(i: int, p: Prompt):
        mi = renderer.model_input(p.text)
        if seed is None:
            return _SharedFuture(sampling_client.sample(mi, num_samples, params(i, first_call_tokens)))
        return _SharedFuture(_Gathered([sampling_client.sample(mi, 1, params(i * num_samples + k, first_call_tokens))
                                        for k in range(num_samples)]))

    group_futures = [_turn0(i, p) for i, p in enumerate(prompts)]
    prompt_tokens = [list(renderer.prompt_tokens(p.text)) for p in prompts]

    out: list[Rollout | None] = [None] * n_episodes

    def run_one(index: int) -> None:
        """Entry point of one episode's thread: take its slice of the group's turn-0 call, then drive
        it to completion and publish it."""
        p_i, k = divmod(index, num_samples)
        seqs = list(group_futures[p_i].result().sequences)
        if len(seqs) != num_samples:
            raise RuntimeError(f"asked for {num_samples} samples, got {len(seqs)}")
        ep = _Episode(index=index, prompt=prompts[p_i], state=env.start(prompts[p_i]),
                      ob=list(prompt_tokens[p_i]))
        rollout = drive(ep, seqs[k])
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
