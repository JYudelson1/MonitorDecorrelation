"""Multi-turn episode driver: render the conversation → sample a turn → hand it to the env → append
the env's replies → repeat.

Single-turn envs sample one completion per prompt (``rl/rollout.py``). Tool-loop envs (the terminal
env) need an agent loop. This one follows tinker-cookbook's own tool-use loop
(``tinker_cookbook.rl.message_env.EnvFromMessageEnv`` + ``tool_use.AgentToolMessageEnv``) and uses its
primitives for everything model-specific, via the policy's cookbook renderer
(``TmlRenderer.chat_renderer``, ``tml_v0`` for Inkling):

- the conversation is a list of cookbook ``Message``s: ``create_conversation_prefix_with_tools(env
  tools)`` + the task as a user message, then per turn the parsed assistant message and the env's
  replies (``tool`` results, and possibly a ``user`` message);
- every observation is ``build_generation_prompt(conversation, effort=…)`` — the WHOLE conversation
  re-rendered, as the cookbook does, never tokens we splice together ourselves;
- every sampled turn is parsed with ``parse_response`` (thinking, visible text, ``tool_calls``) and the
  env runs its tool call through the cookbook's ``handle_tool_call`` (``envs/terminal_verifier.py``).

Generic over any env implementing the ``MultiTurnEnv`` protocol (``check_policy`` / ``tool_specs`` /
``start`` / ``step`` / ``finish`` — see ``envs/base.py``).

Concurrency: **every episode runs free**
--------------------------------------
Each episode gets its own driver thread and advances as fast as its own turns resolve — it never
waits for any other episode. An episode that finishes its command in 20 ms issues its next
generation immediately, while a sibling is still blocked on a 30-second command timeout or a long
generation.

What that means for the pieces:

- **generations** — ``sampling_client.sample`` schedules onto tinker's background event loop and
  returns a ``concurrent.futures.Future``, so calling it from many threads (and awaiting from many
  threads) is safe and genuinely parallel. EVERY request is a single sample — never one
  ``num_samples=G`` request per group: tinker applies one seed to a whole request, so a seeded
  8-sample request returns ~1.4 distinct sequences (measured, base Qwen3-8B), collapsing the GRPO
  group. Turn 0 is one request per episode, issued up front for the whole batch; each episode waits
  only for its own (never for its siblings'); every later turn is one call issued by that episode.
- **env steps** (tool execution) — run in the episode's own thread, uncapped by the driver: every
  episode whose turn is ready runs its command at once. The only bound is the env's own — the
  terminal env takes a cross-process ``globalsem.code_exec_slot`` per command (half the box's
  cores, shared by every run on the box).
- **monitors** — ``on_rollout(index, rollout)`` fires the moment an episode is graded and flattened,
  from that episode's thread, so a caller (``rl/train.py``) can start its judge API calls for a
  finished episode while the rest of the batch is still generating.

The returned list keeps the original order (prompt-major, ``num_samples`` consecutive per prompt)
regardless of who finished first, so GRPO grouping downstream is unchanged.

What comes out is an ordinary ``Rollout`` with two extras in ``meta``:

- ``transitions``: ``[{"ob": [tokens], "ac": [tokens], "logprobs": [...]}, …]``, one per turn —
  ``ob`` is exactly the prompt that turn was sampled from, ``ac`` exactly what was sampled.
  ``rl/grpo.py`` hands them to tinker-cookbook's ``trajectory_to_data``, which folds consecutive
  transitions into ONE datum whenever an ``ob`` extends the previous ``ob + ac`` (observation tokens
  masked, every action token carrying the episode's advantage) and starts a new datum where it does
  not. With Inkling's ``tml_v0`` renderer the re-rendered conversation does extend it — each message is
  framed independently, earlier thinking is kept, and a parsed turn re-renders to the very tokens that
  were sampled (the cookbook declares ``has_extension_property``; checked per turn here) — so an
  episode is normally one datum. ``n_prefix_breaks`` counts the turns where it was not (e.g. a tool
  call whose JSON the model spaced differently from the renderer's canonical form); nothing is lost
  there, the episode just costs more than one datum.
- ``episode``: the env's grading record (``finish()``'s meta) — ``score()`` is then a pure function
  of the rollout, so eval/train scoring code paths stay unchanged.

A turn cut off by ``max_tokens`` ends the episode (as in the cookbook's ``EnvFromMessageEnv``); its
partial thinking / text is recovered with the policy renderer's streaming parser for the record.

Seeding: every sampling call gets its own seed, a pure function of its **position** —
``derive_sample_seed(seed, group, sample, turn)``: the prompt's index in the batch, the episode's
index within that prompt's GRPO group, and the turn's index within the episode — never of the order
calls happen to be issued in. So the seeds are identical whatever the thread interleaving, two
episodes that reached an identical context still get different continuations (which would otherwise
silently collapse a GRPO group's advantage variance), and a run stays reproducible from ``cfg.seed``
alone (the RL loop derives ``seed`` from the run seed, the phase and the RL step — see
``rl/train.py``).
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
    messages: list                       # the conversation so far (cookbook Messages)
    ob: list[int]                        # the rendered conversation the NEXT sampling call starts from
    transitions: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    n_turns: int = 0
    n_truncated: int = 0                 # turns cut off by max_tokens (at most 1: it ends the episode)
    n_prefix_breaks: int = 0             # turns whose ob did not extend the previous ob + ac


def _only_sequence(response, what: str):
    """The one sequence of a single-sample response (every request here asks for exactly one)."""
    seqs = list(response.sequences)
    if len(seqs) != 1:
        raise RuntimeError(f"{what}: asked for 1 sample, got {len(seqs)}")
    return seqs[0]


def _train_tokens(transitions: list[dict]) -> int:
    """Tokens in the training data these transitions make: one datum per run of transitions whose ob
    extends the previous ob + ac (tinker-cookbook ``trajectory_to_data``), each as long as its last
    ob + ac."""
    total, prev = 0, None
    for tr in transitions:
        if prev is not None and tr["ob"][: len(prev)] != prev:
            total += len(prev)  # the datum so far ends here
        prev = tr["ob"] + tr["ac"]
    return total + (len(prev) if prev is not None else 0)


def run_episodes(
    sampling_client,
    renderer,
    env,
    prompts: list[Prompt],
    *,
    num_samples: int = 1,
    max_tokens: int,
    temperature: float = 1.0,
    seed: int | None = None,
    max_turns: int | None = None,
    episode_workers: int | None = None,
    on_rollout: Callable[[int, Rollout], None] | None = None,
) -> list[Rollout]:
    """Run ``num_samples`` episodes per prompt (the GRPO group, kept consecutive in the output) and
    return one ``Rollout`` per episode. Every episode runs in its own thread and advances
    independently (see the module docstring). ``max_turns`` defaults to ``env.max_turns``; each turn
    is one sampling call of ``max_tokens``.

    ``renderer`` is the policy's renderer (``rl.renderers.make_renderer``); the conversation is
    rendered and parsed by its tinker-cookbook ``chat_renderer`` at its reasoning ``effort``. The env
    decides which policies it runs (``env.check_policy``) — the terminal env only Inkling /
    Inkling-Small, whose native tool calls it is built on.

    ``episode_workers`` caps how many episodes are in flight; the default — one thread per episode —
    is what makes the batch fully decoupled. ``on_rollout(index, rollout)`` is called from the
    episode's own thread as soon as that episode is finished and flattened, so callers can start
    per-rollout work (monitor API calls) without waiting for the batch; ``index`` is the rollout's
    position in the returned list.
    """
    model_name = getattr(renderer, "model_name", None)
    if model_name is None:
        raise ValueError(f"multi-turn episodes need a TML (Inkling) renderer, got {type(renderer).__name__}")
    env.check_policy(model_name)
    chat = renderer.chat_renderer
    effort = renderer.effort
    max_turns = max_turns or getattr(env, "max_turns", None)
    if not max_turns or max_turns < 1:
        raise ValueError("run_episodes needs max_turns >= 1 (from the env or the argument)")
    if max_tokens is None or max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens!r}")
    n_episodes = len(prompts) * num_samples
    stop = chat.get_stop_sequences()
    tool_prefix = chat.create_conversation_prefix_with_tools(env.tool_specs(), system_prompt="")

    def params(position: tuple[int, int, int]) -> tinker.SamplingParams:
        """``position`` = (group, sample, turn) of the call — its seed, when the batch is seeded."""
        return tinker.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, stop=stop,
            seed=None if seed is None else derive_sample_seed(seed, *position),
        )

    def render(messages: list) -> list[int]:
        return chat.build_generation_prompt(messages, effort=effort).to_ints()

    def drive(ep: _Episode, first_seq) -> Rollout:
        """The whole life of ONE episode: turn → env step → turn → … → finish. Runs in its own
        thread, touching nothing another episode owns."""
        seq = first_seq
        group, k = divmod(ep.index, num_samples)
        for turn in range(max_turns):
            if turn:
                ob = render(ep.messages)
                prev = ep.transitions[-1]
                ep.n_prefix_breaks += ob[: len(prev["ob"]) + len(prev["ac"])] != prev["ob"] + prev["ac"]
                ep.ob = ob
                seq = _only_sequence(
                    sampling_client.sample(tinker.ModelInput.from_ints(ob), 1, params((group, k, turn))).result(),
                    f"episode {ep.index}, turn {turn}",
                )
            tokens = list(seq.tokens)
            if seq.logprobs is None:
                raise RuntimeError("sampler returned no logprobs — GRPO needs sampling logprobs")
            ep.transitions.append({"ob": list(ep.ob), "ac": tokens, "logprobs": list(seq.logprobs)})
            ep.stop_reason = str(seq.stop_reason)
            ep.n_turns += 1
            truncated = ep.stop_reason != "stop"
            message, termination = chat.parse_response(tokens)
            parse_error = not truncated and not termination.is_clean
            if truncated or parse_error:
                # No clean message to continue from: keep what the streaming parser recovers of the
                # turn (thinking / text up to the break) for the record; the env ends the episode.
                cot, text, _raw = renderer.parse(tokens)
                message = {"role": "assistant",
                           "content": [{"type": "thinking", "thinking": cot}, {"type": "text", "text": text}]}
            ep.n_truncated += int(truncated)
            replies, done = env.step(ep.state, message, truncated=truncated, parse_error=parse_error)
            if done:
                break
            ep.messages = ep.messages + [message] + list(replies)
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
                # Token accounting (one sampling call per transition).
                "n_sampling_calls": len(ep.transitions),
                "input_tokens": sum(len(tr["ob"]) for tr in ep.transitions),
                "output_tokens": sum(len(tr["ac"]) for tr in ep.transitions),
                "train_tokens": _train_tokens(ep.transitions),
                "n_prefix_breaks": ep.n_prefix_breaks,
                "n_truncated_turns": ep.n_truncated,
                "transitions": ep.transitions,
                "episode": view.meta,
            },
        )

    # -- turn 0: issued up front so the whole batch is in flight at once — one single-sample request
    # PER EPISODE (index i*num_samples + k), seeded by its position (i, k, turn 0).
    conversations = [tool_prefix + [{"role": "user", "content": p.text}] for p in prompts]
    prompt_tokens = [render(c) for c in conversations]
    turn0 = []
    for i, toks in enumerate(prompt_tokens):
        mi = tinker.ModelInput.from_ints(toks)
        turn0 += [sampling_client.sample(mi, 1, params((i, k, 0))) for k in range(num_samples)]

    out: list[Rollout | None] = [None] * n_episodes

    def run_one(index: int) -> None:
        """Entry point of one episode's thread: wait for its own turn-0 request, then drive it to
        completion and publish it."""
        p_i = index // num_samples
        first = _only_sequence(turn0[index].result(), f"episode {index}, turn 0")
        ep = _Episode(index=index, prompt=prompts[p_i], state=env.start(prompts[p_i]),
                      messages=list(conversations[p_i]), ob=list(prompt_tokens[p_i]))
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
