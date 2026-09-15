"""Multi-turn episode driver: sample a turn → hand it to the env → append its observation → repeat.

Single-turn envs sample one completion per prompt (``rl/rollout.py``). Tool-loop envs (the terminal
env) need an agent loop: the env executes the turn's command and produces the next user message,
which is rendered and appended to the SAME token sequence the policy just produced, and the policy
continues. This module is that loop, generic over any env implementing the ``MultiTurnEnv`` protocol
(``start`` / ``step`` / ``finish`` — see ``envs/base.py``).

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

Seeding: every sampling call gets its own derived seed (prompt index at turn 0, then per-episode per
call), so two episodes that reached an identical context never get identical continuations — which
would silently collapse a GRPO group's advantage variance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import tinker

from monitordecorrelation.types import Prompt, Rollout


def derive_sample_seed(base_seed: int, call_index: int) -> int:
    """A unique, reproducible seed per sampling call: same run seed → same sequence of call seeds, but
    each call differs (so GRPO groups vary across steps instead of collapsing to one group every step).
    Pure + deterministic so it's unit-testable without tinker."""
    return (base_seed * 1_000_003 + call_index) % (2**31 - 1)


@dataclass
class _Episode:
    prompt: Prompt
    state: Any
    ob: list[int]                        # the observation tokens the NEXT sampling call starts from
    transitions: list[dict] = field(default_factory=list)
    done: bool = False
    stop_reason: str = ""
    turn_tokens: list[int] = field(default_factory=list)   # this turn's tokens (all segments)
    pending: tuple[str, str, bool] | None = None           # (cot, text, truncated) awaiting env.step
    n_turns: int = 0
    n_forced: int = 0                                      # turns whose thinking was budget-forced
    n_truncated: int = 0                                   # turns whose FINAL segment hit max_tokens


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
) -> list[Rollout]:
    """Run ``num_samples`` episodes per prompt (the GRPO group, kept consecutive in the output) and
    return one ``Rollout`` per episode. ``max_turns`` defaults to ``env.max_turns``. With
    ``think_budget`` set, each turn's thinking is capped at that many tokens (then force-closed) and
    the answer gets ``answer_tokens``; otherwise a turn is one call of ``max_tokens``."""
    max_turns = max_turns or getattr(env, "max_turns", None)
    if not max_turns or max_turns < 1:
        raise ValueError("run_episodes needs max_turns >= 1 (from the env or the argument)")
    if think_budget is not None and think_budget < 1:
        raise ValueError("think_budget must be >= 1 (or None for no budget)")
    base_seed = 0 if seed is None else seed
    first_call_tokens = think_budget if think_budget else max_tokens
    calls = 0

    def params(n_tokens: int) -> tinker.SamplingParams:
        nonlocal calls
        calls += 1
        return tinker.SamplingParams(
            max_tokens=n_tokens, temperature=temperature,
            seed=None if seed is None else derive_sample_seed(base_seed, calls - 1),
            **({"stop": renderer.stop_tokens} if getattr(renderer, "stop_tokens", None) else {}),
        )

    def sample_one(ep: _Episode, n_tokens: int):
        return sampling_client.sample(tinker.ModelInput.from_ints(ep.ob), 1, params(n_tokens))

    def finish_turn(ep: _Episode, seq) -> None:
        """Absorb the LAST segment of a turn: record it, then parse the whole turn for the env."""
        _record(ep, seq)
        ep.stop_reason = str(seq.stop_reason)
        truncated = ep.stop_reason != "stop"
        cot, text, raw = renderer.parse(ep.turn_tokens)
        if truncated and not cot and "<think>" in raw and "</think>" not in raw:
            cot, text = text, ""  # cut off mid-thought: what parse() called "answer" is the thinking
        ep.pending = (cot, text, truncated)
        ep.n_turns += 1
        ep.n_truncated += int(truncated)

    def budget_round(eps: list[tuple[_Episode, Any]]) -> None:
        """Given (episode, first-segment seq) pairs: turns cut off inside <think> get the forcing
        suffix + an answer continuation; the rest are complete."""
        second = []
        for ep, seq in eps:
            tokens = list(seq.tokens)
            if think_budget and str(seq.stop_reason) != "stop" and renderer.in_open_think(tokens):
                _record(ep, seq)
                ep.ob = ep.ob + tokens + list(renderer.force_answer_tokens())  # forced = observation
                ep.turn_tokens.extend(renderer.force_answer_tokens())
                ep.n_forced += 1
                second.append((ep, sample_one(ep, answer_tokens)))
            else:
                finish_turn(ep, seq)
        for ep, fut in second:
            finish_turn(ep, list(fut.result().sequences)[0])

    # -- turn 0: one call per prompt, num_samples sequences → num_samples episodes (the group) --------
    episodes: list[_Episode] = []
    futures = []
    for p in prompts:
        ob = list(renderer.prompt_tokens(p.text))
        futures.append((p, ob, sampling_client.sample(renderer.model_input(p.text), num_samples,
                                                      params(first_call_tokens))))
    first_round = []
    for p, ob, fut in futures:
        seqs = list(fut.result().sequences)
        if len(seqs) != num_samples:
            raise RuntimeError(f"asked for {num_samples} samples, got {len(seqs)}")
        for seq in seqs:
            ep = _Episode(prompt=p, state=env.start(p), ob=ob)
            episodes.append(ep)
            first_round.append((ep, seq))
    budget_round(first_round)

    # Command execution is out-of-process work → run the env steps of a turn concurrently.
    with ThreadPoolExecutor(max_workers=max(1, step_workers)) as ex:
        _run_pending_steps(episodes, env, renderer, ex)
        for _turn in range(1, max_turns):
            active = [ep for ep in episodes if not ep.done]
            if not active:
                break
            futs = [(ep, sample_one(ep, first_call_tokens)) for ep in active]
            budget_round([(ep, list(fut.result().sequences)[0]) for ep, fut in futs])
            _run_pending_steps(episodes, env, renderer, ex)

    # -- finish: anything still open hit max_turns; flatten to Rollouts ------------------------------
    out: list[Rollout] = []
    for ep in episodes:
        view = env.finish(ep.state)
        out.append(Rollout(
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
        ))
    return out


def _record(ep: _Episode, seq) -> None:
    """Record one sampled segment as a transition from the episode's current observation."""
    tokens = list(seq.tokens)
    if seq.logprobs is None:
        raise RuntimeError("sampler returned no logprobs — GRPO needs sampling logprobs")
    ep.transitions.append({"ob": list(ep.ob), "ac": tokens, "logprobs": list(seq.logprobs)})
    ep.turn_tokens.extend(tokens)


def _run_pending_steps(episodes: list[_Episode], env, renderer, ex: ThreadPoolExecutor) -> None:
    """Env-step every episode with a pending turn (concurrently — commands run out of process), then
    extend the observation of the ones that continue: ob ← ob + sampled turn + inter-turn framing."""
    pending = [ep for ep in episodes if ep.pending is not None]

    def _step(ep: _Episode):
        cot, text, truncated = ep.pending
        ep.pending = None
        obs, done = env.step(ep.state, cot, text, truncated=truncated)
        return ep, obs, done, truncated

    for ep, obs, done, truncated in ex.map(_step, pending):
        ep.turn_tokens = []
        if done or obs is None:
            ep.done = True
            continue
        ac = ep.transitions[-1]["ac"]
        ep.ob = ep.ob + ac + list(renderer.continuation_tokens(obs, ended_cleanly=not truncated))
