"""The generic Env seam.

An environment supplies prompts and grades rollouts, producing both the RL ``task_reward`` and the
``behavior_present`` oracle. See docs/ENVIRONMENTS.md.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from monitordecorrelation.types import EnvResult, Prompt, Rollout


def is_truncated(rollout: Rollout) -> bool:
    """Sampling stopped on ``max_tokens`` rather than end-of-turn. For a multi-turn episode
    ``stop_reason`` is the last turn's — and a truncated turn always ends the episode."""
    stop = (rollout.meta or {}).get("stop_reason")
    return stop is not None and stop != "stop"


def invalid_reason(env: Any, rollout: Rollout) -> str | None:
    """Why a rollout is INVALID — ``"truncated"`` (cut off by max_tokens) or ``"unparsed"`` (the env
    could not parse its output) — or None if it is valid.

    An invalid rollout gets the flat ``INVALID_ROLLOUT_REWARD`` and is **never shown to a monitor**,
    in training or eval: every monitor statistic (AUROC, d′, class means, the train-against
    suspiciousness) is over the valid rollouts only. Decidable from the rollout alone, so the RL loop
    can skip the judge calls the moment a rollout is sampled, before the env grades it."""
    if is_truncated(rollout):
        return "truncated"
    check = getattr(env, "unparseable", None)
    if check is not None and check(rollout):
        return "unparsed"
    return None


@runtime_checkable
class Env(Protocol):
    """An RL setting with a ground-truth behavior signal."""

    name: str
    behavior_name: str  # the target behavior this env detects, e.g. "sycophancy" — passed to monitors

    def sample_prompt(self) -> Prompt: ...

    def score(self, rollout: Rollout) -> EnvResult: ...

    def score_batch(self, rollouts: list[Rollout]) -> list[EnvResult]:
        """Grade a whole batch at once. Optional — the RL loop calls it when present (``rl/train.py``
        ``_score_env``) and otherwise loops over ``score``. Implement it when grading is expensive and
        parallelizable (e.g. executing generated code in subprocesses)."""
        ...

    def holdout(self, n: int, seed: int = 0) -> list[Prompt]:
        """Pull a fixed held-out eval set of ``n`` prompts and REMOVE them from the training pool
        (eval ⟂ train). Optional — runners fall back to overlapping sampling if absent."""
        ...

    def unparseable(self, rollout: Rollout) -> bool:
        """Could the rollout's output NOT be parsed (no codeblock / answer letter / ``<command>``)?
        Optional — an env without it has no parse failures. Must be cheap and pure (it runs on the
        sampling threads, before ``score``) and agree with ``score``'s ``meta["unparsed"]``; the RL
        loop checks that. See ``invalid_reason``."""
        ...


@runtime_checkable
class MultiTurnEnv(Env, Protocol):
    """A tool-loop env: the policy acts over several turns, the env replies after each one.

    ``multi_turn = True`` is the flag the RL loop dispatches on (``rl/train.py::_sample`` →
    ``backend.sample_episodes`` → ``rl/episodes.py::run_episodes``). The driver threads an opaque
    ``state`` through ``start`` → ``step``* → ``finish``; ``finish`` returns the monitor-facing
    ``cot``/``output`` views plus the grading record, which the driver stores at
    ``Rollout.meta["episode"]`` so ``score`` stays a pure function of the rollout.
    """

    multi_turn: bool
    max_turns: int

    def start(self, prompt: Prompt) -> Any: ...

    def step(self, state: Any, cot: str, text: str, *, truncated: bool = False) -> tuple[str | None, bool]:
        """Consume one assistant turn -> (next user message or None, episode done)."""
        ...

    def finish(self, state: Any) -> Any:
        """Close the episode -> an object with ``.cot``, ``.output`` (str) and ``.meta`` (dict)."""
        ...
