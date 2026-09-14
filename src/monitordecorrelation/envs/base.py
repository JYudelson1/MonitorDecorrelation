"""The generic Env seam.

An environment supplies prompts and grades rollouts, producing both the RL ``task_reward`` and the
``behavior_present`` oracle. See docs/ENVIRONMENTS.md.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from monitordecorrelation.types import EnvResult, Prompt, Rollout


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
