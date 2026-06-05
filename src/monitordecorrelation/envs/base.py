"""The generic Env seam.

An environment supplies prompts and grades rollouts, producing both the RL ``task_reward`` and the
``behavior_present`` oracle. See docs/ENVIRONMENTS.md.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from monitordecorrelation.types import EnvResult, Prompt, Rollout


@runtime_checkable
class Env(Protocol):
    """An RL setting with a ground-truth behavior signal."""

    name: str
    behavior_name: str  # the target behavior this env detects, e.g. "sycophancy" — passed to monitors

    def sample_prompt(self) -> Prompt: ...

    def score(self, rollout: Rollout) -> EnvResult: ...
