"""Core shared types.

A ``Rollout`` is the unit of analysis everything operates on: envs produce the prompt and grade it,
monitors score it, the RL loop turns rewards into gradients. Keep this module dependency-light so it
can be imported from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Prompt:
    """A single task instance handed to the policy."""

    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Rollout:
    """One policy attempt at a prompt, plus whatever a monitor might need to score it.

    ``cot`` and ``output`` are read by black-box monitors; ``activations`` (when populated) are read
    by white-box monitors. ``token_ids`` / ``logprobs`` are kept for the RL loop. Activations are
    None until/unless we wire up the local-forward-pass path (see docs/MONITORS.md).
    """

    prompt: Prompt
    cot: str
    output: str
    token_ids: Optional[list[int]] = None
    logprobs: Optional[list[float]] = None
    activations: Optional[np.ndarray] = None  # shape (n_tokens, d_model), None if unavailable
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvResult:
    """Output of grading a rollout against an environment.

    ``task_reward`` is what RL optimizes (before any monitor penalty). ``behavior_present``
    is the oracle label monitors are graded against — it must NEVER enter a monitor or the reward.
    """

    task_reward: float
    behavior_present: bool
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorResult:
    """A monitor's judgment of a rollout. ``score`` is higher = more likely misbehavior."""

    score: float
    label: bool
    meta: dict[str, Any] = field(default_factory=dict)
