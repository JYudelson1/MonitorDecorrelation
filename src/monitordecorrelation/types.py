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
class Segment:
    """A stretch of a completion that the policy actually sampled, preceded by tokens *we* injected.

    Only non-trivial when generation was interrupted and resumed — today that means a thinking budget
    (``rl/thinking_budget.py``), whose forced ``</think>`` text lands in ``injected``. The split
    matters for training: injected tokens are part of the *observation* the policy was resumed from,
    never an action it chose, so ``rl/grpo.py`` masks them out of the loss.
    """

    injected: list[int]  # spliced in by us, not sampled (no logprobs, not trained on)
    tokens: list[int]  # sampled by the policy
    logprobs: list[float]  # sampling logprobs of ``tokens`` (same length)


@dataclass
class Rollout:
    """One policy attempt at a prompt, plus whatever a monitor might need to score it.

    ``cot`` and ``output`` are read by black-box monitors; ``activations`` (when populated) are read
    by white-box monitors. ``token_ids`` / ``logprobs`` are kept for the RL loop. Activations are
    None until/unless we wire up the local-forward-pass path (see docs/MONITORS.md).

    ``segments`` is None for an ordinary one-shot completion (the overwhelmingly common case) and
    only set when generation was interrupted and resumed; ``token_ids`` is always the full sequence
    either way, with ``logprobs`` NaN at injected positions (those tokens were never sampled, so
    there is no logprob to report — and nothing may treat one as if there were).
    """

    prompt: Prompt
    cot: str
    output: str
    token_ids: Optional[list[int]] = None
    logprobs: Optional[list[float]] = None
    activations: Optional[np.ndarray] = None  # shape (n_tokens, d_model), None if unavailable
    segments: Optional[list[Segment]] = None
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
