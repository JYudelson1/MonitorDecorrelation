"""RL backend abstraction.

The backend is the *substrate*: it samples rollouts from the policy and applies a gradient step
given per-rollout (or per-token) advantages. The RL *algorithm* (GRPO etc.) lives one layer up in
``rl/`` and is backend-agnostic — it computes advantages from rewards and hands them down.

Backends:
- ``tinker`` — the only implemented backend (raw SDK, minimal usage).
- ``verl`` — stub for later; the escape hatch if we need a policy tinker doesn't host (e.g. a small
  DeepSeek-R1 distill). Mildly painful setup, deferred until actually needed.

See docs/INFRA.md.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from monitordecorrelation.types import Prompt, Rollout


@runtime_checkable
class RLBackend(Protocol):
    name: str

    def sample(self, prompts: list[Prompt], **sampling_params: Any) -> list[Rollout]:
        """Generate completions (with token_ids + logprobs) for the given prompts."""
        ...

    def train_step(self, rollouts: list[Rollout], advantages: list[float]) -> dict[str, float]:
        """Apply one policy-gradient update given per-rollout advantages. Returns train metrics."""
        ...

    def save_checkpoint(self, label: str) -> str:
        """Persist current policy weights; returns a handle/path."""
        ...
