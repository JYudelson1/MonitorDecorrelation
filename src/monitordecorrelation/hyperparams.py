"""Learning-rate estimation.

We deliberately delegate to **Thinking Machines' tinker-cookbook** rather than reimplementing the
formula, so these LRs are unambiguously TM's optimal-LoRA-LR heuristic (from "LoRA Without Regret",
https://thinkingmachines.ai/blog/lora/) — not a sweep or a constant of our own.

``get_lr(model_name, is_lora=True)`` returns the recommended LR (for Qwen it fetches the model's
hidden_size via ``transformers.AutoConfig``, so the first call needs network).
"""

from __future__ import annotations

from tinker_cookbook.hyperparam_utils import get_lr

__all__ = ["get_lr"]
