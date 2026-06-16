"""Env factory — the one place that maps a config's ``env`` name to a constructed environment.

Keeps ``run_experiment`` env-agnostic: it asks for ``make_env(cfg)`` and gets back something with the
``Env`` surface (``sample_prompt``/``holdout``/``score`` + ``name``/``behavior_name``). Add a new
setting by adding a branch here (and a value to ``ExperimentConfig.env``'s Literal).

Both envs draw their prompt pool from ``cfg.n_prompts_pool`` and seed off ``cfg.seed``; ``cfg.subset``
only means something for sycophancy (which political/nlp/… slice), so MBPP ignores it.

NOTE (leakage / task #25 split): MBPP's ``holdout`` currently samples random items, NOT coordinated
with which task_ids the iid MBPP probe trained on. The iid-probe-vs-eval split must be unified before a
real run — tracked separately; this factory only handles construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monitordecorrelation.experiment_config import ExperimentConfig


def make_env(cfg: "ExperimentConfig"):
    """Construct the environment named by ``cfg.env``. Raises ValueError on an unknown name."""
    if cfg.env == "sycophancy":
        from monitordecorrelation.envs.sycophancy import SycophancyQAEnv

        return SycophancyQAEnv.from_subset(cfg.subset, n=cfg.n_prompts_pool, seed=cfg.seed)
    if cfg.env == "mbpp_honeypot":
        from monitordecorrelation.envs.mbpp_honeypot import MbppHoneypotEnv

        return MbppHoneypotEnv.from_dataset(n=cfg.n_prompts_pool, seed=cfg.seed)
    raise ValueError(f"unknown env {cfg.env!r}")  # pragma: no cover (pydantic Literal guards this)
