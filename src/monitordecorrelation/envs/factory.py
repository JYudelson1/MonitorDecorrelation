"""Env factory — the one place that maps a config's ``env`` name to a constructed environment.

Keeps ``run_experiment`` env-agnostic: it asks for ``make_env(cfg)`` and gets back something with the
``Env`` surface (``sample_prompt``/``holdout``/``score`` + ``name``/``behavior_name``). Add a new
setting by adding a branch here (and a value to ``ExperimentConfig.env``'s Literal).

Every env draws its prompt pool from ``cfg.n_prompts_pool`` and seeds off ``cfg.seed``. ``cfg.subset``
selects the slice where that means something — sycophancy (political/nlp/…) and impossiblebench
(impossible = oneoff + conflicting, or either alone) and codeforces_ib (the built dataset file, e.g.
``hardest1024``); MBPP ignores it. ``cfg.env_options`` passes env-specific knobs (impossiblebench and
codeforces_ib) straight to the constructor, which validates them.

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
    if cfg.env_options and cfg.env not in ("impossiblebench", "codeforces_ib"):
        raise ValueError(
            f"env_options is only supported by the impossiblebench and codeforces_ib envs, not {cfg.env!r}")
    if cfg.env == "sycophancy":
        from monitordecorrelation.envs.sycophancy import SycophancyQAEnv

        return SycophancyQAEnv.from_subset(cfg.subset, n=cfg.n_prompts_pool, seed=cfg.seed)
    if cfg.env == "mbpp_honeypot":
        from monitordecorrelation.envs.mbpp_honeypot import MbppHoneypotEnv

        return MbppHoneypotEnv.from_dataset(n=cfg.n_prompts_pool, seed=cfg.seed)
    if cfg.env == "impossiblebench":
        from monitordecorrelation.envs.impossiblebench import ImpossibleBenchEnv

        return ImpossibleBenchEnv.from_dataset(
            subset=cfg.subset, n=cfg.n_prompts_pool, seed=cfg.seed, **(cfg.env_options or {})
        )
    if cfg.env == "codeforces_ib":
        from monitordecorrelation.envs.codeforces_ib import CodeforcesIbEnv

        return CodeforcesIbEnv.from_dataset(
            subset=cfg.subset, n=cfg.n_prompts_pool, seed=cfg.seed, **(cfg.env_options or {})
        )
    raise ValueError(f"unknown env {cfg.env!r}")  # pragma: no cover (pydantic Literal guards this)
