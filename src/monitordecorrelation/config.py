"""Run configuration.

Explicit ``train_against`` vs ``held_out`` monitor lists so we never accidentally train on a
held-out detector. See docs/EXPERIMENT_PLAN.md and docs/INFRA.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MonitorSpec:
    """How to construct one monitor. ``kind`` selects the family; ``model_id`` sets strength."""

    name: str
    kind: str  # "cot" | "output" | "probe" | ...
    provider: str = "openrouter"  # judges run on OpenRouter (has Claude + wide family/size spread)
    model_id: str | None = None  # for black-box judges
    checkpoint: str | None = None  # for probes
    threshold: float = 0.5


@dataclass
class LoggingConfig:
    """Local/offline wandb + rollout-sampling for the metric curves."""

    use_wandb: bool = True
    wandb_mode: str = "offline"  # local by default; "online" to sync
    wandb_project: str = "monitor-decorrelation"
    run_name: str | None = None
    log_fraction: float = 1.0  # fraction of rollouts to persist verbatim (1.0 = everything)


@dataclass
class RunConfig:
    env: str = "sycophancy"

    # Policy (the RL'd model). Must natively verbalize CoT — Qwen3-8B has hybrid thinking.
    # NB: on tinker, "Qwen3-4B-Instruct-2507" is the *non-thinking* variant, so 8B is the small
    # verbalizing default. See docs/INFRA.md.
    backend: str = "tinker"
    base_model: str = "Qwen/Qwen3-8B"

    # Detectors. The crux invariant: held_out never enters the gradient.
    train_against: list[MonitorSpec] = field(default_factory=list)
    held_out: list[MonitorSpec] = field(default_factory=list)

    # RL algorithm
    rl_algo: str = "grpo"  # "grpo" | "dr_grpo" | "dapo"
    group_size: int = 8  # rollouts per prompt for group-relative advantage
    penalty_coef: float = 1.0  # λ on the monitor penalty
    kl_coef: float = 0.0

    # Optimization
    batch_size: int = 32
    n_steps: int = 100
    eval_every: int = 10
    lora_rank: int = 16
    learning_rate: float = 1e-5
    seed: int = 0

    logging: LoggingConfig = field(default_factory=LoggingConfig)
