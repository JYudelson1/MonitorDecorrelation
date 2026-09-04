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
    wandb_project: str = "monitor-correlation-impossible-bench"
    wandb_group: str | None = None  # W&B run group — share across a sweep (matrix × seeds) to cluster
    wandb_tags: list[str] = field(default_factory=list)  # e.g. [env, model, "seed0", train-against name]
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
    penalty_coef: float = 1.0  # λ on the monitor penalty (constant unless penalty_schedule is set)
    penalty_schedule: dict | None = None  # None → constant penalty_coef; else {"start_penalty","end_penalty"}
    #   → λ ramps linearly start→end across n_steps (hack-then-hide curriculum). Overrides penalty_coef.
    kl_coef: float = 0.0  # per-token KL-to-base penalty (anchors the policy; 0 = off)
    kl_discount_factor: float = 0.0
    # Length penalty: reward −= length_penalty_coef · (completion tokens, CoT included). 0 = off.
    # Reward units per token, so 1e-4 costs a 1024-token completion 0.1 of a [0,1] task reward.
    length_penalty_coef: float = 0.0

    # Optimization
    batch_size: int = 32
    n_steps: int = 100
    eval_every: int = 10  # run the held-out eval phase every N train steps (+ step 0 + final)
    eval_size: int = 32  # number of distinct held-out eval PROMPTS (fixed set)
    eval_samples_per_prompt: int = 1  # rollouts per eval prompt (>1 → higher-fidelity AUROC/d′ per eval)
    lora_rank: int = 16
    learning_rate: float = 1e-5
    seed: int = 0
    save_every: int = 500  # save checkpoint every N train steps
    # Recorded for the run's own bookkeeping (W&B config + run_info.json); the budget itself is
    # enforced by the backend at sampling time, not by the training loop. None = off.
    thinking_budget: int | None = None

    logging: LoggingConfig = field(default_factory=LoggingConfig)
