"""Experiment: train against a CoT monitor on sycophancy, measure a held-out CoT monitor.

This *is* the first train-against-CoT experiment (a degradation read in its own right): RL Qwen3-8B
against a weak CoT monitor while tracking (a) the ground-truth sycophancy rate and (b) a held-out
stronger CoT monitor's accuracy/AUROC over training. Defaults are a modest-but-real size; tune via
flags. Learning rate is auto-estimated for the policy (tinker-cookbook formula) unless overridden.

Run: uv run python experiments/train_against_cot.py
Plot: uv run python experiments/plot_run.py --run train_against_cot
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from monitordecorrelation.backends.tinker_backend import TinkerBackend
from monitordecorrelation.config import LoggingConfig, RunConfig
from monitordecorrelation.envs.sycophancy import SycophancyQAEnv
from monitordecorrelation.hyperparams import get_lr
from monitordecorrelation.monitors.cot_monitor import CoTMonitor
from monitordecorrelation.rl.train import run_grpo

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="train_against_cot")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16, help="distinct prompts per step")
    ap.add_argument("--group-size", type=int, default=8, help="rollouts per prompt (GRPO group)")
    ap.add_argument("--subset", default="nlp", help="sycophancy subset (nlp has the best monitor signal)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="≥1024: nlp truncates at 768")
    ap.add_argument("--penalty-coef", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="override; default = estimated LoRA LR")
    # Which monitor to train against; the other is held out. Names stay tied to the MODEL
    # (cot_weak=weak model, cot_strong=strong model) across both choices, so runs are comparable.
    ap.add_argument("--train-target", choices=["weak", "strong"], default="weak")
    # On nlp: claude-3.5-haiku ~0.60 AUROC (weak), claude-sonnet-4.6 ~0.80-0.94 (strong).
    ap.add_argument("--weak-monitor", default="anthropic/claude-3.5-haiku")
    ap.add_argument("--strong-monitor", default="anthropic/claude-sonnet-4.6")
    args = ap.parse_args()

    lr = args.lr if args.lr is not None else get_lr(args.base_model)
    print(f"policy={args.base_model}  lr={lr:.2e}  "
          f"{args.batch_size}x{args.group_size} rollouts/step x {args.n_steps} steps")

    cfg = RunConfig(
        base_model=args.base_model,
        rl_algo="grpo",
        batch_size=args.batch_size,
        group_size=args.group_size,
        n_steps=args.n_steps,
        penalty_coef=args.penalty_coef,
        lora_rank=args.lora_rank,
        learning_rate=lr,
        logging=LoggingConfig(run_name=args.run_name, wandb_mode="offline", log_fraction=1.0),
    )

    env = SycophancyQAEnv.from_subset(args.subset, n=512, seed=cfg.seed)
    backend = TinkerBackend(cfg.base_model, lora_rank=cfg.lora_rank, learning_rate=cfg.learning_rate)

    # Names are tied to the model (not the role), so cot_weak/cot_strong mean the same thing whether
    # we train against the weak or the strong one. --train-target picks which enters the gradient.
    # Behavior is passed through so the monitor's rubric targets the env's behavior, not hardcoded.
    cot_weak = CoTMonitor(name="cot_weak", model_id=args.weak_monitor, behavior=env.behavior_name)
    cot_strong = CoTMonitor(name="cot_strong", model_id=args.strong_monitor, behavior=env.behavior_name)
    if args.train_target == "weak":
        train_against, held_out = [cot_weak], [cot_strong]
    else:
        train_against, held_out = [cot_strong], [cot_weak]
    print(f"train-against: {train_against[0].name} ({train_against[0].model_id})  |  "
          f"held-out: {held_out[0].name} ({held_out[0].model_id})")

    run_grpo(
        cfg, env, backend, train_against, held_out,
        max_tokens=args.max_tokens,
        run_info={"experiment": "train_against_cot", "subset": args.subset, "lr": lr,
                  "train_target": args.train_target},
    )
    print("\ntrain_against_cot finished OK")


if __name__ == "__main__":
    main()
