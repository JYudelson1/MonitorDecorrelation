"""GRPO training loop for the decorrelation experiments.

reward = task_reward − penalty_coef · mean(train_against monitor scores). Held-out monitors are
scored every step but never enter the reward. The ground-truth behavior rate is logged as the
primary metric (see docs/EXPERIMENT_PLAN.md "Ground truth is the crux").
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

from monitordecorrelation.config import RunConfig
from monitordecorrelation.envs.base import Env
from monitordecorrelation.eval.metrics import MonitorTracker
from monitordecorrelation.monitors.base import Monitor
from monitordecorrelation.rl.grpo import compute_group_advantages
from monitordecorrelation.types import MonitorResult, Rollout


def _monitor_info(m: Monitor, role: str) -> dict:
    """Best-effort record of what a monitor actually is (so 'cot_weak' is decodable later)."""
    return {
        "name": m.name,
        "role": role,
        "model_id": getattr(m, "model_id", None),
        "behavior": getattr(m, "behavior", None),
        "threshold": getattr(m, "threshold", None),
    }


def run_grpo(
    cfg: RunConfig,
    env: Env,
    backend,
    train_against: Sequence[Monitor],
    held_out: Sequence[Monitor] = (),
    *,
    max_tokens: int = 1024,
    monitor_workers: int = 16,
    extra_rollout_fields: Callable[[Rollout, int], dict] | None = None,
    run_info: dict | None = None,
) -> None:
    """Run GRPO. ``extra_rollout_fields(rollout, idx) -> dict`` lets callers attach arbitrary
    per-rollout metadata to saved rollouts. ``run_info`` is merged into the saved ``run_info.json``
    (use it for anything the caller knows but the loop doesn't, e.g. the dataset subset)."""
    rng = random.Random(cfg.seed)
    tracker = MonitorTracker()
    all_monitors: list[Monitor] = [*train_against, *held_out]
    run = None
    if cfg.logging.use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            mode=cfg.logging.wandb_mode,
            config=cfg.__dict__,
        )

    rollout_log_dir = Path("data/runs") / (cfg.logging.run_name or "smoke")
    rollout_log_dir.mkdir(parents=True, exist_ok=True)

    # Durable record of WHAT this run was: config + which monitor is which (train-against vs
    # held-out, model ids, behavior). Without this the logs' cot_weak/cot_strong are undecodable.
    info = {
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_name": cfg.logging.run_name,
        "policy": cfg.base_model,
        "config": dataclasses.asdict(cfg),
        "env": {"name": env.name, "behavior_name": getattr(env, "behavior_name", None)},
        "train_against": [_monitor_info(m, "train_against") for m in train_against],
        "held_out": [_monitor_info(m, "held_out") for m in held_out],
        **(run_info or {}),
    }
    (rollout_log_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    rollout_log = (rollout_log_dir / "rollouts.jsonl").open("w")
    # Always dump per-step metrics to disk too, so plots are viewable locally without a W&B server.
    metrics_log = (rollout_log_dir / "metrics.jsonl").open("w")

    for step in range(cfg.n_steps):
        prompts = [env.sample_prompt() for _ in range(cfg.batch_size)]
        rollouts = backend.sample(
            prompts, num_samples=cfg.group_size, max_tokens=max_tokens, temperature=1.0
        )

        # Score env (oracle + task reward).
        env_results = [env.score(r) for r in rollouts]
        gt = [er.behavior_present for er in env_results]

        # Score every monitor on every rollout once, reused for the penalty AND the degradation
        # metrics. White-box probes expose ``score_batch`` (one batched local forward — threading a
        # single torch model is pointless/unsafe); API judges are scored per-rollout concurrently.
        results_by_name: dict[str, list[MonitorResult]] = {}
        batched = [m for m in all_monitors if hasattr(m, "score_batch")]
        threaded = [m for m in all_monitors if not hasattr(m, "score_batch")]
        for m in batched:
            results_by_name[m.name] = list(m.score_batch(rollouts))
        if threaded:
            for m in threaded:
                results_by_name[m.name] = [None] * len(rollouts)  # type: ignore[list-item]
            with ThreadPoolExecutor(max_workers=monitor_workers) as ex:
                futs = {
                    ex.submit(m.score, rollouts[i]): (m.name, i)
                    for m in threaded
                    for i in range(len(rollouts))
                }
                for fut in futs:
                    name, i = futs[fut]
                    results_by_name[name][i] = fut.result()
        penalties = [
            mean(results_by_name[m.name][i].score for m in train_against) if train_against else 0.0
            for i in range(len(rollouts))
        ]
        rewards = [er.task_reward - cfg.penalty_coef * pen for er, pen in zip(env_results, penalties)]

        advantages = compute_group_advantages(
            rewards, cfg.group_size, normalize=(cfg.rl_algo == "grpo")
        )
        step_metrics = backend.train_step(rollouts, advantages)

        # Rolling per-monitor accuracy + AUROC vs ground truth (train-against AND held-out).
        monitor_metrics: dict[str, float] = {}
        for m in all_monitors:
            res = results_by_name[m.name]
            monitor_metrics.update(
                tracker.update(m.name, [x.score for x in res], [x.label for x in res], gt)
            )

        gt_rate = mean(float(g) for g in gt)
        loss_keys = [k for k in step_metrics if "loss" in k.lower()]
        loss_val = step_metrics[loss_keys[0]] if loss_keys else float("nan")
        log = {
            "step": step,
            "behavior_rate": gt_rate,  # PRIMARY
            "reward/task_mean": mean(er.task_reward for er in env_results),
            "reward/penalty_mean": mean(penalties),
            "reward/total_mean": mean(rewards),
            "unparsed_rate": mean(float(er.meta["unparsed"]) for er in env_results),
            **{f"train/{k}": v for k, v in step_metrics.items()},
            **monitor_metrics,
        }
        if run is not None:
            run.log(log)
        metrics_log.write(json.dumps(log) + "\n")
        metrics_log.flush()
        mon_summary = " ".join(
            f"{m.name}[acc={monitor_metrics[f'monitor/{m.name}/accuracy']:.2f},"
            f"auc={monitor_metrics[f'monitor/{m.name}/auroc']:.2f}]"
            for m in all_monitors
        )
        print(
            f"step {step}: gt_syco={gt_rate:.2f} task_r={log['reward/task_mean']:.2f} "
            f"penalty={log['reward/penalty_mean']:.2f} loss={loss_val:.1f} | {mon_summary}"
        )

        # Persist a sampled fraction of rollouts with per-rollout metadata: monitor scores/labels
        # (computed anyway), env oracle, reward, advantage — plus any caller-supplied extras.
        for i, (r, er) in enumerate(zip(rollouts, env_results)):
            if rng.random() >= cfg.logging.log_fraction:
                continue
            extra = extra_rollout_fields(r, i) if extra_rollout_fields is not None else {}
            rollout_log.write(
                json.dumps(
                    {
                        "step": step,
                        "question": r.prompt.text,
                        "cot": r.cot,
                        "answer": r.output,
                        "env": {
                            "task_reward": er.task_reward,
                            "behavior_present": er.behavior_present,
                            "choice": er.meta.get("choice"),
                            "unparsed": er.meta.get("unparsed"),
                        },
                        "monitors": {
                            m.name: {
                                "score": results_by_name[m.name][i].score,
                                "label": results_by_name[m.name][i].label,
                            }
                            for m in all_monitors
                        },
                        "reward": rewards[i],
                        "advantage": advantages[i],
                        "extra": extra,
                    }
                )
                + "\n"
            )
        rollout_log.flush()

    rollout_log.close()
    metrics_log.close()
    if run is not None:
        run.finish()

    # Auto-plot on finish (best-effort — never let a plotting hiccup fail a completed run).
    try:
        from monitordecorrelation.eval.plots import plot_run as _plot_run

        for p in _plot_run(rollout_log_dir):
            print(f"plot: {p}")
    except Exception as e:  # noqa: BLE001
        print(f"(auto-plot skipped: {e})")
