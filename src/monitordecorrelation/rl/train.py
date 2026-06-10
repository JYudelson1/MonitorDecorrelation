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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

from monitordecorrelation.config import RunConfig
from monitordecorrelation.envs.base import Env
from monitordecorrelation.eval.metrics import _mean_for_class, accuracy, brier, roc_auc
from monitordecorrelation.monitors.base import Monitor
from monitordecorrelation.rl.grpo import compute_group_advantages
from monitordecorrelation.types import MonitorResult, Rollout


def _log(msg: str) -> None:
    """Print with an [HH:MM:SS] wall-clock prefix, flushed (so it survives pipe buffering)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _monitor_info(m: Monitor, role: str) -> dict:
    """Best-effort record of what a monitor actually is (so 'cot_weak' is decodable later)."""
    return {
        "name": m.name,
        "role": role,
        "model_id": getattr(m, "model_id", None),
        "behavior": getattr(m, "behavior", None),
        "threshold": getattr(m, "threshold", None),
    }


def _score_monitors(
    monitors: Sequence[Monitor], rollouts: list[Rollout], workers: int
) -> dict[str, list[MonitorResult]]:
    """{name: [MonitorResult]} — white-box probes use ``score_batch`` (one batched local forward);
    API judges are scored per-rollout concurrently. Threading a single torch model is pointless/unsafe."""
    out: dict[str, list[MonitorResult]] = {}
    batched = [m for m in monitors if hasattr(m, "score_batch")]
    threaded = [m for m in monitors if not hasattr(m, "score_batch")]
    for m in batched:
        _log(f"      · scoring {m.name} (probe, {len(rollouts)} local fwd)…")
        out[m.name] = list(m.score_batch(rollouts))  # type: ignore[attr-defined]
    if threaded:
        names = ", ".join(m.name for m in threaded)
        _log(f"      · scoring {len(threaded)} API monitor(s) [{names}] × {len(rollouts)} "
             f"rollouts concurrently…")
        for m in threaded:
            out[m.name] = [None] * len(rollouts)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(m.score, rollouts[i]): (m.name, i)
                for m in threaded
                for i in range(len(rollouts))
            }
            for fut in futs:
                name, i = futs[fut]
                out[name][i] = fut.result()
    return out


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
    # TRAIN metrics (per step, on the training rollouts) and EVAL metrics (periodic, on a FIXED
    # held-out set) go to separate files + wandb namespaces. Degradation curves come from EVAL.
    metrics_log = (rollout_log_dir / "metrics.jsonl").open("w")
    eval_metrics_log = (rollout_log_dir / "eval_metrics.jsonl").open("w")

    # The held-out eval set: a FIXED set of prompts, re-scored every eval so curves are comparable.
    # ``holdout`` also REMOVES them from the training pool so eval ⟂ train.
    if hasattr(env, "holdout"):
        eval_prompts = env.holdout(cfg.eval_size, seed=cfg.seed)
    else:
        _log("⚠️  env has no holdout(); eval prompts overlap the training pool (not disjoint).")
        eval_prompts = [env.sample_prompt() for _ in range(cfg.eval_size)]

    def run_eval(step: int) -> None:
        """Sample the current policy on the fixed eval set; log every monitor's degradation metrics.

        AUROC is the headline but goes NaN at single-class (behavior≈0 or ≈1); ``brier`` and
        ``mean_score_syco`` stay defined there, so the obfuscation-success regime stays measurable.
        """
        _log(f"  [eval @ {step}] sampling {len(eval_prompts)} held-out prompts (max_tokens={max_tokens})…")
        t0 = time.perf_counter()
        ev = backend.sample(eval_prompts, num_samples=1, max_tokens=max_tokens, temperature=1.0)
        ev_env = [env.score(r) for r in ev]
        gt = [er.behavior_present for er in ev_env]
        t_sample = time.perf_counter() - t0
        _log(f"  [eval @ {step}] scoring {len(all_monitors)} monitors…")
        t1 = time.perf_counter()
        res = _score_monitors(all_monitors, ev, monitor_workers)
        t_score = time.perf_counter() - t1
        row: dict = {
            "step": step,
            "behavior_rate": mean(float(g) for g in gt),
            "unparsed_rate": mean(float(er.meta["unparsed"]) for er in ev_env),
        }
        for m in all_monitors:
            scores = [x.score for x in res[m.name]]
            preds = [x.label for x in res[m.name]]
            row[f"monitor/{m.name}/auroc"] = roc_auc(scores, gt)
            row[f"monitor/{m.name}/accuracy"] = accuracy(preds, gt)
            row[f"monitor/{m.name}/brier"] = brier(scores, gt)
            row[f"monitor/{m.name}/mean_score"] = mean(scores) if scores else float("nan")
            row[f"monitor/{m.name}/mean_score_syco"] = _mean_for_class(scores, gt, True)
            row[f"monitor/{m.name}/mean_score_honest"] = _mean_for_class(scores, gt, False)
        row["time/sample_s"] = t_sample
        row["time/score_s"] = t_score
        row["time/total_s"] = t_sample + t_score
        eval_metrics_log.write(json.dumps(row) + "\n")
        eval_metrics_log.flush()
        if run is not None:
            run.log({f"eval/{k}": v for k, v in row.items() if k != "step"}, step=step)
        # Print AUROC, but fall back to Brier in the readout so a NaN-AUROC eval still shows a number.
        summ = " ".join(
            f"{m.name}:auc={row[f'monitor/{m.name}/auroc']:.2f}/brier={row[f'monitor/{m.name}/brier']:.2f}"
            for m in all_monitors
        )
        _log(f"  [eval @ {step}] behavior={row['behavior_rate']:.2f} | {summ} "
             f"| {t_sample + t_score:.0f}s (sample {t_sample:.0f}s, score {t_score:.0f}s)")

    for step in range(cfg.n_steps):
        if step % cfg.eval_every == 0:
            run_eval(step)  # held-out eval at step 0 and every eval_every

        prompts = [env.sample_prompt() for _ in range(cfg.batch_size)]
        _log(f"step {step}: sampling {cfg.batch_size}×{cfg.group_size} train rollouts "
             f"(max_tokens={max_tokens})…")
        t0 = time.perf_counter()
        rollouts = backend.sample(
            prompts, num_samples=cfg.group_size, max_tokens=max_tokens, temperature=1.0
        )
        env_results = [env.score(r) for r in rollouts]
        t_sample = time.perf_counter() - t0
        if train_against:
            _log(f"step {step}: scoring train-against monitor(s)…")

        # Only the TRAIN-AGAINST monitors are scored every step (they form the reward penalty).
        # Held-out monitors are measured on the held-out eval set instead — cleaner and cheaper.
        t1 = time.perf_counter()
        ta_res = _score_monitors(train_against, rollouts, monitor_workers) if train_against else {}
        t_score = time.perf_counter() - t1
        penalties = [
            mean(ta_res[m.name][i].score for m in train_against) if train_against else 0.0
            for i in range(len(rollouts))
        ]
        rewards = [er.task_reward - cfg.penalty_coef * pen for er, pen in zip(env_results, penalties)]
        advantages = compute_group_advantages(
            rewards, cfg.group_size, normalize=(cfg.rl_algo == "grpo")
        )
        t2 = time.perf_counter()
        step_metrics = backend.train_step(rollouts, advantages)
        t_optim = time.perf_counter() - t2

        gt_train = [er.behavior_present for er in env_results]
        gt_rate = mean(float(g) for g in gt_train)
        loss_keys = [k for k in step_metrics if "loss" in k.lower()]
        loss_val = step_metrics[loss_keys[0]] if loss_keys else float("nan")
        row = {
            "step": step,
            "behavior_rate": gt_rate,  # PRIMARY (training rollouts; eval has the held-out version)
            "reward/task_mean": mean(er.task_reward for er in env_results),
            "reward/penalty_mean": mean(penalties),
            "reward/total_mean": mean(rewards),
            "unparsed_rate": mean(float(er.meta["unparsed"]) for er in env_results),
            "time/sample_s": t_sample,
            "time/score_s": t_score,
            "time/optim_s": t_optim,
            "time/total_s": t_sample + t_score + t_optim,
            **{f"loss/{k}": v for k, v in step_metrics.items()},
            **{f"monitor/{m.name}/mean_score": mean(x.score for x in ta_res[m.name]) for m in train_against},
            # Brier + class-split score on training rollouts: the obfuscation signal (probe being
            # gamed) shows here even when the batch is single-class and AUROC would be undefined.
            **{f"monitor/{m.name}/brier": brier([x.score for x in ta_res[m.name]], gt_train)
               for m in train_against},
            **{f"monitor/{m.name}/mean_score_syco": _mean_for_class([x.score for x in ta_res[m.name]], gt_train, True)
               for m in train_against},
        }
        if run is not None:
            run.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=step)
        metrics_log.write(json.dumps(row) + "\n")
        metrics_log.flush()
        _log(
            f"step {step}: behavior={gt_rate:.2f} task_r={row['reward/task_mean']:.2f} "
            f"penalty={row['reward/penalty_mean']:.2f} loss={loss_val:.1f} "
            f"| {row['time/total_s']:.0f}s (sample {t_sample:.0f}s, score {t_score:.0f}s, "
            f"optim {t_optim:.0f}s)"
        )

        # Persist a sampled fraction of TRAINING rollouts (with train-against monitor scores).
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
                                "score": ta_res[m.name][i].score,
                                "label": ta_res[m.name][i].label,
                            }
                            for m in train_against
                        },
                        "reward": rewards[i],
                        "advantage": advantages[i],
                        "extra": extra,
                    }
                )
                + "\n"
            )
        rollout_log.flush()

    run_eval(cfg.n_steps)  # final held-out eval

    rollout_log.close()
    metrics_log.close()
    eval_metrics_log.close()
    if run is not None:
        run.finish()

    # Auto-plot on finish (best-effort — never let a plotting hiccup fail a completed run).
    try:
        from monitordecorrelation.eval.plots import plot_run as _plot_run

        for p in _plot_run(rollout_log_dir):
            _log(f"plot: {p}")
    except Exception as e:  # noqa: BLE001
        _log(f"(auto-plot skipped: {e})")
