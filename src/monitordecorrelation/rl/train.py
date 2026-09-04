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

import numpy as np

from monitordecorrelation.config import RunConfig
from monitordecorrelation.envs.base import Env
from monitordecorrelation.eval.rows import eval_row, train_row
from monitordecorrelation.eval.rows import token_metrics as _token_metrics
from monitordecorrelation.monitors.base import Monitor
from monitordecorrelation.types import MonitorResult, Rollout


def _log(msg: str) -> None:
    """Print with an [HH:MM:SS] wall-clock prefix, flushed (so it survives pipe buffering)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_env(env: Env, rollouts: list[Rollout]) -> list:
    """Grade a batch of rollouts, using the env's ``score_batch`` when it has one.

    Public and backend-agnostic, like ``score_monitors``: the NeMo-RL environment actor grades
    through this same function so both backends handle batching and failures identically.

    Envs whose grading is expensive and out-of-process (running generated code) implement
    ``score_batch`` to execute rollouts concurrently; otherwise this is the plain serial loop. Errors
    propagate deliberately — a grading failure is never coerced into a reward."""
    if hasattr(env, "score_batch"):
        results = list(env.score_batch(rollouts))  # type: ignore[attr-defined]
        if len(results) != len(rollouts):
            raise RuntimeError(
                f"{type(env).__name__}.score_batch returned {len(results)} results for "
                f"{len(rollouts)} rollouts"
            )
        return results
    return [env.score(r) for r in rollouts]


def truncate_step_log(path: Path, start_step: int) -> int:
    """Drop every row of a step-keyed ``.jsonl`` log whose ``step`` >= ``start_step`` (in place), so a
    resumed run appends to logs that end exactly where its checkpoint was taken. Returns the number of
    rows kept. A missing file is fine (0 kept). Unparseable rows are dropped too."""
    if not path.exists():
        return 0
    kept = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with path.open() as src, tmp.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                step = int(json.loads(line)["step"])
            except (ValueError, KeyError, TypeError):
                continue
            if step < start_step:
                dst.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    tmp.replace(path)
    return kept


def _monitor_info(m: Monitor, role: str) -> dict:
    """Best-effort record of what a monitor actually is (so 'cot_weak' is decodable later)."""
    return {
        "name": m.name,
        "role": role,
        "model_id": getattr(m, "model_id", None),
        "behavior": getattr(m, "behavior", None),
        "threshold": getattr(m, "threshold", None),
    }


def score_monitors(
    monitors: Sequence[Monitor], rollouts: list[Rollout], workers: int = 16
) -> dict[str, list[MonitorResult]]:
    """{name: [MonitorResult]} — white-box probes use ``score_batch`` (one batched local forward);
    API judges are scored per-rollout concurrently. Threading a single torch model is pointless/unsafe.

    Public and backend-agnostic: the NeMo-RL environment actor scores its monitors through this same
    function (``backends/nemo/env_actor.py``), so failure handling and concurrency behave identically
    on both backends. This module imports nothing backend-specific, so it is importable from
    NeMo-RL's virtualenv."""
    # A failed score must NEVER crash the run — a single transient API hiccup on one of hundreds of
    # rollouts would otherwise abort a multi-hour job. On failure we substitute a NaN sentinel (logged)
    # and the metric code drops it, so that rollout just doesn't count toward this monitor's score.
    def _sentinel(err: str) -> MonitorResult:
        return MonitorResult(score=float("nan"), label=False, meta={"error": err})

    out: dict[str, list[MonitorResult]] = {}
    batched = [m for m in monitors if hasattr(m, "score_batch")]
    threaded = [m for m in monitors if not hasattr(m, "score_batch")]
    for m in batched:
        _log(f"      · scoring {m.name} (probe, {len(rollouts)} local fwd)…")
        try:
            out[m.name] = list(m.score_batch(rollouts))  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 — never let a probe failure kill the run
            _log(f"      ⚠️  {m.name}.score_batch failed ({type(e).__name__}: {e}); NaN-scoring this eval")
            out[m.name] = [_sentinel(f"{type(e).__name__}: {e}") for _ in rollouts]
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
            errors: dict[str, str] = {}
            for fut in futs:
                name, i = futs[fut]
                try:
                    out[name][i] = fut.result()
                except Exception as e:  # noqa: BLE001 — one bad API call shouldn't abort the run
                    out[name][i] = _sentinel(f"{type(e).__name__}: {e}")
                    errors[name] = f"{type(e).__name__}: {e}"
            for name, err in errors.items():  # one concise warning per monitor, not per rollout
                n_failed = sum(1 for r in out[name] if r.meta.get("error"))
                _log(f"      ⚠️  {name}: {n_failed}/{len(rollouts)} scores failed ({err}); NaN-scored")
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
    resume_step: int | None = None,
    wandb_run_id: str | None = None,
) -> None:
    """Run GRPO. ``extra_rollout_fields(rollout, idx) -> dict`` lets callers attach arbitrary
    per-rollout metadata to saved rollouts. ``run_info`` is merged into the saved ``run_info.json``
    (use it for anything the caller knows but the loop doesn't, e.g. the dataset subset).

    ``resume_step``: the backend has already been loaded with the checkpoint saved at that step
    (weights + optimizer), so training starts there instead of at 0. The run directory's step-keyed
    logs are truncated to rows before it and appended to; the eval at that step is re-run (the
    checkpoint's eval row is regenerated, not trusted); the checkpoint save at that step is skipped.
    ``wandb_run_id`` (with ``resume_step``) re-attaches to the original W&B run so its curves continue
    — but W&B ignores rows below the step it already reached (rewind is a private-preview feature),
    so steps between the checkpoint and the crash live in the local logs only."""
    rng = random.Random(cfg.seed)
    # Global RNG seeding for any library that reaches for the default generator (numpy/sklearn paths).
    # The tinker sampler is seeded inside the backend (ServiceClient + per-call SamplingParams seed);
    # env construction + holdout + this rng all take cfg.seed. So one cfg.seed pins the whole run.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _log(f"seed={cfg.seed} (env, holdout, log-sampling, numpy, and the tinker sampler all derive from it)")
    all_monitors: list[Monitor] = [*train_against, *held_out]
    run = None
    if cfg.logging.use_wandb:
        import wandb

        resume_kw = {}
        if resume_step is not None and wandb_run_id:
            # Same run id, so the curves continue on the original W&B page. Rows at a step the run
            # already logged are dropped by W&B (it warns once) — see the docstring.
            resume_kw = {"id": wandb_run_id, "resume": "must"}
            _log(f"resuming W&B run {wandb_run_id} (rows below its last logged step are ignored by "
                 f"W&B; the local metrics.jsonl is complete)")
        elif resume_step is not None:
            _log("⚠️  resuming without a W&B run id — the resumed steps go to a NEW W&B run")
        run = wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            group=cfg.logging.wandb_group,    # cluster all runs of a sweep (matrix × seeds) together
            tags=cfg.logging.wandb_tags or None,
            mode=cfg.logging.wandb_mode,
            config=cfg.__dict__,
            **resume_kw,
        )

    rollout_log_dir = Path("data/runs") / (cfg.logging.run_name or "smoke")
    rollout_log_dir.mkdir(parents=True, exist_ok=True)

    # Durable record of WHAT this run was: config + which monitor is which (train-against vs
    # held-out, model ids, behavior). Without this the logs' cot_weak/cot_strong are undecodable.
    now = datetime.datetime.now().isoformat(timespec="seconds")
    info = {
        "started_at": now,
        "run_name": cfg.logging.run_name,
        "policy": cfg.base_model,
        "config": dataclasses.asdict(cfg),
        "env": {"name": env.name, "behavior_name": getattr(env, "behavior_name", None)},
        "train_against": [_monitor_info(m, "train_against") for m in train_against],
        "held_out": [_monitor_info(m, "held_out") for m in held_out],
        **(run_info or {}),
    }
    if run is not None:
        info["wandb_run_id"] = run.id  # what a later --set resume_from=… re-attaches to
    start_step = 0
    if resume_step is not None:
        # Keep the original run's record (its started_at, its checkpoint paths) and note the resume.
        prev_path = rollout_log_dir / "run_info.json"
        prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
        info = {**prev, **{k: v for k, v in info.items() if k != "started_at"},
                "started_at": prev.get("started_at", now),
                "resumes": [*prev.get("resumes", []), {"at": now, "from_step": resume_step}]}
        start_step = resume_step
        # The logs end where the checkpoint was taken; everything from start_step on is regenerated.
        for name in ("rollouts.jsonl", "metrics.jsonl", "eval_metrics.jsonl", "eval_rollouts.jsonl"):
            kept = truncate_step_log(rollout_log_dir / name, start_step)
            _log(f"resume: {name} truncated to {kept} rows (step < {start_step})")
    (rollout_log_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    mode = "a" if resume_step is not None else "w"
    rollout_log = (rollout_log_dir / "rollouts.jsonl").open(mode)
    # TRAIN metrics (per step, on the training rollouts) and EVAL metrics (periodic, on a FIXED
    # held-out set) go to separate files + wandb namespaces. Degradation curves come from EVAL.
    metrics_log = (rollout_log_dir / "metrics.jsonl").open(mode)
    eval_metrics_log = (rollout_log_dir / "eval_metrics.jsonl").open(mode)
    # Per-eval-prompt record: EVERY monitor's score on EVERY held-out eval prompt+continuation, each
    # eval step (the full bookkeeping for post-hoc recalibration / the suspiciousness-gap analysis).
    eval_rollouts_log = (rollout_log_dir / "eval_rollouts.jsonl").open(mode)

    # The held-out eval set: a FIXED set of prompts, re-scored every eval so curves are comparable.
    # ``holdout`` also REMOVES them from the training pool so eval ⟂ train.
    if hasattr(env, "holdout"):
        eval_prompts = env.holdout(cfg.eval_size, seed=cfg.seed)
    else:
        _log("⚠️  env has no holdout(); eval prompts overlap the training pool (not disjoint).")
        eval_prompts = [env.sample_prompt() for _ in range(cfg.eval_size)]

    def run_eval(step: int, commit: bool = False) -> None:
        """Sample the current policy on the fixed eval set; log every monitor's degradation metrics.

        ``commit`` is the wandb flush flag: an eval inside the loop is followed by the train ``log()``
        for the SAME step, so it stays buffered (commit=False) and both rows flush together; only the
        final eval — with no train step after it — has to commit itself.

        AUROC is the headline but goes NaN at single-class (behavior≈0 or ≈1); ``brier`` and the
        behavior-present mean score (``mean_score_<behavior>``) stay defined there, so the
        obfuscation-success regime stays measurable.
        """
        n_per = getattr(cfg, "eval_samples_per_prompt", 1) or 1
        _log(f"  [eval @ {step}] sampling {len(eval_prompts)}×{n_per} held-out rollouts (max_tokens={max_tokens})…")
        t0 = time.perf_counter()
        ev = backend.sample(eval_prompts, num_samples=n_per, max_tokens=max_tokens, temperature=1.0)
        ev_env = score_env(env, ev)
        gt = [er.behavior_present for er in ev_env]
        t_sample = time.perf_counter() - t0
        _log(f"  [eval @ {step}] scoring {len(all_monitors)} monitors…")
        t1 = time.perf_counter()
        res = score_monitors(all_monitors, ev, monitor_workers)
        t_score = time.perf_counter() - t1
        # The row schema is shared with every other backend — see eval/rows.py. It carries the
        # strict oracle metrics per monitor plus, for envs that define one, the separate `loose/`
        # oracle track (loose ⊇ strict: it also counts honest-but-buggy attempts, so it is a denser
        # but noisier construct and is never merged into the headline numbers).
        row: dict = eval_row(
            step=step, rollouts=ev, env_results=ev_env, monitor_results=res,
            monitor_names=[m.name for m in all_monitors], env=env, max_tokens=max_tokens,
        )
        row["time/sample_s"] = t_sample
        row["time/score_s"] = t_score
        row["time/total_s"] = t_sample + t_score
        eval_metrics_log.write(json.dumps(row) + "\n")
        eval_metrics_log.flush()
        # Per-prompt dump: one record per eval rollout with EVERY monitor's score+label (so monitors
        # can be re-thresholded/recalibrated post-hoc, and the hack-vs-honest gap computed per prompt).
        for i, (r, er, g) in enumerate(zip(ev, ev_env, gt)):
            eval_rollouts_log.write(json.dumps({
                "step": step,
                "task_id": r.prompt.meta.get("task_id") if r.prompt.meta else None,
                "question": r.prompt.text,
                "cot": r.cot,
                "answer": r.output,
                "behavior_present": bool(g),  # = strict_rh
                # per-rollout loose/hardcoding labels too, so loose-oracle AUROC (and the 3-way split) is
                # recomputable post-hoc from this dump, not just the aggregate rates above.
                "loose_rh": bool(er.meta["loose_rh"]) if "loose_rh" in er.meta else None,
                "hardcoding": bool(er.meta["hardcoding"]) if "hardcoding" in er.meta else None,
                "unparsed": er.meta.get("unparsed"),
                "env_meta": er.meta,  # full per-env grading record (see the train-rollout dump above)
                "monitors": {m.name: {"score": res[m.name][i].score, "label": res[m.name][i].label}
                             for m in all_monitors},
            }, default=str) + "\n")
        eval_rollouts_log.flush()
        if run is not None:
            run.log({f"eval/{k}": v for k, v in row.items() if k != "step"},
                    step=step, commit=commit)
        # Print AUROC, but fall back to Brier in the readout so a NaN-AUROC eval still shows a number.
        summ = " ".join(
            f"{m.name}:auc={row[f'monitor/{m.name}/auroc']:.2f}/brier={row[f'monitor/{m.name}/brier']:.2f}"
            for m in all_monitors
        )
        _log(f"  [eval @ {step}] behavior={row['behavior_rate']:.2f} | {summ} "
             f"| {t_sample + t_score:.0f}s (sample {t_sample:.0f}s, score {t_score:.0f}s)")

    cum_tokens = 0.0  # running input+output token count across train steps (tokens/cum_total)
    if start_step:
        # Replay the consumed prompt / sampling-seed streams so the resumed steps see the prompts and
        # seeds the uninterrupted run would have (rather than restarting both from step 0).
        for _ in range(start_step * cfg.batch_size):
            env.sample_prompt()
        n_evals_before = sum(1 for s in range(start_step) if s % cfg.eval_every == 0)
        if hasattr(backend, "_sample_calls"):
            backend._sample_calls = start_step + n_evals_before
        rows = [json.loads(l) for l in (rollout_log_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
        cum_tokens = float(rows[-1].get("tokens/cum_total", 0.0)) if rows else 0.0
        _log(f"resuming at step {start_step}/{cfg.n_steps} (tokens so far {cum_tokens/1e6:.2f}M)")
    for step in range(start_step, cfg.n_steps):
        if step % cfg.eval_every == 0:
            run_eval(step)  # held-out eval at step 0 and every eval_every

        if step % cfg.save_every == 0 and not (resume_step is not None and step == start_step):
            # (a resumed run already has the checkpoint it started from)
            ckpt = backend.save_checkpoint(f"{cfg.logging.run_name or 'run'}-{step}")
            (rollout_log_dir / f"checkpoint_{step}.txt").write_text(ckpt + "\n")
            info[f"checkpoint_{step}"] = ckpt
            _log(f"saved checkpoint at step {step}: {ckpt}")

        prompts = [env.sample_prompt() for _ in range(cfg.batch_size)]
        _log(f"step {step}: sampling {cfg.batch_size}×{cfg.group_size} train rollouts "
             f"(max_tokens={max_tokens})…")
        t0 = time.perf_counter()
        rollouts = backend.sample(
            prompts, num_samples=cfg.group_size, max_tokens=max_tokens, temperature=1.0
        )
        env_results = score_env(env, rollouts)
        t_sample = time.perf_counter() - t0
        if train_against:
            _log(f"step {step}: scoring train-against monitor(s)…")

        # Only the TRAIN-AGAINST monitors are scored every step (they form the reward penalty).
        # Held-out monitors are measured on the held-out eval set instead — cleaner and cheaper.
        t1 = time.perf_counter()
        ta_res = score_monitors(train_against, rollouts, monitor_workers) if train_against else {}
        t_score = time.perf_counter() - t1

        # HARD FAIL if a train-against monitor scored NOTHING: unlike a held-out monitor (just a
        # measurement), a dead train-against monitor means the reward has no penalty signal — the run
        # silently becomes a no-penalty control. Far better to abort (esp. at step 0 = a config/API
        # error) than waste hours training on no signal. Partial failures are tolerated (coalesced to 0).
        for m in train_against:
            if all(r.score != r.score for r in ta_res[m.name]):  # every score is NaN
                raise RuntimeError(
                    f"train-against monitor {m.name!r} failed to score ANY of {len(rollouts)} rollouts "
                    f"at step {step} — aborting (no training signal). Check the monitor / API / model id. "
                    f"First error: {next((r.meta.get('error') for r in ta_res[m.name] if r.meta.get('error')), '?')}"
                )

        # A partially-failed train-against score (NaN sentinel) must not produce a NaN reward → coalesce
        # to 0 (that rollout just isn't penalized this step). Without this one API hiccup would NaN the
        # advantages and corrupt the update.
        def _pen(score: float) -> float:
            return 0.0 if score != score else score

        # ``penalties[i]`` = raw mean train-against suspiciousness for rollout i (BEFORE the coefficient).
        # This is the obfuscation signal (drops as the policy evades) — logged as monitor_suspiciousness_mean.
        penalties = [
            mean(_pen(ta_res[m.name][i].score) for m in train_against) if train_against else 0.0
            for i in range(len(rollouts))
        ]
        # Effective λ this step: constant penalty_coef, OR a linear ramp start→end if penalty_schedule is
        # set (hack-then-hide curriculum). Logged as penalty_coef so the schedule is visible; the applied
        # reward penalty (penalty_mean) is λ·suspiciousness, now decoupled since λ can vary over training.
        pen_coef = cfg.penalty_coef
        if cfg.penalty_schedule:
            frac = step / max(1, cfg.n_steps - 1)  # 0 at step 0 → 1 at the last step
            s, e = cfg.penalty_schedule["start_penalty"], cfg.penalty_schedule["end_penalty"]
            pen_coef = s + (e - s) * frac
        rewards = [er.task_reward - pen_coef * pen for er, pen in zip(env_results, penalties)]
        t2 = time.perf_counter()
        # The backend adapts rollouts+rewards into cookbook trajectory groups and does the whole loss
        # layer (advantage centering, mask/datum assembly, KL penalty, forward_backward) via cookbook
        # primitives — see backends/tinker_backend.py:train_step.
        step_metrics = backend.train_step(rollouts, rewards, cfg.group_size)
        t_optim = time.perf_counter() - t2

        gt_rate = mean(float(er.behavior_present) for er in env_results)
        tok = _token_metrics(rollouts, max_tokens)
        cum_tokens += tok.get("tokens/total", 0.0)  # tokens sampled so far (train batches only)
        loss_keys = [k for k in step_metrics if "loss" in k.lower() and "logprob" not in k.lower()]
        loss_val = step_metrics[loss_keys[0]] if loss_keys else float("nan")
        logp_mean = step_metrics.get("train/logprob_mean", float("nan"))
        # Reward decomposition + per-train-against-monitor metrics, in the schema every backend
        # shares (eval/rows.py); the tinker-specific bits (timings, optimizer metrics, cumulative
        # tokens) are layered on here.
        row = {
            **train_row(
                step=step, rollouts=rollouts, env_results=env_results, monitor_results=ta_res,
                train_against_names=[m.name for m in train_against], env=env,
                max_tokens=max_tokens, penalties=penalties, pen_coef=pen_coef, rewards=rewards,
            ),
            **({"tokens/cum_total": cum_tokens} if tok else {}),  # running total over the run
            "time/sample_s": t_sample,
            "time/score_s": t_score,
            "time/optim_s": t_optim,
            "time/total_s": t_sample + t_score + t_optim,
            "kl/mean": step_metrics.get("kl/mean", 0.0),  # per-token KL to base (0 if kl_coef=0)
            **{f"loss/{k}": v for k, v in step_metrics.items()},
        }
        if run is not None:
            # commit=True flushes this step NOW (with any eval row buffered at the same step).
            # wandb defaults to commit=False whenever `step` is passed explicitly, which would hold
            # every row back until the next step's log() — the metrics land one RL step late.
            run.log({f"train/{k}": v for k, v in row.items() if k != "step"},
                    step=step, commit=True)
        metrics_log.write(json.dumps(row) + "\n")
        metrics_log.flush()
        kl_note = f" kl={row['kl/mean']:.3f}" if row["kl/mean"] else ""
        tok_note = (f" tok={tok['tokens/input_per_rollout']:.0f}in/{tok['tokens/output_per_rollout']:.0f}out"
                    f" per rollout, {tok['tokens/total']/1e3:.1f}k this step" if tok else "")
        if "tokens/budget_forced_rate" in tok:  # thinking budget on: how often it bound + what it cost
            tok_note += (f" | budget forced {tok['tokens/budget_forced_rate']:.0%}, "
                         f"prefill x{tok['tokens/prefill_overhead_ratio']:.2f} "
                         f"(uncached x{tok['tokens/prefill_miss_overhead_ratio']:.2f})")
        # cookbook train_step surfaces no loss metric; show mean training logprob as the proxy signal.
        train_note = f"loss={loss_val:.1f}" if loss_val == loss_val else f"logp={logp_mean:.2f}"
        _log(
            f"step {step}: behavior={gt_rate:.2f} task_r={row['reward/task_mean']:.2f} "
            f"penalty={row['reward/penalty_mean']:.2f} {train_note}{kl_note}{tok_note} "
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
                            # The env's full grading record (per-env keys: MBPP's loose_rh/hardcoding,
                            # ImpossibleBench's frac_passed/passes_original/first_error, …) so a run can
                            # be re-analysed without re-running the env.
                            "meta": er.meta,
                        },
                        "monitors": {
                            m.name: {
                                "score": ta_res[m.name][i].score,
                                "label": ta_res[m.name][i].label,
                            }
                            for m in train_against
                        },
                        "reward": rewards[i],  # advantage is now computed in the backend (cookbook)
                        "extra": extra,
                    },
                    default=str,  # never let an exotic meta value abort a run mid-step
                )
                + "\n"
            )
        rollout_log.flush()

    run_eval(cfg.n_steps, commit=True)  # final held-out eval (nothing follows it → commit itself)

    # Save the final training state on tinker (7-day TTL) so we can resume / sample / inspect the
    # trained model later — important when we don't yet know how long these runs should take.
    if hasattr(backend, "save_checkpoint"):
        try:
            ckpt = backend.save_checkpoint(f"{cfg.logging.run_name or 'run'}-final",
                                           ttl_seconds=7 * 24 * 3600)
            (rollout_log_dir / "final_checkpoint.txt").write_text(ckpt + "\n")
            info["final_checkpoint"] = ckpt
            (rollout_log_dir / "run_info.json").write_text(json.dumps(info, indent=2))
            _log(f"saved final weights on tinker (7-day TTL): {ckpt}")
        except Exception as e:  # noqa: BLE001 — never let a save hiccup fail a completed run
            _log(f"⚠️  final checkpoint save failed: {type(e).__name__}: {e}")

    rollout_log.close()
    metrics_log.close()
    eval_metrics_log.close()
    eval_rollouts_log.close()
    if run is not None:
        run.finish()

    # Auto-plot on finish (best-effort — never let a plotting hiccup fail a completed run).
    try:
        from monitordecorrelation.eval.plots import plot_run as _plot_run

        for p in _plot_run(rollout_log_dir):
            _log(f"plot: {p}")
    except Exception as e:  # noqa: BLE001
        _log(f"(auto-plot skipped: {e})")
