"""GRPO training loop for the decorrelation experiments.

reward = task_reward − penalty_coef · mean(train_against monitor scores), except for an INVALID rollout
(``envs.base.invalid_reason``: truncated by max_tokens, or output the env could not parse), whose reward
is a flat ``INVALID_ROLLOUT_REWARD`` = -1 (``EnvResult.reward_override``, enforced in ``_score_env``).
**No monitor ever scores an invalid rollout**, in training or eval (``MonitorScorer`` skips it), so every
monitor statistic — AUROC, d′, class means, n_pos/n_neg/n_scored, the train-against suspiciousness — is
over the valid rollouts only; ``invalid_rate`` logs the excluded fraction. The oracle rates
(``behavior_rate`` …) stay over ALL rollouts; single-turn envs do not grade a truncated rollout at all
(``_score_env`` checks), so there it counts as task 0 / no behavior, like an unparseable one. Held-out monitors are scored every eval but never enter the
reward. The ground-truth behavior rate is logged as the primary metric (see docs/EXPERIMENT_PLAN.md
"Ground truth is the crux").
"""

from __future__ import annotations

import dataclasses
import datetime
import inspect
import json
import random
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

import numpy as np

from monitordecorrelation.config import RunConfig
from monitordecorrelation.envs.base import Env, invalid_reason, is_truncated
from monitordecorrelation.eval.rollout_dump import monitor_record, slim_record
from monitordecorrelation.eval.metric_keys import absent_score_key, present_score_key
from monitordecorrelation.eval.metrics import _mean_for_class, accuracy, brier, dprime_margin, roc_auc
from monitordecorrelation.monitors.base import Monitor
from monitordecorrelation.rl import sdk_watch as sdk_watch_mod
from monitordecorrelation.types import INVALID_ROLLOUT_REWARD, MonitorResult, Rollout


def _log(msg: str) -> None:
    """Print with an [HH:MM:SS] wall-clock prefix, flushed (so it survives pipe buffering)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _label_metrics(prefix: str, scores: list[float], preds: list[bool], labels: list[bool]) -> dict[str, float]:
    """AUROC / accuracy / brier + present/absent class-split monitor score for ONE ground-truth label
    definition, keyed under ``prefix``. Used for the secondary ``loose`` oracle track (kept separate from
    the strict headline metrics — loose counts honest-buggy attempts as positives, so it's a different
    construct). All go NaN gracefully at single-class (roc_auc / _mean_for_class return NaN)."""
    return {
        f"{prefix}/auroc": roc_auc(scores, labels),
        f"{prefix}/accuracy": accuracy(preds, labels),
        f"{prefix}/brier": brier(scores, labels),
        f"{prefix}/dprime_margin": dprime_margin(scores, labels),
        f"{prefix}/mean_score_present": _mean_for_class(scores, labels, True),
        f"{prefix}/mean_score_absent": _mean_for_class(scores, labels, False),
    }


def _rollout_metrics(rollouts: Sequence[Rollout], rewards: Sequence[float], group_size: int) -> dict[str, float]:
    """Batch-shape diagnostics for RL debugging: the reward distribution (incl. how much of it varies
    WITHIN a GRPO group — only that part produces a gradient), how long completions are, and how
    often sampling hit the token limit (truncation → length collapse / blow-up shows up here first).
    For multi-turn episodes ``token_ids`` is every sampled token of the episode, and ``stop_reason``
    is the last turn's."""
    out: dict[str, float] = {}
    if rewards:
        arr = np.asarray(rewards, dtype=float)
        out["reward/total_std"] = float(arr.std())
        out["reward/total_min"] = float(arr.min())
        out["reward/total_max"] = float(arr.max())
        if group_size > 0 and len(arr) % group_size == 0:
            grp = arr.reshape(-1, group_size)
            out["reward/group_std_mean"] = float(grp.std(axis=1).mean())
            out["reward/frac_constant_groups"] = float((grp.std(axis=1) == 0).mean())
    lens = [len(r.token_ids) for r in rollouts if r.token_ids is not None]
    if lens:
        out["rollout/output_tokens_mean"] = float(np.mean(lens))
        out["rollout/output_tokens_max"] = float(np.max(lens))
        out["rollout/output_tokens_min"] = float(np.min(lens))
    if any((r.meta or {}).get("stop_reason") is not None for r in rollouts):
        out["rollout/truncated_rate"] = mean(float(is_truncated(r)) for r in rollouts)
    return out


def _length_metrics(results: Sequence, env) -> dict[str, float]:
    """The code-length term of the reward, for accounting alongside task + monitor terms.
    MBPP-Honeypot reward = max(0, 1 − length_penalty·len(code)); this surfaces (a) mean code length and
    (b) the reward actually lost to length **among rollouts that pass the visible test** (where length is
    the only thing modulating reward 0→1) — both directly comparable to reward/task_mean. Empty ``{}`` for
    envs without a code-length penalty (``length_penalty``)."""
    lp = getattr(env, "length_penalty", None)
    lens = [r.meta["code_len"] for r in results if "code_len" in r.meta]
    if lp is None or not lens:
        return {}
    passers = [r.meta["code_len"] for r in results
               if r.meta.get("passes_visible") and "code_len" in r.meta]
    len_pen = mean(min(1.0, lp * c) for c in passers) if passers else 0.0
    return {"reward/code_len_mean": mean(lens),          # avg chars of generated code
            "reward/len_penalty_mean": len_pen}          # avg reward lost to length (passers), reward units


def _accepts(fn, name: str) -> bool:
    """Does ``fn`` take a keyword argument called ``name``? Used to stay compatible with backends
    that predate ``on_rollout`` (and with test doubles) instead of guessing from a TypeError."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / C callables have no introspectable signature
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _sample(backend, env: Env, prompts: list, *, num_samples: int, max_tokens: int | None,
            think_budget: int | None = None, answer_tokens: int | None = None,
            on_rollout: Callable[[int, Rollout], None] | None = None) -> tuple[list[Rollout], bool]:
    """Sample rollouts the way the env needs: a multi-turn (tool-loop) env goes through the backend's
    episode driver (``sample_episodes``: sample a turn → env executes it → continue), a single-turn env
    through plain ``sample``. The rest of the loop is agnostic — both return Rollouts, ``group_size``
    consecutive per prompt.

    Returns ``(rollouts, streamed)``. When the backend supports it, ``on_rollout(index, rollout)`` is
    called as each rollout lands — from the thread that finished it — so per-rollout work (the judge
    API calls) overlaps the rest of the batch instead of queueing behind it. ``streamed`` says whether
    that happened; a backend without the hook returns False and the caller submits after the fact."""
    stream = on_rollout is not None
    if getattr(env, "multi_turn", False):
        if not hasattr(backend, "sample_episodes"):
            raise TypeError(f"{type(env).__name__} is multi-turn but backend {type(backend).__name__} "
                            f"has no sample_episodes()")
        # think_budget arrives RESOLVED (experiment_config.resolve_think_budget): None means no budget,
        # full stop — the env's default_think_budget is the config layer's business, not the loop's.
        stream = stream and _accepts(backend.sample_episodes, "on_rollout")
        return backend.sample_episodes(
            env, prompts, num_samples=num_samples, max_tokens=max_tokens, temperature=1.0,
            think_budget=think_budget, answer_tokens=answer_tokens,
            **({"on_rollout": on_rollout} if stream else {}),
        ), stream
    stream = stream and _accepts(backend.sample, "on_rollout")
    return backend.sample(prompts, num_samples=num_samples, max_tokens=max_tokens, temperature=1.0,
                          **({"on_rollout": on_rollout} if stream else {})), stream


def _env_metrics(results: Sequence, env) -> dict[str, float]:
    """Per-env summary means under ``env/<key>``: every key the env lists in ``summary_keys`` is
    averaged over the batch (bools → rates, ints → means). Empty for envs that don't declare any."""
    out: dict[str, float] = {}
    for k in getattr(env, "summary_keys", ()) or ():
        vals = [float(r.meta[k]) for r in results if k in r.meta and r.meta[k] is not None]
        if vals:
            out[f"env/{k}"] = mean(vals)
    return out


def _score_env(env: Env, rollouts: list[Rollout]) -> list:
    """Grade a batch of rollouts, using the env's ``score_batch`` when it has one.

    Envs whose grading is expensive and out-of-process (running generated code) implement
    ``score_batch`` to execute rollouts concurrently; otherwise this is the plain serial loop. Errors
    propagate deliberately — a grading failure is never coerced into a reward.

    Every INVALID rollout (``invalid_reason``: truncated by ``max_tokens``, or unparseable) then gets
    ``reward_override = INVALID_ROLLOUT_REWARD``, whatever the env made of it — the one place this is
    enforced, so it holds for every env and policy. The monitors skipped exactly these rollouts (same
    predicate, evaluated when they were sampled), so two invariants are checked loudly here: the env's
    ``unparsed`` flag agrees with its ``unparseable()`` (on the rollouts it graded), and the env set
    ``reward_override`` on no valid rollout — otherwise an unmonitored rollout could get a monitor-free
    reward, or vice versa.

    A third, for SINGLE-TURN envs: a truncated rollout is not graded at all — the env returns
    ``meta["truncated"] = True``, ``task_reward = 0`` and ``behavior_present = False`` without looking at
    the cut-off text (so no hack is ever counted on a rollout no monitor saw). Multi-turn envs are exempt:
    their episode's earlier, complete turns are still graded (e.g. the terminal env's tampering)."""
    if hasattr(env, "score_batch"):
        results = list(env.score_batch(rollouts))  # type: ignore[attr-defined]
        if len(results) != len(rollouts):
            raise RuntimeError(
                f"{type(env).__name__}.score_batch returned {len(results)} results for "
                f"{len(rollouts)} rollouts"
            )
    else:
        results = [env.score(r) for r in rollouts]
    check = getattr(env, "unparseable", None)
    single_turn = not getattr(env, "multi_turn", False)
    for i, (r, er) in enumerate(zip(rollouts, results)):
        why = invalid_reason(env, r)
        if single_turn and why == "truncated" and not (
                er.meta.get("truncated") is True and er.task_reward == 0.0 and er.behavior_present is False):
            raise RuntimeError(f"{type(env).__name__} graded truncated rollout {i}; a single-turn env must "
                               f"return an ungraded result (meta['truncated']=True, task 0, no behavior)")
        if why != "truncated" and "unparsed" in er.meta and bool(er.meta["unparsed"]) != bool(
                check is not None and check(r)):
            raise RuntimeError(f"{type(env).__name__}: rollout {i} has meta['unparsed']="
                               f"{er.meta['unparsed']} but unparseable() disagrees")
        if why is None and er.reward_override is not None:
            raise RuntimeError(f"{type(env).__name__} set reward_override on valid rollout {i}; it is "
                               f"reserved for invalid (truncated / unparseable) rollouts")
        if why is not None:
            er.reward_override = INVALID_ROLLOUT_REWARD
    return results


def _monitor_info(m: Monitor, role: str) -> dict:
    """Best-effort record of what a monitor actually is (so 'cot_weak' is decodable later).

    Includes the judge's VIEW (use_cot/use_output) and reasoning settings: two runs whose judges
    differ in those are not comparable, and a gemini-3.x judge with no reasoning setting cannot even
    complete a call, so these belong in the durable record rather than only in the config file that
    happened to be passed. ``scripts/verify_runs.py`` checks them post-launch."""
    info = {
        "name": m.name,
        "role": role,
        "model_id": getattr(m, "model_id", None),
        "behavior": getattr(m, "behavior", None),
        "threshold": getattr(m, "threshold", None),
    }
    if getattr(m, "model_id", None):  # LLM judges only — probes have none of these
        info.update({
            "use_cot": getattr(m, "use_cot", None),
            "use_output": getattr(m, "use_output", None),
            "binary_judge": getattr(m, "binary_judge", None),
            # the RESOLVED OpenRouter reasoning object every call sends (model default filled in)
            "reasoning": getattr(m, "reasoning", None),
        })
    return info


# Per-rollout monitor failures are collected rather than raised on the spot, so ONE warning names
# every affected rollout instead of a traceback from whichever thread lost first. They are then
# fatal: _assert_scored aborts the run. Transient errors are already retried indefinitely inside the
# monitor, so anything that still fails here is a real defect, not a hiccup.
def _sentinel(err: str) -> MonitorResult:
    return MonitorResult(score=float("nan"), label=False, meta={"error": err})


def _assert_scored(results: dict[str, list[MonitorResult | None]]) -> None:
    """A NaN score is never data — it is a monitor that did not answer. Silently dropping those
    biases a held-out AUROC (the missing rollouts are exactly the ones the API choked on) and
    silently un-penalizes a train-against rollout, so the run stops instead. Loud beats subtle.
    (``None`` is different: an invalid rollout the monitor was deliberately never shown.)"""
    for name, rs in results.items():
        bad = [i for i, r in enumerate(rs) if r is not None and r.score != r.score]
        if bad:
            err = next((rs[i].meta.get("error") for i in bad if rs[i].meta.get("error")), "?")
            raise RuntimeError(
                f"monitor {name!r} returned NaN for {len(bad)}/{len(rs)} rollouts "
                f"(indices {bad[:10]}{'…' if len(bad) > 10 else ''}) — aborting. First error: {err}"
            )


class MonitorScorer:
    """Scores a set of monitors over a batch of rollouts, **starting each rollout's work the moment
    that rollout exists** rather than after the whole batch has been sampled.

    Two monitor families, deliberately treated differently:

    - **API judges** (no ``score_batch``): one thread-pool task per (monitor, rollout), submitted by
      whichever thread finished the rollout — the episode's own driver thread. This is what decouples
      a fast episode's judge latency from the slowest episode in the batch: by the time sampling ends,
      most judge calls are already done.
    - **White-box probes** (``score_batch``): ONE batched local forward over the whole set, run in
      ``collect`` once every rollout is in. This keeps the batch barrier **on purpose** — a probe is a
      single local torch model, so N one-rollout forwards would be strictly slower than one batched
      forward of N, and threading one torch model is pointless (GIL/CUDA-stream serialized) and
      unsafe. Nothing else waits on the probe: the judges have already run.

    Usage: ``with MonitorScorer(monitors, workers) as sc:`` → ``sc.submit(i, rollout)`` per rollout
    (thread-safe, any order) → ``sc.collect(rollouts)`` → ``{name: [MonitorResult | None]}`` indexed
    exactly like ``rollouts``.

    ``skip(rollout) -> bool`` (the RL loop passes "is it invalid?", ``envs.base.invalid_reason``): a
    skipped rollout is never shown to any monitor — no judge call, not in the probes' batch — and its
    entry in every result list is ``None``. It must be a pure function of the rollout: ``submit`` and
    ``collect`` each evaluate it.
    """

    def __init__(self, monitors: Sequence[Monitor], workers: int,
                 skip: Callable[[Rollout], bool] | None = None) -> None:
        self.monitors = list(monitors)
        self.skip = skip
        self.batched = [m for m in self.monitors if hasattr(m, "score_batch")]
        self.threaded = [m for m in self.monitors if not hasattr(m, "score_batch")]
        self._ex = (ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="monitor")
                    if self.threaded else None)
        self._futs: dict[tuple[str, int], Future] = {}
        self._lock = threading.Lock()

    def __enter__(self) -> "MonitorScorer":
        return self

    def __exit__(self, *exc) -> None:
        if self._ex is not None:
            # cancel_futures: on an aborted step, don't keep paying for judge calls nobody will read.
            self._ex.shutdown(wait=True, cancel_futures=True)
            self._ex = None

    def submit(self, index: int, rollout: Rollout) -> None:
        """Queue every API judge on one rollout. Called from the sampling threads, so it only touches
        ``_futs`` under the lock; ``ThreadPoolExecutor.submit`` is itself thread-safe."""
        if self._ex is None or (self.skip is not None and self.skip(rollout)):
            return
        with self._lock:
            for m in self.threaded:
                key = (m.name, index)
                if key in self._futs:
                    raise RuntimeError(f"rollout {index} submitted twice to monitor {m.name!r}")
                self._futs[key] = self._ex.submit(m.score, rollout)

    def collect(self, rollouts: Sequence[Rollout]) -> dict[str, list[MonitorResult | None]]:
        """Drain the judge futures and run the probes; aborts the run if anything failed to score.
        Skipped (invalid) rollouts get ``None`` from every monitor."""
        n = len(rollouts)
        skipped = [self.skip is not None and self.skip(r) for r in rollouts]
        keep = [i for i in range(n) if not skipped[i]]
        out: dict[str, list[MonitorResult | None]] = {}
        for m in self.batched:
            _log(f"      · scoring {m.name} (probe, {len(keep)} local fwd, one batched pass)…")
            res: list[MonitorResult | None] = [None] * n
            try:
                got = list(m.score_batch([rollouts[i] for i in keep])) if keep else []  # type: ignore[attr-defined]
                if len(got) != len(keep):
                    raise RuntimeError(f"score_batch returned {len(got)} results for {len(keep)} rollouts")
            except Exception as e:  # noqa: BLE001 — never let a probe failure kill the run
                _log(f"      ⚠️  {m.name}.score_batch failed ({type(e).__name__}: {e})")
                got = [_sentinel(f"{type(e).__name__}: {e}") for _ in keep]
            for i, r in zip(keep, got):
                res[i] = r
            out[m.name] = res
        for m in self.threaded:
            res, err = [], None
            for i in range(n):
                if skipped[i]:
                    res.append(None)
                    continue
                fut = self._futs.get((m.name, i))
                if fut is None:
                    raise RuntimeError(
                        f"monitor {m.name!r}: rollout {i}/{n} was never submitted for scoring"
                    )
                try:
                    res.append(fut.result())
                except Exception as e:  # noqa: BLE001 — one bad API call shouldn't abort the run
                    res.append(_sentinel(f"{type(e).__name__}: {e}"))
                    err = f"{type(e).__name__}: {e}"
            out[m.name] = res
            if err is not None:  # one concise warning per monitor, not per rollout
                n_failed = sum(1 for r in res if r is not None and r.meta.get("error"))
                _log(f"      ⚠️  {m.name}: {n_failed}/{n} scores failed ({err})")
        _assert_scored(out)
        return out


def _check_penalty(cfg: RunConfig, train_against: Sequence[Monitor]) -> None:
    """λ is set EXACTLY where it takes effect: one of penalty_coef / penalty_schedule with a
    train_against monitor, neither in a control. Anything else would be silently ignored (a control's
    λ multiplies nothing; a coef next to a schedule is overridden by it), so it is an error instead.
    Mirrors the ExperimentConfig validator, for callers that build a RunConfig by hand."""
    coef, sched = cfg.penalty_coef, cfg.penalty_schedule
    if train_against:
        if (coef is None) == (sched is None):
            raise ValueError(
                f"training against {[m.name for m in train_against]}: set exactly one of penalty_coef "
                f"(constant λ) and penalty_schedule (λ ramp), got penalty_coef={coef!r}, "
                f"penalty_schedule={sched!r}")
    elif coef is not None or sched is not None:
        raise ValueError(
            f"no train_against monitor, so no monitor penalty enters the reward and penalty_coef={coef!r} "
            f"/ penalty_schedule={sched!r} would be ignored — pass neither for a control run")


def run_grpo(
    cfg: RunConfig,
    env: Env,
    backend,
    train_against: Sequence[Monitor],
    held_out: Sequence[Monitor] = (),
    *,
    max_tokens: int | None = None,
    think_budget: int | None = None,
    answer_tokens: int | None = None,
    # Concurrent judge API calls, shared across ALL monitors × rollouts. Now that scoring is
    # pipelined into sampling this is the binding constraint on how much of it can overlap: one eval
    # is len(monitors) × eval rollouts of calls (6 × 64 = 384 on the terminal matrix), so 16 meant 24
    # serial waves. Sized to drain an eval in a handful of waves while leaving headroom for several
    # runs sharing one OpenRouter key (queue_runs.sh -j 4 → 4 × this).
    monitor_workers: int = 64,
    extra_rollout_fields: Callable[[Rollout, int], dict] | None = None,
    run_info: dict | None = None,
) -> None:
    """Run GRPO. ``extra_rollout_fields(rollout, idx) -> dict`` lets callers attach arbitrary
    per-rollout metadata to saved rollouts. ``run_info`` is merged into the saved ``run_info.json``
    (use it for anything the caller knows but the loop doesn't, e.g. the dataset subset).
    Sampling is sized EITHER by ``max_tokens`` (one call per turn) OR by ``think_budget`` +
    ``answer_tokens`` (multi-turn envs only: capped thinking, then a forced answer) — exactly one of
    the two, enforced by ``run_episodes``; ``experiment_config.validate_token_budgets`` is where a
    config gets the same treatment. See
    rl/episodes.py. ``think_budget=None`` is taken literally (no budget); resolve the env default
    before calling (``experiment_config.resolve_think_budget``)."""
    # How a turn is sized, for the sampling logs: exactly one of the two modes is in force
    # (run_episodes enforces it; see the docstring).
    _check_penalty(cfg, train_against)
    _budget_note = (f"think_budget={think_budget}+answer_tokens={answer_tokens}"
                    if think_budget is not None else f"max_tokens={max_tokens}")
    rng = random.Random(cfg.seed)
    # Global RNG seeding for any library that reaches for the default generator (numpy/sklearn paths).
    # The tinker sampler is seeded per call → per SAMPLE inside the backend (one single-sample request
    # per rollout; a seeded n-sample request collapses the GRPO group — rl/rollout.py); env construction
    # + holdout + this rng + the LoRA init all take cfg.seed. So one cfg.seed pins the whole run.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _log(f"seed={cfg.seed} (env, holdout, log-sampling, numpy, LoRA init, and per-sample tinker seeds all derive from it)")
    # The behavior word (reward_hacking / sycophancy / deception) names the class-split score metrics, so
    # charts + W&B read per-env instead of the historical "syco". Slugged to a safe metric-key token.
    behavior = re.sub(r"\W+", "_", getattr(env, "behavior_name", None) or "behavior").strip("_") or "behavior"
    all_monitors: list[Monitor] = [*train_against, *held_out]
    run = None
    if cfg.logging.use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            group=cfg.logging.wandb_group,    # cluster all runs of a sweep (matrix × seeds) together
            tags=cfg.logging.wandb_tags or None,
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

    # Backend (tinker SDK) warnings -> <run>/sdk_warnings.log + a cumulative counter in the metrics
    # rows, so "did this run stall on a queue pause?" is answerable from the COMMITTED artifacts.
    sdk_watch = sdk_watch_mod.install(rollout_log_dir)

    rollout_log = (rollout_log_dir / "rollouts.jsonl").open("w")
    # TRAIN metrics (per step, on the training rollouts) and EVAL metrics (periodic, on a FIXED
    # held-out set) go to separate files + wandb namespaces. Degradation curves come from EVAL.
    metrics_log = (rollout_log_dir / "metrics.jsonl").open("w")
    eval_metrics_log = (rollout_log_dir / "eval_metrics.jsonl").open("w")
    # Per-eval-prompt record: EVERY monitor's score on EVERY held-out eval prompt+continuation, each
    # eval step (the full bookkeeping for post-hoc recalibration / the suspiciousness-gap analysis).
    eval_rollouts_log = (rollout_log_dir / "eval_rollouts.jsonl").open("w")
    # The SLIM companion (labels + every monitor's score per rollout, no text) — small enough to COMMIT;
    # it is what the split-half / cross-fit coupling analyses read. Schema: eval/rollout_dump.SLIM_FIELDS.
    eval_rollouts_slim_log = (rollout_log_dir / "eval_rollouts_slim.jsonl").open("w")

    def is_invalid(r: Rollout) -> bool:
        """The monitors' skip predicate: an invalid rollout is never shown to any monitor."""
        return invalid_reason(env, r) is not None

    # The held-out eval set: a FIXED set of prompts, re-scored every eval so curves are comparable.
    # ``holdout`` also REMOVES them from the training pool so eval ⟂ train.
    if hasattr(env, "holdout"):
        eval_prompts = env.holdout(cfg.eval_size, seed=cfg.seed)
    else:
        _log("⚠️  env has no holdout(); eval prompts overlap the training pool (not disjoint).")
        eval_prompts = [env.sample_prompt() for _ in range(cfg.eval_size)]

    def run_eval(step: int) -> None:
        """Sample the current policy on the fixed eval set; log every monitor's degradation metrics.

        AUROC is the headline but goes NaN at single-class (behavior≈0 or ≈1); ``brier`` and the
        behavior-present mean score (``mean_score_<behavior>``) stay defined there, so the
        obfuscation-success regime stays measurable.

        Monitor metrics are over the VALID eval rollouts only (invalid ones are never monitored), so
        ``n_scored`` = the valid count and ``n_pos``/``n_neg`` split it; the oracle rates are over all.
        """
        n_per = getattr(cfg, "eval_samples_per_prompt", 1) or 1
        _log(f"  [eval @ {step}] sampling {len(eval_prompts)}×{n_per} held-out rollouts ({_budget_note})…")
        t0 = time.perf_counter()
        # Judge calls are pipelined INTO the sampling: each rollout is handed to the monitors the
        # instant its episode finishes, so `sample_s` already contains most of the scoring and
        # `score_s` is only whatever was still in flight when the last episode landed.
        with MonitorScorer(all_monitors, monitor_workers, skip=is_invalid) as scorer:
            ev, streamed = _sample(backend, env, eval_prompts, num_samples=n_per,
                                   max_tokens=max_tokens, think_budget=think_budget,
                                   answer_tokens=answer_tokens, on_rollout=scorer.submit)
            if not streamed:  # backend has no per-rollout hook — score the finished batch
                for i, r in enumerate(ev):
                    scorer.submit(i, r)
            ev_env = _score_env(env, ev)
            gt = [er.behavior_present for er in ev_env]
            invalid = [invalid_reason(env, r) for r in ev]  # None = valid = monitored
            t_sample = time.perf_counter() - t0
            _log(f"  [eval @ {step}] draining {len(all_monitors)} monitors…")
            t1 = time.perf_counter()
            res = scorer.collect(ev)
            t_score = time.perf_counter() - t1
        row: dict = {
            "step": step,
            "behavior_rate": mean(float(g) for g in gt),  # = strict_rh (the AUROC oracle)
            # single-turn envs: a truncated rollout is not graded, so it is NOT unparsed — there,
            # unparsed_rate + truncated_rate = invalid_rate
            "unparsed_rate": mean(float(er.meta.get("unparsed", False)) for er in ev_env),
            # fraction truncated or unparseable = NOT monitored: every monitor metric below is over the rest
            "invalid_rate": mean(float(why is not None) for why in invalid),
            "truncated_rate": mean(float(why == "truncated") for why in invalid),
            **_length_metrics(ev_env, env),  # code length + its reward penalty (MBPP)
            **_env_metrics(ev_env, env),     # env-declared summary rates (terminal env: read/modified/…)
        }
        # MBPP-Honeypot three-way reward-hacking labels (loose ⊇ strict=behavior_rate; + hardcoding).
        # gt_loose = the LOOSE oracle (passes-visible-not-all, incl. honest-buggy) — a separate, denser
        # label track: it's defined at more eval steps than strict (loose_rate ≥ strict_rate) but is a
        # noisier construct, so its monitor metrics live under a separate `loose/` namespace, never merged.
        gt_loose = ([bool(er.meta.get("loose_rh", False)) for er in ev_env]
                    if any("loose_rh" in er.meta for er in ev_env) else None)
        if gt_loose is not None:
            row["loose_rate"] = mean(float(er.meta.get("loose_rh", False)) for er in ev_env)
            row["hardcoding_rate"] = mean(float(er.meta.get("hardcoding", False)) for er in ev_env)
        for m in all_monitors:
            # Every VALID rollout is scored (MonitorScorer.collect aborts the run otherwise) and no
            # invalid one is (None), so this indexes exactly the valid eval rollouts; n_scored = their
            # count, a visible invariant check.
            idx = [i for i, x in enumerate(res[m.name]) if x is not None]
            scores = [res[m.name][i].score for i in idx]
            preds = [res[m.name][i].label for i in idx]
            mgt = [gt[i] for i in idx]  # strict labels (the headline oracle = behavior_present)
            row[f"monitor/{m.name}/auroc"] = roc_auc(scores, mgt)
            row[f"monitor/{m.name}/accuracy"] = accuracy(preds, mgt)
            row[f"monitor/{m.name}/brier"] = brier(scores, mgt)
            row[f"monitor/{m.name}/mean_score"] = mean(scores) if scores else float("nan")
            row[present_score_key(m.name, behavior)] = _mean_for_class(scores, mgt, True)
            row[absent_score_key(m.name, behavior)] = _mean_for_class(scores, mgt, False)
            # margin d′ (non-saturating separation, no AUROC ceiling) + class counts (n_pos·n_neg gives
            # the AUROC resolution, so a d′-via-AUROC clip can be sample-size-aware downstream).
            row[f"monitor/{m.name}/dprime_margin"] = dprime_margin(scores, mgt)
            row[f"monitor/{m.name}/n_pos"] = sum(mgt)
            row[f"monitor/{m.name}/n_neg"] = len(mgt) - sum(mgt)
            row[f"monitor/{m.name}/n_scored"] = len(scores)  # how many rollouts actually scored
            if gt_loose is not None:  # same scores, LOOSE labels — separate track (see note above)
                row.update(_label_metrics(f"monitor/{m.name}/loose", scores, preds, [gt_loose[i] for i in idx]))
        # sample_s now COVERS the judge calls that ran concurrently with sampling; score_s is only
        # the tail still in flight when the last episode landed (plus the probes' batched forward).
        # Their sum is still the true wall clock of the sample+score phase.
        row["time/sample_s"] = t_sample
        row["time/score_s"] = t_score
        row["time/total_s"] = t_sample + t_score
        eval_metrics_log.write(json.dumps(row) + "\n")
        eval_metrics_log.flush()
        # Per-prompt dump: one record per eval rollout with EVERY monitor's score+label (so monitors
        # can be re-thresholded/recalibrated post-hoc, and the hack-vs-honest gap computed per prompt).
        for i, (r, er, g) in enumerate(zip(ev, ev_env, gt)):
            rec = {
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
                # null (valid), "truncated" or "unparsed": an invalid rollout was shown to no monitor,
                # so its `monitors` is {} and it is in none of the monitor metrics.
                "invalid_reason": invalid[i],
                "env_meta": er.meta,  # full per-env grading record (see the train-rollout dump above)
                # {score, label} + for every LLM judge its `call`: the exact request (prompt + API
                # parameters) and response (content + chain of thought) of the successful API call.
                "monitors": {m.name: monitor_record(res[m.name][i]) for m in all_monitors
                             if res[m.name][i] is not None},
            }
            eval_rollouts_log.write(json.dumps(rec, default=str) + "\n")
            eval_rollouts_slim_log.write(json.dumps(slim_record(rec), default=str) + "\n")
        eval_rollouts_log.flush()
        eval_rollouts_slim_log.flush()
        if run is not None:
            run.log({f"eval/{k}": v for k, v in row.items() if k != "step"}, step=step)
        # Print AUROC, but fall back to Brier in the readout so a NaN-AUROC eval still shows a number.
        summ = " ".join(
            f"{m.name}:auc={row[f'monitor/{m.name}/auroc']:.2f}/brier={row[f'monitor/{m.name}/brier']:.2f}"
            for m in all_monitors
        )
        _log(f"  [eval @ {step}] behavior={row['behavior_rate']:.2f} invalid={row['invalid_rate']:.2f} | {summ} "
             f"| {t_sample + t_score:.0f}s (sample {t_sample:.0f}s, score {t_score:.0f}s)")

    for step in range(cfg.n_steps):
        sdk_watch.step = step  # so each persisted warning names the step it landed on
        if step % cfg.eval_every == 0:
            run_eval(step)  # held-out eval at step 0 and every eval_every
            
        if step % cfg.save_every == 0 and hasattr(backend, "save_checkpoint"):
            # TTL only where checkpoints can expire (tinker-hosted state). A backend that writes to
            # local disk says so with checkpoints_expire = False and is asked for no TTL at all —
            # passing one it cannot honour is an error there, not a silently dropped request.
            ttl = {"ttl_seconds": 4 * 7 * 24 * 60 * 60} if getattr(backend, "checkpoints_expire", False) else {}
            ckpt = backend.save_checkpoint(f"{cfg.logging.run_name or 'run'}-{step}", **ttl)
            (rollout_log_dir / f"checkpoint_{step}.txt").write_text(ckpt + "\n")
            info[f"checkpoint_{step}"] = ckpt
            _log(f"saved checkpoint at step {step}: {ckpt}")

        prompts = [env.sample_prompt() for _ in range(cfg.batch_size)]
        _log(f"step {step}: sampling {cfg.batch_size}×{cfg.group_size} train rollouts "
             f"({_budget_note})…")
        t0 = time.perf_counter()
        # Only the TRAIN-AGAINST monitors are scored every step (they form the reward penalty).
        # Held-out monitors are measured on the held-out eval set instead — cleaner and cheaper.
        # As in run_eval, their calls are pipelined into sampling: a rollout is handed to the judges
        # the moment its episode finishes, so t_score below is only the tail still in flight.
        with MonitorScorer(train_against, monitor_workers, skip=is_invalid) as scorer:
            rollouts, streamed = _sample(backend, env, prompts, num_samples=cfg.group_size,
                                         max_tokens=max_tokens, think_budget=think_budget,
                                         answer_tokens=answer_tokens, on_rollout=scorer.submit)
            if not streamed:  # backend has no per-rollout hook — score the finished batch
                for i, r in enumerate(rollouts):
                    scorer.submit(i, r)
            env_results = _score_env(env, rollouts)
            invalid = [invalid_reason(env, r) for r in rollouts]  # None = valid = monitored
            valid = [i for i, why in enumerate(invalid) if why is None]
            t_sample = time.perf_counter() - t0
            if train_against:
                _log(f"step {step}: draining train-against monitor(s)…")
            t1 = time.perf_counter()
            ta_res = scorer.collect(rollouts) if train_against else {}
            t_score = time.perf_counter() - t1

        # ``penalties[i]`` = raw mean train-against suspiciousness for rollout i (BEFORE the coefficient).
        # This is the obfuscation signal (drops as the policy evades) — logged as monitor_suspiciousness_mean.
        # No NaN guard is needed: MonitorScorer.collect aborts the run on the first unscored VALID
        # rollout, so a missing penalty can never silently reach the reward (or NaN the whole group's
        # advantages). An invalid rollout has no score at all (never monitored) → None here; its reward
        # is the -1 override, so no penalty is needed for it.
        penalties = [
            None if invalid[i] is not None
            else mean(ta_res[m.name][i].score for m in train_against) if train_against else 0.0
            for i in range(len(rollouts))
        ]
        # Effective λ this step: constant penalty_coef, OR a linear ramp start→end if penalty_schedule is
        # set (hack-then-hide curriculum). Logged as penalty_coef so the schedule is visible; the applied
        # reward penalty (penalty_mean) is λ·suspiciousness, now decoupled since λ can vary over training.
        # A control (no train_against) has no λ at all — _check_penalty guarantees both keys are None
        # there; its penalties are all 0.0 and the effective λ is 0 by construction.
        pen_coef = cfg.penalty_coef if train_against else 0.0
        if cfg.penalty_schedule is not None:  # (only ever set alongside a train_against monitor)
            frac = step / max(1, cfg.n_steps - 1)  # 0 at step 0 → 1 at the last step
            s, e = cfg.penalty_schedule["start_penalty"], cfg.penalty_schedule["end_penalty"]
            pen_coef = s + (e - s) * frac
        # A ``reward_override`` (truncated by max_tokens / unparseable output → -1) IS the reward: the
        # monitor penalty is not applied to it, though its monitor scores are still logged.
        applied = [0.0 if er.reward_override is not None else pen_coef * pen
                   for er, pen in zip(env_results, penalties)]  # _score_env: override ⇔ invalid ⇔ pen None
        rewards = [er.reward_override if er.reward_override is not None else er.task_reward - ap
                   for er, ap in zip(env_results, applied)]
        t2 = time.perf_counter()
        # The backend adapts rollouts+rewards into cookbook trajectory groups and does the whole loss
        # layer (advantage centering, mask/datum assembly, KL penalty, forward_backward) via cookbook
        # primitives — see backends/tinker_backend.py:train_step.
        step_metrics = backend.train_step(rollouts, rewards, cfg.group_size)
        t_optim = time.perf_counter() - t2

        gt_train = [er.behavior_present for er in env_results]  # strict_rh on the training rollouts
        gt_rate = mean(float(g) for g in gt_train)
        gt_train_loose = ([bool(er.meta.get("loose_rh", False)) for er in env_results]
                          if any("loose_rh" in er.meta for er in env_results) else None)
        # Per-train-against-monitor metrics on the TRAINING rollouts: full set (auroc/accuracy/brier +
        # class-split) vs strict, plus the loose track. AUROC on train is often NaN (single-class batch)
        # but brier + the class-split means stay defined and carry the obfuscation signal.
        # Over the VALID rollouts only — invalid ones were never monitored (see the module docstring).
        mon: dict[str, float] = {}
        gt_v = [gt_train[i] for i in valid]
        for m in train_against:
            s = [ta_res[m.name][i].score for i in valid]
            p = [ta_res[m.name][i].label for i in valid]
            mon[f"monitor/{m.name}/mean_score"] = mean(s) if s else float("nan")
            mon[f"monitor/{m.name}/auroc"] = roc_auc(s, gt_v)
            mon[f"monitor/{m.name}/accuracy"] = accuracy(p, gt_v)
            mon[f"monitor/{m.name}/brier"] = brier(s, gt_v)
            mon[f"monitor/{m.name}/dprime_margin"] = dprime_margin(s, gt_v)
            mon[present_score_key(m.name, behavior)] = _mean_for_class(s, gt_v, True)
            mon[absent_score_key(m.name, behavior)] = _mean_for_class(s, gt_v, False)
            mon[f"monitor/{m.name}/n_pos"] = sum(gt_v)
            mon[f"monitor/{m.name}/n_neg"] = len(gt_v) - sum(gt_v)
            mon[f"monitor/{m.name}/n_scored"] = len(s)
            if gt_train_loose is not None:
                mon.update(_label_metrics(f"monitor/{m.name}/loose", s, p, [gt_train_loose[i] for i in valid]))
        loss_val = step_metrics.get("loss", float("nan"))
        logp_mean = step_metrics.get("train/logprob_mean", float("nan"))
        row = {
            "step": step,
            "behavior_rate": gt_rate,  # = strict_rh (training rollouts; eval has the held-out version)
            **({"loose_rate": mean(float(er.meta.get("loose_rh", False)) for er in env_results),
                "hardcoding_rate": mean(float(er.meta.get("hardcoding", False)) for er in env_results)}
               if any("loose_rh" in er.meta for er in env_results) else {}),
            "reward/task_mean": mean(er.task_reward for er in env_results),
            # raw mean train-against suspiciousness (BEFORE λ) — the obfuscation signal, decoupled from λ.
            # Over the VALID (monitored) rollouts only: averaging in the unmonitored ones as 0 would read
            # a rise in truncation as evasion. NaN if none is valid; 0.0 in a control (no monitor).
            "reward/monitor_suspiciousness_mean": (
                0.0 if not train_against
                else mean(penalties[i] for i in valid) if valid else float("nan")),
            "reward/penalty_coef": pen_coef,                        # effective λ this step (schedule-aware)
            "reward/penalty_mean": mean(applied),  # APPLIED penalty = λ · suspiciousness (0 where overridden)
            "reward/override_rate": mean(float(er.reward_override is not None) for er in env_results),
            # = override_rate (_score_env enforces override ⇔ invalid): the fraction NOT monitored.
            "invalid_rate": mean(float(why is not None) for why in invalid),
            "reward/total_mean": mean(rewards),
            **_rollout_metrics(rollouts, rewards, cfg.group_size),  # reward spread, lengths, truncation
            **_length_metrics(env_results, env),  # code length + its reward penalty (MBPP)
            **_env_metrics(env_results, env),     # env-declared summary rates
            "unparsed_rate": mean(float(er.meta.get("unparsed", False)) for er in env_results),
            "time/sample_s": t_sample,   # includes the judge calls pipelined into sampling
            "time/score_s": t_score,     # only the monitor tail left after the last rollout landed
            "time/optim_s": t_optim,
            "time/total_s": t_sample + t_score + t_optim,
            "kl/mean": step_metrics.get("kl/mean", 0.0),  # per-token KL to base (0 if kl_coef=0)
            **sdk_watch.metrics(),  # cumulative backend queue-pause / SDK warning counts
            **{f"loss/{k}": v for k, v in step_metrics.items()},
            **mon,  # per-train-against-monitor metrics (strict + loose), built above
        }
        if run is not None:
            run.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=step)
        metrics_log.write(json.dumps(row) + "\n")
        metrics_log.flush()
        kl_note = f" kl={row['kl/mean']:.3f}" if row["kl/mean"] else ""
        # The IS loss (tinker's loss:sum, recomputed in rl/grpo.optim_metrics); fall back to the mean
        # training logprob for a backend that reports no loss.
        train_note = f"loss={loss_val:.1f}" if loss_val == loss_val else f"logp={logp_mean:.2f}"
        if "entropy" in step_metrics:
            train_note += f" ent={step_metrics['entropy']:.2f}"
        _log(
            f"step {step}: behavior={gt_rate:.2f} task_r={row['reward/task_mean']:.2f} "
            f"penalty={row['reward/penalty_mean']:.2f} {train_note}{kl_note} "
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
                            "reward_override": er.reward_override,
                            "behavior_present": er.behavior_present,
                            "choice": er.meta.get("choice"),
                            "unparsed": er.meta.get("unparsed"),
                            # The env's full grading record (per-env keys: MBPP's loose_rh/hardcoding,
                            # ImpossibleBench's frac_passed/passes_original/first_error, …) so a run can
                            # be re-analysed without re-running the env.
                            "meta": er.meta,
                        },
                        # {score, label} + each LLM judge's `call` (exact request + response) —
                        # see eval.rollout_dump.monitor_record.
                        # {} for an invalid rollout — never shown to a monitor (see invalid_reason).
                        "monitors": {m.name: monitor_record(ta_res[m.name][i]) for m in train_against
                                     if ta_res[m.name][i] is not None},
                        "invalid_reason": invalid[i],  # null | "truncated" | "unparsed"
                        "reward": rewards[i],  # advantage is now computed in the backend (cookbook)
                        "extra": extra,
                    },
                    default=str,  # never let an exotic meta value abort a run mid-step
                )
                + "\n"
            )
        rollout_log.flush()

    run_eval(cfg.n_steps)  # final held-out eval

    # Save the final training state on tinker (7-day TTL) so we can resume / sample / inspect the
    # trained model later — important when we don't yet know how long these runs should take.
    if hasattr(backend, "save_checkpoint"):
        try:
            # ttl_seconds=None = no expiry (tinker rejects 0: "must be at least 3600"), which is
            # also what a local-disk backend does natively — so no TTL argument either way.
            ckpt = backend.save_checkpoint(f"{cfg.logging.run_name or 'run'}-final")
            (rollout_log_dir / "final_checkpoint.txt").write_text(ckpt + "\n")
            info["final_checkpoint"] = ckpt
            (rollout_log_dir / "run_info.json").write_text(json.dumps(info, indent=2))
            _log(f"saved final weights on tinker (no expiry): {ckpt}")
        except Exception as e:  # noqa: BLE001 — never let a save hiccup fail a completed run
            _log(f"⚠️  final checkpoint save failed: {type(e).__name__}: {e}")

    sdk_watch_mod.uninstall(sdk_watch)
    rollout_log.close()
    metrics_log.close()
    eval_metrics_log.close()
    eval_rollouts_log.close()
    eval_rollouts_slim_log.close()
    if run is not None:
        run.finish()

    # Auto-plot on finish (best-effort — never let a plotting hiccup fail a completed run).
    try:
        from monitordecorrelation.eval.plots import plot_run as _plot_run

        for p in _plot_run(rollout_log_dir):
            _log(f"plot: {p}")
    except Exception as e:  # noqa: BLE001
        _log(f"(auto-plot skipped: {e})")
