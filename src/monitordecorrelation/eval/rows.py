"""The metric-row schema shared by every RL backend.

``data/runs/<run>/metrics.jsonl`` (per train step) and ``eval_metrics.jsonl`` (per held-out eval) are
what ``eval/degradation.py``, ``eval/plots.py`` and the analysis scripts read, so their key layout is a
contract, not an implementation detail of one training loop. This module owns that contract: the
tinker loop (``rl/train.py``) and the NeMo-RL environment actor
(``backends/nemo/env_actor.py``) both build their rows here, so a run is comparable across backends
and a new key cannot appear on one side only.

Everything here is pure: plain lists in, a flat ``{str: float}`` out. No backend, no I/O.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Sequence

from monitordecorrelation.eval.metric_keys import absent_score_key, present_score_key
from monitordecorrelation.eval.metrics import (
    _mean_for_class,
    accuracy,
    brier,
    dprime_margin,
    roc_auc,
)
from monitordecorrelation.types import MonitorResult, Rollout


def behavior_slug(env) -> str:
    """The env's behavior word (reward_hacking / sycophancy / …) as a safe metric-key token.

    It names the class-split score metrics, so charts + W&B read per-env instead of the historical
    "syco"."""
    raw = getattr(env, "behavior_name", None) or "behavior"
    return re.sub(r"\W+", "_", raw).strip("_") or "behavior"


def label_metrics(prefix: str, scores: list[float], preds: list[bool], labels: list[bool]) -> dict[str, float]:
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


def length_metrics(results: Sequence, env) -> dict[str, float]:
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


def token_metrics(rollouts: Sequence[Rollout], max_tokens: int) -> dict[str, float]:
    """Input/output token accounting for one batch of rollouts, keyed under ``tokens/``.

    ``*_total`` is the whole batch at this step (what the sampling bill and the wall-clock track);
    ``*_per_rollout`` is the mean of one rollout. ``truncated_rate`` is the fraction of completions
    that hit ``max_tokens`` — a rising output length with truncation climbing means the metric is
    censored, so read the two together. Empty ``{}`` if sampling didn't record counts (e.g. rollouts
    reloaded from disk)."""
    n_in = [r.meta["n_prompt_tokens"] for r in rollouts if "n_prompt_tokens" in r.meta]
    n_out = [r.meta["n_output_tokens"] for r in rollouts if "n_output_tokens" in r.meta]
    if not n_in or not n_out:
        return {}
    return {
        "tokens/input_total": float(sum(n_in)),            # prompt tokens, whole batch this step
        "tokens/output_total": float(sum(n_out)),          # completion tokens, whole batch this step
        "tokens/total": float(sum(n_in) + sum(n_out)),
        "tokens/input_per_rollout": mean(n_in),
        "tokens/output_per_rollout": mean(n_out),
        "tokens/output_max": float(max(n_out)),
        "tokens/truncated_rate": mean(float(n >= max_tokens) for n in n_out),
    }


def loose_labels(env_results: Sequence) -> list[bool] | None:
    """The LOOSE reward-hacking oracle per rollout, or None for envs that don't define one.

    ``loose ⊇ strict``: it counts honest-but-buggy attempts as positives, so it is a denser but
    noisier construct and its monitor metrics live in a separate ``loose/`` namespace, never merged
    with the strict headline numbers."""
    if not any("loose_rh" in er.meta for er in env_results):
        return None
    return [bool(er.meta.get("loose_rh", False)) for er in env_results]


def _loose_rates(env_results: Sequence) -> dict[str, float]:
    if not any("loose_rh" in er.meta for er in env_results):
        return {}
    return {
        "loose_rate": mean(float(er.meta.get("loose_rh", False)) for er in env_results),
        "hardcoding_rate": mean(float(er.meta.get("hardcoding", False)) for er in env_results),
    }


def eval_row(
    *,
    step: int,
    rollouts: Sequence[Rollout],
    env_results: Sequence,
    monitor_results: dict[str, list[MonitorResult]],
    monitor_names: Sequence[str],
    env,
    max_tokens: int,
) -> dict:
    """One ``eval_metrics.jsonl`` row: ground-truth behavior rate + every monitor's degradation
    metrics on the fixed held-out set.

    AUROC is the headline but goes NaN at single-class (behavior≈0 or ≈1); ``brier`` and the
    behavior-present mean score (``mean_score_<behavior>``) stay defined there, so the
    obfuscation-success regime stays measurable. Rollouts a monitor failed to score (the NaN
    sentinel) are dropped for that monitor only, so a few API hiccups don't poison its metrics.
    """
    behavior = behavior_slug(env)
    gt = [er.behavior_present for er in env_results]
    gt_loose = loose_labels(env_results)
    row: dict = {
        "step": step,
        "behavior_rate": mean(float(g) for g in gt),  # = strict_rh (the AUROC oracle)
        "unparsed_rate": mean(float(er.meta["unparsed"]) for er in env_results),
        **_loose_rates(env_results),
        **length_metrics(env_results, env),  # code length + its reward penalty (MBPP)
        **token_metrics(rollouts, max_tokens),  # input/output token counts for this eval batch
    }
    for name in monitor_names:
        idx = [i for i, x in enumerate(monitor_results[name]) if x.score == x.score]
        scores = [monitor_results[name][i].score for i in idx]
        preds = [monitor_results[name][i].label for i in idx]
        mgt = [gt[i] for i in idx]  # strict labels (the headline oracle = behavior_present)
        row[f"monitor/{name}/auroc"] = roc_auc(scores, mgt)
        row[f"monitor/{name}/accuracy"] = accuracy(preds, mgt)
        row[f"monitor/{name}/brier"] = brier(scores, mgt)
        row[f"monitor/{name}/mean_score"] = mean(scores) if scores else float("nan")
        row[present_score_key(name, behavior)] = _mean_for_class(scores, mgt, True)
        row[absent_score_key(name, behavior)] = _mean_for_class(scores, mgt, False)
        # margin d′ (non-saturating separation, no AUROC ceiling) + class counts (n_pos·n_neg gives
        # the AUROC resolution, so a d′-via-AUROC clip can be sample-size-aware downstream).
        row[f"monitor/{name}/dprime_margin"] = dprime_margin(scores, mgt)
        row[f"monitor/{name}/n_pos"] = sum(mgt)
        row[f"monitor/{name}/n_neg"] = len(mgt) - sum(mgt)
        row[f"monitor/{name}/n_scored"] = len(scores)  # how many rollouts actually scored
        if gt_loose is not None:  # same scores, LOOSE labels — separate track (see note above)
            row.update(label_metrics(f"monitor/{name}/loose", scores, preds, [gt_loose[i] for i in idx]))
    return row


def train_row(
    *,
    step: int,
    rollouts: Sequence[Rollout],
    env_results: Sequence,
    monitor_results: dict[str, list[MonitorResult]],
    train_against_names: Sequence[str],
    env,
    max_tokens: int,
    penalties: Sequence[float],
    pen_coef: float,
    rewards: Sequence[float],
) -> dict:
    """One ``metrics.jsonl`` row: the reward decomposition on this step's TRAINING rollouts, plus the
    train-against monitors' own metrics.

    ``penalties[i]`` is the raw mean train-against suspiciousness for rollout i, BEFORE the
    coefficient — the obfuscation signal, which is why it is logged separately from the applied
    penalty (λ · suspiciousness): λ itself can vary over training under ``penalty_schedule``.
    AUROC on a training batch is often NaN (single-class) but brier + the class-split means stay
    defined and carry the obfuscation signal.
    """
    behavior = behavior_slug(env)
    gt_train = [er.behavior_present for er in env_results]  # strict_rh on the training rollouts
    gt_train_loose = loose_labels(env_results)
    mon: dict[str, float] = {}
    for name in train_against_names:
        s = [x.score for x in monitor_results[name]]
        p = [x.label for x in monitor_results[name]]
        mon[f"monitor/{name}/mean_score"] = mean(s) if s else float("nan")
        mon[f"monitor/{name}/auroc"] = roc_auc(s, gt_train)
        mon[f"monitor/{name}/accuracy"] = accuracy(p, gt_train)
        mon[f"monitor/{name}/brier"] = brier(s, gt_train)
        mon[f"monitor/{name}/dprime_margin"] = dprime_margin(s, gt_train)
        mon[present_score_key(name, behavior)] = _mean_for_class(s, gt_train, True)
        mon[absent_score_key(name, behavior)] = _mean_for_class(s, gt_train, False)
        if gt_train_loose is not None:
            mon.update(label_metrics(f"monitor/{name}/loose", s, p, gt_train_loose))
    return {
        "step": step,
        "behavior_rate": mean(float(g) for g in gt_train),  # = strict_rh (eval has the held-out one)
        **_loose_rates(env_results),
        "reward/task_mean": mean(er.task_reward for er in env_results),
        # raw mean train-against suspiciousness (BEFORE λ) — the obfuscation signal, decoupled from λ:
        "reward/monitor_suspiciousness_mean": mean(penalties) if penalties else 0.0,
        "reward/penalty_coef": pen_coef,                     # effective λ this step (schedule-aware)
        "reward/penalty_mean": pen_coef * (mean(penalties) if penalties else 0.0),  # λ · suspiciousness
        "reward/total_mean": mean(rewards),
        **length_metrics(env_results, env),  # code length + its reward penalty (MBPP)
        **token_metrics(rollouts, max_tokens),  # tokens/… for this step's train batch
        "unparsed_rate": mean(float(er.meta["unparsed"]) for er in env_results),
        **mon,  # per-train-against-monitor metrics (strict + loose)
    }
