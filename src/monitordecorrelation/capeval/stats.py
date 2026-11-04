"""Statistics + report: per-model scores and checkpoint-minus-base deltas with cluster-bootstrap 95% CIs.

Unit of analysis = the item (its score is the mean over its valid samples); the bootstrap resamples
CLUSTERS of items (= the source problem, e.g. CanItEdit's descriptive + lazy variant of one problem),
so correlated items are not counted as independent. A delta is PAIRED: only items with a valid sample
for both the checkpoint and its base model enter it, and the bootstrap resamples those items' clusters
for both models at once — far tighter than comparing two independent CIs.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

N_BOOT = 4000


def item_means(records: list[dict], metric: str) -> dict[str, tuple[str, float, int]]:
    """item id -> (cluster, mean of ``metric`` over the item's valid samples, n samples)."""
    acc: dict[str, list] = {}
    for r in records:
        if r.get("error") or metric not in r.get("scores", {}):
            continue
        v = r["scores"][metric]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        cl, vals = acc.setdefault(r["item"], [r["cluster"], []])
        vals.append(float(v))
    return {i: (cl, float(np.mean(v)), len(v)) for i, (cl, v) in acc.items()}


def _cluster_bootstrap(values: dict[str, tuple[str, float]], seed: int) -> dict:
    """Mean over items + percentile CI from resampling clusters. ``values``: item -> (cluster, value)."""
    if not values:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "se": None, "n_items": 0, "n_clusters": 0}
    sums: dict[str, float] = defaultdict(float)
    cnts: dict[str, int] = defaultdict(int)
    for cl, v in values.values():
        sums[cl] += v
        cnts[cl] += 1
    keys = sorted(sums)
    s = np.array([sums[k] for k in keys])
    c = np.array([cnts[k] for k in keys], dtype=float)
    mean = float(s.sum() / c.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": mean, "ci_lo": float(lo), "ci_hi": float(hi), "se": float(boots.std(ddof=1)),
            "n_items": len(values), "n_clusters": len(keys)}


def score_summary(records: list[dict], metric: str, seed: int = 0) -> dict:
    im = item_means(records, metric)
    out = _cluster_bootstrap({i: (cl, m) for i, (cl, m, _) in im.items()}, seed)
    out["n_samples"] = sum(n for _, _, n in im.values())
    return out


def paired_delta(ckpt_records: list[dict], base_records: list[dict], metric: str, seed: int = 0) -> dict:
    a = item_means(ckpt_records, metric)
    b = item_means(base_records, metric)
    common = sorted(set(a) & set(b))
    out = _cluster_bootstrap({i: (a[i][0], a[i][1] - b[i][1]) for i in common}, seed)
    out["significant"] = (out["mean"] is not None and out["n_items"] > 1
                          and (out["ci_lo"] > 0 or out["ci_hi"] < 0))
    out["n_dropped_unpaired"] = len(set(a) ^ set(b))
    return out


def group_summaries(records: list[dict], metric: str, seed: int = 0) -> dict:
    by: dict[str, list] = defaultdict(list)
    for r in records:
        by[r.get("group", "all")].append(r)
    return {g: score_summary(rs, metric, seed) for g, rs in sorted(by.items())} if len(by) > 1 else {}


def diagnostics(records: list[dict]) -> dict:
    """Confounder diagnostics over one (model, benchmark)'s records."""
    ok = [r for r in records if not r.get("error")]
    d: dict = {"n_records": len(records), "n_errors": len(records) - len(ok)}
    if not ok:
        return d
    diags = [r.get("diag", {}) for r in ok]
    n_gen = sum(x.get("n_gen", 0) for x in diags) or 1
    d["forced_rate"] = sum(x.get("n_forced", 0) for x in diags) / n_gen          # per generation call
    d["truncated_rate"] = sum(x.get("n_truncated", 0) for x in diags) / n_gen
    d["ctx_clipped_rate"] = sum(x.get("n_ctx_clipped", 0) for x in diags) / n_gen
    d["context_exhausted_rate"] = float(np.mean([bool(x.get("context_exhausted")) for x in diags]))
    d["gen_tokens_mean"] = float(np.mean([x.get("gen_tokens", 0) for x in diags]))
    d["gen_tokens_p95"] = float(np.percentile([x.get("gen_tokens", 0) for x in diags], 95))
    for k in ("extract_fail", "turn_limit", "submitted", "no_solution_file", "judge_fail", "empty_reply",
              "passed_public", "public_pass_hidden_fail"):
        vals = [bool(x[k]) for x in diags if k in x]
        if vals:
            d[f"{k}_rate"] = float(np.mean(vals))
    for k in ("turns", "format_errors", "multi_blocks", "premature_submits", "cmd_timeouts", "reply_chars",
              "cjk_frac", "test_frac"):
        vals = [float(x[k]) for x in diags if k in x]
        if vals:
            d[f"{k}_mean"] = float(np.mean(vals))
    ends = defaultdict(int)
    for x in diags:
        if "end_reason" in x:
            ends[x["end_reason"]] += 1
    if ends:
        d["end_reasons"] = dict(ends)
    return d


# ---------------------------------------------------------------------------------------------
# Confounder checks: rates that, if they differ between a checkpoint and its base, could explain a
# score difference by themselves (or that are high enough to cap scores for both).
# ---------------------------------------------------------------------------------------------
# (key, label, absolute level that warrants a warning, checkpoint-vs-base difference that does)
CONFOUNDERS = [
    ("forced_rate", "thinking force-closed at budget", 0.10, 0.05),
    ("truncated_rate", "answer cut off at token cap", 0.02, 0.02),
    ("context_exhausted_rate", "context window full", 0.02, 0.02),
    ("extract_fail_rate", "no extractable final answer", 0.05, 0.05),
    ("turn_limit_rate", "hit turn limit", 0.10, 0.05),
    ("format_errors_mean", "replies w/o bash block per ep", 1.0, 0.5),
    ("multi_blocks_mean", "multi-block replies per ep", 1.0, 0.5),
    ("premature_submits_mean", "premature submits per ep", 0.2, 0.2),
    ("cmd_timeouts_mean", "command timeouts per ep", 1.0, 0.5),
    ("no_solution_file_rate", "no solution file at end", 0.05, 0.05),
    ("judge_fail_rate", "judge gave up (content filter / unparseable; excluded)", 0.0, 0.0),
    ("empty_reply_rate", "empty visible reply", 0.02, 0.02),
]


def confounder_warnings(diag_by_model: dict[str, dict], pairs: dict[str, str], bench: str) -> list[str]:
    """One line per (benchmark, model) listing every rate that is high (> its level) or — for a
    checkpoint — far from its base model's (> its diff: big enough to move the delta by itself)."""
    out = []
    for key, d in diag_by_model.items():
        b = diag_by_model.get(pairs[key], {}) if key in pairs else {}
        flags = []
        if d.get("n_errors"):
            flags.append(f"{d['n_errors']}/{d['n_records']} samples ERRORED, excluded (--resume retries them)")
        for k, label, level, diff in CONFOUNDERS:
            v = d.get(k)
            if v is None:
                continue
            differs = k in b and abs(v - b[k]) > diff
            if v > level or differs:
                ref = f" (base {_fmt_rate(k, b[k])}{' ≠' if differs else ''})" if k in b else ""
                flags.append(f"{label} {_fmt_rate(k, v)}{ref}")
        if flags:
            out.append(f"{bench} · {key}: " + "; ".join(flags))
    return out


def _fmt_rate(k: str, v: float) -> str:
    return f"{v:.2f}" if k.endswith("_mean") else f"{100 * v:.1f}%"


# ---------------------------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------------------------
def _pct(metric: str) -> bool:
    return metric in ("accuracy", "pass", "repetition")


def fmt_val(metric: str, v: float | None, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    if _pct(metric):
        return f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def fmt_ci(metric: str, s: dict, signed: bool = False) -> str:
    if s.get("mean") is None:
        return "n/a"
    half = (s["ci_hi"] - s["ci_lo"]) / 2
    return f"{fmt_val(metric, s['mean'], signed)} ± {fmt_val(metric, half)}"


def render_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))  # noqa: E731
    return "\n".join([line(header), "  ".join("-" * w for w in widths)] + [line(r) for r in rows])
