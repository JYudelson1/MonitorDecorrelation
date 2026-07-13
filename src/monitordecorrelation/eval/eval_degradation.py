"""Absolute eval-degradation matrix: how much does each detector degrade under each training condition.

Complements the coupling matrices (which give the *relative* co-movement β — "when i moves by 1, how
much does j move"). This answers the magnitude question a lab actually asks: **during training on
condition _row_, how much did detector _col_ degrade, on average, over the run** — and how much would
it have degraded anyway (the CONTROL row = no-monitor baseline).

Per run, per detector col, degradation ``D`` (positive = the detector's d′ FELL = it got worse):
  method="slope"    : −(OLS slope of d′ on step) × (measurable span)  — uses every eval point; robust to
                      a noisy endpoint; recovers most of the power a 2-point diff throws away. DEFAULT.
  method="endpoint" : d′(first measurable) − d′(last measurable)      — the literal two-point drop.
Both need ≥2 measurable d′ points and coincide when there are exactly 2. d′ is the coupling HYBRID
(native margin d′ where logged, else *unclipped* AUROC→d′) so a saturated first point can't fake a drop.

Rows pool all penalty regimes for a train-against target → a 7×6 (6 monitors + control). Control merges
all its runs (λ is meaningless with no monitor). Also emits the control-SUBTRACTED matrix (row − control)
= the excess degradation attributable to training on that monitor, beyond the generic drift the policy
causes just by learning to hack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from monitordecorrelation.eval.coupling import (
    _read_jsonl, reliability_series, run_penalty, train_target,
)

# preferred display order (present monitors are filtered to this); control is appended as the last row
MONITOR_ORDER = ["cot_weak", "cot_strong", "out_weak", "out_strong", "probe_iid", "probe_ood"]


def _degradation(dp: np.ndarray, steps: np.ndarray, method: str) -> float:
    """Total d′ drop over a run's measurable window (positive = degraded). NaN if <2 defined points."""
    ok = np.flatnonzero(~np.isnan(np.asarray(dp, dtype=float)))
    if len(ok) < 2:
        return np.nan
    x, y = steps[ok], np.asarray(dp, float)[ok]
    if method == "endpoint":
        return float(y[0] - y[-1])
    if method == "slope":
        slope = np.polyfit(x, y, 1)[0]
        return float(-slope * (x[-1] - x[0]))  # predicted first − predicted last
    raise ValueError(f"unknown method {method!r} (use 'slope' or 'endpoint')")


def run_degradation(run_dir: str | Path, monitors: list[str], method: str = "slope") -> dict:
    """Per-detector degradation D for one run, plus its train-against target and penalty."""
    rows = _read_jsonl(Path(run_dir) / "eval_metrics.jsonl")
    steps = np.array([r.get("step", i) for i, r in enumerate(rows)], dtype=float)
    dp = reliability_series(run_dir, monitors, "dprime")  # [n_monitors, n_steps]
    return {"run": Path(run_dir).name, "target": train_target(run_dir) or "control",
            "penalty": run_penalty(run_dir),
            "D": {m: _degradation(dp[i], steps, method) for i, m in enumerate(monitors)}}


def _boot_ci(vals: np.ndarray, rng, n_boot: int) -> tuple[float, float]:
    if n_boot < 1 or len(vals) < 2:
        return (np.nan, np.nan)
    boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [5, 95])
    return (float(lo), float(hi))


def degradation_matrix(run_dirs, monitors: list[str], method: str = "slope",
                       bootstrap: int = 0, seed: int = 0) -> dict:
    """Assemble the (targets × monitors) mean-degradation matrix + control-subtracted companion.

    Returns ``targets`` (monitors present as train-against + 'control'), ``monitors`` (columns), ``D``
    (mean), ``lo``/``hi`` (bootstrap-over-runs CI), ``n`` (runs with the cell defined); ``sub``/``sub_lo``
    /``sub_hi`` (control-subtracted, monitor rows only)."""
    monitors = [m for m in MONITOR_ORDER if m in monitors] or list(monitors)
    summaries = [run_degradation(d, monitors, method) for d in run_dirs]
    targets = [m for m in monitors] + ["control"]
    by_t: dict[str, list[dict]] = {t: [] for t in targets}
    for s in summaries:
        by_t.setdefault(s["target"], []).append(s)

    R, C = len(targets), len(monitors)
    M = np.full((R, C), np.nan); N = np.zeros((R, C), int)
    LO = np.full((R, C), np.nan); HI = np.full((R, C), np.nan)
    vals_by_cell: dict[tuple[int, int], np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for i, t in enumerate(targets):
        for j, c in enumerate(monitors):
            vals = np.array([s["D"][c] for s in by_t.get(t, [])], dtype=float)
            vals = vals[~np.isnan(vals)]
            vals_by_cell[(i, j)] = vals
            N[i, j] = len(vals)
            if len(vals):
                M[i, j] = float(vals.mean())
                LO[i, j], HI[i, j] = _boot_ci(vals, rng, bootstrap)

    # control-subtracted (monitor rows only): row − control, bootstrapped as a difference of means
    ci = targets.index("control")
    SUB = np.full((C, C), np.nan); SUB_LO = np.full((C, C), np.nan); SUB_HI = np.full((C, C), np.nan)
    for i in range(C):
        for j in range(C):
            rv, cv = vals_by_cell[(i, j)], vals_by_cell[(ci, j)]
            if len(rv) and len(cv):
                SUB[i, j] = float(rv.mean() - cv.mean())
                if bootstrap and len(rv) >= 2 and len(cv) >= 2:
                    diffs = [rng.choice(rv, len(rv), True).mean() - rng.choice(cv, len(cv), True).mean()
                             for _ in range(bootstrap)]
                    SUB_LO[i, j], SUB_HI[i, j] = np.nanpercentile(diffs, [5, 95])
    return {"targets": targets, "monitors": monitors, "method": method,
            "D": M, "n": N, "lo": LO, "hi": HI, "sub": SUB, "sub_lo": SUB_LO, "sub_hi": SUB_HI}
