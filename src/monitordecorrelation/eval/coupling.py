"""Directed detector-COUPLING analysis for the degradation runs (the "chart 7b / chart 10" tools).

The degradation matrix asks "does training against detector i degrade held-out detector j?" Coupling
sharpens that into a continuous, directional quantity:

    β(A→B) = within a run, per unit detector A's reliability changes step-to-step, how much does B's?
           = Σ ΔrA·ΔrB / Σ ΔrA²   (least-squares slope through the origin, pooled over runs)

β is ASYMMETRIC (β(A→B) ≠ β(B→A)) and is the continuous generalisation of "filter to runs where A
degraded" — a run/step where A didn't move contributes ~zero leverage instead of being discarded. Two
forms:

- ``directed_coupling`` — pooled over a set of runs (**chart 7b**). Row = driver A, col = responder B.
- ``directed_coupling_by_target`` — β(trained-against i → held-out j), restricted to the runs trained
  against i (**chart 10**, the causal version). Only rows whose training target actually degraded (its
  reliability is defined) populate; the rest are NaN.

**Reliability metric is pluggable** (we're not sure which is better):
- ``"auroc"``    → d′ = √2·Φ⁻¹(AUROC), with a *resolution-aware* clip ε = 1/(2·n_pos·n_neg) (falls back
                   to 1e-3 if class counts aren't logged) so the ceiling scales with sample size.
- ``"dprime_margin"`` → the natively-logged margin d′ (``monitor/<m>/dprime_margin``); non-saturating,
                   no clip. Only present in runs logged after 2026-07.

Uncertainty via ``bootstrap`` (resample RUNS with replacement — the honest unit, since eval steps are
autocorrelated within a run). Pure/stdlib+numpy so it's unit-testable; plotting lives in
``experiments/analyze_coupling.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def monitor_names(run_dir: str | Path) -> list[str]:
    """Monitor names present in a run's ``eval_metrics.jsonl`` (sorted, stable)."""
    rows = _read_jsonl(Path(run_dir) / "eval_metrics.jsonl")
    names = {k.split("/")[1] for r in rows for k in r if k.startswith("monitor/")}
    return sorted(names)


def train_target(run_dir: str | Path) -> str | None:
    """The single train-against monitor name (from ``run_info.json``), or None (control / unknown)."""
    p = Path(run_dir) / "run_info.json"
    if not p.exists():
        return None
    ta = [m.get("name") for m in json.loads(p.read_text()).get("train_against", [])]
    return ta[0] if ta else None


def reliability_series(run_dir: str | Path, monitors: list[str], metric: str = "auroc") -> np.ndarray:
    """``[n_monitors, n_eval_steps]`` reliability array (NaN where undefined), in a common d′-like scale.

    ``metric="auroc"``: AUROC → d′ with a resolution-aware clip from the logged ``n_pos``/``n_neg``.
    ``metric="dprime_margin"``: the logged margin d′ verbatim.
    """
    rows = _read_jsonl(Path(run_dir) / "eval_metrics.jsonl")
    out = np.full((len(monitors), len(rows)), np.nan)
    for t, r in enumerate(rows):
        for mi, m in enumerate(monitors):
            if metric == "dprime_margin":
                v = r.get(f"monitor/{m}/dprime_margin", np.nan)
                out[mi, t] = float(v) if v is not None else np.nan
            elif metric == "auroc":
                a = r.get(f"monitor/{m}/auroc", np.nan)
                if a is None or a != a:  # None / NaN
                    continue
                np_, nn = r.get(f"monitor/{m}/n_pos"), r.get(f"monitor/{m}/n_neg")
                eps = 1.0 / (2 * np_ * nn) if (np_ and nn) else 1e-3
                out[mi, t] = np.sqrt(2) * _N.inv_cdf(min(max(a, eps), 1 - eps))
            else:
                raise ValueError(f"unknown metric {metric!r} (use 'auroc' or 'dprime_margin')")
    return out


def _run_diffs(run_dirs, monitors, metric, *, require_all=True):
    """Per-run step-to-step reliability differences. Yields (run_dir, target, Δ[n_mon, n_steps-1]).

    ``require_all``: keep only eval steps where EVERY monitor is defined (complete-case, so every pair
    is measured on the same steps — matches how chart 7b was built)."""
    for d in run_dirs:
        R = reliability_series(d, monitors, metric)
        if require_all:
            keep = ~np.isnan(R).any(axis=0)
            R = R[:, keep]
        if R.shape[1] < 3:  # need ≥3 points → ≥2 diffs to estimate a slope
            continue
        yield d, train_target(d), np.diff(R, axis=1)


def _beta_matrix(diffs_list: list[np.ndarray]) -> np.ndarray:
    """β(A→B) = Σ ΔA·ΔB / Σ ΔA² pooled over the runs' Δ arrays. NaN row if a driver never varies."""
    if not diffs_list:
        return np.full((0, 0), np.nan)
    n = diffs_list[0].shape[0]
    D = np.hstack(diffs_list)  # [n_mon, total_pairs]
    B = np.full((n, n), np.nan)
    for i in range(n):
        denom = np.sum(D[i] * D[i])
        if denom > 0:
            for j in range(n):
                B[i, j] = np.sum(D[i] * D[j]) / denom
    return B


def directed_coupling(run_dirs, monitors=None, metric="auroc", *, bootstrap=0, seed=0):
    """Pooled directed-coupling matrix β(row→col) over ``run_dirs`` (**chart 7b**).

    Returns a dict: ``monitors``, ``beta`` [n,n], ``n_runs``, ``n_pairs``, and (if ``bootstrap``>0)
    ``lo``/``hi`` [n,n] percentile-90 CIs from resampling runs with replacement."""
    run_dirs = [str(d) for d in run_dirs]
    monitors = monitors or monitor_names(run_dirs[0])
    diffs = [df for _, _, df in _run_diffs(run_dirs, monitors, metric)]
    beta = _beta_matrix(diffs)
    res = {"monitors": monitors, "beta": beta, "n_runs": len(diffs),
           "n_pairs": int(sum(df.shape[1] for df in diffs)), "metric": metric}
    if bootstrap and diffs:
        rng = np.random.default_rng(seed)
        boots = np.stack([_beta_matrix([diffs[k] for k in rng.integers(0, len(diffs), len(diffs))])
                          for _ in range(bootstrap)])
        res["lo"], res["hi"] = np.nanpercentile(boots, [5, 95], axis=0)
    return res


def directed_coupling_by_target(run_dirs, monitors=None, metric="auroc", *, bootstrap=0, seed=0):
    """β(trained-against i → held-out j), conditioned on the training target (**chart 10**).

    Row i = training target; restricted to the runs trained against i; NaN row if that condition never
    produced a defined reliability trajectory (policy stayed honest). Returns ``targets`` (row order),
    ``monitors`` (cols), ``beta`` [n_targets, n_mon], ``n_runs`` per row, and optional CIs."""
    run_dirs = [str(d) for d in run_dirs]
    monitors = monitors or monitor_names(run_dirs[0])
    by_t: dict[str, list[np.ndarray]] = {}
    for _, tgt, df in _run_diffs(run_dirs, monitors, metric):
        if tgt is not None:
            by_t.setdefault(tgt, []).append(df)
    targets = [m for m in monitors if m in by_t]  # keep monitor order; only populated targets
    B = np.full((len(targets), len(monitors)), np.nan)
    nR = np.zeros(len(targets), int)
    lo = np.full_like(B, np.nan)
    hi = np.full_like(B, np.nan)
    rng = np.random.default_rng(seed)
    for ri, tgt in enumerate(targets):
        diffs = by_t[tgt]
        nR[ri] = len(diffs)
        ti = monitors.index(tgt)
        Dt = np.hstack([df[ti] for df in diffs])
        denom = np.sum(Dt * Dt)
        if denom > 0:
            for j in range(len(monitors)):
                Dj = np.hstack([df[j] for df in diffs])
                B[ri, j] = np.sum(Dt * Dj) / denom
        if bootstrap and denom > 0:
            bs = []
            for _ in range(bootstrap):
                sel = rng.integers(0, len(diffs), len(diffs))
                dt = np.hstack([diffs[k][ti] for k in sel])
                dn = np.sum(dt * dt)
                bs.append([np.sum(dt * np.hstack([diffs[k][j] for k in sel])) / dn
                           if dn > 0 else np.nan for j in range(len(monitors))])
            lo[ri], hi[ri] = np.nanpercentile(np.array(bs), [5, 95], axis=0)
    out = {"targets": targets, "monitors": monitors, "beta": B, "n_runs_per_target": nR, "metric": metric}
    if bootstrap:
        out["lo"], out["hi"] = lo, hi
    return out
