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
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np

from monitordecorrelation.eval.metric_keys import absent_score, present_score

_N = NormalDist()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# Display labels — the ``cot_*`` monitors read CoT AND the final answer (``use_cot=True``); "cot" alone
# reads as cot-only, so charts show "cot+out". Internal ids stay ``cot_weak`` for cross-run continuity.
_DISPLAY = {"cot_weak": "cot+out_weak", "cot_strong": "cot+out_strong"}


def display_name(monitor: str) -> str:
    """Human-facing label for a monitor id (``cot_*`` → ``cot+out_*``); passthrough otherwise."""
    return _DISPLAY.get(monitor, monitor)


def monitor_names(run_dir: str | Path) -> list[str]:
    """Monitor names present in a run's ``eval_metrics.jsonl`` (sorted, stable)."""
    rows = _read_jsonl(Path(run_dir) / "eval_metrics.jsonl")
    names = {k.split("/")[1] for r in rows for k in r if k.startswith("monitor/")}
    return sorted(names)


def train_target(run_dir: str | Path) -> str | None:
    """The single train-against monitor name. Prefers ``run_info.json`` (authoritative); falls back to
    parsing the run-DIR NAME (``mbpp_<model>_<target>_s<seed>[_tag]``) so a copied ``eval_metrics.jsonl``
    in a correctly-named dir is enough — no ``run_info.json`` needed. ``control`` / unparsable → None."""
    p = Path(run_dir) / "run_info.json"
    if p.exists():
        ta = [m.get("name") for m in json.loads(p.read_text()).get("train_against", [])]
        return ta[0] if ta else None  # present-but-empty = control
    m = re.search(r"_([a-z][a-z_]*?)_s\d+", Path(run_dir).name)  # <target> is lowercase; model has caps/digits
    if m and m.group(1) != "control":
        return m.group(1)
    return None


def run_penalty(run_dir: str | Path) -> str:
    """Display string for a run's monitor-penalty λ (shown as α on the charts): ``'0.5'`` constant, or
    ``'0.2→0.6'`` if scheduled; ``''`` if unknown. Reads run_info.json (config) then config.json."""
    for fn, key in (("run_info.json", "config"), ("config.json", None)):
        p = Path(run_dir) / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cfg = d.get(key, {}) if key else d
        sched = cfg.get("penalty_schedule")
        if sched:
            return f"{sched['start_penalty']:g}→{sched['end_penalty']:g}"
        if cfg.get("penalty_coef") is not None:
            return f"{cfg['penalty_coef']:g}"
    return ""


def reliability_series(run_dir: str | Path, monitors: list[str], metric: str = "auroc") -> np.ndarray:
    """``[n_monitors, n_eval_steps]`` reliability array (NaN where undefined), in a common d′-like scale.

    ``metric="auroc"``: AUROC → d′ with a resolution-aware clip from the logged ``n_pos``/``n_neg``.
    ``metric="dprime_margin"``: the logged margin d′ verbatim.
    ``metric="dprime"``: HYBRID — native margin d′ when a run logged it (best, non-saturating), else
    back-calc √2·Φ⁻¹(AUROC) but ONLY where AUROC is non-saturated (interior); saturated points (AUROC
    at the 0/1 boundary, where the back-calc is a clip artifact and would disagree with margin d′) are
    left NaN. Lets the pre-margin-logging runs contribute their trustworthy d′ steps to a combined chart.

    ``metric="suspicion_on_hacks"``: μ_hack — the monitor's MEAN SCORE on actual hacks (behavior-present
    class). This measures co-EVASION (does the held-out monitor stop CATCHING hacks?), immune to the
    honest-class-drift artifact that degrades d′/AUROC (a monitor can keep μ_hack≈1 — still flags every
    hack — while its d′ collapses because honest rollouts started looking suspicious). ``coupling`` β on
    this = "training against i makes held-out j score hacks lower". Note: a DROP is degradation, same as d′.

    ``metric="suspicion_on_clean"``: μ_clean — the monitor's mean score on the NON-hack ('clean') class.
    The complement of μ_hack: a RISE here (honest rollouts looking suspicious) collapses d′ *without* the
    monitor being evaded. β on this measures co-movement of the honest-class drift across monitors.
    """
    rows = _read_jsonl(Path(run_dir) / "eval_metrics.jsonl")
    out = np.full((len(monitors), len(rows)), np.nan)

    def _auroc_dprime(r, m, *, mask_saturated):
        a = r.get(f"monitor/{m}/auroc", np.nan)
        if a is None or a != a:
            return np.nan
        np_, nn = r.get(f"monitor/{m}/n_pos"), r.get(f"monitor/{m}/n_neg")
        eps = 1.0 / (2 * np_ * nn) if (np_ and nn) else 1e-3
        if mask_saturated and (a <= eps or a >= 1 - eps):  # at the resolution boundary → back-calc unreliable
            return np.nan
        return np.sqrt(2) * _N.inv_cdf(min(max(a, eps), 1 - eps))

    for t, r in enumerate(rows):
        for mi, m in enumerate(monitors):
            if metric == "dprime_margin":
                v = r.get(f"monitor/{m}/dprime_margin", np.nan)
                out[mi, t] = float(v) if v is not None else np.nan
            elif metric == "auroc":
                out[mi, t] = _auroc_dprime(r, m, mask_saturated=False)  # clip (keep ceiling values)
            elif metric == "dprime":
                v = r.get(f"monitor/{m}/dprime_margin")
                out[mi, t] = (float(v) if (v is not None and v == v)
                              else _auroc_dprime(r, m, mask_saturated=True))
            elif metric == "suspicion_on_hacks":
                out[mi, t] = present_score(r, m)  # μ_hack (co-evasion metric)
            elif metric == "suspicion_on_clean":
                out[mi, t] = absent_score(r, m)   # μ_clean (honest-class drift — the OTHER driver of d′)
            else:
                raise ValueError(f"unknown metric {metric!r} (use auroc / dprime_margin / dprime / "
                                 "suspicion_on_hacks / suspicion_on_clean)")
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
    ``lo``/``hi`` [n,n] 90% CIs (5/95 percentile) from resampling runs with replacement."""
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
    produced a defined reliability trajectory (policy stayed honest). **All monitors are kept as rows**
    (square matrix — unpopulated targets are NaN rows, not dropped). Returns ``targets`` (= ``monitors``
    order), ``beta`` [n_mon, n_mon], ``n_runs_per_target`` and ``n_pairs_per_target`` (Δd′ pairs), + CIs."""
    run_dirs = [str(d) for d in run_dirs]
    monitors = monitors or monitor_names(run_dirs[0])
    by_t: dict[str, list[np.ndarray]] = {}
    by_pen: dict[str, set] = {}  # penalty λ string(s) contributing to each target row (for the α label)
    for d, tgt, df in _run_diffs(run_dirs, monitors, metric):
        if tgt is not None:
            by_t.setdefault(tgt, []).append(df)
            by_pen.setdefault(tgt, set()).add(run_penalty(d))
    targets = list(monitors)  # ALL monitors as rows → square; targets with no runs stay NaN
    B = np.full((len(targets), len(monitors)), np.nan)
    nR = np.zeros(len(targets), int)
    nP = np.zeros(len(targets), int)
    lo = np.full_like(B, np.nan)
    hi = np.full_like(B, np.nan)
    rng = np.random.default_rng(seed)
    for ri, tgt in enumerate(targets):
        diffs = by_t.get(tgt, [])
        nR[ri] = len(diffs)
        if not diffs:
            continue  # no run trained against this monitor → NaN row
        ti = monitors.index(tgt)
        Dt = np.hstack([df[ti] for df in diffs])
        nP[ri] = len(Dt)
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
    penalty = ["/".join(sorted(p for p in by_pen.get(t, set()) if p)) for t in targets]  # e.g. "0.5" or "0.5/1"
    out = {"targets": targets, "monitors": monitors, "beta": B, "n_runs_per_target": nR,
           "n_pairs_per_target": nP, "penalty_per_target": penalty, "metric": metric}
    if bootstrap:
        out["lo"], out["hi"] = lo, hi
    return out
