"""Turn a sense-(b) miss-overlap matrix into sense-(a) co-degradation PREDICTIONS, and score them
against the measured coupling β once the RL runs exist. (docs/FN_OVERLAP_PLAN.md §4)

The hypothesis: if held-out Y shares X's blind spots at rest, then the hacks the policy discovers to
evade X are ones Y also misses, so Y co-degrades. So a pairwise overlap statistic ``O[X][Y]`` (row =
the monitor trained against, col = held out) should rank the held-out monitors in the same order as
the measured ``β(X→Y)`` from ``eval/coupling.directed_coupling_by_target``.

Two modes:
- **prospective** (before the runs): ``predictions_from_overlap`` → for every training target X, the
  predicted ordering of held-out monitors by co-degradation + a normalized prediction score. Write it
  down before the runs finish (pre-registration).
- **retrospective** (after): ``score_predictions`` → Spearman/Pearson between ``O`` and ``β`` over the
  off-diagonal pairs (pooled, and per training-target row), with a bootstrap-over-pairs CI and a
  permutation p-value. A per-row Spearman is the cleanest test: "given we trained against X, did (b)
  predict WHICH held-out monitors fell?"

Monitor names: the RL runs log ``cot_weak``/``cot_strong`` for the CoT+answer judges while the scored
instance sets (and the sep3 configs) call them ``cot+out_weak``/``cot+out_strong``; ``align_names``
reconciles via ``eval.coupling.display_name``.

Which overlap statistic to feed in is a modelling choice (``stat``): ``lift`` (P(miss_Y|miss_X)/P(miss_Y),
1 = independent), ``cond_miss`` (P(miss_Y|miss_X)), ``yule_q``, ``spearman_pos`` (threshold-free),
``gmean2_shortfall`` or ``pauroc_gain_z`` (ensemble-vs-independence, see miss_overlap). Report several;
the prospective file records all of them.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from monitordecorrelation.eval.coupling import display_name

# Overlap statistics where LARGER means MORE shared blind spot (→ predict MORE co-degradation).
# ``pauroc_gain_z`` / ``ensemble_gap`` are the other way round (large gain = complementary), so they are
# negated before use.
_SIGN = {
    "lift": 1.0, "cond_miss": 1.0, "yule_q": 1.0, "jaccard": 1.0, "spearman_pos": 1.0,
    "gmean2_shortfall": 1.0, "ensemble_gap": -1.0, "pauroc_gain_z": -1.0,
}


def align_names(names: list[str]) -> list[str]:
    """Canonical (display) monitor names so a coupling matrix and an overlap matrix line up."""
    return [display_name(n) for n in names]


def _reindex(mat: np.ndarray, src: list[str], dst: list[str]) -> np.ndarray:
    """Reorder/subset a square matrix from ``src`` name order into ``dst`` order (NaN if absent)."""
    out = np.full((len(dst), len(dst)), np.nan)
    pos = {n: i for i, n in enumerate(src)}
    for i, a in enumerate(dst):
        for j, b in enumerate(dst):
            if a in pos and b in pos:
                out[i, j] = mat[pos[a], pos[b]]
    return out


def predictions_from_overlap(overlap: dict, stat: str = "lift") -> dict:
    """Prospective predictions from one ``analyze`` result dict (``eval/miss_overlap``).

    Returns per training target X: the held-out monitors ranked most→least predicted co-degradation,
    and a row-normalized score in [0, 1] (rank-based, so the scale of ``stat`` doesn't matter)."""
    names = align_names(list(overlap["monitors"]))
    M = np.array(overlap[stat], dtype=float) * _SIGN[stat]
    np.fill_diagonal(M, np.nan)
    preds = {}
    for i, x in enumerate(names):
        row = M[i]
        order = [names[j] for j in np.argsort(-np.nan_to_num(row, nan=-np.inf)) if j != i and not np.isnan(row[j])]
        ranks = {}
        valid = [(names[j], row[j]) for j in range(len(names)) if j != i and not np.isnan(row[j])]
        if len(valid) > 1:
            vals = np.array([v for _, v in valid])
            r = vals.argsort().argsort() / (len(vals) - 1)  # 0 = least, 1 = most predicted co-degradation
            ranks = {n: float(s) for (n, _), s in zip(valid, r)}
        preds[x] = {"ranking_most_to_least": order, "normalized_score": ranks,
                    "raw": {names[j]: (None if np.isnan(row[j]) else float(row[j])) for j in range(len(names)) if j != i}}
    return {"stat": stat, "sign": _SIGN[stat], "monitors": names, "predictions": preds,
            "label_key": overlap.get("label_key"), "fpr": overlap.get("fpr")}


def score_predictions(overlap: dict, coupling: dict, stat: str = "lift", *, n_boot: int = 2000,
                      n_perm: int = 2000, seed: int = 42) -> dict:
    """Retrospective: how well does ``overlap[stat]`` (rows = trained-against X) predict the measured
    ``coupling['beta']`` (by-target β, rows = training target)?

    Pooled over all off-diagonal pairs with both values defined: Spearman ρ + Pearson r, bootstrap
    90 % CI over pairs, and a permutation p-value (shuffle β within each row → destroys the pairing but
    keeps each target's β distribution). Per-target-row Spearman as well (n_valid per row)."""
    o_names = align_names(list(overlap["monitors"]))
    c_names = align_names(list(coupling["monitors"]))
    if list(coupling.get("targets", c_names)) != list(coupling["monitors"]):
        raise ValueError("expected a square by-target coupling (targets == monitors)")
    names = [n for n in o_names if n in c_names]
    O = _reindex(np.array(overlap[stat], dtype=float) * _SIGN[stat], o_names, names)
    B = _reindex(np.array(coupling["beta"], dtype=float), c_names, names)
    np.fill_diagonal(O, np.nan)
    np.fill_diagonal(B, np.nan)
    mask = ~(np.isnan(O) | np.isnan(B))
    xs, ys = O[mask], B[mask]
    rng = np.random.default_rng(seed)
    res = {"stat": stat, "monitors": names, "n_pairs": int(mask.sum()), "pairs": [
        {"target": names[i], "held_out": names[j], "overlap": float(O[i, j]), "beta": float(B[i, j])}
        for i in range(len(names)) for j in range(len(names)) if mask[i, j]]}
    if mask.sum() < 3:
        res.update({"spearman": np.nan, "pearson": np.nan, "note": "fewer than 3 defined pairs"})
        return res
    rho = spearmanr(xs, ys).statistic
    r = pearsonr(xs, ys).statistic
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xs), len(xs))
        if len(set(xs[idx])) > 1 and len(set(ys[idx])) > 1:
            boots.append(spearmanr(xs[idx], ys[idx]).statistic)
    # permutation null: shuffle β within each target row (keeps row-level magnitude, breaks pairing)
    perm = []
    for _ in range(n_perm):
        Bp = B.copy()
        for i in range(len(names)):
            cols = np.where(mask[i])[0]
            Bp[i, cols] = B[i, rng.permutation(cols)]
        perm.append(spearmanr(xs, Bp[mask]).statistic)
    perm = np.array(perm)
    per_row = {}
    for i, x in enumerate(names):
        cols = np.where(mask[i])[0]
        if len(cols) >= 3:
            per_row[x] = {"spearman": float(spearmanr(O[i, cols], B[i, cols]).statistic), "n": int(len(cols))}
        else:
            per_row[x] = {"spearman": None, "n": int(len(cols))}
    res.update({
        "spearman": float(rho), "pearson": float(r),
        "spearman_ci90": [float(np.nanpercentile(boots, 5)), float(np.nanpercentile(boots, 95))] if boots else None,
        "perm_p_one_sided": float(np.mean(perm >= rho)),  # H1: overlap predicts MORE co-degradation
        "per_target": per_row,
    })
    return res
