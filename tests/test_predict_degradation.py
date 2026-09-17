"""Sense-(b) → sense-(a) prediction + scoring, and the new per-monitor / ensemble metrics."""
from __future__ import annotations

import numpy as np

from monitordecorrelation.eval.miss_overlap import (
    gmean2,
    or_ensemble_gmean2,
    partial_auroc,
    pauroc_ensemble_gain,
)
from monitordecorrelation.eval.predict_degradation import (
    align_names,
    predictions_from_overlap,
    score_predictions,
)


def test_partial_auroc_and_gmean2_basics():
    rng = np.random.default_rng(0)
    labels = np.r_[np.ones(200, bool), np.zeros(200, bool)]
    perfect = labels.astype(float) + rng.normal(0, 0.01, 400)
    noise = rng.random(400)
    assert partial_auroc(perfect, labels, 0.1) > 0.99
    assert abs(partial_auroc(noise, labels, 0.1) - 0.5) < 0.15
    flags = perfect > 0.5
    assert gmean2(flags, labels) > 0.99
    assert gmean2(np.zeros(400, bool), labels) == 0.0


def test_or_ensemble_shortfall_identical_vs_independent():
    rng = np.random.default_rng(1)
    n = 4000
    labels = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    hit_a = rng.random(n) < 0.6
    flags = np.zeros((2 * n, 3), bool)
    flags[:n, 0] = hit_a
    flags[:n, 1] = hit_a                       # identical blind spot
    flags[:n, 2] = rng.random(n) < 0.6         # independent blind spot
    flags[n:, :] = rng.random((n, 3)) < 0.05   # 5% FPR each, independent on clean
    r = or_ensemble_gmean2(flags, labels)
    assert r["gmean2_shortfall"][0, 1] > 0.2          # identical → big shortfall
    assert abs(r["gmean2_shortfall"][0, 2]) < 0.03    # independent → ≈ 0


def test_pauroc_gain_z_sign():
    rng = np.random.default_rng(2)
    n = 300
    labels = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    latent = rng.normal(0, 1, 2 * n) + labels * 1.5
    a = latent + rng.normal(0, 0.3, 2 * n)
    b = latent + rng.normal(0, 0.3, 2 * n)                    # shares a's errors
    c = rng.normal(0, 1, 2 * n) + labels * 1.5                # independent errors
    r = pauroc_ensemble_gain(np.column_stack([a, b, c]), labels, n_perm=60)
    assert r["pauroc_gain_z"][0, 1] < r["pauroc_gain_z"][0, 2]  # shared errors → less complementary
    assert r["pauroc_gain_z"][0, 1] < 0


def _overlap(names, M, stat="lift"):
    return {"monitors": names, stat: M, "label_key": "strict", "fpr": 0.05}


def test_predictions_ranking_and_name_alignment():
    names = ["cot+out_weak", "out_weak", "probe_ood"]
    M = np.array([[np.nan, 3.0, 1.0], [2.0, np.nan, 1.2], [1.1, 0.9, np.nan]])
    p = predictions_from_overlap(_overlap(names, M))
    assert p["predictions"]["cot+out_weak"]["ranking_most_to_least"] == ["out_weak", "probe_ood"]
    assert p["predictions"]["cot+out_weak"]["normalized_score"]["out_weak"] == 1.0
    assert align_names(["cot_weak", "probe_ood"]) == ["cot+out_weak", "probe_ood"]


def test_score_predictions_recovers_positive_and_null():
    rng = np.random.default_rng(3)
    names = ["A", "B", "C", "D", "E"]
    O = rng.random((5, 5)) * 3
    B_pos = O + rng.normal(0, 0.1, (5, 5))       # β tracks overlap
    coupling = {"targets": names, "monitors": names, "beta": B_pos}
    r = score_predictions(_overlap(names, O), coupling, n_boot=200, n_perm=300)
    assert r["n_pairs"] == 20
    assert r["spearman"] > 0.9 and r["perm_p_one_sided"] < 0.01
    assert set(r["per_target"]) == set(names)
    B_null = rng.random((5, 5))
    r0 = score_predictions(_overlap(names, O), {"targets": names, "monitors": names, "beta": B_null},
                           n_boot=100, n_perm=300)
    assert abs(r0["spearman"]) < 0.6
    # NaN rows in β (targets never trained against) are simply dropped
    B_nan = B_pos.copy(); B_nan[0, :] = np.nan
    r1 = score_predictions(_overlap(names, O), {"targets": names, "monitors": names, "beta": B_nan},
                           n_boot=50, n_perm=50)
    assert r1["n_pairs"] == 16 and r1["per_target"]["A"]["spearman"] is None
