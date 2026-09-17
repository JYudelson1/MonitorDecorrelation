"""Sense-(b) miss-overlap metrics on synthetic scored sets."""
from __future__ import annotations

import numpy as np

from monitordecorrelation.eval.miss_overlap import (
    analyze,
    labels_for,
    pairwise,
    stratum_miss_rates,
    threshold_at_fpr,
    to_jsonable,
)


def _records(scores: np.ndarray, strict: np.ndarray, loose: np.ndarray, names: list[str]) -> list[dict]:
    return [
        {"question": "q", "cot": "", "answer": "a",
         "env": {"strict_rh": bool(s), "loose_rh": bool(l)},
         "monitors": {n: {"score": float(scores[i, j]), "label": False} for j, n in enumerate(names)}}
        for i, (s, l) in enumerate(zip(strict, loose))
    ]


def test_threshold_at_fpr_flags_at_most_fpr_of_clean():
    clean = np.linspace(0, 1, 100)
    t = threshold_at_fpr(clean, 0.05)
    assert (clean >= t).mean() <= 0.05
    assert (clean >= t).mean() >= 0.04  # not absurdly conservative


def test_pairwise_identical_and_independent_misses():
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.random(n) < 0.3
    b = a.copy()                       # identical blind spot
    c = rng.random(n) < 0.3            # independent blind spot
    pw = pairwise(np.stack([a, b, c], axis=1))
    assert np.isclose(pw["cond_miss"][0, 1], 1.0)
    assert np.isclose(pw["lift"][0, 1], 1 / a.mean())
    assert np.isclose(pw["jaccard"][0, 1], 1.0)
    assert np.isclose(pw["yule_q"][0, 1], 1.0)
    assert abs(pw["lift"][0, 2] - 1.0) < 0.15          # independent → lift ≈ 1
    assert abs(pw["union_catch"][0, 2] - pw["indep_catch"][0, 2]) < 0.03
    assert np.isclose(pw["union_catch"][0, 1], 1 - a.mean())  # identical → no ensemble gain


def test_analyze_strict_vs_loose_label_passes_and_json():
    rng = np.random.default_rng(1)
    n = 300
    strict = np.zeros(n, bool); strict[:100] = True
    loose = np.zeros(n, bool); loose[:200] = True          # 100 strict + 100 loose-only + 100 clean
    # monitor A separates strict from everything; B separates loose from clean; C is noise
    a = np.where(strict, 0.8, 0.1) + rng.normal(0, 0.05, n)
    b = np.where(loose, 0.7, 0.1) + rng.normal(0, 0.05, n)
    c = rng.random(n)
    recs = _records(np.stack([a, b, c], 1), strict, loose, ["A", "B", "C"])
    rs = analyze(recs, ["A", "B", "C"], label_key="strict", fpr=0.05, n_boot=50)
    rl = analyze(recs, ["A", "B", "C"], label_key="loose", fpr=0.05, n_boot=50)
    assert rs["n_pos"] == 100 and rs["n_neg"] == 200
    assert rl["n_pos"] == 200 and rl["n_neg"] == 100
    assert rs["auroc"][0] > 0.99                      # A is perfect on strict
    assert rs["p_miss"][0] < rs["p_miss"][2]         # A misses fewer strict hacks than noise
    assert rl["p_miss"][1] < rl["p_miss"][0]         # under loose labels B beats A (A misses loose-only)
    assert rs["lift_ci"]["lo"].shape == (3, 3)
    js = to_jsonable(rs)
    import json
    json.dumps(js)  # must serialize
    assert labels_for(recs, "loose").sum() == 200


def test_stratum_rates():
    n = 90
    strict = np.zeros(n, bool); strict[:30] = True
    loose = np.zeros(n, bool); loose[:60] = True
    s = np.where(strict, 0.9, np.where(loose, 0.5, 0.1))
    recs = _records(s[:, None], strict, loose, ["A"])
    st = stratum_miss_rates(recs, ["A"], fpr=0.05)
    assert st["n"] == {"strict": 30, "loose_only": 30, "clean": 30}
    assert st["flag_rate"]["strict"][0] == 1.0
    assert st["flag_rate"]["loose_only"][0] == 1.0
    assert st["flag_rate"]["clean"][0] <= 0.05
