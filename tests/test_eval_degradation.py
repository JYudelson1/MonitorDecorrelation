"""eval_degradation: per-run d′ drop (slope vs endpoint) + the target×detector matrix incl. control."""

import json

import numpy as np

from monitordecorrelation.eval.eval_degradation import (
    _degradation, degradation_matrix,
)


def test_degradation_slope_vs_endpoint():
    steps = np.array([0.0, 1.0, 2.0, 3.0])
    dp = np.array([3.0, 3.0, 3.0, 0.0])  # flat then a late drop → the two methods disagree
    assert _degradation(dp, steps, "endpoint") == 3.0          # first − last
    assert abs(_degradation(dp, steps, "slope") - 2.7) < 1e-9  # −OLS slope × span, uses all 4 points
    assert np.isnan(_degradation(np.array([np.nan, 1.0]), steps[:2], "slope"))  # <2 defined → NaN


def _run(tmp, name, *, target, dp):  # dp: {monitor: [d′ per step]}
    d = tmp / name
    d.mkdir()
    n = len(next(iter(dp.values())))
    rows = []
    for t in range(n):
        row = {"step": t, "behavior_rate": 0.5}
        for m, vals in dp.items():
            row[f"monitor/{m}/dprime_margin"] = vals[t]
        rows.append(row)
    (d / "eval_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ta = [] if target is None else [{"name": target}]
    (d / "run_info.json").write_text(json.dumps({"train_against": ta, "config": {"penalty_coef": 0.5}}))
    return str(d)


def test_matrix_control_row_and_subtraction(tmp_path):
    # NB: MONITOR_ORDER only keeps known names, so use real monitor ids for the columns
    r_a = _run(tmp_path, "ra", target="cot_weak",
               dp={"cot_weak": [3, 3, 3, 0], "cot_strong": [2, 2, 1, 1]})
    ctl = _run(tmp_path, "ctl", target=None,
               dp={"cot_weak": [1, 1, 1, 1], "cot_strong": [1, 1, 0, 0]})
    R = degradation_matrix([r_a, ctl], ["cot_weak", "cot_strong"], method="endpoint")
    assert R["targets"] == ["cot_weak", "cot_strong", "control"]
    ri, ci = R["targets"].index("cot_weak"), R["monitors"].index("cot_weak")
    assert R["D"][ri][ci] == 3.0 and R["n"][ri][ci] == 1          # cot_weak degraded 3.0 under its own training
    cri = R["targets"].index("control")
    assert R["D"][cri][ci] == 0.0                                 # control: cot_weak d′ flat → 0 drop
    assert R["D"][cri][R["monitors"].index("cot_strong")] == 1.0  # control: cot_strong dropped 1
    assert np.isnan(R["D"][R["targets"].index("cot_strong")][ci])  # no run trained against cot_strong
    # excess = row − control: cot_weak row, cot_weak col = 3.0 − 0.0 = 3.0
    assert R["sub"][ri][ci] == 3.0
