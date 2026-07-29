"""threshold_reliability: fraction-of-eval-time below a bar, per (target × detector) incl. control."""

import json

import numpy as np

from monitordecorrelation.eval.threshold_reliability import run_frac_below, threshold_matrix


def _run(tmp, name, *, target, dp):  # dp: {monitor: [d′ per step]}
    d = tmp / name
    d.mkdir()
    n = len(next(iter(dp.values())))
    rows = [{"step": t, "behavior_rate": 0.5, **{f"monitor/{m}/dprime_margin": vals[t] for m, vals in dp.items()}}
            for t in range(n)]
    (d / "eval_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ta = [] if target is None else [{"name": target}]
    (d / "run_info.json").write_text(json.dumps({"train_against": ta, "config": {"penalty_coef": 0.5}}))
    return str(d)


def test_run_frac_below(tmp_path):
    r = _run(tmp_path, "r", target="cot_weak", dp={"cot_weak": [3.0, 3.0, 1.0, 1.0]})  # 2 of 4 < 2.0
    assert run_frac_below(r, ["cot_weak"], 2.0)["frac"]["cot_weak"] == 0.5
    assert run_frac_below(r, ["cot_weak"], 5.0)["frac"]["cot_weak"] == 1.0   # all below 5
    # NaN d′ points are excluded from the denominator
    r2 = _run(tmp_path, "r2", target="cot_weak", dp={"cot_weak": [np.nan, 1.0, 1.0]})
    assert run_frac_below(r2, ["cot_weak"], 2.0)["frac"]["cot_weak"] == 1.0  # 2/2 defined below


def test_threshold_matrix_control_row(tmp_path):
    r = _run(tmp_path, "r", target="cot_weak",
             dp={"cot_weak": [3, 3, 1, 1], "cot_strong": [4, 4, 4, 4]})
    ctl = _run(tmp_path, "c", target=None,
               dp={"cot_weak": [1, 1, 1, 1], "cot_strong": [3, 3, 1, 1]})
    R = threshold_matrix([r, ctl], ["cot_weak", "cot_strong"], 2.0)
    assert R["targets"] == ["cot_weak", "cot_strong", "control"]
    ci = R["monitors"].index("cot_weak")
    assert R["P"][R["targets"].index("cot_weak")][ci] == 0.5          # trained row: cot_weak below 2 half the time
    assert R["P"][R["targets"].index("control")][ci] == 1.0           # control: cot_weak always below 2
    assert R["P"][0][R["monitors"].index("cot_strong")] == 0.0        # cot_strong never below 2 under cot_weak-training
    assert np.isnan(R["P"][R["targets"].index("cot_strong")][ci])     # no run trained against cot_strong
