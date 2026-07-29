"""threshold_reliability: fraction-of-eval-time below a bar, per (target × detector) incl. control."""

import json

import numpy as np

from monitordecorrelation.eval.threshold_reliability import run_final_counts, threshold_matrix


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


def test_run_final_counts(tmp_path):
    r = _run(tmp_path, "r", target="cot_weak", dp={"cot_weak": [3.0, 3.0, 1.0, 1.0]})  # steps 0..3
    assert run_final_counts(r, ["cot_weak"], 2.0, final_frac=1.0)["counts"]["cot_weak"] == (2, 4)  # whole run
    assert run_final_counts(r, ["cot_weak"], 2.0, final_frac=0.5)["counts"]["cot_weak"] == (2, 2)   # steps 2,3
    # NaN d′ snapshots are excluded from the window
    r2 = _run(tmp_path, "r2", target="cot_weak", dp={"cot_weak": [np.nan, 1.0, 1.0]})
    assert run_final_counts(r2, ["cot_weak"], 2.0, final_frac=1.0)["counts"]["cot_weak"] == (2, 2)


def test_threshold_matrix_control_row(tmp_path):
    r = _run(tmp_path, "r", target="cot_weak",
             dp={"cot_weak": [3, 3, 1, 1], "cot_strong": [4, 4, 4, 4]})
    ctl = _run(tmp_path, "c", target=None,
               dp={"cot_weak": [1, 1, 1, 1], "cot_strong": [3, 3, 1, 1]})
    R = threshold_matrix([r, ctl], ["cot_weak", "cot_strong"], 2.0, final_frac=1.0)
    assert R["targets"] == ["cot_weak", "cot_strong", "control"]
    ci = R["monitors"].index("cot_weak")
    ri = R["targets"].index("cot_weak")
    assert (R["below"][ri][ci], R["total"][ri][ci]) == (2, 4) and R["P"][ri][ci] == 0.5  # 2/4 snapshots below
    assert R["P"][R["targets"].index("control")][ci] == 1.0            # control: cot_weak always below 2
    assert R["P"][0][R["monitors"].index("cot_strong")] == 0.0         # cot_strong never below 2 here
    assert np.isnan(R["P"][R["targets"].index("cot_strong")][ci])      # no run trained against cot_strong
