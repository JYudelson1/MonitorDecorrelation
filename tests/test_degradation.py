"""Cross-run degradation grids: AUROC-drop, suspiciousness-gap-drop, end-AUROC. Synthetic run dirs,
fully offline. Run: uv run pytest tests/test_degradation.py -q."""

from __future__ import annotations

import json

from monitordecorrelation.eval.degradation import build_degradation_matrix


def _make_run(d, target, series):
    """series: {monitor: [(step, auroc, gap_pos_mean, honest_mean), ...]}."""
    d.mkdir(parents=True, exist_ok=True)
    ta = [] if target is None else [{"name": target}]
    (d / "run_info.json").write_text(json.dumps({"train_against": ta, "held_out": []}))
    with (d / "eval_metrics.jsonl").open("w") as f:
        steps = sorted({s for rows in series.values() for (s, *_ ) in rows})
        for step in steps:
            row = {"step": step}
            for mon, rows in series.items():
                rec = next(r for r in rows if r[0] == step)
                _, auroc, syco, honest = rec
                row[f"monitor/{mon}/auroc"] = auroc
                row[f"monitor/{mon}/mean_score_syco"] = syco
                row[f"monitor/{mon}/mean_score_honest"] = honest
            f.write(json.dumps(row) + "\n")


def test_three_grids(tmp_path):
    # run A trains against m1: m1 degrades hard (auroc 0.9->0.6, gap 0.6->0.1); m2 stays (0.8->0.78)
    _make_run(tmp_path / "runA", "m1", {
        "m1": [(0, 0.9, 0.7, 0.1), (10, 0.6, 0.3, 0.2)],
        "m2": [(0, 0.8, 0.6, 0.1), (10, 0.78, 0.58, 0.11)],
    })
    # run B trains against m2: m2 degrades, m1 stays
    _make_run(tmp_path / "runB", "m2", {
        "m1": [(0, 0.88, 0.66, 0.1), (10, 0.86, 0.64, 0.1)],
        "m2": [(0, 0.85, 0.65, 0.1), (10, 0.55, 0.2, 0.2)],
    })

    res = build_degradation_matrix([tmp_path / "runA", tmp_path / "runB"],
                                   edge=1, rolling=1, out_dir=tmp_path / "out")

    # AUROC-drop diagonal (train against i degrades i) should be large + positive
    assert res["matrix"]["m1"]["m1"] > 0.25
    assert res["matrix"]["m2"]["m2"] > 0.25
    # off-diagonal (held-out) much smaller
    assert abs(res["matrix"]["m1"]["m2"]) < 0.1

    gap = res["gap_drop_matrix"]["matrix"]
    assert gap["m1"]["m1"] > 0.3            # gap collapses on the trained-against monitor
    assert gap["m2"]["m2"] > 0.3

    end = res["end_auroc_matrix"]["matrix"]
    assert abs(end["m1"]["m1"] - 0.6) < 1e-6   # final AUROC = late value
    assert abs(end["m2"]["m2"] - 0.55) < 1e-6

    # all three grids written to disk
    for stem in ("degradation_matrix", "gap_drop_matrix", "end_auroc_matrix"):
        assert (tmp_path / "out" / f"{stem}.json").exists()
        assert (tmp_path / "out" / f"{stem}.csv").exists()
