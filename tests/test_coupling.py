"""Directed-coupling recovery on synthetic eval logs."""
from __future__ import annotations

import json

import numpy as np

from monitordecorrelation.eval.coupling import directed_coupling, directed_coupling_by_target


def _make_run(tmp, name, series, target=None, penalty=0.5):
    """Write a minimal run dir: eval_metrics.jsonl with monitor/<m>/dprime_margin per step."""
    d = tmp / name
    d.mkdir()
    rows = []
    n = len(next(iter(series.values())))
    for t in range(n):
        row = {"step": t * 2}
        for m, vals in series.items():
            row[f"monitor/{m}/dprime_margin"] = vals[t]
        rows.append(row)
    (d / "eval_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if target is not None:
        (d / "run_info.json").write_text(json.dumps(
            {"train_against": [{"name": target}], "config": {"penalty_coef": penalty}}))
    return str(d)


def test_directed_coupling_recovers_known_slope(tmp_path):
    # B moves at 0.5× A's step-to-step change; C is independent noise
    rng = np.random.default_rng(0)
    dirs = []
    for k in range(4):
        a = np.cumsum(rng.normal(0, 1, 12))
        b = 0.5 * a + 3.0                      # ΔB = 0.5·ΔA exactly
        c = np.cumsum(rng.normal(0, 1, 12))    # independent
        dirs.append(_make_run(tmp_path, f"run{k}", {"A": a, "B": b, "C": c}, target="A"))
    r = directed_coupling(dirs, ["A", "B", "C"], metric="dprime_margin", bootstrap=200)
    B = r["beta"]; mons = r["monitors"]
    ia, ib, ic = mons.index("A"), mons.index("B"), mons.index("C")
    assert abs(B[ia, ib] - 0.5) < 0.05          # β(A→B) ≈ 0.5
    assert abs(B[ia, ia] - 1.0) < 1e-9          # self = 1
    assert abs(B[ia, ic]) < 0.25                # A→C ≈ 0 (independent)
    assert r["lo"][ia, ib] < 0.5 < r["hi"][ia, ib]  # CI brackets the truth

    # conditioned on training target: all runs trained against A → SQUARE matrix, row A populated,
    # B/C rows kept but NaN (no run trained against them); both n's tracked.
    bt = directed_coupling_by_target(dirs, ["A", "B", "C"], metric="dprime_margin")
    assert bt["targets"] == ["A", "B", "C"]           # square: all monitors as rows
    assert abs(bt["beta"][0, ib] - 0.5) < 0.05        # row A populated
    assert np.isnan(bt["beta"][1]).all() and np.isnan(bt["beta"][2]).all()  # B, C rows NaN
    assert list(bt["n_runs_per_target"]) == [4, 0, 0]
    assert bt["n_pairs_per_target"][0] == 4 * 11 and bt["n_pairs_per_target"][1] == 0  # 4 runs × 11 diffs
    from monitordecorrelation.eval.coupling import run_penalty
    assert bt["penalty_per_target"][0] == "0.5"          # α label for the populated (A) row
    assert run_penalty(dirs[0]) == "0.5"
