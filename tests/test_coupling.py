"""Directed-coupling recovery on synthetic eval logs."""
from __future__ import annotations

import json

import numpy as np

from monitordecorrelation.eval.coupling import directed_coupling, directed_coupling_by_target


def _make_run(tmp, name, series, target=None):
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
        (d / "run_info.json").write_text(json.dumps({"train_against": [{"name": target}]}))
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

    # conditioned on training target: all runs trained against A → row A populated, B/C rows absent
    bt = directed_coupling_by_target(dirs, ["A", "B", "C"], metric="dprime_margin")
    assert bt["targets"] == ["A"]
    assert abs(bt["beta"][0, ib] - 0.5) < 0.05
