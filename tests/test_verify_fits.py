"""experiments/verify_fits.py on the synthetic probes of tests/test_fit_probes.py.

Checks:
- The lazy selection (cv_grid.csv's rounded cells plus layers recomputed on demand) equals ``probe_fit.select`` on
  the full grid. Three grids: a peaked one with a tied twin layer, one where many cells are perfect (SE 0), and a
  flat one.
- Fitted probes verify: every recomputed cell matches cv_grid.csv, the selection is re-derived, and the refit equals
  the saved probe bit for bit.
- A perturbed probe fails: its weights, or one cell of its cv_grid.csv.
"""

from __future__ import annotations

import csv
import glob
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from test_fit_probes import SIGNAL_LAYER, make_world

from monitordecorrelation.whitebox.probe_fit import C_GRID, auroc, select

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("verify_fits", os.path.join(REPO, "experiments", "verify_fits.py"))
vf = importlib.util.module_from_spec(spec)
sys.modules["verify_fits"] = vf  # its dataclass looks itself up there
spec.loader.exec_module(vf)


def _lazy(oof, y, groups, layers, tie_metric=None):
    """select on the full grid, and derive_selection on the rounded grid it saves, recomputing layers on demand."""
    full = select(oof, y, groups, layers, C_GRID, tie_metric=tie_metric)
    stored = {(r["layer"], r["C"]): (f"{r['cv_auroc']:.5f}", f"{r['cv_auroc_sd']:.5f}") for r in full.grid}
    g = vf.Grid(stored, layers, y)
    while True:
        try:
            return full, vf.derive_selection(g, groups, tie_metric), g
        except vf.Missing as m:
            for layer, j in m.need.items():
                g.add(layer, oof[layers.index(layer), : j + 1])


def _same(full, derived):
    assert (derived["layer"], derived["C"], derived["flat"]) == (full.layer, full.C, full.flat)
    assert derived["cv_auroc"] == full.cv_auroc
    assert derived["cv_se"] == full.cv_se or (np.isnan(derived["cv_se"]) and np.isnan(full.cv_se))


@pytest.mark.parametrize("kind", ["peaked", "perfect", "flat"])
def test_lazy_selection_equals_select(kind):
    rng = np.random.default_rng(3)
    n, n_l, n_r = 240, 10, 3
    y = (rng.random(n) < (0.5 if kind == "peaked" else 0.15)).astype(int)
    groups = np.repeat(np.arange(n // 4), 4)
    layers = list(range(1, n_l + 1))
    base = rng.standard_normal((n_r, n))
    oof = np.empty((n_l, len(C_GRID), n_r, n), np.float32)
    for i in range(n_l):
        for j in range(len(C_GRID)):
            if kind == "peaked":  # layer 6 (index 5) best, and layer 7 its exact twin (set below)
                strength = 2.5 * np.exp(-((i - 5) ** 2) / 3) * (1 - 0.05 * abs(j - 3))
                oof[i, j] = y * strength + base + (0 if i == 6 else 0.3 * rng.standard_normal((n_r, n)))
            elif kind == "perfect":  # separable from layer 4 on at C ≥ 1e-4, so the best is 1 and its SE is 0
                oof[i, j] = y * (10.0 if i >= 3 and j >= 1 else 0.6) + base + 0.1 * rng.standard_normal((n_r, n))
            else:  # every cell nearly the same
                oof[i, j] = y + base + 1e-3 * rng.standard_normal((n_r, n))
    if kind == "peaked":
        oof[6] = oof[5]
    hard = rng.random(n) < 0.5
    full, derived, g = _lazy(oof, y, groups, layers, tie_metric=lambda s: auroc(s[hard], y[hard]))
    _same(full, derived)
    if kind == "peaked":
        assert derived["best_cell"][0] == 6 and len(g.oof) < n_l  # the twins' tie goes to the first; few layers
    if kind == "perfect":
        assert full.cv_se == 0.0 and not full.flat and len(derived["candidates"]) > 1
    if kind == "flat":
        assert full.flat and list(g.oof) == [layers[n_l // 2]]


def _run(args, timeout=1800):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=REPO, timeout=timeout)


def _verify(tmp_path, tag, data, feats, probes, *extra):
    out = tmp_path / f"verify_{tag}"
    cp = _run([os.path.join(REPO, "experiments", "verify_fits.py"), "--probes", str(probes), "--datasets", str(data),
               "--check", f"synthetic/followup,router_all={feats}", "--out", str(out), "--jobs", "2",
               "--fit-jobs", "2", *extra])
    runs = glob.glob(str(out / "*" / "summary.json"))
    assert len(runs) == 1, cp.stdout[-3000:] + cp.stderr[-3000:]
    return cp, json.loads(open(runs[0]).read()), os.path.dirname(runs[0])


def test_verify_fits_reproduces_the_fitted_probes_and_catches_changes(tmp_path):
    data, feats = make_world(tmp_path)
    probes = tmp_path / "probes"
    cp = _run([os.path.join(REPO, "experiments", "fit_probes.py"), "--reader", "synthetic", "--features", str(feats),
               "--datasets", str(data), "--out", str(probes / "synthetic"), "--repeats", "2", "--jobs", "2",
               "--readouts", "followup,router_all", "--skip-baselines"])
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]

    cp, summary, out = _verify(tmp_path, "clean", data, feats, probes)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    assert len(summary["probes"]) == 6
    for name, r in summary["probes"].items():
        assert r["verdict"] == "pass" and r["selection"]["status"] == "reproduced", (name, r)
        assert 0 < r["cv"]["cells_checked"] == r["cv"]["cells_matched"], (name, r["cv"])
        assert r["refit"]["weights"]["identical"] and r["refit"]["thresholds"]["identical"], (name, r["refit"])
        assert r["refit"]["metrics"]["compared"] > 5 and r["refit"]["test_logits"]["same_ids"], (name, r["refit"])
    iid = summary["probes"]["synthetic/terminal_prerl/with_reasoning/followup"]
    assert iid["cv"]["layers_slice"][0] == SIGNAL_LAYER and iid["inputs"]["counts"]
    assert set(iid["cv"]["layers_slice"]) <= set(iid["cv"]["layers_recomputed"])
    assert open(os.path.join(out, "report.md")).read().startswith("# Do today's")

    # the weights of one probe, and a cell of another's grid in its selected layer
    target = probes / "synthetic" / "terminal_prerl" / "with_reasoning" / "followup"
    z = dict(np.load(target / "probe.npz"))
    z["coef"] = z["coef"] * (1 + 1e-6)
    np.savez(target / "probe.npz", **z)
    ood = probes / "synthetic" / "diverse_deception" / "without_reasoning" / "followup"
    rows = list(csv.reader(open(ood / "cv_grid.csv")))
    layer = str(json.load(open(ood / "meta.json"))["layer"])
    k = next(i for i, r in enumerate(rows) if r[0] == layer and r[1] == "1.0")
    rows[k][2] = f"{float(rows[k][2]) - 0.001:.5f}"
    with open(ood / "cv_grid.csv", "w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    cp, summary, _ = _verify(tmp_path, "perturbed", data, feats, probes, "--only",
                             "with_reasoning/followup,diverse_deception/without_reasoning/followup")
    assert cp.returncode == 1 and len(summary["probes"]) == 2
    r = summary["probes"]["synthetic/terminal_prerl/with_reasoning/followup"]
    assert r["verdict"] == "fail" and not r["refit"]["weights"]["ok"] and r["refit"]["metrics"]["ok"]
    assert r["cv"]["cells_matched"] == r["cv"]["cells_checked"]
    r = summary["probes"]["synthetic/diverse_deception/without_reasoning/followup"]
    assert r["verdict"] == "fail" and r["refit"]["weights"]["ok"]
    assert r["cv"]["cells_matched"] == r["cv"]["cells_checked"] - 1
    assert r["cv"]["mismatches"][0]["C"] == 1.0
