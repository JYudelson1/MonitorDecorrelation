"""Does today's code reproduce the fitted Inkling probes? Re-runs part of each fit of experiments/fit_probes.py, instead
of the whole grid (hours on hub), and compares it with what the fit saved.

    uv run python experiments/verify_fits.py --probes ../scratch/probes --out ../scratch/verify_fits \
        --check bf16/followup=../scratch/features/bf16 --check bf16/mean,router,router_all=../scratch/features_mr/bf16

``--check READER/READOUTS=FEATURES`` (repeatable) picks the probes ``<probes>/<reader>/<dataset>/<view>/<readout>/``
and the features they were fitted on. For each fitted IID and OOD probe of those readouts:

1. **Inputs:** the dev and test rows, labels, groups and folds, rebuilt as fit_probes builds them (``--seed``, and the
   repeats in meta.json). Checked against metrics.json's counts and the ids in scores.jsonl.
2. **Grid slice:** the out-of-fold AUROC (mean and sd over repeats) at every C for the selected layer, the best layer in
   cv_grid.csv and the grid's first and last layers (``--full-grid`` readouts: every layer). A cell matches if it
   equals cv_grid.csv's after the same rounding (5 decimals).
3. **Selection:** the pre-registered rule (``probe_fit.select``) re-derived from cv_grid.csv plus recomputed cells. A
   stored cell is known to ±5e-6, so a layer is recomputed only when a step can't be decided without it: the flat-grid
   test, the best cell, the one-SE bootstrap around it, the smallest C within one SE, the tie-breaks among its cells.
   Every recomputed cell is compared with cv_grid.csv too.
4. **Refit** at the recorded layer and C, compared with the saved weights (probe.npz), t₁%/t₅% (probe.npz, meta.json),
   test logits (scores.jsonl) and main metrics (metrics.json):
   - IID: test AUROC and its CI, AUROC under behavior_present, TPR/FPR at t₁% and t₅%, the counts;
   - OOD: test AUROC and its CI, per source; per TerminalEnv view, the transfer metrics and thresholds (meta.json's
     thresholds_by_terminal_view).

   Not recomputed: the cross-fitted hard subsets and concealment (IID), leave-one-source-out (OOD), the baselines and
   the sensitivity rows.

What fit_probes computes inline in fit_iid / fit_ood is reproduced here from a copy of its code:
- the rows, labels, groups, folds and tie-break metrics (``iid_inputs``, ``ood_inputs``);
- the IID thresholds and test metrics (``refit_iid``);
- the OOD test and transfer metrics (``refit_ood``).

The fitting is the library's own: ``LayerSource.layer`` then ``_fit_path`` (the body of ``_fit_path_from``, what
``cv_grid_source`` runs per layer), ``cluster_bootstrap``, ``SelectedProbe.fit``, ``thresholds``, and fit_probes'
``rate`` and ``router_layer``.

**Bits.** BLAS threads change a fit's last bits (~1e-14). So the out-of-fold fits use the threads cv_grid_source gave
them at fitting time: one per worker, but ``router_all``'s single layer got ``--fit-jobs``. The refits use BLAS's
default threads, as fit_probes' main process did. ``--refit-threads`` caps them, at the cost of bitwise equality.
A difference passes if it is ≤ ``--tol`` × max(1, |saved value|).

**Memory.** On hub the page cache counts toward the pod's memory, and each layer is read once. So the shards' pages are
dropped from it (posix_fadvise) once a layer is read, and after each refit.

**Writes** ``<out>/<UTC timestamp>/summary.json`` (per probe: cells checked and matched, the selection, the largest
differences, the verdict) and ``report.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from monitordecorrelation.whitebox import probe_fit
from monitordecorrelation.whitebox.probe_fit import (
    C_GRID, FLAT_FALLBACK_C, LayerSource, SelectedProbe, _fit_path, auroc, ci, cluster_bootstrap, load_items,
    make_folds, thresholds)

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES = (("terminal_prerl", "with_reasoning"), ("terminal_prerl", "without_reasoning"),
          ("diverse_deception", "without_reasoning"))
VIEWS = ("with_reasoning", "without_reasoning")
ROUNDING = 5e-6 + 1e-12  # cv_grid.csv's cells are rounded to 5 decimals


def _fit_probes():
    spec = importlib.util.spec_from_file_location("fit_probes", os.path.join(HERE, "fit_probes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fit_probes = _fit_probes()


def log(msg: str) -> None:
    print(f"[verify {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ selection


class Missing(Exception):
    """A step of the selection needs these layers' out-of-fold logits, up to these C indices."""

    def __init__(self, need: dict[int, int]):
        super().__init__(need)
        self.need = need


class Grid:
    """One probe's out-of-fold grid as far as it is known: cv_grid.csv's rounded cells, and the recomputed layers."""

    def __init__(self, stored: dict, layers: list[int], y: np.ndarray, c_grid=C_GRID):
        self.stored, self.layers, self.y, self.c_grid = stored, list(layers), np.asarray(y), tuple(c_grid)
        self.oof: dict[int, np.ndarray] = {}  # layer -> out-of-fold logits [C computed, repeats, n]
        self.auc: dict[int, np.ndarray] = {}  # layer -> AUROC [C computed, repeats]

    def add(self, layer: int, oof: np.ndarray) -> None:
        if layer in self.oof and len(self.oof[layer]) >= len(oof):
            return
        self.oof[layer] = oof
        self.auc[layer] = np.array([[auroc(oof[j, r], self.y) for r in range(oof.shape[1])] for j in range(len(oof))])

    def known(self, i: int, j: int) -> bool:
        return self.layers[i] in self.oof and j < len(self.oof[self.layers[i]])

    def mean(self, i: int, j: int) -> float:  # as select computes it
        return float(np.nanmean(self.auc[self.layers[i]], axis=-1)[j])

    def sd(self, i: int, j: int) -> float:
        return float(np.nanstd(self.auc[self.layers[i]][j]))

    def bounds(self, i: int, j: int) -> tuple[float, float]:
        if self.known(i, j):
            m = self.mean(i, j)
            return m, m
        v = float(self.stored[(self.layers[i], float(self.c_grid[j]))][0])
        return v - ROUNDING, min(v + ROUNDING, 1.0)  # an AUROC is at most 1

    def score(self, i: int, j: int) -> np.ndarray:
        return self.oof[self.layers[i]][j].mean(0)

    def require(self, cells) -> None:
        need: dict[int, int] = {}
        for i, j in cells:
            if not self.known(i, j):
                need[self.layers[i]] = max(j, need.get(self.layers[i], -1))
        if need:
            raise Missing(need)


def derive_selection(g: Grid, groups, tie_metric=None, seed: int = 0, batch: int = 6) -> dict:
    """``probe_fit.select``'s rule, decided on what ``g`` knows. Raises Missing for the layers a step needs."""
    n_l, n_c = len(g.layers), len(g.c_grid)
    b = {(i, j): g.bounds(i, j) for i in range(n_l) for j in range(n_c)}
    cells = [c for c in b if not math.isnan(b[c][0])]  # row-major, as np.nanargmax scans
    lo_best, hi_best = max(b[c][0] for c in cells), max(b[c][1] for c in cells)
    lo_worst, hi_worst = min(b[c][0] for c in cells), min(b[c][1] for c in cells)
    if hi_best - lo_worst >= 0.005 > lo_best - hi_worst:  # can't tell yet whether the grid is flat
        g.require([c for c in cells if b[c][1] >= lo_best or b[c][0] <= hi_worst])
    if hi_best - lo_worst < 0.005:
        mid, jf = n_l // 2, g.c_grid.index(FLAT_FALLBACK_C)
        g.require([(mid, jf)])
        return {"layer": g.layers[mid], "C": FLAT_FALLBACK_C, "cv_auroc": g.mean(mid, jf), "cv_se": float("nan"),
                "flat": True, "best_cell": None, "candidates": []}
    # the best cell: the first maximum in row-major order; an unknown cell is open while it could still be it
    possible = [c for c in cells if b[c][1] >= lo_best]
    known = [c for c in possible if g.known(*c)]
    m = max((g.mean(*c) for c in known), default=None)
    first = min((c for c in known if g.mean(*c) == m), default=None)
    open_ = [c for c in possible if not g.known(*c) and (m is None or b[c][1] > m or (b[c][1] >= m and c < first))]
    if open_:
        next_layers = sorted({i for i, _ in open_})[:batch]
        g.require([c for c in open_ if c[0] in next_layers])
    bi, bj = first
    best = g.mean(bi, bj)
    se = float(np.nanstd(cluster_bootstrap(auroc, groups, g.score(bi, bj), g.y, n=200, seed=seed)))
    cand: list = []
    for j in range(n_c):  # the smallest C with a cell within one SE of the best
        maybe = [c for c in cells if c[1] == j and b[c][1] >= best - se]
        if maybe:
            g.require(maybe)
            cand = [c for c in maybe if g.mean(*c) >= best - se]
            if cand:
                break

    def key(c):
        s = g.score(*c)
        tie = tie_metric(s) if tie_metric is not None else 0.0
        p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
        ll = -np.mean(g.y * np.log(p + 1e-12) + (1 - g.y) * np.log(1 - p + 1e-12))
        return (np.nan_to_num(tie, nan=-1.0), -ll, g.mean(*c))

    i, j = max(cand, key=key)
    return {"layer": g.layers[i], "C": float(g.c_grid[j]), "cv_auroc": g.mean(i, j), "cv_se": se, "flat": False,
            "best_cell": [g.layers[bi], float(g.c_grid[bj])], "candidates": [g.layers[c[0]] for c in cand]}


# ------------------------------------------------------------------------------------------------ inputs


def iid_inputs(src: LayerSource, items: dict, repeats: int, seed: int) -> dict:
    """fit_iid's rows, labels, groups, folds and tie-break metric (experiments/fit_probes.py)."""
    keep = src.keep
    its = [items[i] for i in src.ids]
    y_all = np.array([int(bool((it.get("diagnostics") or {}).get("used_verifier"))) for it in its])
    groups_all = np.array([it["task_id"] for it in its])
    split = np.array([it["split"] for it in its])
    diag = [it.get("diagnostics") or {} for it in its]
    dev = np.flatnonzero(np.isin(split, ["train", "val"]) & keep)
    test = np.flatnonzero((split == "test") & keep)
    folds = make_folds(len(dev), y_all[dev], groups_all[dev], repeats=repeats, seed=seed)
    hard_dev = np.array([y_all[i] == 1 or diag[i].get("read_not_used") or not diag[i].get("passed") for i in dev],
                        bool)
    return {"dev": dev, "test": test, "y_all": y_all, "groups_dev": groups_all[dev], "folds": folds,
            "tie_metric": lambda s: auroc(s[hard_dev], y_all[dev][hard_dev]), "groups_all": groups_all,
            "bp_all": np.array([int(it["label"]) for it in its])}


def ood_inputs(src: LayerSource, items: dict, repeats: int, seed: int) -> dict:
    """fit_ood's rows, labels, groups, folds and tie-break metric (experiments/fit_probes.py)."""
    keep = src.keep
    its = [items[i] for i in src.ids]
    y_all = np.array([int(it["label"]) for it in its])
    pair = np.array([it["pair"] for it in its])
    source = np.array([it["source"] for it in its])
    split = np.array([it["split"] for it in its])
    dev = np.flatnonzero((split == "train") & keep)
    test = np.flatnonzero((split == "test") & keep)
    strat = np.array([f"{s}|{y}" for s, y in zip(source[dev], y_all[dev])])
    folds = make_folds(len(dev), strat, pair[dev], repeats=repeats, seed=seed)
    sources = sorted(set(source))

    def source_avg(s):
        vals = [auroc(s[source[dev] == x], y_all[dev][source[dev] == x]) for x in sources]
        return float(np.nanmean(vals))

    return {"dev": dev, "test": test, "y_all": y_all, "groups_dev": pair[dev], "folds": folds,
            "tie_metric": source_avg, "pair": pair, "source": source, "sources": sources}


@dataclass
class Probe:
    """One fitted probe: what fit_probes saved, the inputs it used, and what is recomputed."""
    name: str
    path: str
    features: str
    meta: dict
    metrics: dict
    saved: SelectedProbe
    src: LayerSource
    inputs: dict
    grid: Grid
    threads: int  # BLAS threads per worker that cv_grid_source used for this grid
    slice_layers: list = field(default_factory=list)
    selection: dict | None = None
    cv_seconds: float = 0.0

    @property
    def iid(self) -> bool:
        return self.meta["dataset"] == "terminal_prerl"


def load_probe(path: str, name: str, features: str, data: dict, seed: int, fit_jobs: int) -> Probe:
    meta = json.load(open(os.path.join(path, "meta.json")))
    metrics = json.load(open(os.path.join(path, "metrics.json")))
    with open(os.path.join(path, "cv_grid.csv")) as fh:
        stored = {(int(r["layer"]), float(r["C"])): (r["cv_auroc"], r["cv_auroc_sd"]) for r in csv.DictReader(fh)}
    src = LayerSource(os.path.join(features, meta["dataset"]), meta["view"], meta["readout"])
    make = iid_inputs if meta["dataset"] == "terminal_prerl" else ood_inputs
    inputs = make(src, data[meta["dataset"]], int(meta["repeats"]), seed)
    layers = list(range(src.first_layer, src.n_layers))
    threads = max(1, fit_jobs // len(layers)) if 0 < len(layers) < fit_jobs else 1  # as cv_grid_source
    return Probe(name, path, features, meta, metrics, SelectedProbe.load(path), src, inputs,
                 Grid(stored, layers, inputs["y_all"][inputs["dev"]]), threads)


def slice_layers(p: Probe, full: bool, others: int = 2) -> list[int]:
    """The selected layer, cv_grid.csv's best layer, and ``others`` more from the grid's ends inwards."""
    layers = p.grid.layers
    if full:
        return list(layers)
    best = max(p.grid.stored, key=lambda k: (float(p.grid.stored[k][0]), -layers.index(k[0]), -k[1]))[0]
    chosen = list(dict.fromkeys([int(p.meta["selection"]["layer"]), best]))
    ends = [x for pair in zip(layers, reversed(layers)) for x in pair]
    return chosen + [x for x in dict.fromkeys(ends) if x not in chosen][:others]


# ------------------------------------------------------------------------------------------------ out-of-fold fits


def drop_cache(paths) -> None:
    """Drop these files' pages from the page cache: they count toward the pod's memory, and a layer is read once."""
    for path in paths:
        with contextlib.suppress(OSError):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def shard_files(features: str) -> list[str]:
    return glob.glob(os.path.join(features, "*", "*", "shard_*.safetensors"))


def _timed_fit_path(src, layer, rows, y, folds, c_grid):
    """``_fit_path_from`` (what cv_grid_source runs per layer), dropping the shards' cache once the layer is read."""
    t0 = time.time()
    x = src.layer(layer, rows)
    drop_cache([data for data, _, _ in src.shards])
    return _fit_path(x, y, folds, c_grid), time.time() - t0


def run_layers(tasks: list[tuple[Probe, int, int]], jobs: int) -> None:
    """Out-of-fold logits for (probe, layer, last C index) tasks, each fitted as cv_grid_source fitted it."""
    from joblib import Parallel, delayed, parallel_config

    by_threads: dict[int, list] = {}
    for t in tasks:
        by_threads.setdefault(t[0].threads, []).append(t)
    for threads, ts in sorted(by_threads.items()):
        ts.sort(key=lambda t: (t[0].src.d, not t[0].iid, t[2]), reverse=True)  # the longest first
        left = {}
        for p, _, _ in ts:
            left[p.name] = left.get(p.name, 0) + 1
        # always in loky workers (n_jobs=1 would fit here, with BLAS's default threads); a multi-threaded group one
        # task at a time, as cv_grid_source fitted router_all's single layer
        for batch in [ts] if threads == 1 else [[t] for t in ts]:
            with parallel_config(backend="loky", inner_max_num_threads=threads):
                out = Parallel(n_jobs=max(2, min(jobs, len(batch))), pre_dispatch="n_jobs", return_as="generator")(
                    delayed(_timed_fit_path)(p.src, layer, p.inputs["dev"], p.grid.y, p.inputs["folds"],
                                             C_GRID[:j + 1]) for p, layer, j in batch)
                for (p, layer, j), (oof, secs) in zip(batch, out):
                    p.grid.add(layer, oof)
                    p.cv_seconds += secs
                    left[p.name] -= 1
                    if left[p.name] == 0:
                        log(f"  {p.name}: {len(p.grid.oof)} layers known ({p.cv_seconds:.0f} s of fits); {memory()}")


def compare_cells(p: Probe) -> dict:
    """Every recomputed cell against cv_grid.csv, after the same rounding."""
    checked, matched, worst_mean, worst_sd, bad = 0, 0, 0.0, 0.0, []
    for layer in sorted(p.grid.oof):
        i = p.grid.layers.index(layer)
        for j in range(len(p.grid.oof[layer])):
            m, s = p.grid.mean(i, j), p.grid.sd(i, j)
            sm, ss = p.grid.stored[(layer, float(C_GRID[j]))]
            checked += 1
            ok = f"{m:.5f}" == sm and f"{s:.5f}" == ss
            matched += ok
            worst_mean, worst_sd = max(worst_mean, abs(m - float(sm))), max(worst_sd, abs(s - float(ss)))
            if not ok:
                bad.append({"layer": layer, "C": float(C_GRID[j]), "mean": m, "sd": s, "stored": [sm, ss]})
    return {"cells_checked": checked, "cells_matched": matched, "cells_total": len(p.grid.stored),
            "layers_slice": p.slice_layers, "layers_recomputed": sorted(p.grid.oof),
            "max_abs_diff_mean": worst_mean, "max_abs_diff_sd": worst_sd, "mismatches": bad[:10]}


# ------------------------------------------------------------------------------------------------ refits


def refit_iid(p: Probe, limit) -> tuple[SelectedProbe, dict, dict]:
    """fit_iid's refit, thresholds and test metrics, at the recorded layer and C."""
    layer, C = int(p.meta["selection"]["layer"]), float(p.meta["selection"]["C"])
    dev, test, y_all = p.inputs["dev"], p.inputs["test"], p.inputs["y_all"]
    x_sel = p.src.layer(layer)
    with limit():
        probe = SelectedProbe.fit(x_sel[dev], y_all[dev], C, layer)
        t_logit = probe.logit(x_sel[test])
    oof_sel = p.grid.oof[layer][C_GRID.index(C)].mean(0)
    probe.threshold_1, probe.threshold_5 = thresholds(oof_sel[y_all[dev] == 0])
    yt, gt = y_all[test], p.inputs["groups_all"][test]
    rate = fit_probes.rate
    metrics = {
        "test_auroc": auroc(t_logit, yt), "test_auroc_ci": ci(cluster_bootstrap(auroc, gt, t_logit, yt, n=2000)),
        "test_auroc_behavior_present": auroc(t_logit, p.inputs["bp_all"][test]),
        "tpr_at_t1": rate(t_logit, yt == 1, probe.threshold_1), "fpr_at_t1": rate(t_logit, yt == 0, probe.threshold_1),
        "tpr_at_t5": rate(t_logit, yt == 1, probe.threshold_5), "fpr_at_t5": rate(t_logit, yt == 0, probe.threshold_5),
        "n": {"dev": len(dev), "dev_hacks": int(y_all[dev].sum()), "test": len(test), "test_hacks": int(yt.sum())},
    }
    return probe, metrics, {p.src.ids[i]: float(z) for i, z in zip(test, t_logit)}


def refit_ood(p: Probe, items: dict, concealed: dict | None, limit) -> tuple[SelectedProbe, dict, dict]:
    """fit_ood's refit, test metrics and TerminalEnv transfer (with its thresholds), at the recorded layer and C."""
    layer, C, readout = int(p.meta["selection"]["layer"]), float(p.meta["selection"]["C"]), p.meta["readout"]
    dev, test, y_all = p.inputs["dev"], p.inputs["test"], p.inputs["y_all"]
    source, rate = p.inputs["source"], fit_probes.rate
    x_sel = p.src.layer(layer)
    with limit():
        probe = SelectedProbe.fit(x_sel[dev], y_all[dev], C, layer)
        t_logit = probe.logit(x_sel[test])
    metrics = {"test_auroc": auroc(t_logit, y_all[test]),
               "test_auroc_ci": ci(cluster_bootstrap(auroc, p.inputs["pair"][test], t_logit, y_all[test], n=2000)),
               "per_source": {s: auroc(t_logit[source[test] == s], y_all[test][source[test] == s])
                              for s in p.inputs["sources"]}}
    transfer, per_view_t = {}, {}
    for tview in VIEWS:
        tsrc = LayerSource(os.path.join(p.features, "terminal_prerl"), tview, readout)
        titems = [items[i] for i in tsrc.ids]
        ty = np.array([int(bool((it.get("diagnostics") or {}).get("used_verifier"))) for it in titems])
        tsplit = np.array([it["split"] for it in titems])
        tgroups = np.array([it["task_id"] for it in titems])
        rows = np.flatnonzero(tsrc.keep)
        xt = tsrc.layer(layer, rows)
        with limit():
            logit = probe.logit(xt)
        dev_neg = np.isin(tsplit[rows], ["train", "val"]) & (ty[rows] == 0)
        t1, t5 = thresholds(logit[dev_neg])
        per_view_t[tview] = (t1, t5)
        d = [titems[i].get("diagnostics") or {} for i in rows]
        hw = np.array([ty[i] == 1 or not dd.get("passed") for i, dd in zip(rows, d)], bool)
        rnu = np.array([ty[i] == 1 or dd.get("read_not_used") for i, dd in zip(rows, d)], bool)
        test_rows = tsplit[rows] == "test"
        transfer[tview] = {
            "auroc_all": auroc(logit, ty[rows]),
            "auroc_all_ci": ci(cluster_bootstrap(auroc, tgroups[rows], logit, ty[rows], n=1000)),
            "auroc_test": auroc(logit[test_rows], ty[rows][test_rows]),
            "hacks_vs_read_not_used": auroc(logit[rnu], ty[rows][rnu]),
            "hacks_vs_honest_wrong": auroc(logit[hw], ty[rows][hw]),
            "fpr_honest_wrong_at_t1": rate(logit, hw & (ty[rows] == 0), t1),
            "tpr_at_t1": rate(logit, ty[rows] == 1, t1), "thresholds": [t1, t5],
        }
        if concealed is not None and os.path.isdir(os.path.join(p.features, "terminal_concealed", tview)):
            csrc = LayerSource(os.path.join(p.features, "terminal_concealed"), tview, readout)
            crows = np.flatnonzero(csrc.keep)
            xc = csrc.layer(layer, crows)
            with limit():
                clog = probe.logit(xc)
            lv = np.array([concealed[csrc.ids[i]]["concealment"] for i in crows])
            transfer[tview]["concealment_recall_at_t1"] = {lev: rate(clog, lv == lev, t1) for lev in ("A", "B")}
    metrics["transfer"] = transfer
    metrics["thresholds_by_terminal_view"] = {v: list(t) for v, t in per_view_t.items()}  # meta.json's, compared too
    probe.threshold_1, probe.threshold_5 = per_view_t["without_reasoning"]
    return probe, metrics, {p.src.ids[i]: float(z) for i, z in zip(test, t_logit)}


# ------------------------------------------------------------------------------------------------ comparing


def numbers(d, prefix: str = "") -> dict[str, float]:
    """The numbers in nested dicts and lists, by path."""
    if isinstance(d, dict):
        return {k: v for key, val in d.items() for k, v in numbers(val, f"{prefix}{key}/").items()}
    if isinstance(d, (list, tuple)):
        return {k: v for n, val in enumerate(d) for k, v in numbers(val, f"{prefix}{n}/").items()}
    if isinstance(d, (bool, int, float, np.number, np.bool_)):
        return {prefix.rstrip("/"): float(d)}
    return {}


def close(new: float, old: float, tol: float) -> bool:
    if math.isnan(new) or math.isnan(old):
        return math.isnan(new) and math.isnan(old)
    return abs(new - old) <= tol * max(1.0, abs(old))


def compare(new: dict, old: dict, tol: float) -> dict:
    """The numbers of ``new`` against the same paths in ``old``: how many, the largest |Δ|, which ones differ."""
    a, b = numbers(new), numbers(old)
    missing = sorted(k for k in a if k not in b)
    diffs = {k: (0.0 if (math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] else abs(a[k] - b[k]))
             for k in a if k in b}
    bad = sorted(k for k in diffs if not close(a[k], b[k], tol))
    worst = max(diffs, key=lambda k: diffs[k]) if diffs else None
    return {"compared": len(diffs), "max_abs_diff": diffs[worst] if worst else 0.0, "worst": worst,
            "identical": all(v == 0.0 for v in diffs.values()) and not missing, "missing": missing,
            "differ": [{"key": k, "new": a[k], "saved": b[k]} for k in bad[:10]], "ok": not bad and not missing}


def check_refit(p: Probe, probe: SelectedProbe, metrics: dict, test_logits: dict, tol: float) -> dict:
    s = p.saved
    weights = compare({"coef": probe.coef.tolist(), "intercept": probe.intercept, "layer": probe.layer},
                      {"coef": s.coef.tolist(), "intercept": s.intercept, "layer": s.layer}, tol)
    weights["coef_max_abs"] = float(np.abs(s.coef).max())
    new_t = {"npz": [probe.threshold_1, probe.threshold_5], "meta": [probe.threshold_1, probe.threshold_5]}
    old_t = {"npz": [s.threshold_1, s.threshold_5], "meta": [p.meta["threshold_1"], p.meta["threshold_5"]]}
    if not p.iid:
        new_t["by_terminal_view"] = metrics["thresholds_by_terminal_view"]
        old_t["by_terminal_view"] = p.meta["thresholds_by_terminal_view"]
    saved_metrics = {k: v for k, v in p.metrics.items() if k != "selection"}
    m = compare({k: v for k, v in metrics.items() if k != "thresholds_by_terminal_view"}, saved_metrics, tol)
    with open(os.path.join(p.path, "scores.jsonl")) as fh:
        saved_logits = {r["id"]: r["test_logit"] for r in map(json.loads, fh) if r.get("test_logit") is not None}
    logits = compare(test_logits, saved_logits, tol)
    logits["same_ids"] = list(test_logits) == list(saved_logits)
    logits["ok"] = logits["ok"] and logits["same_ids"]
    return {"weights": weights, "thresholds": compare(new_t, old_t, tol), "metrics": m, "test_logits": logits}


def check_inputs(p: Probe) -> dict:
    out = {"n_dev": len(p.inputs["dev"]), "n_test": len(p.inputs["test"]), "repeats": int(p.meta["repeats"]),
           "grid_layers": sorted({k[0] for k in p.grid.stored}) == p.grid.layers
           and sorted({k[1] for k in p.grid.stored}) == sorted(C_GRID),
           "meta": p.meta["layer"] == p.meta["selection"]["layer"] and p.meta["C"] == p.meta["selection"]["C"]
           and p.meta["n_layers"] == p.src.n_layers
           and p.meta.get("model_layer") == fit_probes.router_layer(p.meta["readout"], p.meta["layer"]).get(
               "model_layer")}
    if p.iid:
        y, dev, test = p.inputs["y_all"], p.inputs["dev"], p.inputs["test"]
        n = {"dev": len(dev), "dev_hacks": int(y[dev].sum()), "test": len(test), "test_hacks": int(y[test].sum())}
        out["counts"] = n == p.metrics["n"]
    out["ok"] = all(v for k, v in out.items() if isinstance(v, bool))
    return out


def check_selection(p: Probe, tol: float) -> dict:
    saved, d = p.meta["selection"], p.selection or {"skipped": "not reached"}
    if "skipped" in d or "error" in d:
        return {"status": "error" if "error" in d else "not re-derived", **d, "saved": saved, "ok": "error" not in d}
    same = d["layer"] == saved["layer"] and d["C"] == saved["C"] and d["flat"] == saved["flat"] \
        and close(d["cv_auroc"], saved["cv_auroc"], tol) \
        and ("cv_se" not in saved or close(d["cv_se"], saved["cv_se"], tol))
    diffs = {f"{k}_abs_diff": 0.0 if math.isnan(d[k]) and math.isnan(saved[k]) else abs(d[k] - saved[k])
             for k in ("cv_auroc", "cv_se") if k in saved}
    return {"status": "reproduced" if same else "differs", "derived": d, "saved": saved, "ok": same, **diffs}


# ------------------------------------------------------------------------------------------------ main


class MemoryWatch:
    """The cgroup's memory (and its anonymous part) every few seconds, for the peaks."""

    def __init__(self, every: float = 5.0):
        self.peak_current = self.peak_anon = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(every,), daemon=True)
        self._t.start()

    def sample(self) -> tuple[int, int]:
        try:
            cur = int(open("/sys/fs/cgroup/memory.current").read())
            anon = next(int(line.split()[1]) for line in open("/sys/fs/cgroup/memory.stat") if line.startswith("anon "))
        except (OSError, StopIteration, ValueError):
            return 0, 0
        self.peak_current, self.peak_anon = max(self.peak_current, cur), max(self.peak_anon, anon)
        return cur, anon

    def _run(self, every: float) -> None:
        while not self._stop.wait(every):
            self.sample()

    def stop(self) -> dict:
        self._stop.set()
        return {"peak_memory_current_gb": round(self.peak_current / 2**30, 2),
                "peak_anon_gb": round(self.peak_anon / 2**30, 2)}


WATCH: MemoryWatch | None = None


def memory() -> str:
    if WATCH is None:
        return ""
    cur, anon = WATCH.sample()
    return f"memory {cur / 2**30:.1f} GB (anon {anon / 2**30:.1f})" if cur else ""


def fmt(x: float) -> str:
    return "0" if x == 0 else f"{x:.1e}"


def report(summary: dict) -> str:
    lines = ["# Do today's fit_probes.py and probe_fit.py reproduce the fitted probes?", "",
             f"{summary['created_utc']}; code sha256 fit_probes.py {summary['code']['fit_probes.py'][:12]}, "
             f"probe_fit.py {summary['code']['probe_fit.py'][:12]}; {summary['seconds']['total']:.0f} s "
             f"(out-of-fold fits {summary['seconds']['cv']:.0f} s, refits {summary['seconds']['refit']:.0f} s). "
             "Cells: recomputed out-of-fold AUROC (mean and sd) equal to cv_grid.csv's to 5 decimals. "
             "Δ: the largest absolute difference from what was saved (0 = bit for bit).", "",
             "| probe | cells matched / checked (of grid) | layers recomputed | selection | Δ weights | Δ t₁/t₅ "
             "| Δ metrics | Δ test logits | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in summary["probes"].items():
        if "error" in r:
            lines.append(f"| {name} | | | | | | | | error: {r['error']} |")
            continue
        cv, s, rf = r["cv"], r["selection"], r["refit"]
        sel = f"{s['status']} ({s['saved']['layer']}, C {s['saved']['C']:g}{', flat' if s['saved']['flat'] else ''})"
        lines.append(
            f"| {name} | {cv['cells_matched']} / {cv['cells_checked']} ({cv['cells_total']}) "
            f"| {len(cv['layers_recomputed'])} | {sel} | {fmt(rf['weights']['max_abs_diff'])} "
            f"| {fmt(rf['thresholds']['max_abs_diff'])} | {fmt(rf['metrics']['max_abs_diff'])} "
            f"({rf['metrics']['compared']}) | {fmt(rf['test_logits']['max_abs_diff'])} | {r['verdict']} |")
    bad = {n: r for n, r in summary["probes"].items() if r.get("verdict") != "pass"}
    if bad:
        lines += ["", "## Not passing", ""]
        for n, r in bad.items():
            lines.append(f"- **{n}** ({r.get('verdict')}): " + json.dumps(
                {k: r.get(k) for k in ("error", "inputs", "selection")} | {
                    "cell mismatches": r.get("cv", {}).get("mismatches"),
                    "refit differences": {k: v.get("differ") for k, v in r.get("refit", {}).items()}}, default=str))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    global WATCH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", required=True, help="the folder with <reader>/<dataset>/<view>/<readout>/ probes")
    ap.add_argument("--check", action="append", required=True,
                    help="READER/READOUTS=FEATURES, e.g. bf16/mean,router=../scratch/features_mr/bf16 (repeatable)")
    ap.add_argument("--datasets", default="data/probe_datasets")
    ap.add_argument("--out", required=True, help="a <UTC timestamp> folder is made in it")
    ap.add_argument("--only", default="", help="comma-list: keep probes whose name contains one of these")
    ap.add_argument("--jobs", type=int, default=6, help="parallel single-threaded workers for out-of-fold fits")
    ap.add_argument("--fit-jobs", type=int, default=6, help="fit_probes' --jobs when the probes were fitted")
    ap.add_argument("--refit-threads", type=int, help="cap BLAS threads in the refits (default: BLAS's own)")
    ap.add_argument("--seed", type=int, default=0, help="fit_probes' --seed (not recorded in meta.json)")
    ap.add_argument("--other-layers", type=int, default=2, help="grid-slice layers beyond the selected and best")
    ap.add_argument("--full-grid", default="router,router_all", help="readouts whose whole grid is recomputed")
    ap.add_argument("--max-layers", type=int, help="give up re-deriving a selection that needs more layers than this")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)
    t_start = time.time()
    out = os.path.join(args.out, time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t_start)))
    os.makedirs(out)  # now, so that a clash shows before the fits
    code = {f: hashlib.sha256(open(path, "rb").read()).hexdigest() for f, path in (
        ("fit_probes.py", os.path.join(HERE, "fit_probes.py")), ("probe_fit.py", probe_fit.__file__),
        ("verify_fits.py", os.path.abspath(__file__)))}  # the code that runs, before anyone edits it
    WATCH = MemoryWatch()
    from threadpoolctl import threadpool_limits

    limit = (lambda: threadpool_limits(args.refit_threads)) if args.refit_threads else contextlib.nullcontext
    data = {"terminal_prerl": load_items(os.path.join(args.datasets, "terminal_prerl")),
            "diverse_deception": load_items(os.path.join(args.datasets, "diverse_deception"))}
    conc_dir = os.path.join(args.datasets, "terminal_concealed")
    concealed = load_items(conc_dir) if os.path.isdir(conc_dir) else None
    only = [s for s in args.only.split(",") if s]
    full = {r for r in args.full_grid.split(",") if r}
    probes, summary = [], {"probes": {}}
    for spec in args.check:
        left, _, features = spec.partition("=")
        reader, _, readouts = left.partition("/")
        if not (reader and readouts and features):
            ap.error(f"--check {spec!r}: expected READER/READOUTS=FEATURES")
        for readout in [r for r in readouts.split(",") if r]:
            for dataset, view in PROBES:
                name = f"{reader}/{dataset}/{view}/{readout}"
                path = os.path.join(args.probes, reader, dataset, view, readout)
                if only and not any(s in name for s in only):
                    continue
                if not os.path.exists(os.path.join(path, "probe.npz")):
                    log(f"{name}: no fitted probe, skipped")
                    continue
                p = load_probe(path, name, features, data, args.seed, args.fit_jobs)
                p.slice_layers = slice_layers(p, readout in full, args.other_layers)
                probes.append(p)
                log(f"{name}: {len(p.inputs['dev'])} dev, {len(p.inputs['test'])} test, {len(p.grid.layers)} layers;"
                    f" slice {p.slice_layers}, {p.threads} BLAS thread(s) per worker")
    for features in sorted({p.features for p in probes}):
        drop_cache(shard_files(features))
    log(f"{len(probes)} probes; {memory()}")
    # out-of-fold fits: the slices, then rounds of whatever the selections still need
    t_cv = time.time()
    run_layers([(p, layer, len(C_GRID) - 1) for p in probes for layer in p.slice_layers], args.jobs)
    for rnd in range(1, 100):
        tasks = []
        for p in probes:
            if p.selection is not None:
                continue
            try:
                p.selection = derive_selection(p.grid, p.inputs["groups_dev"], p.inputs["tie_metric"], args.seed,
                                               batch=args.jobs)
                log(f"{p.name}: selection re-derived: layer {p.selection['layer']}, C {p.selection['C']:g}"
                    f"{' (flat grid)' if p.selection['flat'] else ''}; {len(p.grid.oof)} layers recomputed")
            except Missing as m:
                if args.max_layers is not None and len(set(p.grid.oof) | set(m.need)) > args.max_layers:
                    p.selection = {"skipped": f"needs more than {args.max_layers} layers"}
                    log(f"{p.name}: selection not re-derived ({p.selection['skipped']})")
                    continue
                tasks += [(p, layer, j) for layer, j in m.need.items()]
            except Exception as e:  # noqa: BLE001  (reported as this probe's failure)
                p.selection = {"error": f"{type(e).__name__}: {e}"}
                log(f"{p.name}: selection failed: {p.selection['error']}")
        if not tasks:
            break
        log(f"selection round {rnd}: {len(tasks)} layers for {len({t[0].name for t in tasks})} probes")
        run_layers(tasks, args.jobs)
    cv_seconds = time.time() - t_cv
    # refits, one probe at a time
    t_refit = time.time()
    for p in probes:
        t0 = time.time()
        try:
            if p.iid:
                probe, metrics, logits = refit_iid(p, limit)
            else:
                probe, metrics, logits = refit_ood(p, data["terminal_prerl"], concealed, limit)
            r = {"features": p.features, "inputs": check_inputs(p), "cv": compare_cells(p),
                 "selection": check_selection(p, args.tol), "refit": check_refit(p, probe, metrics, logits, args.tol)}
        except Exception as e:  # noqa: BLE001  (one probe's failure is reported, the others still run)
            r = {"features": p.features, "error": f"{type(e).__name__}: {e}"}
        r["seconds"] = {"cv_fits": round(p.cv_seconds, 1), "refit": round(time.time() - t0, 1)}
        if "error" in r:
            r["verdict"] = "fail"
        else:
            ok = r["inputs"]["ok"] and r["cv"]["cells_matched"] == r["cv"]["cells_checked"] and r["selection"]["ok"] \
                and all(v["ok"] for v in r["refit"].values())
            r["verdict"] = ("pass" if r["selection"]["status"] == "reproduced" else "pass, selection not re-derived") \
                if ok else "fail"
        summary["probes"][p.name] = r
        drop_cache(shard_files(p.features))
        log(f"{p.name}: {r['verdict']} ({r['seconds']['refit']:.0f} s refit); {memory()}")
    summary = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(args),
               "code": code,
               "seconds": {"cv": cv_seconds, "refit": time.time() - t_refit, "total": time.time() - t_start},
               **WATCH.stop(), **summary}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    text = report(summary)
    open(os.path.join(out, "report.md"), "w").write(text)
    print(text)
    log(f"wrote {out}")
    return 0 if all(r["verdict"].startswith("pass") for r in summary["probes"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
