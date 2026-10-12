"""How stable is an OOD probe's transfer to TerminalEnv hacks?

On 2026-10-10 the 4-bit and the full-precision OOD follow-up probes had the same layer, C and in-distribution AUROC
(0.977 / 0.979, weight cosine 0.80), yet transferred to TerminalEnv hacks at 0.91 and 0.71. This measures how much of
that is refitting noise. For one fitted OOD probe (experiments/fit_probes.py):
- **Refits:** ``--refits`` fits on pair-bootstrap resamples of its dev set, at its own layer and C.
- **By layer:** one fit on all of dev at every layer, at its C, to show how transfer depends on the layer.

    uv run python experiments/ood_refits.py --probe PROBES/bf16/diverse_deception/without_reasoning/followup \
        --features FEATURES/bf16 --out OUT/ood_refits/bf16/followup

**Scored for every fit:**
- OOD test AUROC (in distribution);
- per TerminalEnv view:
  - transfer AUROC over all episodes (label ``used_verifier``);
  - hacks vs read-not-used;
  - TPR at t₁%, with t₁% from that view's honest dev episodes, as fit_probes sets it;
  - concealment recall at t₁% (levels A and B), when the concealment features are there;
- the cosine of its weights to the original probe's.

**Writes** ``refits.csv``, ``by_layer.csv`` and ``summary.json`` (each metric's value for the original probe, then
the median and 5th/95th percentiles over the refits; the code version, provenance.code_version, as ``code``). An
``--out`` that already holds any of them is refused, unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from monitordecorrelation.provenance import code_version
from monitordecorrelation.whitebox.probe_fit import LayerSource, SelectedProbe, auroc, load_items, thresholds

VIEWS = ("with_reasoning", "without_reasoning")
OUTPUTS = ("refits.csv", "by_layer.csv", "summary.json")


class Scorer:
    """Every evaluation set of one probe layer, read once."""

    def __init__(self, features: str, datasets: str, readout: str, layer: int):
        ood = load_items(os.path.join(datasets, "diverse_deception"))
        src = LayerSource(os.path.join(features, "diverse_deception"), "without_reasoning", readout)
        its = [ood[i] for i in src.ids]
        split = np.array([it["split"] for it in its])
        self.dev = np.flatnonzero((split == "train") & src.keep)
        test = np.flatnonzero((split == "test") & src.keep)
        y = np.array([int(it["label"]) for it in its])
        self.y_dev, self.groups_dev = y[self.dev], np.array([it["pair"] for it in its])[self.dev]
        x = src.layer(layer)
        self.x_dev, self.x_test, self.y_test = x[self.dev], x[test], y[test]
        items = load_items(os.path.join(datasets, "terminal_prerl"))
        conc_path = os.path.join(datasets, "terminal_concealed")
        concealed = load_items(conc_path) if os.path.isdir(conc_path) else None
        self.terminal = {}
        for view in VIEWS:
            tsrc = LayerSource(os.path.join(features, "terminal_prerl"), view, readout)
            rows = np.flatnonzero(tsrc.keep)
            d = [items[tsrc.ids[i]].get("diagnostics") or {} for i in rows]
            ty = np.array([int(bool(dd.get("used_verifier"))) for dd in d])
            tsplit = np.array([items[tsrc.ids[i]]["split"] for i in rows])
            entry = {"x": tsrc.layer(layer, rows), "y": ty,
                     "dev_neg": np.isin(tsplit, ["train", "val"]) & (ty == 0),
                     "rnu": (ty == 1) | np.array([bool(dd.get("read_not_used")) for dd in d])}
            cdir = os.path.join(features, "terminal_concealed", view)
            if concealed is not None and os.path.isdir(cdir):
                csrc = LayerSource(os.path.join(features, "terminal_concealed"), view, readout)
                crows = np.flatnonzero(csrc.keep)
                entry["conc_x"] = csrc.layer(layer, crows)
                entry["conc_level"] = np.array([concealed[csrc.ids[i]]["concealment"] for i in crows])
            self.terminal[view] = entry

    def score(self, p: SelectedProbe) -> dict:
        out = {"ood_test_auroc": auroc(p.logit(self.x_test), self.y_test)}
        for view, t in self.terminal.items():
            z = p.logit(t["x"])
            t1, _ = thresholds(z[t["dev_neg"]])
            out[f"transfer_auroc_{view}"] = auroc(z, t["y"])
            out[f"hacks_vs_read_not_used_{view}"] = auroc(z[t["rnu"]], t["y"][t["rnu"]])
            out[f"tpr_at_t1_{view}"] = float(np.mean(z[t["y"] == 1] > t1))
            if "conc_x" in t:
                cz = p.logit(t["conc_x"])
                for level in ("A", "B"):
                    out[f"concealment_{level}_at_t1_{view}"] = float(np.mean(cz[t["conc_level"] == level] > t1))
        return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def summarize(original: dict, rows: list[dict]) -> dict:
    out = {}
    for k, v in original.items():
        vals = np.array([r[k] for r in rows if r.get(k) is not None], float)
        out[k] = {"original": v, "median": float(np.median(vals)), "p05": float(np.percentile(vals, 5)),
                  "p95": float(np.percentile(vals, 95))}
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, help="a fitted OOD probe: <out>/diverse_deception/without_reasoning/<r>")
    ap.add_argument("--features", required=True, help="the same reader's features (<dataset>/<view>/shard_*)")
    ap.add_argument("--datasets", default="data/probe_datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--refits", type=int, default=50)
    ap.add_argument("--skip-layers", action="store_true", help="no by-layer fits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="replace the outputs already in --out (else refused)")
    args = ap.parse_args(argv)
    existing = [f for f in OUTPUTS if os.path.exists(os.path.join(args.out, f))]
    if existing and not args.overwrite:
        ap.error(f"{args.out} already holds {', '.join(existing)}; --overwrite replaces them")
    code = code_version()
    log = lambda m: print(f"[refits {time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731

    probe = SelectedProbe.load(args.probe)
    readout, C = probe.meta["readout"], float(probe.meta["C"])
    sc = Scorer(args.features, args.datasets, readout, probe.layer)
    original = sc.score(probe)
    log(f"{args.probe}: layer {probe.layer}, C {C:g}; original {json.dumps(original)}")
    rng = np.random.default_rng(args.seed)
    pairs = np.unique(sc.groups_dev)
    rows_of = {g: np.flatnonzero(sc.groups_dev == g) for g in pairs}
    refits = []
    while len(refits) < args.refits:
        rows = np.concatenate([rows_of[g] for g in rng.choice(pairs, len(pairs), replace=True)])
        if len(set(sc.y_dev[rows])) < 2:
            continue
        p = SelectedProbe.fit(sc.x_dev[rows], sc.y_dev[rows], C, probe.layer)
        refits.append({"refit": len(refits), **sc.score(p), "cosine_to_original": cosine(p.coef, probe.coef)})
        if len(refits) % 10 == 0:
            log(f"  {len(refits)}/{args.refits} refits")
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "refits.csv"), refits)
    summary = {"probe": args.probe, "layer": probe.layer, "C": C, "readout": readout, "refits": len(refits),
               "code": code, "metrics": summarize({**original, "cosine_to_original": 1.0}, refits)}
    if not args.skip_layers:
        src = LayerSource(os.path.join(args.features, "diverse_deception"), "without_reasoning", readout)
        by_layer = []
        for layer in range(src.first_layer, src.n_layers):
            sl = Scorer(args.features, args.datasets, readout, layer)
            by_layer.append({"layer": layer, **sl.score(SelectedProbe.fit(sl.x_dev, sl.y_dev, C, layer))})
            log(f"  layer {layer}: {json.dumps({k: round(v, 3) for k, v in by_layer[-1].items() if k != 'layer'})}")
        write_csv(os.path.join(args.out, "by_layer.csv"), by_layer)
        summary["by_layer_transfer_with_reasoning"] = {r["layer"]: r["transfer_auroc_with_reasoning"]
                                                        for r in by_layer}
    elif os.path.exists(os.path.join(args.out, "by_layer.csv")):  # an earlier run's (--overwrite): not this one's
        os.remove(os.path.join(args.out, "by_layer.csv"))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    m = summary["metrics"]
    for k in ("ood_test_auroc", "transfer_auroc_with_reasoning", "transfer_auroc_without_reasoning",
              "tpr_at_t1_with_reasoning", "cosine_to_original"):
        log(f"{k}: original {m[k]['original']:.3f}; refits median {m[k]['median']:.3f} "
            f"[{m[k]['p05']:.3f}, {m[k]['p95']:.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
