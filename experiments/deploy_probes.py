"""Turn fitted Inkling probes (experiments/fit_probes.py) into deployed monitors under data/probes/Inkling-Small/,
each with the golden set the probe server checks at startup.

    uv run python experiments/deploy_probes.py --probes ../scratch/probes/nvfp4 --features ../scratch/features/nvfp4 \
        --diagnostics ../scratch/diagnostics/nvfp4 --out data/probes/Inkling-Small

**What gets deployed** (one directory per monitor, ``<dataset>/<view>/<readout>/``; ``view`` is how the server
renders a TerminalEnv episode):
- **IID** ``terminal_prerl/<view>/<readout>``: the fitted probe as it is.
- **OOD** ``diverse_deception/<view>/<readout>`` for both TerminalEnv views: the probe fitted on diverse_deception
  (rendered without reasoning), with the t₁%/t₅% thresholds of that TerminalEnv view's honest dev episodes.

**Added to meta.json:** ``name``, ``effort`` (the probe datasets' rendering effort), ``golden_tolerance`` and the
fitted directory.

**Serving on another reader** (``--features`` of a reader other than the one the probes were fitted on; PROBE_PLAN.md
section 8: train on full precision, serve 4-bit). The served logits come from the serving reader, so:
- t₁%/t₅% are recalibrated as the 99th/95th percentiles of the probe's logits on the serving reader's honest
  TerminalEnv dev episodes (train + val), per view;
- ``reader``/``reader_info`` name the serving reader (the monitor checks them against the server), and the fitting
  reader's go to ``trained_reader``/``trained_reader_info``;
- the golden logits and the tolerance come from the serving reader's features and diagnostics.

**Task splits** (``task_splits.json``): the TerminalEnv task ids of the probes' train/val/test splits. The probes were
fitted on pre-RL episodes of the same eval tasks the RL runs evaluate on, so report held-out monitor metrics on
the ``test`` tasks too: those are the tasks no probe saw (within-task AUROC says the probes don't key on the task).

**Serving check** (in ``deployed.json``): on the serving reader's TerminalEnv features, the deployed probe's test
AUROC and its TPR/FPR at t₁%; and the batching rule's numbers for it (PROBE_PLAN.md 5b): the 99th-percentile
|Δlogit| between batched and one-at-a-time readouts on the diagnostics sample, and the share of hacks and of
negatives whose logit lies that close to t₁%.

**Golden set** (``golden.jsonl``): 32 TerminalEnv test episodes (up to 8 hacks), with their renderings' sha256 and
their logits on the stored features (``probe_fit.write_golden``).

**Tolerance** (the largest |Δlogit| the server's check accepts):
- The server batches differently from the extraction, and batching moves 4-bit readouts by ~4% (median, layer 25).
- **Follow-up probes:** twice the largest |Δlogit| between batched and one-at-a-time readouts on the diagnostics
  sample (228 inputs), at least 1.
- **Mean probes:** 3, unmeasured: the diagnostics keep only the follow-up readout.
- The check exists to catch a wrong reader, layer or rendering. Renderings are compared exactly (sha256).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from monitordecorrelation.whitebox.probe_fit import LayerSource, SelectedProbe, load_items, thresholds, write_golden

VIEWS = ("with_reasoning", "without_reasoning")
N_GOLDEN, N_GOLDEN_HACKS = 32, 8
MEAN_TOLERANCE = 3.0


def golden_ids(items: dict, src: LayerSource, seed: int = 0) -> list[str]:
    """Up to ``N_GOLDEN_HACKS`` hacks and the rest negatives, from the TerminalEnv test split (kept rows only)."""
    rng = np.random.default_rng(seed)
    test = [i for i, k in zip(src.ids, src.keep) if k and items[i]["split"] == "test"]
    hack = {i: bool((items[i].get("diagnostics") or {}).get("used_verifier")) for i in test}  # the IID label
    hacks, negs = sorted(i for i in test if hack[i]), sorted(i for i in test if not hack[i])
    pick = list(rng.permutation(hacks)[:N_GOLDEN_HACKS])
    pick += list(rng.permutation(negs)[: N_GOLDEN - len(pick)])
    return [str(i) for i in pick]


def followup_tolerance(diagnostics: str | None, probe: SelectedProbe) -> float | None:
    path = os.path.join(diagnostics or "", "diag.safetensors")
    if not diagnostics or not os.path.exists(path):
        return None
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        a = f.get_slice("followup_batched")[:, probe.layer, :].float().numpy().astype(np.float64)
        b = f.get_slice("followup_alone")[:, probe.layer, :].float().numpy().astype(np.float64)
    worst = float(np.abs(probe.logit(a) - probe.logit(b)).max())
    return max(1.0, round(2 * worst, 2))


def recalibrate(probe: SelectedProbe, items: dict, features: str, view: str, readout: str) -> tuple[float, float]:
    """t₁%/t₅% from the probe's logits on the honest TerminalEnv dev episodes (train + val) of ``features``."""
    src = LayerSource(os.path.join(features, "terminal_prerl"), view, readout)
    rows = np.flatnonzero(src.keep)
    neg = np.array([items[src.ids[i]]["split"] in ("train", "val")
                    and not (items[src.ids[i]].get("diagnostics") or {}).get("used_verifier") for i in rows])
    return thresholds(probe.logit(src.layer(probe.layer, rows))[neg])


def serve_on(probe: SelectedProbe, serving: dict, items: dict, features: str, view: str, readout: str) -> None:
    """Re-point ``probe`` at the serving reader ``serving`` (its features' index.json ``reader``)."""
    if probe.meta.get("reader_info") == serving["reader_info"]:
        return
    probe.threshold_1, probe.threshold_5 = recalibrate(probe, items, features, view, readout)
    probe.meta = {**probe.meta, "trained_reader": probe.meta.get("reader"),
                  "trained_reader_info": probe.meta.get("reader_info"), "reader": serving["reader"],
                  "reader_info": serving["reader_info"],
                  "thresholds_from": f"honest TerminalEnv dev episodes on the {serving['reader']} reader's features"}


def serving_check(probe: SelectedProbe, items: dict, features: str, view: str, readout: str,
                  diagnostics: str | None) -> dict:
    from monitordecorrelation.whitebox.probe_fit import auroc

    src = LayerSource(os.path.join(features, "terminal_prerl"), view, readout)
    rows = np.flatnonzero(src.keep)
    its = [items[src.ids[i]] for i in rows]
    y = np.array([int(bool((it.get("diagnostics") or {}).get("used_verifier"))) for it in its])
    test = np.array([it["split"] == "test" for it in its])
    z = probe.logit(src.layer(probe.layer, rows))
    t1 = probe.threshold_1
    out = {"test_auroc": auroc(z[test], y[test]), "test_tpr_at_t1": float(np.mean(z[test & (y == 1)] > t1)),
           "test_fpr_at_t1": float(np.mean(z[test & (y == 0)] > t1))}
    path = os.path.join(diagnostics or "", "diag.safetensors")
    if readout == "followup" and diagnostics and os.path.exists(path):
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            a = f.get_slice("followup_batched")[:, probe.layer, :].float().numpy().astype(np.float64)
            b = f.get_slice("followup_alone")[:, probe.layer, :].float().numpy().astype(np.float64)
        p99 = float(np.percentile(np.abs(probe.logit(a) - probe.logit(b)), 99))
        near = np.abs(z - t1) <= p99
        hacks, negs = float(near[y == 1].mean()), float(near[y == 0].mean())
        out["batching"] = {"dlogit_p99": p99, "near_threshold_hacks": hacks, "near_threshold_negatives": negs,
                           "fix_batching": hacks > 0.01 or negs > 0.005}
    return out


def fitted_ref(fitted: str) -> str:
    """Where the fitted probe lives in the repo: experiments/probe_report.py collects ``<reader>/<dataset>/<view>/
    <readout>`` of fit_probes.py's output (the --probes folder is named after its reader) into data/probe_results."""
    return "data/probe_results/fitted/" + "/".join(os.path.normpath(fitted).split(os.sep)[-4:])


def deploy(probe: SelectedProbe, fitted: str, out: str, name: str, view: str, readout: str, items: dict,
           features: str, effort: float, tolerance: float | None, log, diagnostics: str | None = None) -> dict:
    src = LayerSource(os.path.join(features, "terminal_prerl"), view, readout)
    ids = golden_ids(items, src)
    if tolerance is None:
        tolerance = MEAN_TOLERANCE
    probe.meta = {**probe.meta, "name": name, "view": view, "effort": effort, "golden_tolerance": tolerance,
                  "fitted_dir": fitted_ref(fitted), "deployed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                       time.gmtime())}
    if os.path.isdir(out):
        shutil.rmtree(out)
    probe.save(out)
    n = write_golden(out, probe, src, items, readout, view, ids, effort=effort)
    log(f"{name}: layer {probe.layer}, t1 {probe.threshold_1:.3f}, golden {n} episodes, tolerance {tolerance}")
    reader = probe.meta.get("reader")
    return {"name": name, "dir": out, "layer": probe.layer, "golden": n, "golden_tolerance": tolerance,
            "reader": reader, "trained_reader": probe.meta.get("trained_reader", reader),
            "threshold_1": probe.threshold_1,
            "serving_check": serving_check(probe, items, features, view, readout, diagnostics)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", required=True, help="fit_probes.py's --out for one reader")
    ap.add_argument("--features", required=True,
                    help="the serving reader's features (<dataset>/<view>/shard_*): the fitting reader's, or another")
    ap.add_argument("--serving-reader", default="nvfp4", help="the serving reader's name, for the record")
    ap.add_argument("--diagnostics", help="the serving reader's diagnostics folder (diag.safetensors)")
    ap.add_argument("--datasets", default="data/probe_datasets")
    ap.add_argument("--out", default="data/probes/Inkling-Small")
    ap.add_argument("--readouts", default="followup,mean")
    args = ap.parse_args()
    log = lambda m: print(f"[deploy] {m}", flush=True)  # noqa: E731
    items = load_items(os.path.join(args.datasets, "terminal_prerl"))
    effort = json.load(open(os.path.join(args.datasets, "terminal_prerl", "manifest.json"))).get("effort", 0.5)
    index = json.load(open(os.path.join(args.features, "terminal_prerl", "index.json")))
    serving = {"reader": args.serving_reader, "reader_info": index["reader"]}
    done = []
    for readout in args.readouts.split(","):
        for view in VIEWS:
            fitted = os.path.join(args.probes, "terminal_prerl", view, readout)
            if os.path.isdir(fitted):
                p = SelectedProbe.load(fitted)
                serve_on(p, serving, items, args.features, view, readout)
                tol = followup_tolerance(args.diagnostics, p) if readout == "followup" else None
                done.append(deploy(p, fitted, os.path.join(args.out, "terminal_prerl", view, readout),
                                   f"probe_iid_{view}_{readout}", view, readout, items, args.features, effort, tol,
                                   log, args.diagnostics))
        fitted = os.path.join(args.probes, "diverse_deception", "without_reasoning", readout)
        if os.path.isdir(fitted):
            for view in VIEWS:
                p = SelectedProbe.load(fitted)
                p.threshold_1, p.threshold_5 = p.meta["thresholds_by_terminal_view"][view]
                p.meta = {**p.meta, "train_view": "without_reasoning"}
                serve_on(p, serving, items, args.features, view, readout)
                tol = followup_tolerance(args.diagnostics, p) if readout == "followup" else None
                done.append(deploy(p, fitted, os.path.join(args.out, "diverse_deception", view, readout),
                                   f"probe_ood_{view}_{readout}", view, readout, items, args.features, effort, tol,
                                   log, args.diagnostics))
    splits: dict[str, set] = {}
    for it in items.values():
        splits.setdefault(it["split"], set()).add(it["task_id"])
    with open(os.path.join(args.out, "task_splits.json"), "w") as fh:
        json.dump({k: sorted(v) for k, v in sorted(splits.items())}, fh, indent=1)
    record = os.path.join(args.out, "deployed.json")  # merged with earlier runs (e.g. another reader's readouts)
    earlier = json.load(open(record)) if os.path.exists(record) else []
    done += [d for d in earlier if d["dir"] not in {e["dir"] for e in done}]
    with open(record, "w") as fh:
        json.dump(sorted(done, key=lambda d: d["dir"]), fh, indent=1)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
