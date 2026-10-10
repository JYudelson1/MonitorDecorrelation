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

**Task splits** (``task_splits.json``): the TerminalEnv task ids of the probes' train/val/test splits. The probes were
fitted on pre-RL episodes of the same eval tasks the RL runs evaluate on, so report held-out monitor metrics on
the ``test`` tasks too: those are the tasks no probe saw (within-task AUROC says the probes don't key on the task).

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

from monitordecorrelation.whitebox.probe_fit import LayerSource, SelectedProbe, load_items, write_golden

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


def deploy(probe: SelectedProbe, fitted: str, out: str, name: str, view: str, readout: str, items: dict,
           features: str, effort: float, tolerance: float | None, log) -> dict:
    src = LayerSource(os.path.join(features, "terminal_prerl"), view, readout)
    ids = golden_ids(items, src)
    if tolerance is None:
        tolerance = MEAN_TOLERANCE
    probe.meta = {**probe.meta, "name": name, "view": view, "effort": effort, "golden_tolerance": tolerance,
                  "fitted_dir": os.path.relpath(fitted), "deployed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                       time.gmtime())}
    if os.path.isdir(out):
        shutil.rmtree(out)
    probe.save(out)
    n = write_golden(out, probe, src, items, readout, view, ids, effort=effort)
    log(f"{name}: layer {probe.layer}, t1 {probe.threshold_1:.3f}, golden {n} episodes, tolerance {tolerance}")
    return {"name": name, "dir": out, "layer": probe.layer, "golden": n, "golden_tolerance": tolerance}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", required=True, help="fit_probes.py's --out for one reader")
    ap.add_argument("--features", required=True, help="that reader's features (<dataset>/<view>/shard_*)")
    ap.add_argument("--diagnostics", help="that reader's diagnostics folder (diag.safetensors)")
    ap.add_argument("--datasets", default="data/probe_datasets")
    ap.add_argument("--out", default="data/probes/Inkling-Small")
    ap.add_argument("--readouts", default="followup,mean")
    args = ap.parse_args()
    log = lambda m: print(f"[deploy] {m}", flush=True)  # noqa: E731
    items = load_items(os.path.join(args.datasets, "terminal_prerl"))
    effort = json.load(open(os.path.join(args.datasets, "terminal_prerl", "manifest.json"))).get("effort", 0.5)
    done = []
    for readout in args.readouts.split(","):
        for view in VIEWS:
            fitted = os.path.join(args.probes, "terminal_prerl", view, readout)
            if os.path.isdir(fitted):
                p = SelectedProbe.load(fitted)
                tol = followup_tolerance(args.diagnostics, p) if readout == "followup" else None
                done.append(deploy(p, fitted, os.path.join(args.out, "terminal_prerl", view, readout),
                                   f"probe_iid_{view}_{readout}", view, readout, items, args.features, effort, tol, log))
        fitted = os.path.join(args.probes, "diverse_deception", "without_reasoning", readout)
        if os.path.isdir(fitted):
            for view in VIEWS:
                p = SelectedProbe.load(fitted)
                p.threshold_1, p.threshold_5 = p.meta["thresholds_by_terminal_view"][view]
                p.meta = {**p.meta, "train_view": "without_reasoning"}
                tol = followup_tolerance(args.diagnostics, p) if readout == "followup" else None
                done.append(deploy(p, fitted, os.path.join(args.out, "diverse_deception", view, readout),
                                   f"probe_ood_{view}_{readout}", view, readout, items, args.features, effort, tol,
                                   log))
    splits: dict[str, set] = {}
    for it in items.values():
        splits.setdefault(it["split"], set()).add(it["task_id"])
    with open(os.path.join(args.out, "task_splits.json"), "w") as fh:
        json.dump({k: sorted(v) for k, v in sorted(splits.items())}, fh, indent=1)
    with open(os.path.join(args.out, "deployed.json"), "w") as fh:
        json.dump(done, fh, indent=1)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
