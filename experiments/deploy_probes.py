"""Turn fitted Inkling probes (experiments/fit_probes.py) into deployed monitors under data/probes/Inkling-Small/,
each with the golden set the probe server checks at startup.

    uv run python experiments/deploy_probes.py --probes PROBES/bf16 --features FEATURES/nvfp4 \
        --diagnostics DIAGNOSTICS/nvfp4 --out data/probes/Inkling-Small
    uv run python experiments/deploy_probes.py --grids-only    # only grids.json (below), from the dataset's items

**v2** (docs/inkling_probes_amendment_1.md and _2.md), fitted and served on the full-precision reader, in a root of
its own:

    uv run python experiments/deploy_probes.py --iid-dataset terminal_prerl_v2 --probes PROBES/bf16 \
        --features FEATURES/bf16 --diagnostics DIAGNOSTICS-v2/bf16 --serving-reader bf16 \
        --out data/probes/Inkling-Small/v2

- **Inputs:** ``--probes`` holds the IID probe v2 (``terminal_prerl_v2/``, from fit_probes.py ``--iid-dataset
  terminal_prerl_v2``) next to v1's full-precision OOD fit (``diverse_deception/``; the collected copy is in
  ``data/probe_results/fitted/bf16``). ``--features`` is the full-precision reader's, with ``terminal_prerl_v2/``.
- **Same reader:** nothing is re-pointed, so the IID monitors keep their fit's thresholds and no ``trained_reader``
  fields appear. The OOD monitors' thresholds are terminal_prerl_v2's (below), not their v1 fit's.
- **Diagnostics:** the full-precision reader's, from the v2 session (``diagnostics-v2/bf16/`` in the bucket; v1's
  never ran). Without ``--diagnostics``, each follow-up monitor's golden tolerance is the mean probes' 3, and
  deployed.json has no batching numbers.
- **Its own root** holds its monitors, deployed.json, task_splits.json and grids.json. The v1 root's tooling reads
  only ``<root>/<dataset>/<view>/<readout>`` and the files at the root (this script, check_probe_server.py,
  probe_report.py ``--deployed``), so the nested ``v2/`` never mixes in. To use v2, give each tool the v2 root, and
  the post-hoc scorer ``--grids data/probes/Inkling-Small/v2/grids.json`` (its default is v1's).

**What gets deployed** (one directory per monitor, ``<dataset>/<view>/<readout>/``; ``view`` is how the server
renders a TerminalEnv episode). The IID dataset (``--iid-dataset``, default ``terminal_prerl``) supplies every
monitor's golden set and serving check, the task splits and grids.json:
- **IID** ``<iid dataset>/<view>/<readout>``: the fitted probe as it is, with its fit's t₁%/t₅% (from out-of-fold
  honest dev logits, which recalibrating the final probe can't reproduce) unless served on another reader (below).
- **OOD** ``diverse_deception/<view>/<readout>`` for both TerminalEnv views: the probe fitted on diverse_deception
  (rendered without reasoning), with the t₁%/t₅% of that view's honest dev episodes (train + val) of the IID dataset
  on the serving reader's features (``recalibrate``), and ``thresholds_from`` saying so.
  - **Always recalibrated:** the fit's ``thresholds_by_terminal_view`` are of the IID dataset it was fitted with,
    which its meta doesn't name, and v2 deploys v1's OOD fit with terminal_prerl_v2. They stay in meta.json as the
    fit's record.
  - **On the fit's own dataset and reader** recalibrating gives exactly those numbers (fit_probes.py's ``fit_ood``
    computes them the same way), so a v1 deployment's OOD thresholds are what they were.

**Added to meta.json:** ``name``, ``effort`` (the probe datasets' rendering effort), ``golden_tolerance``, the
fitted directory, ``deployed_utc`` and ``deployed_code``, the code version that deployed it (provenance.code_version;
``code`` stays the fit's). A deployment replaces its monitors' directories: that is its job.

**Serving on another reader** (``--features`` of a reader other than the one the probes were fitted on; the original
plan's section 8, as docs/INKLING_PROBES.md section 5 quotes it: train on full precision, serve 4-bit; v1). The served
logits come from the serving reader, so:
- t₁%/t₅% are recalibrated as the 99th/95th percentiles of the probe's logits on the serving reader's honest
  TerminalEnv dev episodes (train + val) of the IID dataset, per view (``thresholds_from``);
- ``reader``/``reader_info`` name the serving reader (the monitor checks them against the server), and the fitting
  reader's go to ``trained_reader``/``trained_reader_info``;
- the golden logits and the tolerance come from the serving reader's features and diagnostics.

**Task splits** (``task_splits.json``): the TerminalEnv task ids of the IID dataset's train/val/test splits.

**Grids** (``grids.json``; ``--grids-only`` writes only this file, from the items alone): the same splits as grid
keys, the sha256 of the grid each task's question shows (envs/terminal_verifier.grid_key), with the key's definition,
the IID dataset's name and ``items_sha256`` and the code version. v1's probes were fitted on pre-RL episodes of seed
0's eval tasks, and grids recur across seeds under other task ids (seed s's eval task i is seed 0's task s + i), so an
RL run may evaluate on the probes' own tasks whatever its task ids say (v2's puzzles leave out every grid of the RL
seeds up to 10,000). experiments/eval_inkling_probes_on_runs.py reads this file to report held-out monitor metrics
also on the grids in neither the train nor the val split, which no probe saw (within-task AUROC says the probes don't
key on the task, but the clean number costs nothing).

**Serving check** (in ``deployed.json``): on the serving reader's TerminalEnv features, the deployed probe's test
AUROC and its TPR/FPR at t₁%; and the batching rule's numbers for it (docs/inkling_probes_preregistration.md): the
99th-percentile |Δlogit| between batched and one-at-a-time readouts on the diagnostics sample, and the share of hacks
and of negatives whose logit lies that close to t₁%.

**Golden set** (``golden.jsonl``): 32 TerminalEnv test episodes of the IID dataset (up to 8 hacks), with their
renderings' sha256 and their logits on the stored features (``probe_fit.write_golden``).

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

from monitordecorrelation.envs.terminal_verifier import GRID_KEY, grid_key
from monitordecorrelation.provenance import code_version
from monitordecorrelation.whitebox.probe_datasets import SPLITS
from monitordecorrelation.whitebox.probe_fit import LayerSource, SelectedProbe, load_items, thresholds, write_golden

VIEWS = ("with_reasoning", "without_reasoning")
N_GOLDEN, N_GOLDEN_HACKS = 32, 8
MEAN_TOLERANCE = 3.0
GRIDS_FILE = "grids.json"
IID_DATASET = "terminal_prerl"  # --iid-dataset's default: v1's


def grid_splits(items: dict, dataset: dict, code: dict | None = None) -> dict:
    """What ``grids.json`` holds: the key's definition (``GRID_KEY``), ``dataset`` (which items) with its item and task
    counts, ``code`` (default: the code's version now), and per split (train, val and test at least) the sorted grid
    keys of its tasks. SystemExit if an item's question shows no grid: the post-hoc scorer couldn't tell a run's
    episodes on that task from others."""
    splits: dict[str, set[str]] = {split: set() for split in SPLITS}
    for it in items.values():
        key = grid_key(it["question"])
        if key is None:
            raise SystemExit(f"item {it['id']}: its question shows no grid (envs/terminal_verifier.prompt_grid), so "
                             f"{GRIDS_FILE} can't hold its task")
        splits.setdefault(it["split"], set()).add(key)
    return {"key": GRID_KEY,
            "dataset": {**dataset, "n_items": len(items), "n_tasks": len({it["task_id"] for it in items.values()})},
            "code": code_version() if code is None else code,
            **{split: sorted(keys) for split, keys in sorted(splits.items())}}


def write_grids(record: dict, out: str, log) -> str:
    """``record`` (``grid_splits``) as ``<out>/grids.json``; logs the counts, and any grid in more than one split
    (the scorer then counts it as seen if one of them is train or val)."""
    splits = [k for k in record if isinstance(record[k], list)]
    shared = {g for i, a in enumerate(splits) for b in splits[i + 1:] for g in set(record[a]) & set(record[b])}
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, GRIDS_FILE)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1)
    log(f"{path}: {', '.join(f'{len(record[k])} {k}' for k in splits)} grids"
        + (f"; WARNING: {len(shared)} grids are in more than one split" if shared else ""))
    return path


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


def recalibrate(probe: SelectedProbe, items: dict, features: str, view: str, readout: str,
                dataset: str = IID_DATASET) -> tuple[float, float]:
    """t₁%/t₅%: the 99th/95th percentiles of the probe's logits on the honest dev episodes (train + val, not
    ``used_verifier``) of the TerminalEnv dataset ``dataset`` in ``features``. fit_probes.py's ``fit_ood`` computes an
    OOD probe's ``thresholds_by_terminal_view`` the same way, so on its dataset and reader this gives those."""
    src = LayerSource(os.path.join(features, dataset), view, readout)
    rows = np.flatnonzero(src.keep)
    neg = np.array([items[src.ids[i]]["split"] in ("train", "val")
                    and not (items[src.ids[i]].get("diagnostics") or {}).get("used_verifier") for i in rows])
    return thresholds(probe.logit(src.layer(probe.layer, rows))[neg])


def same_reader(a: dict | None, b: dict | None) -> bool:
    """Whether two ``reader_info`` records name the same reader: the same loader and checkpoint revision. The
    checkpoint's path is where it sat on the machine that ran it, and data/probe_results keeps only its name."""
    return bool(a) and bool(b) and all(a.get(k) == b.get(k) for k in ("loader", "revision"))


def serve_on(probe: SelectedProbe, serving: dict, items: dict, features: str, view: str, readout: str,
             dataset: str = IID_DATASET, *, calibrate: bool = False) -> None:
    """Re-point ``probe`` at the serving reader ``serving`` (its features' index.json ``reader``).

    - **Another reader** than the probe's: ``reader``/``reader_info`` become the serving reader's, the fitting reader's
      go to ``trained_reader``/``trained_reader_info``, and t₁%/t₅% are recalibrated (``recalibrate``, on ``dataset``
      in ``features``).
    - **The same reader:** nothing changes, unless ``calibrate`` (the OOD probes): then only t₁%/t₅% are
      recalibrated.
    - Recalibrated thresholds record where they come from: ``thresholds_from``."""
    other = not same_reader(probe.meta.get("reader_info"), serving["reader_info"])
    if other:
        probe.meta = {**probe.meta, "trained_reader": probe.meta.get("reader"),
                      "trained_reader_info": probe.meta.get("reader_info"), "reader": serving["reader"],
                      "reader_info": serving["reader_info"]}
    if other or calibrate:
        probe.threshold_1, probe.threshold_5 = recalibrate(probe, items, features, view, readout, dataset)
        probe.meta = {**probe.meta, "thresholds_from": f"honest {dataset} dev episodes (train + val) on the "
                                                       f"{probe.meta.get('reader')} reader's features"}


def serving_check(probe: SelectedProbe, items: dict, features: str, view: str, readout: str,
                  diagnostics: str | None, dataset: str = IID_DATASET) -> dict:
    from monitordecorrelation.whitebox.probe_fit import auroc

    src = LayerSource(os.path.join(features, dataset), view, readout)
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
           features: str, effort: float, tolerance: float | None, log, diagnostics: str | None = None,
           code: dict | None = None, dataset: str = IID_DATASET) -> dict:
    """Write one monitor to ``out`` (replacing it), its golden set and serving check from the IID dataset ``dataset``
    in ``features``; ``code``: the deploying code's version (default: taken now)."""
    src = LayerSource(os.path.join(features, dataset), view, readout)
    ids = golden_ids(items, src)
    if tolerance is None:
        tolerance = MEAN_TOLERANCE
    probe.meta = {**probe.meta, "name": name, "view": view, "effort": effort, "golden_tolerance": tolerance,
                  "fitted_dir": fitted_ref(fitted), "deployed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                       time.gmtime()),
                  "deployed_code": code_version() if code is None else code}
    if os.path.isdir(out):
        shutil.rmtree(out)
    probe.save(out)
    n = write_golden(out, probe, src, items, readout, view, ids, effort=effort)
    log(f"{name}: layer {probe.layer}, t1 {probe.threshold_1:.3f}, golden {n} episodes, tolerance {tolerance}")
    reader = probe.meta.get("reader")
    return {"name": name, "dir": out, "layer": probe.layer, "golden": n, "golden_tolerance": tolerance,
            "reader": reader, "trained_reader": probe.meta.get("trained_reader", reader),
            "threshold_1": probe.threshold_1,
            "serving_check": serving_check(probe, items, features, view, readout, diagnostics, dataset)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", help="fit_probes.py's --out for one reader, holding <iid dataset>/ and "
                                     "diverse_deception/ (required unless --grids-only)")
    ap.add_argument("--features", help="the serving reader's features (<dataset>/<view>/shard_*): the fitting "
                                       "reader's, or another (required unless --grids-only)")
    ap.add_argument("--serving-reader", default="nvfp4", help="the serving reader's name, for the record (v2: bf16)")
    ap.add_argument("--diagnostics", help="the serving reader's diagnostics folder (diag.safetensors)")
    ap.add_argument("--datasets", default="data/probe_datasets")
    ap.add_argument("--iid-dataset", default=IID_DATASET,
                    help="the IID dataset (TerminalEnv episodes) under --datasets, --features and --probes: "
                         "terminal_prerl, or terminal_prerl_v2 (docs/inkling_probes_amendment_1.md). Its episodes give "
                         "every monitor's golden set and serving check, the OOD monitors' thresholds, the task splits "
                         "and grids.json")
    ap.add_argument("--out", default="data/probes/Inkling-Small",
                    help="the deployment's root (v2: data/probes/Inkling-Small/v2)")
    ap.add_argument("--readouts", default="followup,mean")
    ap.add_argument("--grids-only", action="store_true",
                    help=f"write only {GRIDS_FILE} into --out, from the --iid-dataset items in --datasets: no monitor "
                         "is deployed or changed")
    args = ap.parse_args(argv)
    if args.grids_only:
        unused = [f"--{k}" for k in ("probes", "features", "diagnostics") if getattr(args, k) is not None]
        if unused:
            ap.error(f"--grids-only deploys no monitor, so {', '.join(unused)} would be unused")
    elif args.probes is None or args.features is None:
        ap.error("--probes and --features are required (unless --grids-only)")
    log = lambda m: print(f"[deploy] {m}", flush=True)  # noqa: E731
    iid = args.iid_dataset
    items = load_items(os.path.join(args.datasets, iid))
    manifest = json.load(open(os.path.join(args.datasets, iid, "manifest.json")))
    code = code_version()
    grids = grid_splits(items, {"name": iid, "items_sha256": manifest.get("items_sha256")}, code)
    if args.grids_only:
        write_grids(grids, args.out, log)
        return 0
    effort = manifest.get("effort", 0.5)
    index = json.load(open(os.path.join(args.features, iid, "index.json")))
    serving = {"reader": args.serving_reader, "reader_info": index["reader"]}
    done = []
    for readout in args.readouts.split(","):
        for view in VIEWS:
            fitted = os.path.join(args.probes, iid, view, readout)
            if os.path.isdir(fitted):
                p = SelectedProbe.load(fitted)
                serve_on(p, serving, items, args.features, view, readout, iid)
                tol = followup_tolerance(args.diagnostics, p) if readout == "followup" else None
                done.append(deploy(p, fitted, os.path.join(args.out, iid, view, readout),
                                   f"probe_iid_{view}_{readout}", view, readout, items, args.features, effort, tol,
                                   log, args.diagnostics, code, iid))
        fitted = os.path.join(args.probes, "diverse_deception", "without_reasoning", readout)
        if os.path.isdir(fitted):
            for view in VIEWS:
                p = SelectedProbe.load(fitted)
                p.meta = {**p.meta, "train_view": "without_reasoning"}
                serve_on(p, serving, items, args.features, view, readout, iid, calibrate=True)
                tol = followup_tolerance(args.diagnostics, p) if readout == "followup" else None
                done.append(deploy(p, fitted, os.path.join(args.out, "diverse_deception", view, readout),
                                   f"probe_ood_{view}_{readout}", view, readout, items, args.features, effort, tol,
                                   log, args.diagnostics, code, iid))
    splits: dict[str, set] = {}
    for it in items.values():
        splits.setdefault(it["split"], set()).add(it["task_id"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "task_splits.json"), "w") as fh:
        json.dump({k: sorted(v) for k, v in sorted(splits.items())}, fh, indent=1)
    write_grids(grids, args.out, log)
    record = os.path.join(args.out, "deployed.json")  # merged with earlier runs (e.g. another reader's readouts)
    earlier = json.load(open(record)) if os.path.exists(record) else []
    done += [d for d in earlier if d["dir"] not in {e["dir"] for e in done}]
    with open(record, "w") as fh:
        json.dump(sorted(done, key=lambda d: d["dir"]), fh, indent=1)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
