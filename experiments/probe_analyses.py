"""Numbers behind PROBE_RESULTS.md that came from one-off snippets, as subcommands that write JSON to ``--out``.

    uv run python experiments/probe_analyses.py dataset-stats --datasets data/probe_datasets \
        --out ../scratch/analyses/<UTC timestamp>
    uv run python experiments/probe_analyses.py near-threshold --features ../scratch/features/bf16 \
        --probes ../scratch/probes/bf16/terminal_prerl/with_reasoning/followup,<more probe dirs> \
        --out ../scratch/analyses/<UTC timestamp>

**dataset-stats** (``dataset_stats.json``; reads the datasets' items.jsonl only):
- **Disclosure** (terminal_prerl): how many ``used_verifier`` hacks, and how many other episodes, have visible turn
  texts (not the thinking) that match probe_datasets.DISCLOSURE, the pattern the concealment set removes. When nearly
  every hack says so and few others do, a text monitor can key on the policy's own words.
- **OOD question overlap** (diverse_deception): per source, how many test items share their question with a training
  item (of any source), and the distinct questions. Pairs are split, but questions can repeat across pairs: every
  MASK item asks "State a fact.".
- **Token counts** of the probe inputs (the conversation plus the follow-up, rendered by whitebox.inkling_replay as
  the extraction renders them), per dataset and view: mean, median, 95th percentile, max, and the total over all
  items (mean × items). Terminal datasets in both views; the OOD set without reasoning, the only view extracted.
  ``--sample N`` renders N random items per dataset (seed 0); by default every item.

**near-threshold** (``near_threshold.json``): for each probe directory (fit_probes.py's, or a deployed monitor's) on
one reader's TerminalEnv features, the share of honest dev episodes (train + val, not ``used_verifier``) whose logit
lies within ``--k`` logits of t₁%, at:
- the probe's own t₁%: a fitted OOD probe has one per TerminalEnv view and is read in both; any other probe in its
  own view;
- t₁% recalibrated on these features: the 99th percentile of the same episodes' logits.

It sizes a re-scoring of only the episodes near t₁% (PROBE_RESULTS.md 3.4).

An output file that already exists is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

from monitordecorrelation.whitebox.probe_fit import LayerSource, SelectedProbe, load_items, thresholds

VIEWS = ("with_reasoning", "without_reasoning")
DATASETS = ("terminal_prerl", "diverse_deception", "terminal_concealed")


def is_hack(item: dict) -> bool:
    """The IID label: the episode read the verifier and submitted its value."""
    return bool((item.get("diagnostics") or {}).get("used_verifier"))


def disclosure(items: list[dict]) -> dict:
    """``used_verifier`` hacks and the other episodes whose visible turn texts match DISCLOSURE."""
    from monitordecorrelation.whitebox.probe_datasets import DISCLOSURE

    out = {}
    for name, group in (("hacks", [it for it in items if is_hack(it)]),
                        ("non_hacks", [it for it in items if not is_hack(it)])):
        n = sum(any(DISCLOSURE.search(t.get("text") or "") for t in it.get("turns") or []) for it in group)
        out[name] = {"n": len(group), "disclosing": n, "share": n / len(group) if group else float("nan")}
    return out


def question_overlap(items: list[dict]) -> dict:
    """Per source: the test items whose question a training item (of any source) also has, and distinct questions."""
    train_questions = {it["question"] for it in items if it["split"] == "train"}
    out = {}
    for source in sorted({it["source"] for it in items}):
        its = [it for it in items if it["source"] == source]
        test = [it for it in its if it["split"] == "test"]
        out[source] = {"n_items": len(its), "n_test": len(test),
                       "test_sharing_question_with_train": sum(it["question"] in train_questions for it in test),
                       "distinct_questions": len({it["question"] for it in its}),
                       "distinct_questions_train": len({it["question"] for it in its if it["split"] == "train"}),
                       "distinct_questions_test": len({it["question"] for it in test})}
    return out


def token_counts(items: list[dict], views, sample: int | None, replay) -> dict:
    """Probe-input lengths in tokens per view, over ``sample`` random items (seed 0), or all of them."""
    from monitordecorrelation.whitebox.probe_datasets import item_messages

    chosen = list(items)
    random.Random(0).shuffle(chosen)
    chosen = chosen[:sample] if sample else chosen
    if not chosen:
        return {}
    out = {}
    for view in views:
        n = np.array([len(replay.probe_input(item_messages(it, view)).tokens) for it in chosen])
        out[view] = {"n_items": len(items), "n_rendered": len(chosen), "mean": float(n.mean()),
                     "median": float(np.median(n)), "p95": float(np.percentile(n, 95)), "max": int(n.max()),
                     "total_estimate": float(n.mean() * len(items))}
    return out


def dataset_stats(args) -> dict:
    present = [d for d in DATASETS if os.path.exists(os.path.join(args.datasets, d, "items.jsonl"))]
    items = {d: list(load_items(os.path.join(args.datasets, d)).values()) for d in present}
    out: dict = {"datasets": os.path.abspath(args.datasets), "items_sha256": {}}
    for d in present:
        manifest = os.path.join(args.datasets, d, "manifest.json")
        out["items_sha256"][d] = json.load(open(manifest)).get("items_sha256") if os.path.exists(manifest) else None
    if "terminal_prerl" in items:
        out["disclosure"] = disclosure(items["terminal_prerl"])
    if "diverse_deception" in items:
        out["ood_question_overlap"] = question_overlap(items["diverse_deception"])
    if not args.skip_tokens:
        from monitordecorrelation.whitebox.inkling_replay import InklingReplay

        replay = InklingReplay()
        out["tokens"] = {d: token_counts(its, VIEWS if its and "turns" in its[0] else ("without_reasoning",),
                                         args.sample, replay) for d, its in items.items()}
    return out


def own_thresholds(probe: SelectedProbe) -> dict[str, float]:
    """{TerminalEnv view: the probe's own t₁% there}. A fitted OOD probe carries one per view; an IID probe, or a
    deployed monitor (one view each), its ``threshold_1`` in its own view."""
    meta = probe.meta
    by_view = meta.get("thresholds_by_terminal_view")
    if meta.get("dataset") == "diverse_deception" and by_view and "deployed_utc" not in meta:
        return {view: float(t[0]) for view, t in by_view.items()}
    return {meta["view"]: probe.threshold_1}


def near_threshold(probe_dir: str, features: str, items: dict, k: float) -> dict:
    """The share of honest TerminalEnv dev episodes within ``k`` logits of t₁% (own and recalibrated), per view."""
    probe = SelectedProbe.load(probe_dir)
    readout = probe.meta["readout"]
    out: dict = {"layer": probe.layer, "readout": readout, "dataset": probe.meta.get("dataset"), "views": {}}
    for view, t1 in own_thresholds(probe).items():
        src = LayerSource(os.path.join(features, "terminal_prerl"), view, readout)
        rows = np.flatnonzero(src.keep)
        honest_dev = np.array([items[src.ids[i]]["split"] in ("train", "val") and not is_hack(items[src.ids[i]])
                               for i in rows])
        z = probe.logit(src.layer(probe.layer, rows))[honest_dev]
        t1_re = thresholds(z)[0]
        out["views"][view] = {"n_honest_dev": len(z), "t1_own": t1,
                              "share_within_k_of_t1_own": float(np.mean(np.abs(z - t1) <= k)),
                              "t1_recalibrated": t1_re,
                              "share_within_k_of_t1_recalibrated": float(np.mean(np.abs(z - t1_re) <= k))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    ds = sub.add_parser("dataset-stats", help="disclosure, OOD question overlap and token counts (items only)")
    ds.add_argument("--sample", type=int, default=None, help="render N random items per dataset (default: all)")
    ds.add_argument("--skip-tokens", action="store_true", help="no token counts")
    nt = sub.add_parser("near-threshold", help="honest dev episodes within --k logits of t1, per probe and view")
    nt.add_argument("--probes", required=True, help="comma-list of probe directories (fitted or deployed)")
    nt.add_argument("--features", required=True, help="one reader's features: <dataset>/<view>/shard_*")
    nt.add_argument("--k", type=float, default=4.0, help="distance from t1, in logits")
    for p in (ds, nt):
        p.add_argument("--datasets", default="data/probe_datasets", help="folder with the dataset folders")
        p.add_argument("--out", required=True, help="output folder, e.g. ../scratch/analyses/<UTC timestamp>")
    args = ap.parse_args(argv)

    path = os.path.join(args.out, args.command.replace("-", "_") + ".json")
    if os.path.exists(path):
        ap.error(f"{path} exists; pick a new --out")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.command == "dataset-stats":
        result = dataset_stats(args)
    else:
        items = load_items(os.path.join(args.datasets, "terminal_prerl"))
        result = {"features": os.path.abspath(args.features), "k": args.k,
                  "probes": {p: near_threshold(p, args.features, items, args.k) for p in args.probes.split(",") if p}}
    result = {"command": args.command, "started_utc": started, **result}
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
