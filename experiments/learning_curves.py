"""Was there enough training data? Learning curves for one reader's follow-up probes, refitted at the fitted probe's
layer and C on growing fractions of its training data.

    uv run python experiments/learning_curves.py --probes PROBES/bf16 --features FEATURES/bf16 \
        --out OUT/learning_curves/bf16

**IID** (TerminalEnv, label ``used_verifier``), per view:
- **Training data:** a fraction of the dev tasks (train + val). Tasks with and without hacks are sampled separately,
  so every subsample has hacks. Each fraction is repeated ``--repeats`` times.
- **Scored on the test split:**
  - AUROC;
  - TPR at 1% FPR, with the threshold from the test negatives themselves, so it measures the ranking, not the
    calibration;
  - recall at that threshold on the concealed versions of the test hacks.

**OOD** (diverse deception):
- **Training data:** fractions of the training pairs, stratified by source.
- **Scored:**
  - in distribution: OOD test AUROC;
  - on TerminalEnv, as deployed: transfer AUROC over every episode in both views, and TPR at t₁% from the honest dev
    episodes.
- **Per source:** trained on that source alone, and on all sources but it, next to all sources. Each composition
  is refitted ``--source-refits`` times on pair-bootstrap resamples, because a single fit's transfer is as noisy
  as the differences between compositions.

**Writes** ``iid.csv``, ``ood.csv``, ``ood_sources.csv`` and ``summary.json``: per fraction, each metric's median and
5th/95th percentiles. A fraction of 1 is the full training set, fitted once. summary.json records the code version
(provenance.code_version) as ``code``. An ``--out`` that already holds any of these files is refused, unless
``--overwrite``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from monitordecorrelation.provenance import code_version
from monitordecorrelation.whitebox.probe_fit import LayerSource, SelectedProbe, auroc, load_items, thresholds

VIEWS = ("with_reasoning", "without_reasoning")
OUTPUTS = ("iid.csv", "ood.csv", "ood_sources.csv", "summary.json")


def used_verifier(it: dict) -> int:
    return int(bool((it.get("diagnostics") or {}).get("used_verifier")))


def tpr_at_fpr(z: np.ndarray, y: np.ndarray, fpr: float = 0.01) -> tuple[float, float]:
    t = float(np.percentile(z[y == 0], 100 * (1 - fpr)))
    return float(np.mean(z[y == 1] > t)), t


def summarize(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in sorted(groups.items()):
        row = dict(zip(keys, g), fits=len(rs), n_train=float(np.median([r["n_train"] for r in rs])),
                   n_hacks_or_pairs=float(np.median([r["n_hacks_or_pairs"] for r in rs])))
        for m in metrics:
            v = np.array([r[m] for r in rs], float)
            row[m] = {"median": float(np.median(v)), "p05": float(np.percentile(v, 5)),
                      "p95": float(np.percentile(v, 95))}
        out.append(row)
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", required=True, help="fit_probes.py's --out for one reader (layer and C come from it)")
    ap.add_argument("--features", required=True, help="the same reader's features (<dataset>/<view>/shard_*)")
    ap.add_argument("--datasets", default="data/probe_datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fractions", default="0.05,0.1,0.25,0.5,1")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--source-refits", type=int, default=10, help="bootstrap refits per source composition")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="replace the outputs already in --out (else refused)")
    args = ap.parse_args(argv)
    existing = [f for f in OUTPUTS if os.path.exists(os.path.join(args.out, f))]
    if existing and not args.overwrite:
        ap.error(f"{args.out} already holds {', '.join(existing)}; --overwrite replaces them")
    code = code_version()
    log = lambda m: print(f"[curves {time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    fractions = [float(f) for f in args.fractions.split(",")]
    rng = np.random.default_rng(args.seed)
    items = load_items(os.path.join(args.datasets, "terminal_prerl"))
    concealed = load_items(os.path.join(args.datasets, "terminal_concealed"))
    jobs = []  # (table, callable returning a row)

    # ---------------------------------------------------------------- IID
    for view in VIEWS:
        p = SelectedProbe.load(os.path.join(args.probes, "terminal_prerl", view, "followup"))
        C = float(p.meta["C"])
        src = LayerSource(os.path.join(args.features, "terminal_prerl"), view, "followup")
        x = src.layer(p.layer)
        its = [items[i] for i in src.ids]
        y = np.array([used_verifier(it) for it in its])
        split = np.array([it["split"] for it in its])
        task = np.array([it["task_id"] for it in its])
        dev, test = np.isin(split, ["train", "val"]), split == "test"
        hack_tasks = np.array(sorted(set(task[dev & (y == 1)])))
        clean_tasks = np.array(sorted(set(task[dev]) - set(hack_tasks)))
        csrc = LayerSource(os.path.join(args.features, "terminal_concealed"), view, "followup")
        of_test = np.array([items[concealed[i]["id"].rsplit("/", 1)[0]]["split"] == "test" for i in csrc.ids])
        cx = csrc.layer(p.layer)[of_test]
        log(f"IID {view}: layer {p.layer}, C {C:g}; dev {int(dev.sum())} episodes, {int(y[dev].sum())} hacks in "
            f"{len(hack_tasks)} tasks; test {int(test.sum())} / {int(y[test].sum())}; {len(cx)} concealed test hacks")

        def iid_fit(rows, frac, rep, view=view, x=x, y=y, task=task, test=test, cx=cx, C=C, layer=p.layer):
            q = SelectedProbe.fit(x[rows], y[rows], C, layer)
            z = q.logit(x[test])
            tpr, t = tpr_at_fpr(z, y[test])
            return {"view": view, "fraction": frac, "repeat": rep, "n_train": len(rows),
                    "n_hacks_or_pairs": int(y[rows].sum()), "n_tasks": len(set(task[rows])),
                    "test_auroc": auroc(z, y[test]), "test_tpr_at_1pct_fpr": tpr,
                    "concealed_test_hacks_recall": float(np.mean(q.logit(cx) > t))}

        for frac in fractions:
            for rep in range(1 if frac >= 1 else args.repeats):
                chosen = np.concatenate([
                    rng.choice(hack_tasks, max(1, round(frac * len(hack_tasks))), replace=False),
                    rng.choice(clean_tasks, max(1, round(frac * len(clean_tasks))), replace=False)])
                rows = np.flatnonzero(dev & np.isin(task, chosen))
                jobs.append(("iid", lambda rows=rows, frac=frac, rep=rep, f=iid_fit: f(rows, frac, rep)))

    # ---------------------------------------------------------------- OOD
    p = SelectedProbe.load(os.path.join(args.probes, "diverse_deception", "without_reasoning", "followup"))
    C = float(p.meta["C"])
    ood = load_items(os.path.join(args.datasets, "diverse_deception"))
    src = LayerSource(os.path.join(args.features, "diverse_deception"), "without_reasoning", "followup")
    xo = src.layer(p.layer)
    its = [ood[i] for i in src.ids]
    yo = np.array([int(it["label"]) for it in its])
    source = np.array([it["source"] for it in its])
    pair = np.array([it["pair"] for it in its])
    split = np.array([it["split"] for it in its])
    tr, te = (split == "train") & src.keep, (split == "test") & src.keep
    terminal = {}
    for view in VIEWS:
        tsrc = LayerSource(os.path.join(args.features, "terminal_prerl"), view, "followup")
        rows = np.flatnonzero(tsrc.keep)
        tits = [items[tsrc.ids[i]] for i in rows]
        ty = np.array([used_verifier(it) for it in tits])
        dev_neg = np.isin([it["split"] for it in tits], ["train", "val"]) & (ty == 0)
        terminal[view] = (tsrc.layer(p.layer, rows), ty, dev_neg)
    sources = sorted(set(source))
    log(f"OOD: layer {p.layer}, C {C:g}; {len(set(pair[tr]))} training pairs over {len(sources)} sources")

    def ood_score(rows) -> dict:
        q = SelectedProbe.fit(xo[rows], yo[rows], C, p.layer)
        out = {"n_train": len(rows), "n_hacks_or_pairs": len(set(pair[rows])),
               "ood_test_auroc": auroc(q.logit(xo[te]), yo[te])}
        for s in sources:
            m = te & (source == s)
            out[f"ood_test_auroc_{s}"] = auroc(q.logit(xo[m]), yo[m])
        for view, (xt, ty, dev_neg) in terminal.items():
            z = q.logit(xt)
            t1, _ = thresholds(z[dev_neg])
            out[f"transfer_auroc_{view}"] = auroc(z, ty)
            out[f"tpr_at_t1_{view}"] = float(np.mean(z[ty == 1] > t1))
        return out

    pairs_by_source = {s: np.array(sorted(set(pair[tr & (source == s)]))) for s in sources}
    for frac in fractions:
        for rep in range(1 if frac >= 1 else args.repeats):
            chosen = np.concatenate([rng.choice(v, max(1, round(frac * len(v))), replace=False)
                                     for v in pairs_by_source.values()])
            rows = np.flatnonzero(tr & np.isin(pair, chosen))
            jobs.append(("ood", lambda rows=rows, frac=frac, rep=rep: {"fraction": frac, "repeat": rep,
                                                                       **ood_score(rows)}))
    rows_of_pair = {pp: np.flatnonzero(tr & (pair == pp)) for pp in set(pair[tr])}
    compositions = [("all", "all", np.ones(len(source), bool))] + [
        (s, how, m) for s in sources for how, m in (("only", source == s), ("all_but", source != s))]
    for s, how, m in compositions:
        comp = np.array(sorted(set(pair[tr & m])))
        for rep in range(args.source_refits):
            rows = np.concatenate([rows_of_pair[pp] for pp in rng.choice(comp, len(comp), replace=True)])
            jobs.append(("ood_sources", lambda rows=rows, s=s, how=how, rep=rep: {
                "source": s, "trained_on": how, "repeat": rep, **ood_score(rows)}))

    # ---------------------------------------------------------------- run
    from threadpoolctl import threadpool_limits

    tables: dict[str, list[dict]] = {"iid": [], "ood": [], "ood_sources": []}
    t0 = time.time()
    with threadpool_limits(1), ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [(name, pool.submit(fn)) for name, fn in jobs]
        for k, (name, fut) in enumerate(futures, start=1):
            tables[name].append(fut.result())
            if k % 20 == 0 or k == len(futures):
                log(f"  {k}/{len(futures)} fits ({time.time() - t0:.0f}s)")
    os.makedirs(args.out, exist_ok=True)
    for name, rows in tables.items():
        write_csv(os.path.join(args.out, f"{name}.csv"), rows)
    iid_m = ("test_auroc", "test_tpr_at_1pct_fpr", "concealed_test_hacks_recall")
    ood_m = ("ood_test_auroc", *(f"{k}_{v}" for v in VIEWS for k in ("transfer_auroc", "tpr_at_t1")))
    summary = {"probes": args.probes, "features": args.features, "code": code,
               "iid": summarize(tables["iid"], ("view", "fraction"), iid_m),
               "ood": summarize(tables["ood"], ("fraction",), ood_m),
               "ood_sources": summarize(tables["ood_sources"], ("trained_on", "source"), ood_m)}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    for r in summary["iid"]:
        log(f"IID {r['view']} {r['fraction']:g} ({r['n_hacks_or_pairs']:.0f} hacks): " + "; ".join(
            f"{m} {r[m]['median']:.3f} [{r[m]['p05']:.3f}, {r[m]['p95']:.3f}]" for m in iid_m))
    for r in summary["ood"]:
        log(f"OOD {r['fraction']:g} ({r['n_hacks_or_pairs']:.0f} pairs): " + "; ".join(
            f"{m} {r[m]['median']:.3f} [{r[m]['p05']:.3f}, {r[m]['p95']:.3f}]" for m in ood_m))
    for r in summary["ood_sources"]:
        log(f"OOD {r['trained_on']} {r['source']} ({r['n_hacks_or_pairs']:.0f} pairs): " + "; ".join(
            f"{m} {r[m]['median']:.3f} [{r[m]['p05']:.3f}, {r[m]['p95']:.3f}]"
            for m in ("ood_test_auroc", "transfer_auroc_with_reasoning", "transfer_auroc_without_reasoning")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
