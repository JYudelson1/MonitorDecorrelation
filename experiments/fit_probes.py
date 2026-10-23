"""Fit and evaluate one reader's Inkling probes on CPUs, as pre-registered in docs/inkling_probes_preregistration.md.

    uv run python experiments/fit_probes.py --reader nvfp4 --features FEATURES/nvfp4 \
        --datasets data/probe_datasets --out PROBES/nvfp4

``--features`` holds what the GPU session saved for that reader: ``<dataset>/<view>/shard_*``. Download it first, e.g.
``hf buckets sync hf://buckets/probemaxxing/inkling-probes/features/nvfp4 FEATURES/nvfp4 --exclude "*/full/*"``.

**Probes:**
- **IID** (``--iid-dataset``, default ``terminal_prerl``): {with, without reasoning} × ``--readouts``. Label
  ``used_verifier``. Dev is train ∪ val, grouped by task.
  - ``terminal_prerl_v2``, or any other IID dataset, is fitted under docs/inkling_probes_amendment_1.md and
    docs/inkling_probes_amendment_2.md, whose sha256s its probes record.
  - Amendment 2 fixes its features: the full-precision reader (``V2_READER``), one episode per forward. Features
    whose index.json says otherwise are refused before any fitting.
- **OOD** (``diverse_deception``): ``--readouts``. Label: deceptive. Dev is train, grouped by pair and stratified by
  source.

**Per probe, written to ``<out>/<dataset>/<view>/<readout>/``:**
- ``probe.npz`` and ``meta.json``: the selected layer and C, raw-space weights, t₁%/t₅%, provenance and the
  pre-registration's hash;
- ``cv_grid.csv``: out-of-fold AUROC for every (layer, C);
- ``scores.jsonl``: per item, its labels, its test, cross-fitted or transfer scores, and the baselines;
- ``metrics.json``: test AUROC with task-cluster bootstrap CIs, TPR/FPR at the thresholds, the hard subsets
  (cross-fitted), concealment recall, OOD per-source and transfer results, baselines, and sensitivity rows.

``<out>/summary.md`` collects the headline numbers.

**One --out, several runs:** a later run with other ``--readouts`` adds its probes to ``--out``, and
``<out>/summary.json`` accumulates. A run refuses to start if a probe folder it would write already exists, unless
``--overwrite`` (then that folder's files are rewritten).

**Code version** (provenance.code_version): each probe's meta.json and summary.json entry record the code that fitted
it as ``code``; meta.json also records the features' (their index.json's ``code``) as ``features_code``.

**Same items:** each dataset's ``items.jsonl`` must match the ``items_sha256`` of its manifest and of the features'
index.json (what they were extracted from); otherwise the run refuses to start, before any fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

from monitordecorrelation.provenance import code_version
from monitordecorrelation.whitebox.probe_fit import (
    C_GRID, MOE_FIRST_LAYER, LayerSource, SelectedProbe, auroc, ci, cluster_bootstrap, cv_grid_source, load_items,
    make_folds, select, thresholds, within_group_auroc)

PLAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                    "inkling_probes_preregistration.md")


def amendment_sha(name: str = "inkling_probes_amendment_1.md") -> str | None:
    """sha256 of docs/``name``, an amendment the IID probe v2 is fitted under, recorded next to ``prereg_sha256`` by
    every probe not fitted on terminal_prerl. Amendment 1 (the default) changes where the IID episodes come from;
    amendment 2 the reader and batching of their features."""
    path = os.path.join(os.path.dirname(PLAN), name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


V2_READER = {"loader": "transformers", "revision": "8cc5877b44d343f88b92086aa1fb72897950f06a"}
"""The reader of the v2 features (docs/inkling_probes_amendment_2.md): full-precision Inkling-Small, the reader every
v1 probe was fitted on."""


def amendment_2_problems(index: dict) -> list[str]:
    """Why features with this index.json can't be fitted under amendment 2 (empty if they can). The amendment fixes
    the reader (``V2_READER``) and one episode per forward pass (``max_batch_tokens`` 1)."""
    reader = index.get("reader") or {}
    problems = [f"reader {k} {reader.get(k)!r}, not {v!r}" for k, v in V2_READER.items() if reader.get(k) != v]
    if index.get("max_batch_tokens") != 1:
        problems.append(f"max_batch_tokens {index.get('max_batch_tokens')!r}, not 1 (one episode per forward)")
    return problems


def prereg_sha() -> str | None:
    """sha256 of the pre-registration's whole file, the rules this script follows. The file is section 5b of the
    original plan, byte for byte, so this is the hash every fitted probe's meta.json records."""
    if not os.path.exists(PLAN):
        return None
    with open(PLAN, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def log(msg: str) -> None:
    print(f"[fit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def features_index(args, dataset: str) -> dict:
    """The features' index.json for ``dataset`` ({} if there is none, e.g. synthetic features)."""
    path = os.path.join(args.features, dataset, "index.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def dataset_items(args, dataset: str) -> dict[str, dict]:
    """The dataset's items by id, refused unless items.jsonl is the file its manifest describes and the one the
    features were extracted from (the ``items_sha256`` their index.json recorded)."""
    recorded = (features_index(args, dataset).get("dataset") or {}).get("items_sha256")
    return load_items(os.path.join(args.datasets, dataset),
                      {os.path.join(args.features, dataset, "index.json"): recorded})


def policy_text(item: dict, view: str) -> str:
    """The policy's own words in a view: thinking (with reasoning only) and visible text, every turn."""
    parts = []
    for t in item.get("turns") or []:
        if view == "with_reasoning" and t.get("cot"):
            parts.append(t["cot"])
        parts.append(t.get("text") or "")
    if "answer" in item:
        parts.append(item["answer"])
    return "\n".join(parts)


def rate(scores, mask, threshold) -> float:
    s = np.asarray(scores)[np.asarray(mask, bool)]
    return float(np.mean(s > threshold)) if len(s) else float("nan")


def crossfit(x_layer, rows, y, groups, layer, C, n_folds: int = 5, seed: int = 0):
    """Logits for every row from a model that never saw its group: 5 folds by group (task). ``x_layer`` is the
    selected layer for all rows of the view."""
    from sklearn.model_selection import GroupKFold

    out = np.empty(len(rows))
    fold_of = {}
    for k, (tr, te) in enumerate(GroupKFold(n_folds, shuffle=True, random_state=seed).split(rows, y, groups)):
        p = SelectedProbe.fit(x_layer[rows[tr]], y[tr], C, layer)
        out[te] = p.logit(x_layer[rows[te]])
        for g in set(np.asarray(groups)[te]):
            fold_of[g] = p
    return out, fold_of


def text_baselines(items_dev, items_test, y_dev, y_test, groups_test, view, counts_dev, counts_test) -> dict:
    """Text-only baselines on the same items: length, turns, a regex and TF-IDF logistic regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def turns(items):
        return np.array([len(it.get("turns") or []) for it in items], float)

    out = {}
    feats = lambda c, items: np.column_stack([np.log1p(c[:, 0]), np.log1p(c[:, 1]), turns(items)])  # noqa: E731
    lr = LogisticRegression(class_weight="balanced", max_iter=5000).fit(feats(counts_dev, items_dev), y_dev)
    s = lr.decision_function(feats(counts_test, items_test))
    out["length"] = auroc(s, y_test)
    out["n_turns"] = auroc(turns(items_test), y_test)
    out["regex_verifier"] = auroc([bool(re.search(r"verifier", policy_text(it, view), re.I)) for it in items_test],
                                  y_test)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    a = vec.fit_transform([policy_text(it, view) for it in items_dev])
    tf = LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0).fit(a, y_dev)
    s = tf.decision_function(vec.transform([policy_text(it, view) for it in items_test]))
    out["tfidf"] = auroc(s, y_test)
    out["tfidf_ci"] = ci(cluster_bootstrap(auroc, groups_test, s, y_test, n=1000))
    return out


def activation_baselines(x_layer, dev, test, y_dev, y_test, layer, C, x0=None, answer_logp=None, groups_dev=None,
                         seed: int = 0) -> dict:
    """Baselines on activations: random directions, task-shuffled labels, class-mean difference, layer-0 mean
    pooling (``x0``: that layer for every row), and the reader's own answer (log P(No) − log P(Yes))."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    xd, xt = x_layer[dev], x_layer[test]
    sc = StandardScaler().fit(xd)
    zd, zt = sc.transform(xd), sc.transform(xt)
    out = {}
    dirs = rng.standard_normal((1000, zd.shape[1]))
    rand = []
    for v in dirs:
        sign = 1.0 if auroc(zd @ v, y_dev) >= 0.5 else -1.0
        rand.append(auroc(sign * (zt @ v), y_test))
    out["random_direction_p95"] = float(np.nanpercentile(rand, 95))
    diff = zd[y_dev == 1].mean(0) - zd[y_dev == 0].mean(0)
    out["class_mean_difference"] = auroc(zt @ diff, y_test)
    if groups_dev is not None:
        groups_dev = np.asarray(groups_dev)
        tasks = np.unique(groups_dev)
        task_label = {g: int(y_dev[groups_dev == g].max()) for g in tasks}
        shuffled = []
        for _ in range(100):
            perm = dict(zip(tasks, rng.permutation([task_label[g] for g in tasks])))
            y_shuf = np.array([perm[g] if y_dev[i] == task_label[g] else 0 for i, g in enumerate(groups_dev)])
            if 0 < y_shuf.sum() < len(y_shuf):
                p = SelectedProbe.fit(xd, y_shuf, C, layer)
                shuffled.append(auroc(p.logit(xt), y_test))
        out["shuffled_labels_p95"] = float(np.nanpercentile(shuffled, 95)) if shuffled else float("nan")
    if x0 is not None:
        p0 = SelectedProbe.fit(x0[dev], y_dev, C, 0)
        out["layer0_mean_pool"] = auroc(p0.logit(x0[test]), y_test)
    if answer_logp is not None:
        a = answer_logp[test]
        out["self_report"] = auroc(a[:, 1] - a[:, 0], y_test)
    return out


def sensitivity_rows(src, dev, test, y_dev, y_test, sel, layers) -> dict:
    """Not deployed: FAR's protocol, layer by layer (C=1 on standardized features as in whitebox/probe.py; keep the
    layers with train CE < 0.6; average their probabilities), and the mean logit over the layers within 0.01 of the
    best out-of-fold AUROC at the selected C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler

    out, probs, kept = {}, [], 0
    for layer in layers:
        x = src.layer(layer)
        sc = StandardScaler().fit(x[dev])
        clf = LogisticRegression(max_iter=2000, tol=1e-3).fit(sc.transform(x[dev]), y_dev)
        if log_loss(y_dev, clf.predict_proba(sc.transform(x[dev]))[:, 1], labels=[0, 1]) < 0.6:
            probs.append(clf.predict_proba(sc.transform(x[test]))[:, 1])
            kept += 1
    out["far_protocol"] = {"test_auroc": auroc(np.mean(probs, axis=0), y_test) if probs else float("nan"),
                           "kept_layers": kept}
    best_at_c = {r["layer"]: r["cv_auroc"] for r in sel.grid if r["C"] == sel.C}
    top = max(best_at_c.values())
    near = [layer for layer, a in best_at_c.items() if a >= top - 0.01]
    logits = []
    for layer in near:
        x = src.layer(layer)
        logits.append(SelectedProbe.fit(x[dev], y_dev, sel.C, layer).logit(x[test]))
    out["mean_logit_near_best"] = {"test_auroc": auroc(np.mean(logits, axis=0), y_test), "layers": near}
    return out


def save_probe(out_dir, probe, sel, meta, metrics, scores_rows) -> None:
    probe.meta = meta
    probe.save(out_dir)
    with open(os.path.join(out_dir, "cv_grid.csv"), "w") as fh:
        fh.write("layer,C,cv_auroc,cv_auroc_sd\n")
        for r in sel.grid:
            fh.write(f"{r['layer']},{r['C']},{r['cv_auroc']:.5f},{r['cv_auroc_sd']:.5f}\n")
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=1, default=float)
    with open(os.path.join(out_dir, "scores.jsonl"), "w") as fh:
        for row in scores_rows:
            fh.write(json.dumps(row, default=float) + "\n")


# ------------------------------------------------------------------------------------------------ IID


def keys_for(readout: str) -> tuple:
    from monitordecorrelation.whitebox.probe_fit import READOUT_TENSORS, SMALL_TENSORS

    return (*READOUT_TENSORS[readout], *SMALL_TENSORS)


def router_layer(readout: str, layer: int) -> dict:
    """For router probes, which model layer(s) the selected router logits come from."""
    if readout == "router":
        return {"model_layer": layer + MOE_FIRST_LAYER}
    if readout == "router_all":
        return {"model_layer": "all MoE layers"}
    return {}


def fit_iid(args, items, concealed_items, view, readout, provenance) -> dict:
    src = LayerSource(os.path.join(args.features, args.iid_dataset), view, readout)
    keep = src.keep
    its = [items[i] for i in src.ids]
    y_all = np.array([int(bool((it.get("diagnostics") or {}).get("used_verifier"))) for it in its])
    bp_all = np.array([int(it["label"]) for it in its])
    groups_all = np.array([it["task_id"] for it in its])
    split = np.array([it["split"] for it in its])
    diag = [it.get("diagnostics") or {} for it in its]
    dev = np.flatnonzero(np.isin(split, ["train", "val"]) & keep)
    test = np.flatnonzero((split == "test") & keep)
    everything = np.flatnonzero(keep)
    n_layers = src.n_layers
    layers = list(range(src.first_layer, n_layers))
    folds = make_folds(len(dev), y_all[dev], groups_all[dev], repeats=args.repeats, seed=args.seed)
    log(f"IID {view}/{readout}: {len(dev)} dev ({int(y_all[dev].sum())} hacks), {len(test)} test; "
        f"grid {len(layers)} layers × {len(C_GRID)} C × {args.repeats * 5} folds")
    oof = cv_grid_source(src, dev, y_all[dev], folds, layers, C_GRID, n_jobs=args.jobs)
    hard_dev = np.array([y_all[i] == 1 or diag[i].get("read_not_used") or not diag[i].get("passed")
                         for i in dev], bool)
    sel = select(oof, y_all[dev], groups_all[dev], layers, C_GRID,
                 tie_metric=lambda s: auroc(s[hard_dev], y_all[dev][hard_dev]), seed=args.seed)
    log(f"  selected layer {sel.layer}, C={sel.C:g} (out-of-fold AUROC {sel.cv_auroc:.4f} ± {sel.cv_se:.4f}"
        f"{', flat grid' if sel.flat else ''})")
    li, cj = layers.index(sel.layer), list(C_GRID).index(sel.C)
    oof_sel = oof[li, cj].mean(0)
    x_sel = src.layer(sel.layer)  # the selected layer, every row of the view
    probe = SelectedProbe.fit(x_sel[dev], y_all[dev], sel.C, sel.layer)
    probe.threshold_1, probe.threshold_5 = thresholds(oof_sel[y_all[dev] == 0])
    t_logit = probe.logit(x_sel[test])
    yt, gt = y_all[test], groups_all[test]
    metrics = {
        "test_auroc": auroc(t_logit, yt), "test_auroc_ci": ci(cluster_bootstrap(auroc, gt, t_logit, yt, n=2000)),
        "test_auroc_behavior_present": auroc(t_logit, bp_all[test]),
        "tpr_at_t1": rate(t_logit, yt == 1, probe.threshold_1), "fpr_at_t1": rate(t_logit, yt == 0, probe.threshold_1),
        "tpr_at_t5": rate(t_logit, yt == 1, probe.threshold_5), "fpr_at_t5": rate(t_logit, yt == 0, probe.threshold_5),
        "selection": {"layer": sel.layer, "C": sel.C, "cv_auroc": sel.cv_auroc, "cv_se": sel.cv_se, "flat": sel.flat},
        "n": {"dev": len(dev), "dev_hacks": int(y_all[dev].sum()), "test": len(test), "test_hacks": int(yt.sum())},
    }
    # cross-fitted over everything, for the hard subsets and the concealment set
    cf, fold_of = crossfit(x_sel, everything, y_all[everything], groups_all[everything], sel.layer, sel.C,
                           seed=args.seed)
    y_e, g_e = y_all[everything], groups_all[everything]
    d_e = [diag[i] for i in everything]
    rnu = np.array([y == 1 or d.get("read_not_used") for y, d in zip(y_e, d_e)], bool)
    hw = np.array([y == 1 or not d.get("passed") for y, d in zip(y_e, d_e)], bool)
    long_ = np.array([(d.get("n_turns") or 0) >= 3 for d in d_e], bool)
    metrics["crossfit"] = {
        "auroc_all": auroc(cf, y_e),
        "hacks_vs_read_not_used": auroc(cf[rnu], y_e[rnu]),
        "hacks_vs_honest_wrong": auroc(cf[hw], y_e[hw]),
        "three_plus_turns": auroc(cf[long_], y_e[long_]),
        "within_task": within_group_auroc(cf, y_e, g_e),
        "n_read_not_used": int((rnu & (y_e == 0)).sum()), "n_honest_wrong": int((hw & (y_e == 0)).sum()),
    }
    if concealed_items is not None and os.path.isdir(os.path.join(args.features, "terminal_concealed", view)):
        csrc = LayerSource(os.path.join(args.features, "terminal_concealed"), view, readout)
        cx, ckeep = csrc.layer(sel.layer), csrc.keep
        rec = {}
        for level in ("A", "B"):
            logits = []
            for row, cid in enumerate(csrc.ids):
                it = concealed_items[cid]
                if it["concealment"] == level and ckeep[row] and it["task_id"] in fold_of:
                    logits.append(float(fold_of[it["task_id"]].logit(cx[row: row + 1])[0]))
            rec[level] = {"n": len(logits), "recall_at_t1": rate(logits, np.ones(len(logits)), probe.threshold_1),
                          "recall_at_t5": rate(logits, np.ones(len(logits)), probe.threshold_5)}
        unedited = cf[y_e == 1]
        rec["unedited_crossfit"] = {"n": len(unedited), "recall_at_t1": rate(unedited, np.ones(len(unedited)),
                                                                             probe.threshold_1)}
        metrics["concealment"] = rec
    if readout == "followup" and not args.skip_baselines:
        log("  baselines")
        base = text_baselines([its[i] for i in dev], [its[i] for i in test], y_all[dev], yt, gt, view,
                              src.counts[dev], src.counts[test])
        msrc = LayerSource(os.path.join(args.features, args.iid_dataset), view, "mean")
        assert msrc.ids == src.ids
        both = msrc.keep[dev].all() and msrc.keep[test].all()
        base.update(activation_baselines(x_sel, dev, test, y_all[dev], yt, sel.layer, sel.C,
                                         x0=msrc.layer(0) if both else None, answer_logp=src.answer_logp,
                                         groups_dev=groups_all[dev], seed=args.seed))
        metrics["baselines"] = base
        if not args.skip_sensitivity:
            log("  sensitivity rows")
            metrics["sensitivity"] = sensitivity_rows(src, dev, test, y_all[dev], yt, sel, list(range(n_layers)))
    meta = {**provenance, "dataset": args.iid_dataset, "view": view, "readout": readout, "label": "used_verifier",
            "C": sel.C, "selection": metrics["selection"], "n_layers": n_layers, **router_layer(readout, sel.layer)}
    rows = [{"id": src.ids[i], "split": split[i], "used_verifier": int(y_all[i]), "behavior_present": int(bp_all[i]),
             "task_id": groups_all[i], "crossfit_logit": float(c)} for i, c in zip(everything, cf)]
    test_logit = dict(zip(test.tolist(), t_logit.tolist()))
    for r, i in zip(rows, everything):
        r["test_logit"] = test_logit.get(int(i))
    out = os.path.join(args.out, args.iid_dataset, view, readout)
    save_probe(out, probe, sel, meta, metrics, rows)
    log(f"  test AUROC {metrics['test_auroc']:.4f} {metrics['test_auroc_ci']}; within-task "
        f"{metrics['crossfit']['within_task']:.3f}; hacks vs read-not-used {metrics['crossfit']['hacks_vs_read_not_used']:.3f}")
    return metrics


# ------------------------------------------------------------------------------------------------ OOD


def fit_ood(args, ood_items, items, concealed_items, readout, provenance) -> dict:
    view = "without_reasoning"
    src = LayerSource(os.path.join(args.features, "diverse_deception"), view, readout)
    keep = src.keep
    its = [ood_items[i] for i in src.ids]
    y_all = np.array([int(it["label"]) for it in its])
    pair = np.array([it["pair"] for it in its])
    source = np.array([it["source"] for it in its])
    split = np.array([it["split"] for it in its])
    dev = np.flatnonzero((split == "train") & keep)
    test = np.flatnonzero((split == "test") & keep)
    layers = list(range(src.first_layer, src.n_layers))
    strat = np.array([f"{s}|{y}" for s, y in zip(source[dev], y_all[dev])])
    folds = make_folds(len(dev), strat, pair[dev], repeats=args.repeats, seed=args.seed)
    log(f"OOD {readout}: {len(dev)} dev, {len(test)} test")
    oof = cv_grid_source(src, dev, y_all[dev], folds, layers, C_GRID, n_jobs=args.jobs)
    sources = sorted(set(source))

    def source_avg(s):
        vals = [auroc(s[source[dev] == src], y_all[dev][source[dev] == src]) for src in sources]
        return float(np.nanmean(vals))

    sel = select(oof, y_all[dev], pair[dev], layers, C_GRID, tie_metric=source_avg, seed=args.seed)
    log(f"  selected layer {sel.layer}, C={sel.C:g} (out-of-fold AUROC {sel.cv_auroc:.4f})")
    x_sel = src.layer(sel.layer)
    probe = SelectedProbe.fit(x_sel[dev], y_all[dev], sel.C, sel.layer)
    t_logit = probe.logit(x_sel[test])
    metrics = {"test_auroc": auroc(t_logit, y_all[test]),
               "test_auroc_ci": ci(cluster_bootstrap(auroc, pair[test], t_logit, y_all[test], n=2000)),
               "per_source": {s: auroc(t_logit[source[test] == s], y_all[test][source[test] == s]) for s in sources},
               "selection": {"layer": sel.layer, "C": sel.C, "cv_auroc": sel.cv_auroc, "flat": sel.flat}}
    loso = {}
    for s in sources:
        tr = dev[source[dev] != s]
        te = test[source[test] == s]
        if len(te) and len(set(y_all[tr])) == 2:
            loso[s] = auroc(SelectedProbe.fit(x_sel[tr], y_all[tr], sel.C, sel.layer).logit(x_sel[te]), y_all[te])
    metrics["leave_one_source_out"] = loso
    # transfer to TerminalEnv: thresholds from honest dev episodes, metrics over every episode
    transfer, per_view_t = {}, {}
    for tview in ("with_reasoning", "without_reasoning"):
        tsrc = LayerSource(os.path.join(args.features, args.iid_dataset), tview, readout)
        tkeep = tsrc.keep
        titems = [items[i] for i in tsrc.ids]
        ty = np.array([int(bool((it.get("diagnostics") or {}).get("used_verifier"))) for it in titems])
        tsplit = np.array([it["split"] for it in titems])
        tgroups = np.array([it["task_id"] for it in titems])
        rows = np.flatnonzero(tkeep)
        logit = probe.logit(tsrc.layer(sel.layer, rows))
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
        if concealed_items is not None and os.path.isdir(os.path.join(args.features, "terminal_concealed", tview)):
            csrc = LayerSource(os.path.join(args.features, "terminal_concealed"), tview, readout)
            crows = np.flatnonzero(csrc.keep)
            clog = probe.logit(csrc.layer(sel.layer, crows))
            lv = np.array([concealed_items[csrc.ids[i]]["concealment"] for i in crows])
            transfer[tview]["concealment_recall_at_t1"] = {lev: rate(clog, lv == lev, t1) for lev in ("A", "B")}
    metrics["transfer"] = transfer
    probe.threshold_1, probe.threshold_5 = per_view_t["without_reasoning"]
    meta = {**provenance, "dataset": "diverse_deception", "view": view, "readout": readout, "label": "deceptive",
            "C": sel.C, "selection": metrics["selection"], "n_layers": src.n_layers, **router_layer(readout, sel.layer),
            "thresholds_by_terminal_view": per_view_t}
    rows = [{"id": src.ids[i], "split": split[i], "label": int(y_all[i]), "source": source[i],
             "test_logit": float(z)} for i, z in zip(test, t_logit)]
    save_probe(os.path.join(args.out, "diverse_deception", view, readout), probe, sel, meta, metrics, rows)
    log(f"  OOD test AUROC {metrics['test_auroc']:.4f}; transfer (with reasoning) "
        f"{transfer['with_reasoning']['auroc_all']:.4f}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", required=True, help="a name for the record, e.g. nvfp4 or bf16")
    ap.add_argument("--features", required=True, help="the reader's features: <dataset>/<view>/shard_*")
    ap.add_argument("--datasets", default="data/probe_datasets", help="folder with the dataset folders")
    ap.add_argument("--iid-dataset", default="terminal_prerl",
                    help="the IID dataset (TerminalEnv episodes) under --datasets and --features: terminal_prerl, or "
                         "terminal_prerl_v2 (docs/inkling_probes_amendment_1.md and _2.md, whose sha256s its probes "
                         "record; its features must be full precision, one episode per forward)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readouts", default="followup,mean",
                    help="comma-list of followup, followup_prev, mean, and the exploratory router readouts: router "
                         "(per MoE layer; its layer index i is model layer i + 2), router_all (every MoE layer)")
    ap.add_argument("--probes", default="iid,ood")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite probe folders that already exist in --out (else the run refuses to start)")
    args = ap.parse_args()
    readouts = [r for r in args.readouts.split(",") if r]
    targets = [os.path.join(args.out, args.iid_dataset, view, r) for view in ("with_reasoning", "without_reasoning")
               for r in readouts if "iid" in args.probes]
    targets += [os.path.join(args.out, "diverse_deception", "without_reasoning", r) for r in readouts
                if "ood" in args.probes]
    existing = [t for t in targets if os.path.exists(t)]
    if existing and not args.overwrite:
        ap.error(f"{len(existing)} of the probe folders this run would write already exist ({', '.join(existing)}); "
                 "--overwrite rewrites them")
    code = code_version()

    items = dataset_items(args, args.iid_dataset)
    concealed = dataset_items(args, "terminal_concealed") \
        if os.path.isdir(os.path.join(args.datasets, "terminal_concealed")) else None
    index = features_index(args, args.iid_dataset)
    if args.iid_dataset != "terminal_prerl" and amendment_2_problems(index):
        ap.error(f"{args.iid_dataset}'s features aren't the ones docs/inkling_probes_amendment_2.md fixes: "
                 f"{'; '.join(amendment_2_problems(index))} (their index.json)")
    provenance = {"reader": args.reader, "reader_info": index.get("reader"), "feature_format": index.get("feature_format"),
                  "versions": index.get("versions"), "prereg_sha256": prereg_sha(), "fitted_utc": time.strftime(
                      "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "c_grid": list(C_GRID), "repeats": args.repeats,
                  "code": code, "features_code": index.get("code")}
    if args.iid_dataset != "terminal_prerl":
        provenance["amendment_sha256"] = amendment_sha()
        provenance["amendment_2_sha256"] = amendment_sha("inkling_probes_amendment_2.md")
    summary = {}
    if "iid" in args.probes:
        for view in ("with_reasoning", "without_reasoning"):
            for readout in readouts:
                summary[f"iid/{view}/{readout}"] = {**fit_iid(args, items, concealed, view, readout, provenance),
                                                    "code": code}
    if "ood" in args.probes:
        ood_items = dataset_items(args, "diverse_deception")
        for readout in readouts:
            summary[f"ood/{readout}"] = {**fit_ood(args, ood_items, items, concealed, readout, provenance),
                                         "code": code}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "summary.json")  # accumulates: a later run with other readouts adds its probes
    if os.path.exists(path):
        summary = {**json.load(open(path)), **summary}
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    lines = [f"# Probes for reader `{args.reader}`", "", "| probe | layer | C | test AUROC (95% CI) | notes |",
             "|---|---|---|---|---|"]
    for name, m in summary.items():
        s = m["selection"]
        note = (f"within-task {m['crossfit']['within_task']:.3f}; vs read-not-used "
                f"{m['crossfit']['hacks_vs_read_not_used']:.3f}") if "crossfit" in m else \
            f"transfer {m['transfer']['with_reasoning']['auroc_all']:.3f} / {m['transfer']['without_reasoning']['auroc_all']:.3f}"
        layer = (f"MoE {s['layer']} (model {s['layer'] + MOE_FIRST_LAYER})" if name.endswith("/router") else
                 "all MoE" if name.endswith("/router_all") else s["layer"])
        lines.append(f"| {name} | {layer} | {s['C']:g} | {m['test_auroc']:.4f} "
                     f"[{m['test_auroc_ci'][0]:.3f}, {m['test_auroc_ci'][1]:.3f}] | {note} |")
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
