"""Compare the 4-bit and full-precision readers, and apply the pre-registered reader, batching and readout rules
(docs/inkling_probes_preregistration.md). Runs on CPUs after experiments/fit_probes.py has fitted both readers.

    uv run python experiments/compare_readers.py --probes PROBES --features FEATURES \
        --diagnostics DIAGNOSTICS --datasets data/probe_datasets --out OUT/compare

Inputs:
- ``--probes``: ``<reader>/<dataset>/<view>/<readout>/`` from fit_probes.py.
- ``--features``: ``<reader>/<dataset>/<view>/shard_*``.
- ``--diagnostics``: ``<reader>/{diag.safetensors,done.json}`` from the GPU session.

**Reader rule.** Keep 4-bit only if, for every probe compared:
- **(a) Agreement:** the full-precision probe flags nearly the same TerminalEnv val+test episodes at t₁% on both
  readers' features (Jaccard ≥ 0.8), with Spearman ≥ 0.95.
- **(b) No AUROC loss:** the paired ΔAUROC (4-bit − full, each reader's own probe) is ≥ −0.01, with the bootstrap
  CI's lower bound ≥ −0.02.

**(c) Noise.** (a) and (b) are read against two floors:
- **Jitter:** the diagnostics' batched vs one-at-a-time readouts (|Δlogit| at the deployed probe's layer).
- **Refit:** the full-precision probe vs 20 task-bootstrap refits of itself (same reader, layer and C): Jaccard at t₁%
  and Spearman on the same episodes. Each probe's t₁% there is the 99th percentile of its logits on that view's
  honest dev episodes.
- How (c) is applied (fixed 2026-10-10 03:52 UTC, before any full-precision result): agreement passes if it meets
  (a)'s numbers, or if both its Jaccard and its Spearman are at least the refit floor's medians in each view, i.e. the
  reader moves the flagged set no more than a typical refit does. Near t₁% the flagged honest episodes are a dense
  tail, so a fixed Jaccard bar can fail for two equally good readers.

**Batching rule.** Fix the batching if more than 1% of hacks, or more than 0.5% of negatives, have logits within the
99th-percentile jitter |Δlogit| of t₁%.

**Readout rule.** "Yes" counts as sink-like if its median norm ratio to the preceding context is > 5 in layers 10–35.
The pre-registration's other criterion, "or the same ≤ 8 massive dimensions dominate its norm", has no number to apply,
so it is not decided here; its numbers are reported (massive dimensions, below).

The batching and readout rules are decided over the readers whose check ran. If none ran (the follow-up readout was not
compared, or there are no diagnostics), ``fix_batching`` / ``sink_like_yes`` are None, with the reason in
``decision.not_evaluated``.

**Report only** (no rule reads these), for every compared probe:
- **Weight cosine** between the two readers' probes; none when they sit at different layers.
- **Score shift:** the full-precision probe's logits on 4-bit minus on full-precision features, over the same
  TerminalEnv val+test episodes, per view: mean, sd, 99th percentile of |Δ|, and Pearson r.
- **Cross-reader scoring:** each reader's probe on both readers' features.
  - IID: test AUROC, and TPR/FPR at the probe's own t₁% and at t₁% recalibrated on that reader's honest dev episodes
    (train + val), as experiments/deploy_probes.py recalibrates for serving on another reader.
  - OOD: OOD test AUROC, and transfer AUROC over every TerminalEnv episode of each view.
- **Batching floor** (follow-up probes; the diagnostics keep only that readout): on the diagnostics' TerminalEnv rows
  of the view(s) the probe scores, the mean and sd of its one-at-a-time − batched logits, and Spearman between them.
- **Massive dimensions** of "Yes" (``massive_dims``, per reader with diagnostics; follow-up comparisons): per layer,
  the share of its squared norm in each input's top-8 dimensions, how consistent those dimensions are across inputs,
  and the share in the layer's 8 most frequent ones; and their medians over layers 10–35.

**Writes** ``decision.json`` and ``report.md``; decision.json records the code version (provenance.code_version) as
``code``. An ``--out`` that already holds either is refused, unless ``--overwrite``.

**--from-json DECISION** re-applies the rules to an earlier decision.json and rewrites both files, recomputing nothing.
It rewrites the folder DECISION is in without ``--overwrite`` (that is what it is for); another folder that holds a
comparison needs it. ``code`` stays that of the run that computed the numbers; the code that re-applied the rules is
recorded as ``from_json_code``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from monitordecorrelation.provenance import code_version
from monitordecorrelation.whitebox.probe_fit import (
    LayerSource, SelectedProbe, auroc, ci, cluster_bootstrap, load_items, thresholds)

READERS = ("nvfp4", "bf16")
OUTPUTS = ("decision.json", "report.md")
VIEWS = ("with_reasoning", "without_reasoning")


def probes_for(readouts: list[str]) -> list[tuple[str, str, str]]:
    """The probes compared for each readout: IID in both views, and OOD."""
    return [(dataset, view, r) for r in readouts for dataset, view in
            (("terminal_prerl", "with_reasoning"), ("terminal_prerl", "without_reasoning"),
             ("diverse_deception", "without_reasoning"))]


def router_topk(features_dir: str, view: str) -> dict:
    """{id: top-6 experts at "Yes" per MoE layer [M, k]} from a reader's shards, or {} if they don't have them."""
    import glob

    from safetensors import safe_open

    out = {}
    for marker in sorted(glob.glob(os.path.join(features_dir, view, "shard_*.jsonl"))):
        data = marker[: -len(".jsonl")] + ".safetensors"
        if not os.path.exists(data):
            continue
        with safe_open(data, framework="pt") as f:
            if "router_topk" not in f.keys():
                return {}
            topk = f.get_tensor("router_topk").numpy()
        ids = [json.loads(line)["id"] for line in open(marker)]
        out.update(zip(ids, topk))
    return out


def routing_agreement_at_yes(args) -> dict:
    """The share of matching top-6 experts at "Yes" between the readers, per MoE layer, over every item both readers
    extracted (the stored features; the diagnostics' version covers every token of a small sample)."""
    out = {}
    for dataset, view in (("terminal_prerl", "with_reasoning"), ("terminal_prerl", "without_reasoning"),
                          ("diverse_deception", "without_reasoning")):
        a = router_topk(os.path.join(args.features, "nvfp4", dataset), view)
        b = router_topk(os.path.join(args.features, "bf16", dataset), view)
        ids = [i for i in a if i in b]
        if not ids:
            continue
        x, y = np.stack([a[i] for i in ids]).astype(np.int64), np.stack([b[i] for i in ids]).astype(np.int64)
        same = (x[..., :, None] == y[..., None, :]).any(-1).mean(-1)  # [items, M]: share of a's experts in b's
        per_layer = same.mean(0)
        out[f"{dataset}/{view}"] = {"n": len(ids), "mean": float(per_layer.mean()),
                                    "min_layer": int(per_layer.argmin()), "min": float(per_layer.min()),
                                    "per_moe_layer": [round(float(v), 4) for v in per_layer]}
    return out


def jaccard(a, b) -> float:
    a, b = set(np.flatnonzero(a)), set(np.flatnonzero(b))
    return len(a & b) / len(a | b) if a | b else 1.0


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def logits_on(probe: SelectedProbe, features_dir, dataset, view, readout, ids=None):
    """The probe's logits for every kept item of a view, read one layer from disk (to keep memory low)."""
    src = LayerSource(os.path.join(features_dir, dataset), view, readout)
    rows = np.flatnonzero(src.keep)
    z = probe.logit(src.layer(probe.layer, rows))
    out = dict(zip([src.ids[i] for i in rows], z))
    return out if ids is None else np.array([out[i] for i in ids])


def hack_labels(items, ids) -> np.ndarray:
    """The IID label, ``used_verifier``, for TerminalEnv ``ids``."""
    return np.array([int(bool((items[i].get("diagnostics") or {}).get("used_verifier"))) for i in ids])


def rate(scores, threshold: float) -> float:
    s = np.asarray(scores, float)
    return float(np.mean(s > threshold)) if len(s) else float("nan")


def weight_cosine(a: SelectedProbe, b: SelectedProbe) -> float | None:
    """The cosine between two probes' weights; None when they read different layers."""
    if a.layer != b.layer or a.coef.shape != b.coef.shape:
        return None
    return float(a.coef @ b.coef / (np.linalg.norm(a.coef) * np.linalg.norm(b.coef)))


def score_shift(on_full, on_4bit) -> dict:
    """How one probe's logits move from full-precision to 4-bit features on the same episodes (Δ = 4-bit − full)."""
    a, b = np.asarray(on_full, float), np.asarray(on_4bit, float)
    if len(a) < 2:
        return {"n": len(a), "mean": float("nan"), "sd": float("nan"), "p99_abs": float("nan"), "pearson": float("nan")}
    d = b - a
    return {"n": len(a), "mean": float(d.mean()), "sd": float(d.std()), "p99_abs": float(np.percentile(np.abs(d), 99)),
            "pearson": float(np.corrcoef(a, b)[0, 1])}


def cross_reader(args, dataset, view, readout, probes: dict, items, ood_items) -> dict:
    """Each reader's probe on both readers' features: ``{probe's reader: {features' reader: metrics}}``.

    - IID: test AUROC, and TPR/FPR on test at the probe's own t₁% and at t₁% recalibrated on the features' honest dev
      episodes (train + val, not ``used_verifier``).
    - OOD: OOD test AUROC, and transfer AUROC over every TerminalEnv episode of each view."""
    out: dict = {}
    for p_reader, probe in probes.items():
        out[p_reader] = {}
        for f_reader in READERS:
            feats = os.path.join(args.features, f_reader)
            z = logits_on(probe, feats, dataset, view, readout)
            if dataset == "terminal_prerl":
                ids = list(z)
                s, y = np.array([z[i] for i in ids]), hack_labels(items, ids)
                split = np.array([items[i]["split"] for i in ids])
                test, dev_neg = split == "test", np.isin(split, ["train", "val"]) & (y == 0)
                m = {"n_test": int(test.sum()), "test_auroc": auroc(s[test], y[test])}
                for name, t1 in (("own", probe.threshold_1), ("recalibrated", thresholds(s[dev_neg])[0])):
                    m.update({f"t1_{name}": float(t1), f"tpr_at_t1_{name}": rate(s[test & (y == 1)], t1),
                              f"fpr_at_t1_{name}": rate(s[test & (y == 0)], t1)})
            else:
                ids = [i for i in z if ood_items[i]["split"] == "test"]
                m = {"ood_test_auroc": auroc([z[i] for i in ids], [int(ood_items[i]["label"]) for i in ids]),
                     "transfer_auroc": {}}
                for tview in VIEWS:
                    zt = logits_on(probe, feats, "terminal_prerl", tview, readout)
                    m["transfer_auroc"][tview] = auroc(list(zt.values()), hack_labels(items, zt))
            out[p_reader][f_reader] = m
    return out


def batching_floor(args, dataset, view, probes: dict, items) -> dict:
    """How each reader's follow-up probe moves between the diagnostics' batched and one-at-a-time forwards, on the
    diagnostics' TerminalEnv rows of the view(s) the probe scores (an OOD probe scores both):
    ``{diagnostics' reader: {probe's reader: {view: n, mean and sd of alone − batched, Spearman}}}``. Report only."""
    from safetensors import safe_open

    tviews = VIEWS if dataset == "diverse_deception" else (view,)
    out: dict = {}
    for d_reader in READERS:
        path, done = (os.path.join(args.diagnostics, d_reader, f) for f in ("diag.safetensors", "done.json"))
        if not (os.path.exists(path) and os.path.exists(done)):
            continue
        rows_meta = json.load(open(done)).get("items") or []
        out[d_reader] = {}
        for p_reader, probe in probes.items():
            with safe_open(path, framework="pt") as f:
                batched, alone = [probe.logit(f.get_slice(k)[:, probe.layer, :].float().numpy().astype(np.float64))
                                  for k in ("followup_batched", "followup_alone")]
            if len(rows_meta) != len(batched):
                raise ValueError(f"{done} lists {len(rows_meta)} items for {len(batched)} diagnostics rows")
            out[d_reader][p_reader] = {}
            for tview in tviews:
                rows = [k for k, r in enumerate(rows_meta) if r["view"] == tview and r["id"] in items]
                a, b = batched[rows], alone[rows]
                enough = len(rows) >= 2
                out[d_reader][p_reader][tview] = {
                    "n": len(rows), "mean": float(np.mean(b - a)) if enough else float("nan"),
                    "sd": float(np.std(b - a)) if enough else float("nan"),
                    "spearman": spearman(a, b) if enough else float("nan")}
    return out


def compare_probe(args, dataset, view, readout, items, ood_items) -> dict:
    p4 = SelectedProbe.load(os.path.join(args.probes, "nvfp4", dataset, view, readout))
    pf = SelectedProbe.load(os.path.join(args.probes, "bf16", dataset, view, readout))
    out = {"layers": {"nvfp4": p4.layer, "bf16": pf.layer}}
    # (a) the full-precision probe on both readers' TerminalEnv val+test features, in both views
    agree, shift = {}, {}
    for tview in ("with_reasoning", "without_reasoning"):
        on_f = logits_on(pf, os.path.join(args.features, "bf16"), "terminal_prerl", tview, readout)
        on_4 = logits_on(pf, os.path.join(args.features, "nvfp4"), "terminal_prerl", tview, readout)
        ids = [i for i in on_f if i in on_4 and items[i]["split"] in ("val", "test")]
        a, b = np.array([on_f[i] for i in ids]), np.array([on_4[i] for i in ids])
        t1 = pf.meta.get("thresholds_by_terminal_view", {}).get(tview, [pf.threshold_1])[0]
        agree[tview] = {"n": len(ids), "jaccard_at_t1": jaccard(a > t1, b > t1), "spearman": spearman(a, b),
                        "max_abs_dlogit": float(np.max(np.abs(a - b))) if len(ids) else float("nan")}
        shift[tview] = score_shift(a, b)
    out["agreement"] = agree
    # (b) paired ΔAUROC, each reader's own probe on its own features
    if dataset == "diverse_deception":
        ids = [i for i, it in ood_items.items() if it["split"] == "test"]
        z4 = logits_on(p4, os.path.join(args.features, "nvfp4"), dataset, view, readout)
        zf = logits_on(pf, os.path.join(args.features, "bf16"), dataset, view, readout)
        ids = [i for i in ids if i in z4 and i in zf]
        y = np.array([ood_items[i]["label"] for i in ids])
        g = np.array([ood_items[i]["pair"] for i in ids])
        deltas = {"ood_test": (np.array([z4[i] for i in ids]), np.array([zf[i] for i in ids]), y, g)}
        for tview in ("with_reasoning", "without_reasoning"):  # IID transfer
            t4 = logits_on(p4, os.path.join(args.features, "nvfp4"), "terminal_prerl", tview, readout)
            tf = logits_on(pf, os.path.join(args.features, "bf16"), "terminal_prerl", tview, readout)
            tids = [i for i in t4 if i in tf]
            deltas[f"iid_transfer_{tview}"] = (
                np.array([t4[i] for i in tids]), np.array([tf[i] for i in tids]),
                np.array([int(bool(items[i]["diagnostics"].get("used_verifier"))) for i in tids]),
                np.array([items[i]["task_id"] for i in tids]))
    else:
        z4 = logits_on(p4, os.path.join(args.features, "nvfp4"), dataset, view, readout)
        zf = logits_on(pf, os.path.join(args.features, "bf16"), dataset, view, readout)
        ids = [i for i in z4 if i in zf and items[i]["split"] == "test"]
        deltas = {"iid_test": (np.array([z4[i] for i in ids]), np.array([zf[i] for i in ids]),
                               np.array([int(bool(items[i]["diagnostics"].get("used_verifier"))) for i in ids]),
                               np.array([items[i]["task_id"] for i in ids]))}
    out["delta_auroc"] = {}
    for name, (s4, sf, y, g) in deltas.items():
        boot = cluster_bootstrap(lambda a, b, yy: auroc(a, yy) - auroc(b, yy), g, s4, sf, y, n=2000)
        out["delta_auroc"][name] = {"auroc_nvfp4": auroc(s4, y), "auroc_bf16": auroc(sf, y),
                                    "delta": auroc(s4, y) - auroc(sf, y), "ci": ci(boot)}
    out["refit_floor"] = refit_floor(args, "bf16", dataset, view, readout, pf, items, ood_items)
    # report only: no rule reads these
    out["weight_cosine"] = weight_cosine(p4, pf)
    out["score_shift"] = shift
    out["cross_reader"] = cross_reader(args, dataset, view, readout, {"nvfp4": p4, "bf16": pf}, items, ood_items)
    if readout == "followup":
        out["batching_floor"] = batching_floor(args, dataset, view, {"nvfp4": p4, "bf16": pf}, items)
    return out


def jitter(args, reader, probe: SelectedProbe, items) -> dict:
    """|Δlogit| between the session's batching and one item per forward, on the diagnostics sample."""
    import torch
    from safetensors.torch import load_file

    path = os.path.join(args.diagnostics, reader, "diag.safetensors")
    if not os.path.exists(path):
        return {}
    d = load_file(path)
    meta = json.load(open(os.path.join(args.diagnostics, reader, "done.json")))
    a = d["followup_batched"][:, probe.layer, :].float().numpy().astype(np.float64)
    b = d["followup_alone"][:, probe.layer, :].float().numpy().astype(np.float64)
    dz = np.abs(probe.logit(a) - probe.logit(b))
    p99 = float(np.percentile(dz, 99))
    out = {"n": len(dz), "dlogit_p50": float(np.median(dz)), "dlogit_p99": p99,
           "topk_agree_jitter_mean": meta.get("topk_agree_jitter_mean")}
    if "routing_topk" in d:
        out["routing_topk_shape"] = list(d["routing_topk"].shape)
    _ = torch  # loaded for safetensors
    return out


def batching_rule(args, reader, probe: SelectedProbe, items, p99: float) -> dict:
    """The share of hacks and negatives whose logit lies within ``p99`` of t₁%."""
    rows = []
    for view in ("with_reasoning", "without_reasoning"):
        z = logits_on(probe, os.path.join(args.features, reader), "terminal_prerl", view, "followup")
        rows += [(z[i], bool(items[i]["diagnostics"].get("used_verifier"))) for i in z]
    z = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    near = np.abs(z - probe.threshold_1) <= p99
    hacks, negs = float(near[y].mean()) if y.any() else 0.0, float(near[~y].mean())
    return {"near_threshold_hacks": hacks, "near_threshold_negatives": negs,
            "fix_batching": hacks > 0.01 or negs > 0.005}


N_REFITS = 20


def refit_floor(args, reader, dataset, view, readout, probe: SelectedProbe, items, ood_items, n: int = N_REFITS,
                seed: int = 0) -> dict:
    """Rule (c)'s refit floor: how far the flagged set and the ranking move when ``probe`` is refitted on task (or
    pair) bootstrap resamples of its own dev set, on TerminalEnv val+test in both views."""
    feats = os.path.join(args.features, reader)
    src = LayerSource(os.path.join(feats, dataset), view, readout)
    if dataset == "terminal_prerl":
        its = [items[i] for i in src.ids]
        y = np.array([int(bool((it.get("diagnostics") or {}).get("used_verifier"))) for it in its])
        groups = np.array([it["task_id"] for it in its])
        dev = np.isin([it["split"] for it in its], ["train", "val"])
    else:
        its = [ood_items[i] for i in src.ids]
        y = np.array([int(it["label"]) for it in its])
        groups = np.array([it["pair"] for it in its])
        dev = np.array([it["split"] for it in its]) == "train"
    dev = np.flatnonzero(dev & src.keep)
    x, y, groups = src.layer(probe.layer, dev), y[dev], groups[dev]
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in uniq}
    C = float(probe.meta.get("C", probe.meta.get("selection", {}).get("C")))
    refits = []
    while len(refits) < n:
        rows = np.concatenate([rows_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(set(y[rows])) == 2:
            refits.append(SelectedProbe.fit(x[rows], y[rows], C, probe.layer))
    out = {"n": n, "C": C}
    for tview in ("with_reasoning", "without_reasoning"):
        tsrc = LayerSource(os.path.join(feats, "terminal_prerl"), tview, readout)
        rows = np.flatnonzero(tsrc.keep)
        xt = tsrc.layer(probe.layer, rows)
        tits = [items[tsrc.ids[i]] for i in rows]
        split = np.array([it["split"] for it in tits])
        ty = np.array([int(bool((it.get("diagnostics") or {}).get("used_verifier"))) for it in tits])
        dev_neg, ev = np.isin(split, ["train", "val"]) & (ty == 0), np.isin(split, ["val", "test"])

        def flags(p):
            z = p.logit(xt)
            return z[ev] > np.percentile(z[dev_neg], 99), z[ev]

        f0, z0 = flags(probe)
        jac, sp = [], []
        for r in refits:
            f, z = flags(r)
            jac.append(jaccard(f0, f))
            sp.append(spearman(z0, z))
        out[tview] = {"jaccard_median": float(np.median(jac)), "jaccard_p05": float(np.percentile(jac, 5)),
                      "spearman_median": float(np.median(sp)), "spearman_p05": float(np.percentile(sp, 5))}
    return out


def routing_agreement(args) -> dict:
    from safetensors.torch import load_file

    paths = {r: os.path.join(args.diagnostics, r, "diag.safetensors") for r in ("nvfp4", "bf16")}
    if not all(os.path.exists(p) for p in paths.values()):
        return {}
    a, b = (load_file(paths[r])["routing_topk"].long() for r in ("nvfp4", "bf16"))
    same = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean(-1)  # [T, M]
    per_layer = same.mean(0).tolist()
    return {"mean": float(np.mean(per_layer)), "per_moe_layer": [round(x, 4) for x in per_layer]}


SINK_LAYERS = slice(10, 36)  # layers 10–35, where the readout rule looks


def sink_check(args, reader) -> dict:
    path = os.path.join(args.diagnostics, reader, "done.json")
    if not os.path.exists(path):
        return {}
    ratios = json.load(open(path))["yes_norm_ratio_median_by_layer"]
    mid = ratios[SINK_LAYERS]
    return {"median_ratio_layers_10_35": float(np.median(mid)), "sink_like": float(np.median(mid)) > 5}


def massive_dims(args, reader) -> dict:
    """Report only: the numbers behind the readout rule's second half ("Yes" is also sink-like if "the same ≤ 8 massive
    dimensions dominate its norm"), for which the pre-registration sets no threshold and none is set here. From the
    diagnostics' 8 largest-magnitude dimensions of "Yes" per layer (``massive_idx``) and its readouts from the same
    one-item forwards (``followup_alone``), per layer:
    - ``top_share``: the median over inputs of the share of "Yes"'s squared norm in its own top-8 dimensions;
    - ``consistency``: how much the inputs agree on those dimensions: the mean share of an input's top 8 that are among
      the layer's 8 most frequent (1: every input has the same 8);
    - ``common_share``: the median share of the squared norm in those 8 most frequent dimensions, the same 8 for every
      input; and the median of each over layers 10–35."""
    from safetensors import safe_open

    path = os.path.join(args.diagnostics, reader, "diag.safetensors")
    if not os.path.exists(path):
        return {}
    with safe_open(path, framework="pt") as f:
        if not {"massive_idx", "followup_alone"} <= set(f.keys()):
            return {}
        idx = f.get_tensor("massive_idx").long().numpy()  # [inputs, layers, k]
        yes = f.get_tensor("followup_alone")  # [inputs, layers, d], bf16
    n, n_layers, k = idx.shape
    top, consistency, common_share, common_dims = [], [], [], []
    for layer in range(n_layers):
        y = yes[:, layer].double().numpy()
        sq = y ** 2
        total = sq.sum(1)
        top.append(float(np.median(np.take_along_axis(sq, idx[:, layer], 1).sum(1) / total)))
        dims, counts = np.unique(idx[:, layer], return_counts=True)
        common = dims[np.argsort(-counts, kind="stable")[:k]]  # ties: the lower dimension
        consistency.append(float(np.isin(idx[:, layer], common).sum(1).mean() / k))
        common_share.append(float(np.median(sq[:, common].sum(1) / total)))
        common_dims.append(sorted(int(d) for d in common))

    def mid(values: list[float]) -> float | None:
        return float(np.median(values[SINK_LAYERS])) if values[SINK_LAYERS] else None

    return {"n": n, "k": k, "top_share_by_layer": [round(v, 4) for v in top],
            "consistency_by_layer": [round(v, 4) for v in consistency],
            "common_share_by_layer": [round(v, 4) for v in common_share], "common_dims_by_layer": common_dims,
            "median_layers_10_35": {"top_share": mid(top), "consistency": mid(consistency),
                                    "common_share": mid(common_share)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes")
    ap.add_argument("--features")
    ap.add_argument("--diagnostics")
    ap.add_argument("--datasets", default="data/probe_datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readouts", default="followup",
                    help="comma-list: the probes compared (followup, mean, router, router_all); the jitter, batching "
                         "and sink checks are about the follow-up readout and run only with it")
    ap.add_argument("--from-json", help="re-apply the rules to an earlier decision.json and rewrite the report, "
                                        "without recomputing anything; may rewrite the folder that JSON is in")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the decision.json and report.md already in --out (else the run refuses to start)")
    args = ap.parse_args(argv)
    existing = [f for f in OUTPUTS if os.path.exists(os.path.join(args.out, f))]
    rerender = bool(args.from_json) and \
        os.path.realpath(os.path.dirname(os.path.abspath(args.from_json))) == os.path.realpath(args.out)
    if existing and not (args.overwrite or rerender):
        ap.error(f"{args.out} already holds {' and '.join(existing)}; --overwrite replaces them")
    if args.from_json:
        report = json.load(open(args.from_json))
        finish({**report, "from_json_code": code_version()}, args.out)
        return 0
    if not (args.probes and args.features and args.diagnostics):
        ap.error("--probes, --features and --diagnostics are required (unless --from-json)")
    readouts = [r for r in args.readouts.split(",") if r]
    items = load_items(os.path.join(args.datasets, "terminal_prerl"))
    ood_items = load_items(os.path.join(args.datasets, "diverse_deception"))
    report = {"readouts": readouts, "code": code_version(), "probes": {}, "jitter": {}, "batching": {}, "sink": {},
              "massive_dims": {}, "routing_agreement": routing_agreement(args),
              "routing_agreement_at_yes": routing_agreement_at_yes(args)}
    for dataset, view, readout in probes_for(readouts):
        name = f"{dataset}/{view}/{readout}"
        if all(os.path.isdir(os.path.join(args.probes, r, dataset, view, readout)) for r in ("nvfp4", "bf16")):
            report["probes"][name] = compare_probe(args, dataset, view, readout, items, ood_items)
    for reader in ("nvfp4", "bf16") if "followup" in readouts else ():
        deployed = os.path.join(args.probes, reader, "terminal_prerl", "with_reasoning", "followup")
        if os.path.isdir(deployed):
            probe = SelectedProbe.load(deployed)
            report["jitter"][reader] = jitter(args, reader, probe, items)
            if report["jitter"][reader]:
                report["batching"][reader] = batching_rule(args, reader, probe, items,
                                                           report["jitter"][reader]["dlogit_p99"])
        report["sink"][reader] = sink_check(args, reader)
        report["massive_dims"][reader] = massive_dims(args, reader)  # report only
    finish(report, args.out)
    return 0


def any_reader(results: dict, key: str) -> bool | None:
    """``key`` is true for any reader whose check ran (a non-empty result); None if it ran for none."""
    ran = [r[key] for r in results.values() if r]
    return any(ran) if ran else None


def why_not_evaluated(report: dict, rule: str, needs: str) -> str:
    readouts = report.get("readouts")
    if readouts is not None and "followup" not in readouts:
        return (f"the {rule} is about the follow-up readout, which this comparison left out (readouts: "
                f"{', '.join(readouts)})")
    return f"no reader had {needs}"


def decided(decision: dict, key: str) -> str:
    """A rule's outcome for report.md: True / False, or why it was not evaluated."""
    value = decision.get(key)
    if value is None:
        return f"not evaluated ({(decision.get('not_evaluated') or {}).get(key, 'no reader was checked')})"
    return str(value)


def finish(report: dict, out: str) -> None:
    """Apply the rules to the compared probes (one decision per readout: a readout keeps 4-bit only if every probe on
    it passes), then write decision.json and report.md.

    The batching and readout rules are decided over the readers they ran for. A rule that ran for none (follow-up not
    compared, or no diagnostics) is recorded as None, with the reason in ``decision.not_evaluated``: never as False.
    Decision files written before 2026-10-12 recorded False there; ``--from-json`` decides them again."""
    ok_by_readout: dict[str, list[bool]] = {}
    for name, r in report["probes"].items():
        agree_ok = all(a["jaccard_at_t1"] >= 0.8 and a["spearman"] >= 0.95 for a in r["agreement"].values())
        fl = r["refit_floor"]
        within_noise = all(a["jaccard_at_t1"] >= fl[v]["jaccard_median"] and a["spearman"] >= fl[v]["spearman_median"]
                           for v, a in r["agreement"].items())
        delta_ok = all(d["delta"] >= -0.01 and d["ci"][0] >= -0.02 for d in r["delta_auroc"].values())
        r["passes"] = {"agreement": agree_ok, "agreement_within_refit_noise": within_noise, "delta_auroc": delta_ok}
        ok_by_readout.setdefault(name.rsplit("/", 1)[1], []).append((agree_ok or within_noise) and delta_ok)
    by_readout = {r: "nvfp4" if all(ok) else "bf16" for r, ok in ok_by_readout.items()}
    not_evaluated = {}
    fix_batching = any_reader(report.get("batching") or {}, "fix_batching")
    if fix_batching is None:
        not_evaluated["fix_batching"] = why_not_evaluated(
            report, "batching rule", "both diagnostics and a fitted follow-up probe (terminal_prerl/with_reasoning)")
    sink_like = any_reader(report.get("sink") or {}, "sink_like")
    if sink_like is None:
        not_evaluated["sink_like_yes"] = why_not_evaluated(report, "readout rule", "diagnostics (done.json)")
    decision = {"reader": "nvfp4" if by_readout and all(v == "nvfp4" for v in by_readout.values())
                else ("bf16" if by_readout else "undecided: both readers not fitted"),
                "by_readout": by_readout, "fix_batching": fix_batching, "sink_like_yes": sink_like,
                "not_evaluated": not_evaluated}
    report["decision"] = decision
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "decision.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    per = "; ".join(f"{r} → `{v}`" for r, v in by_readout.items())
    lines = [f"# Reader comparison ({', '.join(report.get('readouts', by_readout))})", "",
             f"**Decision per readout:** {per or 'none compared'}. Fix batching = {decided(decision, 'fix_batching')}; "
             f"'Yes' sink-like = {decided(decision, 'sink_like_yes')}", "",
             "| probe | Jaccard@t1 (with / without reasoning) | Spearman | refit floor: Jaccard / Spearman (medians) "
             "| ΔAUROC (4-bit − full) [CI] | passes |",
             "|---|---|---|---|---|---|"]
    for name, r in report["probes"].items():
        ag = r["agreement"]
        dl = "; ".join(f"{k}: {v['delta']:+.4f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
                       for k, v in r["delta_auroc"].items())
        fl = r["refit_floor"]
        lines.append(f"| {name} | {ag['with_reasoning']['jaccard_at_t1']:.3f} / "
                     f"{ag['without_reasoning']['jaccard_at_t1']:.3f} | "
                     f"{min(a['spearman'] for a in ag.values()):.4f} | "
                     f"{fl['with_reasoning']['jaccard_median']:.3f} / {fl['without_reasoning']['jaccard_median']:.3f}; "
                     f"{min(fl[v]['spearman_median'] for v in ag):.4f} | {dl} | {r['passes']} |")
    if report.get("routing_agreement"):
        lines += ["", f"Routing agreement (share of matching top-6 experts, all tokens of the diagnostics sample): "
                      f"{report['routing_agreement']['mean']:.4f}"]
    for name, r in report.get("routing_agreement_at_yes", {}).items():  # absent in reports before 2026-10-10 14:00
        lines.append(f"Routing agreement at \"Yes\" between the readers, {name} ({r['n']} items): mean "
                     f"{r['mean']:.4f}, lowest {r['min']:.4f} at MoE layer {r['min_layer']}")
    lines += ["", f"Jitter: {json.dumps(report['jitter'])}", f"Batching: {json.dumps(report['batching'])}",
              f"Sink check: {json.dumps(report['sink'])}"]
    lines += report_only_lines(report)
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def report_only_lines(report: dict) -> list[str]:
    """report.md's lines for the report-only fields (absent from decision.json files written before 2026-10-12)."""
    probes = {name: r for name, r in report["probes"].items() if "cross_reader" in r}
    massive = massive_lines(report.get("massive_dims") or {})
    if not (probes or massive):
        return []
    lines = ["", "## Report only (no rule reads these)"]
    if probes:
        lines += probe_lines(probes)
    return lines + massive


def massive_lines(by_reader: dict) -> list[str]:
    """report.md's table for ``massive_dims``: per layer and reader, the top-8 share, consistency and common-8
    share."""
    readers = [r for r in READERS if by_reader.get(r)]
    if not readers:
        return []
    lines = ["", "Massive dimensions of \"Yes\" (the readout rule's second half, \"the same ≤ 8 massive dimensions "
                 "dominate its norm\", has no pre-registered threshold; the rule above uses the norm ratio only). Per "
                 "layer: the median share of the squared norm in each input's own top 8 dimensions; how consistent "
                 "those dimensions are across inputs (share of an input's top 8 among the layer's 8 most frequent); "
                 "the median share in those 8 most frequent dimensions.", ""]
    for r in readers:
        med = by_reader[r]["median_layers_10_35"]
        lines.append(f"- {r} ({by_reader[r]['n']} diagnostics inputs), medians over layers 10–35: top-8 share "
                     f"{pct_or_dash(med['top_share'])}, consistency {pct_or_dash(med['consistency'])}, common-8 "
                     f"share {pct_or_dash(med['common_share'])}")
    lines += ["", "| layer | " + " | ".join(f"{r}: top-8 share / consistency / common-8 share" for r in readers)
              + " |", "|---|" + "---|" * len(readers)]
    for layer in range(max(len(by_reader[r]["top_share_by_layer"]) for r in readers)):
        cells = []
        for r in readers:
            d = by_reader[r]
            cells.append(" / ".join(pct_or_dash(d[key][layer] if layer < len(d[key]) else None) for key in
                                    ("top_share_by_layer", "consistency_by_layer", "common_share_by_layer")))
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    return lines


def pct_or_dash(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def probe_lines(probes: dict) -> list[str]:
    """report.md's report-only tables for the compared probes."""
    lines = ["",
             "Score shift: the full-precision probe's logits on 4-bit minus on full-precision features, TerminalEnv "
             "val+test.", "",
             "| probe | weight cosine (4-bit vs full) | shift, with reasoning: mean (sd), p99 abs, Pearson r "
             "| shift, without reasoning |", "|---|---|---|---|"]
    for name, r in probes.items():
        c = r["weight_cosine"]
        cos = f"{c:.3f}" if c is not None else \
            f"layers differ (nvfp4 {r['layers']['nvfp4']}, bf16 {r['layers']['bf16']})"
        sh = [f"{s['mean']:+.3f} ({s['sd']:.3f}), {s['p99_abs']:.2f}, {s['pearson']:.3f}"
              for s in (r["score_shift"][v] for v in VIEWS)]
        lines.append(f"| {name} | {cos} | {sh[0]} | {sh[1]} |")
    lines += ["", "Cross-reader scoring, IID probes (TerminalEnv test):", "",
              "| probe | probe's reader | features | test AUROC | TPR / FPR at own t1 | TPR / FPR at recalibrated t1 |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {name} | {pr} | {fr} | {m['test_auroc']:.5f} | {m['tpr_at_t1_own']:.1%} / {m['fpr_at_t1_own']:.2%} | "
              f"{m['tpr_at_t1_recalibrated']:.1%} / {m['fpr_at_t1_recalibrated']:.2%} |"
              for name, r in probes.items() for pr, by_f in r["cross_reader"].items() for fr, m in by_f.items()
              if "test_auroc" in m]
    lines += ["", "Cross-reader scoring, OOD probes:", "",
              "| probe | probe's reader | features | OOD test AUROC | transfer AUROC (with / without reasoning) |",
              "|---|---|---|---|---|"]
    lines += [f"| {name} | {pr} | {fr} | {m['ood_test_auroc']:.4f} | {m['transfer_auroc']['with_reasoning']:.3f} / "
              f"{m['transfer_auroc']['without_reasoning']:.3f} |"
              for name, r in probes.items() for pr, by_f in r["cross_reader"].items() for fr, m in by_f.items()
              if "ood_test_auroc" in m]
    floor = [(name, dr, pr, v, s) for name, r in probes.items() for dr, by_p in r.get("batching_floor", {}).items()
             for pr, by_v in by_p.items() for v, s in by_v.items()]
    if floor:
        lines += ["", "Batching floor (diagnostics' TerminalEnv rows; the batching rule is unchanged):", "",
                  "| probe | probe's reader | diagnostics' reader | view | n | one-at-a-time − batched: mean (sd) | "
                  "Spearman |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {name} | {pr} | {dr} | {v} | {s['n']} | {s['mean']:+.3f} ({s['sd']:.3f}) | {s['spearman']:.3f} |"
                  for name, dr, pr, v, s in floor]
    return lines


if __name__ == "__main__":
    sys.exit(main())
