"""Numbers behind docs/INKLING_PROBES.md that came from one-off snippets, as subcommands that write JSON to ``--out``.

    uv run python experiments/probe_analyses.py dataset-stats --datasets data/probe_datasets \
        --out OUT/analyses/<UTC timestamp>
    uv run python experiments/probe_analyses.py near-threshold --features FEATURES/bf16 \
        --probes PROBES/bf16/terminal_prerl/with_reasoning/followup,<more probe dirs> \
        --out OUT/analyses/<UTC timestamp>
    uv run python experiments/probe_analyses.py massive-dims --per-token FEATURES/nvfp4/terminal_prerl \
        --per-token-features FEATURES/nvfp4 --features FEATURES/bf16 --features-prev FEATURES_PREV/bf16 \
        --out OUT/analyses/<UTC timestamp>
    uv run python experiments/probe_analyses.py readout-gap --probes PROBES/bf16/diverse_deception/without_reasoning \
        --readouts followup,followup_prev --out OUT/analyses/<UTC timestamp>

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

It sizes a re-scoring of only the episodes near t₁% (the batching rule's fix, docs/INKLING_PROBES.md section 7).

**massive-dims** (``massive_dims.json``): the readout rule's second half calls "Yes" sink-like if "the same ≤ 8 massive
dimensions dominate its norm". The pre-registration sets no number for it, and none is set here. These numbers tell
whether that concentration is specific to "Yes" or found at every token. Two parts; give either or both:
- **Per token** (``--per-token``): extract_probe_features.py's ``--full-activations`` files,
  ``<view>/full/<item id, "/" as "__">.safetensors``, read one file and one layer at a time.
  - Each file's item (``--per-token-dataset``) is rendered as the extraction renders it, and must give the file's
    tokens.
  - Per layer and token: the share of its squared norm in its own top 8 dimensions; its norm over the episode's median
    token norm; and, against the same episode's "Yes" at that layer, how many of its top 8 are "Yes"'s and the share
    of its squared norm in "Yes"'s 8.
  - Summarized per role (``token_roles``: "Yes", the token before it, other special tokens, the policy's thinking and
    text, everything else) and for the first token (also counted as special). Per layer: the median top-8 share, the
    share of tokens above 50%, the median relative norm, the mean overlap with "Yes"'s 8 (as a share of 8), the share
    with exactly "Yes"'s 8, and the median share in "Yes"'s 8. Each also as its median over the layers of each band:
    4–16, 17–24, 25–35, and 10–35, where the readout rule looks.
  - Per file, the sink check's ratio for "Yes" and for the token before: the norm over the median of the 64
    conversation tokens before the follow-up, as whitebox/diagnostics.py defines it.
  - With ``--per-token-features`` (the per-item features of the files' reader): whether each file's "Yes" equals its
    item's ``followup`` readout there, bit for bit.
- **Per item** (``--features``; ``--features-prev``, default the same folder): one reader's ``followup`` ("Yes") and
  ``followup_prev`` (the token before) readouts of every item of ``--readout-datasets``, in each view both hold, one
  shard at a time. Per layer: each readout's median top-8 share, the median norm ratio ‖"Yes"‖ / ‖token before‖, and
  the mean share of the 8 they have in common; each also as its median over each band. ``--max-shards N`` reads only
  each view's first N shards, for a quick check.

**readout-gap** (``readout_gap.json``): for each ``--probes`` folder (fit_probes.py's ``<reader>/<dataset>/<view>/``,
with a probe folder per readout), the test AUROC difference between its two ``--readouts`` (the second minus the
first; by default followup_prev − followup):
- a paired cluster bootstrap (probe_fit.cluster_bootstrap: the same resampled clusters for both readouts; ``--n-boot``
  resamples, ``--seed``): its sd, 95% CI and share of resamples above 0;
- Spearman between the two probes' test logits.
Clusters and labels as fit_probes.py's test AUROC: the OOD set's pairs and its deceptive label, or TerminalEnv tasks
and ``used_verifier``. Both probes must score the same test items with the same labels, and each recomputed test AUROC
must equal its metrics.json's, else the subcommand refuses.

Each subcommand writes its own file, so all can share one ``--out``. A subcommand refuses to start if its file is
already there, unless ``--overwrite``. Every file records the code version (provenance.code_version) as ``code``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

from monitordecorrelation.provenance import code_version
from monitordecorrelation.whitebox.probe_fit import (
    LayerSource, SelectedProbe, auroc, ci, cluster_bootstrap, load_items, thresholds)

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


MASSIVE_K = 8  # the readout rule's "≤ 8 massive dimensions" (diagnostics.N_MASSIVE)
ROLES = ("yes", "prev", "special", "policy", "other")
BANDS = {"4-16": (4, 16), "17-24": (17, 24), "25-35": (25, 35), "10-35": (10, 35)}  # inclusive; 10-35: the rule's
DOMINATES = 0.5  # a token's top-k dimensions dominate its norm: they hold more than half of its squared norm


def top_k_mask(sq: np.ndarray, k: int = MASSIVE_K) -> np.ndarray:
    """[n, d] bool: each row's ``k`` largest entries of ``sq`` (squared activations, so the largest magnitudes). Ties
    at the k-th value go to the lower dimensions, so the set never depends on how a sort orders equal values."""
    d = sq.shape[1]
    kth = np.partition(sq, d - k, axis=1)[:, d - k, None]  # each row's k-th largest value
    above = sq > kth
    tied = sq == kth
    return above | (tied & (np.cumsum(tied, axis=1, dtype=np.int32) <= k - above.sum(1, keepdims=True)))


def top_k_share(sq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The share of each row's squared norm in its ``mask`` dimensions; NaN for a zero row."""
    total = sq.sum(1, dtype=np.float64)
    top = np.where(mask, sq, 0).sum(1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, top / total, np.nan)


def squares(x) -> np.ndarray:
    """Squared bfloat16 activations as float32: exact, since a bfloat16 has 8 significant bits and float32 24."""
    h = x.float().numpy()
    return h * h


def rounded(values) -> list:
    return [round(float(v), 4) for v in values]


def band_medians(by_layer: dict) -> dict:
    """Each per-layer statistic's median over the layers of each band; None for a band the model's layers don't
    cover."""
    return {band: {stat: float(np.median(v[lo: hi + 1])) if hi < len(v) else None for stat, v in by_layer.items()}
            for band, (lo, hi) in BANDS.items()}


def token_roles(tokens, readout: int, n_conversation_tokens: int, sp: dict[str, int], special) -> np.ndarray:
    """Each position's role, as an index into ROLES (the first that applies):
    - ``yes``: the readout, the follow-up answer "Yes";
    - ``prev``: the token before it (the ``followup_prev`` readout, ``<|content_text|>``);
    - ``special``: any other token for which ``special(token id)`` holds: the role and channel markers;
    - ``policy``: the policy's own content, its thinking and visible text: features.role_masks, the tokens the
      mean-pooled probe averages;
    - ``other``: the rest: the system message, the task, the command outputs and the follow-up question."""
    from monitordecorrelation.whitebox.features import role_masks

    roles = np.full(len(tokens), ROLES.index("other"), dtype=np.int8)
    reasoning, action = role_masks(tokens, n_conversation_tokens, sp)
    roles[np.asarray(reasoning + action, dtype=int)] = ROLES.index("policy")
    roles[np.asarray([i for i, t in enumerate(tokens) if special(int(t))], dtype=int)] = ROLES.index("special")
    roles[readout - 1] = ROLES.index("prev")
    roles[readout] = ROLES.index("yes")
    return roles


def per_token_file(path: str, item: dict, view: str, replay, sp: dict[str, int], special,
                   k: int = MASSIVE_K) -> dict:
    """One per-token file (``hidden`` [T, L, d] bf16, ``tokens`` [T]), read one layer at a time.

    Its item is rendered as the extraction renders it (whitebox/inkling_replay.py), and must give the file's tokens:
    otherwise the roles would be wrong, and the file is refused (SystemExit). Returns, per layer [L, T]:
    - ``share``: each token's top-k share;
    - ``rel_norm``: its norm over the episode's median token norm at that layer;
    - ``overlap``: how many of its top k are among "Yes"'s;
    - ``yes_dims_share``: the share of its squared norm in "Yes"'s top k.
    Also: ``ctx_ratio`` [2, L], "Yes"'s and the token before's norm over the median norm of the
    ``diagnostics.CONTEXT`` conversation tokens before the follow-up (the sink check's ratio), ``roles``, ``tokens``,
    ``readout`` and ``yes_hidden`` ("Yes"'s [L, d] as stored)."""
    from safetensors import safe_open

    from monitordecorrelation.whitebox.diagnostics import CONTEXT
    from monitordecorrelation.whitebox.probe_datasets import item_messages

    p = replay.probe_input(item_messages(item, view))
    with safe_open(path, framework="pt") as f:
        tokens = f.get_tensor("tokens").tolist()
        if tokens != p.tokens:
            raise SystemExit(f"{path}: its tokens are not those of {item['id']} rendered in {view}, so its tokens' "
                             "roles can't be told; refusing")
        hidden = f.get_slice("hidden")
        n_tok, n_layers, _ = hidden.get_shape()
        out = {"share": np.empty((n_layers, n_tok), np.float32), "rel_norm": np.empty((n_layers, n_tok), np.float32),
               "overlap": np.empty((n_layers, n_tok), np.int8),
               "yes_dims_share": np.empty((n_layers, n_tok), np.float32), "ctx_ratio": np.empty((2, n_layers))}
        ctx = slice(max(0, p.n_conversation_tokens - CONTEXT), p.n_conversation_tokens)
        for layer in range(n_layers):
            sq = squares(hidden[:, layer, :])  # [T, d]
            mask = top_k_mask(sq, k)
            norm = np.sqrt(sq.sum(1, dtype=np.float64))
            out["share"][layer] = top_k_share(sq, mask)
            out["rel_norm"][layer] = norm / np.median(norm)
            out["overlap"][layer] = (mask & mask[p.readout]).sum(1)
            out["yes_dims_share"][layer] = top_k_share(sq, np.broadcast_to(mask[p.readout], mask.shape))
            out["ctx_ratio"][:, layer] = norm[[p.readout, p.readout - 1]] / np.median(norm[ctx])
            del sq, mask
        out["yes_hidden"] = hidden[p.readout]
    out.update(roles=token_roles(p.tokens, p.readout, p.n_conversation_tokens, sp, special), tokens=tokens,
               readout=p.readout)
    return out


def token_summary(share, rel_norm, overlap, yes_dims_share, k: int) -> dict:
    """Over a group of tokens ([L, n] arrays, as per_token_file's), per layer and as band medians: the median top-k
    share, the share of tokens above DOMINATES, the median relative norm, the mean overlap with "Yes"'s top k (as a
    share of k), the share of tokens with exactly "Yes"'s top k, and the median share in "Yes"'s top k."""
    n = share.shape[1]
    if not n:
        return {"n": 0, "by_layer": None, "bands": None}
    by_layer = {"top_share": np.median(share, 1), "share_dominated": (share > DOMINATES).mean(1),
                "rel_norm": np.median(rel_norm, 1), "yes_overlap": overlap.mean(1) / k,
                "yes_same": (overlap == k).mean(1), "yes_dims_share": np.median(yes_dims_share, 1)}
    return {"n": int(n), "by_layer": {s: rounded(v) for s, v in by_layer.items()}, "bands": band_medians(by_layer)}


def per_token_paths(spec: str) -> list[str]:
    """The files of a comma-list of per-token files and folders (a folder: every ``full/*.safetensors`` below it)."""
    import glob

    paths = []
    for p in (s for s in spec.split(",") if s):
        if os.path.isdir(p):
            found = glob.glob(os.path.join(p, "**", "full", "*.safetensors"), recursive=True)
            if not found:
                raise SystemExit(f"--per-token {p}: no full/*.safetensors below it")
            paths += found
        elif os.path.isfile(p):
            paths.append(p)
        else:
            raise SystemExit(f"--per-token {p}: no such file or folder")
    return sorted({os.path.abspath(p) for p in paths})


def index_hashes(roots, dataset: str) -> dict[str, str | None]:
    """{index.json: the items_sha256 it recorded} for each features folder that has one for ``dataset``."""
    out = {}
    for root in roots:
        path = os.path.join(root, dataset, "index.json")
        if os.path.exists(path):
            with open(path) as fh:
                out[path] = (json.load(fh).get("dataset") or {}).get("items_sha256")
    return out


def shard_rows(features_view: str) -> dict[str, tuple[str, int]]:
    """id -> (shard file, row) over one view's complete shards (a .jsonl and its .safetensors)."""
    import glob

    out = {}
    for marker in sorted(glob.glob(os.path.join(features_view, "shard_*.jsonl"))):
        data = marker[: -len(".jsonl")] + ".safetensors"
        if os.path.exists(data):
            with open(marker) as fh:
                for row, line in enumerate(fh):
                    out[json.loads(line)["id"]] = (data, row)
    return out


def per_token(args) -> dict:
    """massive-dims' per-token part, one file at a time: per file, its roles and sink-check ratios; per role (and for
    the first token), token_summary over every file's tokens; the sink check's median ratios."""
    import torch
    from safetensors import safe_open

    from monitordecorrelation.whitebox.features import special_ids
    from monitordecorrelation.whitebox.inkling_replay import InklingReplay

    paths = per_token_paths(args.per_token)
    dataset = args.per_token_dataset
    check = args.per_token_features
    items = load_items(os.path.join(args.datasets, dataset), index_hashes([check] if check else [], dataset))
    replay = InklingReplay()
    sp, special = special_ids(replay), replay.renderer.tokenizer.is_special_token
    files, parts, rows, token_ids = {}, [], {}, set()
    for path in paths:
        view = os.path.basename(os.path.dirname(os.path.dirname(path)))
        item_id = os.path.basename(path)[: -len(".safetensors")].replace("__", "/")
        if os.path.basename(os.path.dirname(path)) != "full" or view not in VIEWS:
            raise SystemExit(f"{path}: not <view>/full/<item>.safetensors, with a view in {VIEWS}")
        if item_id not in items:
            raise SystemExit(f"{path}: {item_id} is not an item of {dataset}")
        r = per_token_file(path, items[item_id], view, replay, sp, special, args.top_k)
        entry = {"file": os.path.basename(path), "hack": is_hack(items[item_id]), "n_tokens": len(r["tokens"]),
                 "n_by_role": {role: int((r["roles"] == i).sum()) for i, role in enumerate(ROLES)},
                 "yes_position": r["readout"], "yes_ratio_to_context": rounded(r["ctx_ratio"][0]),
                 "prev_ratio_to_context": rounded(r["ctx_ratio"][1])}
        if check:
            if view not in rows:
                rows[view] = shard_rows(os.path.join(check, dataset, view))
            where = rows[view].get(item_id)
            if where is None:
                entry["yes_equals_features"] = None  # the item is not in those features
            else:
                with safe_open(where[0], framework="pt") as f:
                    entry["yes_equals_features"] = bool(torch.equal(f.get_slice("followup")[where[1]],
                                                                    r["yes_hidden"]))
        files[f"{view}/{item_id}"] = entry
        token_ids.update(r["tokens"])
        first = np.zeros(len(r["tokens"]), bool)
        first[0] = True
        parts.append({**{key: r[key] for key in ("share", "rel_norm", "overlap", "yes_dims_share", "roles",
                                                  "ctx_ratio")}, "first": first})
        print(f"[massive-dims] {view}/{item_id}: {len(r['tokens'])} tokens", file=sys.stderr, flush=True)
    cat = {key: np.concatenate([p[key] for p in parts], axis=-1) for key in
           ("share", "rel_norm", "overlap", "yes_dims_share", "roles", "first")}

    def summary(columns) -> dict:
        return token_summary(*(cat[key][:, columns] for key in ("share", "rel_norm", "overlap", "yes_dims_share")),
                             args.top_k)

    ratios = np.median(np.stack([p["ctx_ratio"] for p in parts]), axis=0)  # [2, L]: the median over files
    lo, hi = BANDS["10-35"]
    return {
        "sources": [os.path.abspath(s) for s in args.per_token.split(",") if s], "dataset": dataset,
        "n_files": len(files), "n_tokens": int(cat["roles"].size),
        "special_ids_seen": sorted(int(t) for t in token_ids if special(int(t))),
        "special_ids_from_200000": all((t >= 200000) == bool(special(int(t))) for t in token_ids),
        "roles": {role: summary(cat["roles"] == i) for i, role in enumerate(ROLES)},
        "first_token": summary(cat["first"]),
        "sink_check": {name: {"median_ratio_by_layer": rounded(ratios[j]),
                              "median_ratio_layers_10_35": float(np.median(ratios[j][lo: hi + 1]))
                              if hi < ratios.shape[1] else None}
                       for j, name in enumerate(("yes", "prev"))},
        "files": files,
    }


def readout_pair(features: str, features_prev: str, dataset: str, view: str, items: dict, k: int,
                 max_shards: int | None) -> dict:
    """"Yes" (``followup``) against the token before (``followup_prev``) on one reader's per-item readouts of one
    dataset view, read one shard at a time: per layer and as band medians, each readout's median top-k share, the
    median norm ratio ‖"Yes"‖ / ‖token before‖, and the mean share of the top k they have in common."""
    from safetensors import safe_open

    a = LayerSource(os.path.join(features, dataset), view, "followup")
    b = LayerSource(os.path.join(features_prev, dataset), view, "followup_prev")
    if a.ids != b.ids or [n for _, n, _ in a.shards] != [n for _, n, _ in b.shards]:
        raise SystemExit(f"{dataset}/{view}: the followup and followup_prev shards don't hold the same items in the "
                         "same order")
    missing = [i for i in a.ids if i not in items]
    if missing:
        raise SystemExit(f"{dataset}/{view}: {len(missing)} feature rows are not items of {dataset}, e.g. {missing[0]}")
    shards = list(zip(a.shards, b.shards))[:max_shards]
    n = sum(rows for (_, rows, _), _ in shards)
    stats = {key: np.empty((a.n_layers, n)) for key in ("followup", "followup_prev", "ratio", "common")}
    start = 0
    for (path_a, rows, _), (path_b, _, _) in shards:
        cols = slice(start, start + rows)
        with safe_open(path_a, framework="pt") as fa, safe_open(path_b, framework="pt") as fb:
            ya, yb = fa.get_tensor("followup"), fb.get_tensor("followup_prev")  # [rows, L, d] bf16
            for layer in range(a.n_layers):
                qa, qb = squares(ya[:, layer]), squares(yb[:, layer])
                ma, mb = top_k_mask(qa, k), top_k_mask(qb, k)
                stats["followup"][layer, cols] = top_k_share(qa, ma)
                stats["followup_prev"][layer, cols] = top_k_share(qb, mb)
                stats["ratio"][layer, cols] = np.sqrt(qa.sum(1, dtype=np.float64) / qb.sum(1, dtype=np.float64))
                stats["common"][layer, cols] = (ma & mb).sum(1) / k
            del ya, yb
        start += rows
    by_layer = {"top_share_followup": np.median(stats["followup"], 1),
                "top_share_followup_prev": np.median(stats["followup_prev"], 1),
                "norm_ratio": np.median(stats["ratio"], 1), "top_dims_in_common": stats["common"].mean(1)}
    return {"n": n, "shards": len(shards), "of_shards": len(a.shards),
            "by_layer": {s: rounded(v) for s, v in by_layer.items()}, "bands": band_medians(by_layer)}


def readouts(args) -> dict:
    """massive-dims' per-item part: readout_pair for every view of ``--readout-datasets`` both feature folders hold."""
    prev = args.features_prev or args.features
    out = {}
    for dataset in (d for d in args.readout_datasets.split(",") if d):
        views = [v for v in VIEWS
                 if all(os.path.isdir(os.path.join(root, dataset, v)) for root in (args.features, prev))]
        if not views:
            raise SystemExit(f"--readout-datasets {dataset}: neither view is in both {args.features} and {prev}")
        items = load_items(os.path.join(args.datasets, dataset), index_hashes([args.features, prev], dataset))
        for view in views:
            out[f"{dataset}/{view}"] = readout_pair(args.features, prev, dataset, view, items, args.top_k,
                                                    args.max_shards)
            print(f"[massive-dims] {dataset}/{view}: {out[f'{dataset}/{view}']['n']} items", file=sys.stderr,
                  flush=True)
    return out


def massive_dims(args) -> dict:
    out: dict = {"k": args.top_k, "dominates": DOMINATES, "bands": {b: list(r) for b, r in BANDS.items()}}
    if args.per_token:
        out["per_token"] = per_token(args)
    if args.features:
        out["readouts"] = {"features": os.path.abspath(args.features),
                           "features_prev": os.path.abspath(args.features_prev or args.features),
                           "max_shards": args.max_shards, "views": readouts(args)}
    return out


def test_scores(probe_dir: str, items: dict[str, dict], ood: bool) -> dict[str, tuple[float, int, object]]:
    """A fitted probe's test items from its scores.jsonl, by id: (test logit, label, cluster), as fit_probes.py's test
    AUROC reads them: OOD, the deceptive label and the pair; IID, ``used_verifier`` and the task. Labels come from the
    items, and a row that says otherwise is refused."""
    out = {}
    with open(os.path.join(probe_dir, "scores.jsonl")) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("split") != "test" or row.get("test_logit") is None:
                continue
            it = items.get(row["id"])
            if it is None:
                raise SystemExit(f"{probe_dir}: {row['id']} is not an item of the dataset")
            label = int(it["label"]) if ood else int(is_hack(it))
            if int(row["label" if ood else "used_verifier"]) != label:
                raise SystemExit(f"{probe_dir}: {row['id']}'s label in scores.jsonl is not its item's")
            out[row["id"]] = (float(row["test_logit"]), label, it["pair"] if ood else it["task_id"])
    return out


def readout_gap(probes: str, readouts: list[str], datasets: str, n_boot: int, seed: int) -> dict:
    """The test AUROC difference between two readouts' probes (the second minus the first), with its paired cluster
    bootstrap, and Spearman between their test logits."""
    from scipy.stats import spearmanr

    dirs = [os.path.join(probes, r) for r in readouts]
    metas = []
    for d, readout in zip(dirs, readouts):
        with open(os.path.join(d, "meta.json")) as fh:
            metas.append(json.load(fh))
        if metas[-1].get("readout", readout) != readout:
            raise SystemExit(f"{d}: its meta.json says readout {metas[-1]['readout']!r}, not {readout!r}")
    dataset, view = metas[0].get("dataset"), metas[0].get("view")
    if dataset not in ("diverse_deception", "terminal_prerl") or \
            any((m.get("dataset"), m.get("view")) != (dataset, view) for m in metas):
        raise SystemExit(f"{probes}: want two probes of one dataset (diverse_deception or terminal_prerl) and view, "
                         f"got {[(m.get('dataset'), m.get('view')) for m in metas]}")
    ood = dataset == "diverse_deception"
    items = load_items(os.path.join(datasets, dataset))
    scores = [test_scores(d, items, ood) for d in dirs]
    if set(scores[0]) != set(scores[1]):
        raise SystemExit(f"{probes}: the two probes' test items differ; a paired comparison needs the same items")
    ids = sorted(scores[0])
    a, b = (np.array([s[i][0] for i in ids]) for s in scores)
    y = np.array([scores[0][i][1] for i in ids])
    groups = np.array([scores[0][i][2] for i in ids])
    test_auroc = {r: auroc(z, y) for r, z in zip(readouts, (a, b))}
    for d, r in zip(dirs, readouts):
        path = os.path.join(d, "metrics.json")
        if os.path.exists(path):
            with open(path) as fh:
                recorded = json.load(fh).get("test_auroc")
            if recorded is not None and abs(recorded - test_auroc[r]) > 1e-9:
                raise SystemExit(f"{d}: test AUROC {test_auroc[r]} from scores.jsonl, but {recorded} in metrics.json")
    boot = cluster_bootstrap(lambda x, z, yy: auroc(z, yy) - auroc(x, yy), groups, a, b, y, n=n_boot, seed=seed)
    ok = boot[~np.isnan(boot)]
    return {"dataset": dataset, "view": view, "cluster": "pair" if ood else "task_id", "n": len(ids),
            "n_positive": int(y.sum()), "n_clusters": len(np.unique(groups)),
            "layers": {r: m.get("layer") for r, m in zip(readouts, metas)}, "test_auroc": test_auroc,
            "difference": test_auroc[readouts[1]] - test_auroc[readouts[0]],
            "bootstrap": {"n": n_boot, "seed": seed, "n_nan": int(len(boot) - len(ok)),
                          "sd": float(np.std(ok)) if len(ok) else float("nan"), "ci": ci(boot),
                          "share_above_zero": float(np.mean(ok > 0)) if len(ok) else float("nan")},
            "spearman": float(spearmanr(a, b).statistic)}


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
    md = sub.add_parser("massive-dims", help="massive dimensions at every token by role, and \"Yes\" vs the token "
                                             "before on the per-item readouts")
    md.add_argument("--per-token", help="comma-list of per-token files and folders (a folder: every "
                                        "<view>/full/*.safetensors below it)")
    md.add_argument("--per-token-dataset", help="the dataset the per-token files' items are of (default: "
                                                "terminal_prerl)")
    md.add_argument("--per-token-features", help="the per-token files' reader's features (<dataset>/<view>/shard_*): "
                                                 "check each file's \"Yes\" against its item's followup readout")
    md.add_argument("--features", help="per-item followup readouts: <dataset>/<view>/shard_*")
    md.add_argument("--features-prev", help="per-item followup_prev readouts, if not in --features")
    md.add_argument("--readout-datasets", help="comma-list: the datasets whose per-item readouts are compared, in "
                                               "every view found (default: terminal_prerl,diverse_deception)")
    md.add_argument("--max-shards", type=int, default=None, help="only each view's first N shards of the readouts")
    md.add_argument("--top-k", type=int, default=MASSIVE_K, help="the top dimensions counted per token")
    rg = sub.add_parser("readout-gap", help="paired cluster bootstrap of the test AUROC difference between two "
                                            "readouts' probes")
    rg.add_argument("--probes", required=True, help="comma-list of fit_probes.py folders <reader>/<dataset>/<view>, "
                                                    "each with a probe folder per readout")
    rg.add_argument("--readouts", default="followup,followup_prev", help="A,B: the difference is B's AUROC − A's")
    rg.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    rg.add_argument("--seed", type=int, default=0, help="the bootstrap's seed")
    for p in (ds, nt, md, rg):
        p.add_argument("--datasets", default="data/probe_datasets", help="folder with the dataset folders")
        p.add_argument("--out", required=True, help="output folder, e.g. OUT/analyses/<UTC timestamp>")
        p.add_argument("--overwrite", action="store_true", help="replace this subcommand's file in --out")
    args = ap.parse_args(argv)
    if args.command == "massive-dims":
        if not (args.per_token or args.features):
            ap.error("massive-dims: give --per-token, --features or both")
        ignored = [f"{flag} needs {part}" for flag, given, part in (
            ("--per-token-dataset", args.per_token_dataset, "--per-token"),
            ("--per-token-features", args.per_token_features, "--per-token"),
            ("--features-prev", args.features_prev, "--features"),
            ("--readout-datasets", args.readout_datasets, "--features"),
            ("--max-shards", args.max_shards is not None, "--features"))
            if given and not getattr(args, part[2:].replace("-", "_"))]
        if ignored:
            ap.error(f"massive-dims: {'; '.join(ignored)} (it would be ignored)")
        args.per_token_dataset = args.per_token_dataset or "terminal_prerl"
        args.readout_datasets = args.readout_datasets or "terminal_prerl,diverse_deception"
    if args.command == "readout-gap":
        pair = [r for r in args.readouts.split(",") if r]
        if len(pair) != 2 or pair[0] == pair[1]:
            ap.error(f"readout-gap: --readouts wants two different readouts A,B, got {args.readouts!r}")

    path = os.path.join(args.out, args.command.replace("-", "_") + ".json")
    if os.path.exists(path) and not args.overwrite:
        ap.error(f"{path} exists; pick a new --out, or --overwrite replaces it")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    code = code_version()
    if args.command == "dataset-stats":
        result = dataset_stats(args)
    elif args.command == "near-threshold":
        items = load_items(os.path.join(args.datasets, "terminal_prerl"))
        result = {"features": os.path.abspath(args.features), "k": args.k,
                  "probes": {p: near_threshold(p, args.features, items, args.k) for p in args.probes.split(",") if p}}
    elif args.command == "massive-dims":
        result = massive_dims(args)
    else:
        result = {"readouts": pair, "difference": f"{pair[1]} - {pair[0]}",
                  "probes": {os.path.abspath(p): readout_gap(p, pair, args.datasets, args.n_boot, args.seed)
                             for p in args.probes.split(",") if p}}
    result = {"command": args.command, "started_utc": started, "code": code, **result}
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
