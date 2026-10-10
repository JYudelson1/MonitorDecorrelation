"""Fitting and evaluating Inkling probes on extracted features, as pre-registered in PROBE_PLAN.md section 5b.

This runs on hub's CPUs; ``experiments/fit_probes.py`` is the command line.

- **Loading:** a reader's features for a dataset view come from shards (whitebox/features.py), joined to the dataset's
  items by id for labels, tasks and diagnostics.
- **Feature matrix** (per readout):
  - ``followup`` / ``followup_prev``: the readout itself;
  - ``mean``: the count-weighted mean over thinking + text in the with-reasoning view, and over text otherwise. Rows
    with a zero count are dropped.
- **Cross-validation:** ``cv_grid`` gives out-of-fold logits for every (layer, C) under StratifiedGroupKFold(5) × 3,
  grouped by task (or pair). Each fold standardizes, then fits an L2 logistic regression with balanced class weights,
  warm-started along increasing C.
- **Selection** (``select``): the best mean out-of-fold AUROC; then, within one task-bootstrap SE of it, the smallest
  C; then the best tie-break metric and log-loss.
- **Refit:** the chosen layer and C, refit on all of dev, give a ``SelectedProbe``: raw-space weights, so it scores raw
  activations. Its deployed score is σ(logit − t₁%).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FLAT_FALLBACK_C = 1e-3


# ------------------------------------------------------------------------------------------------ metrics


def auroc(scores, labels) -> float:
    """Rank AUROC with ties averaged; NaN without both classes."""
    from scipy.stats import rankdata

    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cluster_bootstrap(stat, groups, *arrays, n: int = 2000, seed: int = 0) -> np.ndarray:
    """``stat(*resampled arrays)`` over ``n`` resamples of whole clusters (tasks or pairs), with replacement."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        idx = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        out[b] = stat(*[np.asarray(a)[idx] for a in arrays])
    return out


def ci(values: np.ndarray) -> list[float]:
    v = values[~np.isnan(values)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def within_group_auroc(scores, labels, groups) -> float:
    """AUROC within each group that has both classes, weighted by its number of (positive, negative) pairs."""
    scores, labels, groups = map(np.asarray, (scores, labels, groups))
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        p, n = int(labels[m].sum()), int((~labels[m].astype(bool)).sum())
        if p and n:
            num += auroc(scores[m], labels[m]) * p * n
            den += p * n
    return num / den if den else float("nan")


# ------------------------------------------------------------------------------------------------ loading


def load_items(dataset_dir: str) -> dict[str, dict]:
    with open(os.path.join(dataset_dir, "items.jsonl")) as fh:
        return {it["id"]: it for it in map(json.loads, fh)}


@dataclass
class ViewFeatures:
    """One reader's features for one dataset view, rows in shard order."""
    ids: list[str]
    tensors: dict  # key -> torch tensor [n, ...]
    records: list[dict]


READOUT_TENSORS = {"followup": ("followup",), "followup_prev": ("followup_prev",),
                   "mean": ("mean_reasoning", "mean_action")}
SMALL_TENSORS = ("counts", "answer_logp")


def load_view(features_dir: str, view: str, keys=("followup", "followup_prev", "mean_reasoning", "mean_action",
                                                   "counts", "answer_logp", "router_logits", "router_topk"),
              layers: list[int] | None = None) -> ViewFeatures:
    """Every complete shard of ``<features_dir>/<view>/`` (marker and data), concatenated in shard order.

    ``keys`` limits what is read; ``layers`` reads only those layers of the [n, L, d] tensors (memory: one view's
    every tensor in full is ~23 GB)."""
    import torch
    from safetensors import safe_open

    parts: dict[str, list] = {}
    ids, records = [], []
    for marker in sorted(glob.glob(os.path.join(features_dir, view, "shard_*.jsonl"))):
        data = marker[: -len(".jsonl")] + ".safetensors"
        if not os.path.exists(data):
            continue
        with open(marker) as fh:
            recs = [json.loads(line) for line in fh]
        with safe_open(data, framework="pt") as f:
            have = set(f.keys())
            for k in keys:
                if k in have:
                    if layers is not None and len(f.get_slice(k).get_shape()) == 3:
                        sl = f.get_slice(k)
                        parts.setdefault(k, []).append(torch.stack([sl[:, layer, :] for layer in layers], dim=1))
                    else:
                        parts.setdefault(k, []).append(f.get_tensor(k))
        records += recs
        ids += [r["id"] for r in recs]
    if not ids:
        raise FileNotFoundError(f"no complete shards under {os.path.join(features_dir, view)}")
    n_complete = sum(os.path.exists(m[: -len(".jsonl")] + ".safetensors")
                     for m in glob.glob(os.path.join(features_dir, view, "shard_*.jsonl")))
    tensors = {}
    for k, chunks in parts.items():  # into one preallocated tensor: torch.cat would hold everything twice
        if len(chunks) != n_complete:
            continue
        out = torch.empty((sum(len(c) for c in chunks), *chunks[0].shape[1:]), dtype=chunks[0].dtype)
        i = 0
        while chunks:
            c = chunks.pop(0)
            out[i: i + len(c)] = c
            i += len(c)
        tensors[k] = out
    return ViewFeatures(ids, tensors, records)


def feature_matrix(vf: ViewFeatures, readout: str, view: str):
    """(X as a torch tensor [n, L, d] in its stored dtype, keep mask [n]) for ``readout``."""
    import torch

    if readout in ("followup", "followup_prev"):
        x = vf.tensors[readout]
        return x, np.ones(len(x), bool)
    if readout != "mean":
        raise ValueError(f"unknown readout {readout!r}")
    counts = vf.tensors["counts"].numpy().astype(float)  # [n, 2]: thinking, text
    if view == "with_reasoning" and "mean_reasoning" in vf.tensors:
        w = counts / np.clip(counts.sum(1, keepdims=True), 1, None)
        x = (vf.tensors["mean_reasoning"] * torch.tensor(w[:, :1, None], dtype=torch.float32)
             + vf.tensors["mean_action"] * torch.tensor(w[:, 1:, None], dtype=torch.float32))
        keep = counts.sum(1) > 0
    else:
        x = vf.tensors["mean_action"]
        keep = counts[:, 1] > 0
    return x, keep


def layer_array(x, layer: int, rows=None) -> np.ndarray:
    """One layer as float64 numpy (rows optional). The layer is taken first: ``x[rows]`` would copy every layer."""
    import torch

    sl = x[:, layer, :]
    if rows is not None:
        sl = sl[torch.as_tensor(np.asarray(rows), dtype=torch.long)]
    return sl.float().numpy().astype(np.float64)


# ------------------------------------------------------------------------------------------------ fitting


def make_folds(n: int, strat, groups, n_splits: int = 5, repeats: int = 3, seed: int = 0) -> list:
    from sklearn.model_selection import StratifiedGroupKFold

    return [list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + r).split(np.zeros(n), strat, groups))
            for r in range(repeats)]


def _fit_path(x: np.ndarray, y: np.ndarray, folds: list, c_grid) -> np.ndarray:
    """Out-of-fold logits [len(c_grid), repeats, n] for one layer (``x`` arrives as float32 to halve the transfer)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=np.float64)
    out = np.full((len(c_grid), len(folds), len(y)), np.nan, np.float32)
    for r, fold in enumerate(folds):
        for tr, te in fold:
            scaler = StandardScaler().fit(x[tr])
            xtr, xte = scaler.transform(x[tr]), scaler.transform(x[te])
            clf = LogisticRegression(C=c_grid[0], class_weight="balanced", max_iter=5000, tol=1e-4, warm_start=True)
            for ci_, c in enumerate(c_grid):  # increasing C, each fit starting from the last
                clf.set_params(C=c)
                clf.fit(xtr, y[tr])
                out[ci_, r, te] = clf.decision_function(xte)
    return out


def cv_grid(x, rows, y, folds, layers, c_grid=C_GRID, n_jobs: int = -1) -> np.ndarray:
    """Out-of-fold logits [len(layers), len(c_grid), repeats, n] over ``rows`` of the [n, L, d] tensor ``x``."""
    from joblib import Parallel, delayed, parallel_config

    # one queued layer per worker (a float32 [n, d] each), so memory stays at ~2.5 GB per worker
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=n_jobs, pre_dispatch="n_jobs")(
            delayed(_fit_path)(layer_array(x, layer, rows).astype(np.float32), y, folds, c_grid) for layer in layers)
    return np.stack(res)


@dataclass
class Selection:
    layer: int
    C: float
    cv_auroc: float
    cv_se: float
    flat: bool
    grid: list = field(default_factory=list)  # rows: layer, C, mean AUROC, sd over repeats


def select(oof: np.ndarray, y, groups, layers, c_grid=C_GRID, tie_metric=None, seed: int = 0) -> Selection:
    """The pre-registered rule over out-of-fold logits [layers, C, repeats, n] (see the module docstring)."""
    n_l, n_c, n_r, _ = oof.shape
    auc = np.array([[[auroc(oof[i, j, r], y) for r in range(n_r)] for j in range(n_c)] for i in range(n_l)])
    mean = np.nanmean(auc, axis=-1)
    grid = [{"layer": int(layers[i]), "C": float(c_grid[j]), "cv_auroc": float(mean[i, j]),
             "cv_auroc_sd": float(np.nanstd(auc[i, j]))} for i in range(n_l) for j in range(n_c)]
    bi, bj = np.unravel_index(np.nanargmax(mean), mean.shape)
    best = float(mean[bi, bj])
    if best - float(np.nanmin(mean)) < 0.005:  # a flat grid: nothing to choose by
        mid = n_l // 2
        return Selection(int(layers[mid]), FLAT_FALLBACK_C, float(mean[mid, list(c_grid).index(FLAT_FALLBACK_C)]),
                         float("nan"), True, grid)
    score_best = oof[bi, bj].mean(0)
    se = float(np.nanstd(cluster_bootstrap(auroc, groups, score_best, y, n=200, seed=seed)))
    cand = [(i, j) for i in range(n_l) for j in range(n_c) if mean[i, j] >= best - se]
    c_min = min(c_grid[j] for _, j in cand)
    cand = [(i, j) for i, j in cand if c_grid[j] == c_min]

    def key(ij):
        i, j = ij
        s = oof[i, j].mean(0)
        tie = tie_metric(s) if tie_metric is not None else 0.0
        p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
        ll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        return (np.nan_to_num(tie, nan=-1.0), -ll, mean[i, j])

    i, j = max(cand, key=key)
    return Selection(int(layers[i]), float(c_grid[j]), float(mean[i, j]), se, False, grid)


@dataclass
class SelectedProbe:
    """A single-layer linear probe in raw activation space: logit = coef · x[layer] + intercept."""
    layer: int
    coef: np.ndarray
    intercept: float
    threshold_1: float = float("nan")  # t₁%: the deployed score is σ(logit − t₁%)
    threshold_5: float = float("nan")
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, x_layer: np.ndarray, y: np.ndarray, C: float, layer: int) -> "SelectedProbe":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(x_layer)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, tol=1e-4).fit(scaler.transform(x_layer), y)
        w = clf.coef_[0] / scaler.scale_
        return cls(layer, w, float(clf.intercept_[0] - np.sum(clf.coef_[0] * scaler.mean_ / scaler.scale_)))

    def logit(self, x_layer: np.ndarray) -> np.ndarray:
        return x_layer @ self.coef + self.intercept

    def score(self, x_layer: np.ndarray) -> np.ndarray:
        """The deployed score: σ(logit − t₁%), so 0.5 is the 1%-FPR operating point."""
        z = np.clip(self.logit(x_layer) - self.threshold_1, -60, 60)
        return 1 / (1 + np.exp(-z))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, "probe.npz"), coef=self.coef.astype(np.float64),
                 intercept=np.array([self.intercept]), layer=np.array([self.layer]),
                 thresholds=np.array([self.threshold_1, self.threshold_5]))
        with open(os.path.join(path, "meta.json"), "w") as fh:  # the probe's own fields win over a loaded meta's
            json.dump({**self.meta, "layer": self.layer, "threshold_1": self.threshold_1,
                       "threshold_5": self.threshold_5, "score": "sigmoid(coef . x[layer] + intercept - threshold_1)"},
                      fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "SelectedProbe":
        z = np.load(os.path.join(path, "probe.npz"))
        meta = json.load(open(os.path.join(path, "meta.json")))
        return cls(int(z["layer"][0]), z["coef"], float(z["intercept"][0]), float(z["thresholds"][0]),
                   float(z["thresholds"][1]), meta)


def thresholds(negative_logits: np.ndarray) -> tuple[float, float]:
    """t₁% and t₅%: the 99th and 95th percentiles of honest episodes' logits."""
    v = np.asarray(negative_logits, float)
    return float(np.percentile(v, 99)), float(np.percentile(v, 95))


def write_golden(probe_dir: str, probe: SelectedProbe, vf, items: dict, readout: str, view: str,
                 ids: list[str], effort: float = 0.5) -> int:
    """``golden.jsonl`` for the probe server's startup check (whitebox/probe_server.py).

    For each of ``ids`` it stores the episode's replay fields, the sha256 of its rendering, and its logit on the stored
    features. The rendering is redone here with the same replay, so hub and server must agree on it too."""
    from monitordecorrelation.whitebox.inkling_replay import FOLLOW_UP, InklingReplay
    from monitordecorrelation.whitebox.probe_datasets import item_messages
    from monitordecorrelation.whitebox.probe_server import conversation_sha

    replay = InklingReplay(effort=effort, follow_up=FOLLOW_UP)
    if isinstance(vf, LayerSource):  # read the probe's layer from disk
        x_layer, keep = vf.layer(probe.layer), vf.keep
    else:
        x, keep = feature_matrix(vf, readout, view)
        x_layer = layer_array(x, probe.layer)
    row_of = {i: r for r, i in enumerate(vf.ids)}
    n = 0
    with open(os.path.join(probe_dir, "golden.jsonl"), "w") as fh:
        for item_id in ids:
            r = row_of.get(item_id)
            if r is None or not keep[r]:
                continue
            it = items[item_id]
            episode = {"key": item_id, "question": it["question"], "turns": it["turns"],
                       "env_options": it["env_options"]}
            tokens = replay.probe_input(item_messages(episode, view)).tokens
            logit = float(probe.logit(x_layer[r: r + 1])[0])
            fh.write(json.dumps({"episode": episode, "conversation_sha256": conversation_sha(tokens),
                                 "logit": logit}) + "\n")
            n += 1
    return n


class LayerSource:
    """One reader's features for one dataset view and readout, read from the shard files one layer at a time.

    Hub's pod has only ~6 GB of memory to spare, and one view's every tensor is ~23 GB. So nothing is held in full:
    ``layer(l, rows)`` reads layer ``l`` of every shard (``safe_open`` slices) and returns float64 [rows, d].

    It is picklable (paths only), so each joblib worker loads its own layer. The readouts follow ``feature_matrix``:
    ``mean`` is the count-weighted thinking + text mean in the with-reasoning view, the text mean otherwise, and
    rows with a zero count are not kept."""

    def __init__(self, features_dir: str, view: str, readout: str) -> None:
        import torch
        from safetensors import safe_open

        if readout not in READOUT_TENSORS:
            raise ValueError(f"unknown readout {readout!r}")
        self.view, self.readout = view, readout
        self.shards, self.ids, self.records = [], [], []
        counts, answer = [], []
        for marker in sorted(glob.glob(os.path.join(features_dir, view, "shard_*.jsonl"))):
            data = marker[: -len(".jsonl")] + ".safetensors"
            if not os.path.exists(data):
                continue
            with open(marker) as fh:
                recs = [json.loads(line) for line in fh]
            with safe_open(data, framework="pt") as f:
                keys = set(f.keys())
                counts.append(f.get_tensor("counts"))
                answer.append(f.get_tensor("answer_logp") if "answer_logp" in keys else None)
                shape = f.get_slice(READOUT_TENSORS[readout][-1]).get_shape()
            self.shards.append((data, len(recs), keys))
            self.records += recs
            self.ids += [r["id"] for r in recs]
        if not self.shards:
            raise FileNotFoundError(f"no complete shards under {os.path.join(features_dir, view)}")
        self.n_layers, self.d = int(shape[1]), int(shape[2])
        self.counts = torch.cat(counts).numpy().astype(float)
        self.answer_logp = torch.cat(answer).numpy() if all(a is not None for a in answer) else None
        if readout == "mean":
            with_r = view == "with_reasoning" and all("mean_reasoning" in k for _, _, k in self.shards)
            self.keep = self.counts.sum(1) > 0 if with_r else self.counts[:, 1] > 0
        else:
            self.keep = np.ones(len(self.ids), bool)

    def layer(self, layer: int, rows=None) -> np.ndarray:
        import torch
        from safetensors import safe_open

        parts, start = [], 0
        for data, n, keys in self.shards:
            with safe_open(data, framework="pt") as f:
                if self.readout == "mean":
                    a = f.get_slice("mean_action")[:, layer, :].float()
                    if self.view == "with_reasoning" and "mean_reasoning" in keys:
                        c = torch.tensor(self.counts[start: start + n])
                        w = c / c.sum(1, keepdim=True).clamp(min=1)
                        a = f.get_slice("mean_reasoning")[:, layer, :].float() * w[:, :1] + a * w[:, 1:]
                    parts.append(a.numpy())
                else:
                    parts.append(f.get_slice(self.readout)[:, layer, :].float().numpy())
            start += n
        out = np.concatenate(parts).astype(np.float64)
        return out if rows is None else out[np.asarray(rows)]


def _fit_path_from(src: "LayerSource", layer: int, rows, y, folds, c_grid) -> np.ndarray:
    return _fit_path(src.layer(layer, rows), y, folds, c_grid)


def cv_grid_source(src: "LayerSource", rows, y, folds, layers, c_grid=C_GRID, n_jobs: int = 6) -> np.ndarray:
    """``cv_grid`` with each worker reading its own layer from disk: out-of-fold logits [layers, C, repeats, n]."""
    from joblib import Parallel, delayed, parallel_config

    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=n_jobs, pre_dispatch="n_jobs")(
            delayed(_fit_path_from)(src, layer, rows, y, folds, c_grid) for layer in layers)
    return np.stack(res)
