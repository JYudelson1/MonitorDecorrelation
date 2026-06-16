"""``LinearProbe``: per-layer logistic regression on residual-stream activations (Atlas §4.3).

Fit one logistic regression per layer, drop layers whose train cross-entropy exceeds a threshold
(default 0.6 — "weak" layers), and score a rollout as the **mean sigmoid over the kept layers**. The
probe is agnostic to which dataset(s) produced the activations; that choice lives in the trainer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from monitordecorrelation.eval.metrics import roc_auc

_CE_THRESHOLD = 0.6  # keep layers with train cross-entropy below this


def _fit_layer(acts: np.ndarray, layer: int, y: np.ndarray, max_iter: int, tol: float):
    """Fit one standardized per-layer LR, fold the scaler into raw-space weights, return (clf, ce).
    Module-level (not a closure) so joblib can ship it to worker processes; takes the full ``acts`` as
    an arg so joblib memmaps it once and shares it read-only across workers instead of copying."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(acts[:, layer, :], dtype=np.float64)  # bf16-derived floats → f64 for a stable solve
    scaler = StandardScaler().fit(x)                      # scale_ is 1.0 for zero-variance cols (no div0)
    clf = LogisticRegression(max_iter=max_iter, tol=tol).fit(scaler.transform(x), y)
    # fold standardization into raw-space weights: logit = w·(x−μ)/σ + b
    #   ⇒ raw coef = w/σ,  raw intercept = b − Σ w·μ/σ   (so score/save/load use RAW activations)
    w, mu, sigma = clf.coef_[0], scaler.mean_, scaler.scale_
    clf.coef_ = (w / sigma).reshape(1, -1)
    clf.intercept_ = np.array([clf.intercept_[0] - float(np.sum(w * mu / sigma))])
    ce = float(log_loss(y, clf.predict_proba(x)[:, 1], labels=[0, 1]))  # CE on raw x (post-fold)
    return clf, ce


class LinearProbe:
    def __init__(self, threshold: float = 0.5, ce_threshold: float = _CE_THRESHOLD) -> None:
        self.threshold = threshold
        self.ce_threshold = ce_threshold
        self._models: dict[int, object] = {}  # layer -> fitted LogisticRegression
        self.kept_layers: list[int] = []
        self.layer_ce: dict[int, float] = {}
        self.meta: dict = {}

    def fit(self, acts: np.ndarray, labels, *, progress: bool = False, n_jobs: int = 1) -> "LinearProbe":
        """``acts`` [n, L, d], ``labels`` [n] in {0,1}. Fits a LR per layer, keeps low-CE layers.
        ``progress`` shows a tqdm bar over the per-layer fits; ``n_jobs`` parallelizes the (independent)
        layer fits across CPU cores (the trainer passes ``-1`` = all cores; default 1 keeps tests fast).

        Each layer is standardized then its scaler folded into raw-space weights (see ``_fit_layer``) +
        ``tol=1e-3``: high-dim activations are near-separable + collinear, so the default ``tol=1e-4``
        makes lbfgs grind to max_iter every layer for precision a probe doesn't need. score/save/load
        operate on RAW activations — no scaler to persist.
        """
        from contextlib import nullcontext

        from joblib import Parallel, delayed, parallel_config

        y = np.asarray(labels).astype(int)
        n, n_layers, _ = acts.shape
        self._models, self.layer_ce = {}, {}
        tasks = (delayed(_fit_layer)(acts, layer, y, 2000, 1e-3) for layer in range(n_layers))
        # When parallelizing, pin each worker's BLAS to one thread (else N workers × BLAS threads
        # oversubscribe the cores); inner_max_num_threads needs an explicit backend. For n_jobs=1 skip
        # it entirely so the default stays a plain in-process run (fast for tests). return_as generator
        # yields in submission order → enumerate gives the layer index + a tqdm bar that ticks on COMPLETE.
        ctx = (parallel_config(backend="loky", inner_max_num_threads=1)
               if n_jobs not in (1, None) else nullcontext())
        with ctx:
            gen = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
            if progress:
                from tqdm import tqdm

                gen = tqdm(gen, total=n_layers, desc="fit per-layer LR", unit="layer")
            for layer, (clf, ce) in enumerate(gen):
                self._models[layer] = clf
                self.layer_ce[layer] = ce
        self.kept_layers = [l for l, ce in self.layer_ce.items() if ce < self.ce_threshold]
        if not self.kept_layers:
            # Tiny models may clear no layer; keep the single best so the probe still functions.
            best = min(self.layer_ce, key=self.layer_ce.get)
            self.kept_layers = [best]
            self.meta["no_layer_below_ce"] = True
        self.meta.update({"n_layers": n_layers, "d_model": int(acts.shape[2]), "n_train": int(n)})
        return self

    def score(self, acts: np.ndarray) -> np.ndarray:
        """-> [n] mean P(deceptive) over kept layers."""
        if not self.kept_layers:
            raise RuntimeError("probe not fitted")
        cols = [self._models[l].predict_proba(acts[:, l, :])[:, 1] for l in self.kept_layers]
        return np.mean(np.stack(cols, axis=1), axis=1)

    def evaluate(self, acts: np.ndarray, labels) -> float:
        """Held-out ROC-AUC (reuses the project's NaN-safe rank AUROC)."""
        scores = self.score(acts).tolist()
        return roc_auc(scores, [bool(v) for v in np.asarray(labels).astype(int)])

    # --- persistence: npz of per-layer weights + sibling meta.json ---
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for layer in self.kept_layers:
            clf = self._models[layer]
            arrays[f"coef_{layer}"] = clf.coef_
            arrays[f"intercept_{layer}"] = clf.intercept_
        np.savez(path / "probe.npz", **arrays)
        meta = {
            "threshold": self.threshold,
            "ce_threshold": self.ce_threshold,
            "kept_layers": self.kept_layers,
            "layer_ce": self.layer_ce,
            **self.meta,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe":
        from sklearn.linear_model import LogisticRegression

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        probe = cls(threshold=meta["threshold"], ce_threshold=meta.get("ce_threshold", _CE_THRESHOLD))
        probe.kept_layers = list(meta["kept_layers"])
        probe.layer_ce = {int(k): v for k, v in meta.get("layer_ce", {}).items()}
        probe.meta = {k: meta[k] for k in meta if k not in ("threshold", "ce_threshold", "kept_layers", "layer_ce")}
        npz = np.load(path / "probe.npz")
        for layer in probe.kept_layers:
            clf = LogisticRegression()
            clf.coef_ = npz[f"coef_{layer}"]
            clf.intercept_ = npz[f"intercept_{layer}"]
            clf.classes_ = np.array([0, 1])
            probe._models[layer] = clf
        return probe

    # --- HuggingFace Hub: push/pull a probe for reproducibility + sharing ---
    def push_to_hub(
        self, repo_id: str, *, private: bool = True, repo_type: str = "model", token: str | None = None
    ) -> str:
        """Upload this probe (npz + meta) to a HF repo. The meta records the base model, datasets, and
        follow-up, so a pulled probe is self-describing. Needs `HF_TOKEN` (or pass `token`)."""
        import tempfile

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, repo_type=repo_type, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            self.save(d)
            api.upload_folder(folder_path=d, repo_id=repo_id, repo_type=repo_type)
        return f"https://huggingface.co/{repo_id}"

    @classmethod
    def from_hub(cls, repo_id: str, *, repo_type: str = "model", token: str | None = None) -> "LinearProbe":
        """Download + load a probe previously pushed with :meth:`push_to_hub`."""
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id, repo_type=repo_type, token=token)
        return cls.load(local)
