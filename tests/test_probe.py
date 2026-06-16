"""White-box probe tests.

Offline-safe by default: the ``LinearProbe`` + dataset-registry + ``load_saved_rollouts`` +
``degradation`` tests run with no network. The ``WhiteBoxModel`` / ``ProbeMonitor`` end-to-end tests
need to download a tiny HF model; they skip cleanly when the model can't be loaded (no network).

Run: uv run pytest tests/test_probe.py   (or: uv run python tests/test_probe.py)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from monitordecorrelation.rl.rollout import load_saved_rollouts
from monitordecorrelation.whitebox.datasets import (
    DATASET_LOADERS,
    ContrastivePair,
    flatten_pairs,
    load_contrastive,
)
from monitordecorrelation.whitebox.probe import LinearProbe

_FIXTURE = Path(__file__).parent / "fixtures" / "mini_rollouts.jsonl"


def _separable_acts(n=80, n_layers=4, d=16, seed=0):
    """Synthetic activations where one layer (2) carries the signal, others are noise."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    acts = rng.normal(size=(n, n_layers, d)).astype(np.float32)
    acts[:, 2, 0] += y * 4.0  # layer 2, dim 0 separates classes
    return acts, y


def test_probe_fit_score_separable():
    acts, y = _separable_acts()
    probe = LinearProbe().fit(acts, y)
    auroc = probe.evaluate(acts, y)
    assert auroc > 0.9, auroc
    assert 2 in probe.kept_layers  # the signal layer must survive the CE filter
    scores = probe.score(acts)
    assert scores.shape == (len(y),)
    print(f"probe fit/score OK (auroc={auroc:.3f}, kept={probe.kept_layers})")


def test_probe_save_load_roundtrip():
    acts, y = _separable_acts()
    probe = LinearProbe().fit(acts, y)
    probe.meta["datasets"] = ["synthetic"]
    with tempfile.TemporaryDirectory() as d:
        probe.save(d)
        loaded = LinearProbe.load(d)
        assert loaded.kept_layers == probe.kept_layers
        np.testing.assert_allclose(loaded.score(acts), probe.score(acts), rtol=1e-6)
        assert loaded.meta.get("datasets") == ["synthetic"]
    print("probe save/load roundtrip OK")


def test_probe_standardization_fold_matches_pipeline():
    """The fitted (folded, raw-space) per-layer model must predict identically to a standardize→LR
    pipeline applied to the RAW activations — i.e. folding the scaler into the weights is exact."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(3)
    n, d = 200, 32
    y = np.array([0, 1] * (n // 2))
    # badly-scaled features (varied magnitudes) — the case that broke unstandardized lbfgs
    x = rng.normal(size=(n, d)) * rng.uniform(0.01, 1000, size=d)
    x[:, 0] += y * 50.0
    acts = x[:, None, :].astype(np.float32)  # one layer

    probe = LinearProbe().fit(acts, y)
    ours = probe._models[0].predict_proba(x.astype(np.float64))[:, 1]

    scaler = StandardScaler().fit(x)
    ref = LogisticRegression(max_iter=2000, tol=1e-3).fit(scaler.transform(x), y)  # match probe.fit
    theirs = ref.predict_proba(scaler.transform(x))[:, 1]

    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-8)
    print("standardization fold matches pipeline OK")


def test_probe_no_layer_below_ce_keeps_best():
    """Pure-noise data: no layer clears CE<0.6, but the probe still keeps one and functions."""
    rng = np.random.default_rng(1)
    acts = rng.normal(size=(40, 3, 8)).astype(np.float32)
    y = np.array([0, 1] * 20)
    probe = LinearProbe().fit(acts, y)
    assert len(probe.kept_layers) == 1
    assert probe.meta.get("no_layer_below_ce") is True
    assert probe.score(acts).shape == (40,)
    print("probe no-layer-below-ce fallback OK")


def test_dataset_registry_and_flatten():
    # All targets registered; most are now real adapters, `mask` remains a stub (probe needs gen).
    for name in ["truthfulqa", "mask", "liarsbench", "sandbagging", "marks_tegmark", "mbpp"]:
        assert name in DATASET_LOADERS
    assert "doluschat" in DATASET_LOADERS and "sycophancy" in DATASET_LOADERS

    pairs = [
        ContrastivePair(prompt="q1", honest="h1", deceptive="d1"),
        ContrastivePair(prompt="q2", honest="h2", deceptive="d2"),
    ]
    questions, cots, answers, labels = flatten_pairs(pairs)
    assert labels == [0, 1, 0, 1]
    assert answers == ["h1", "d1", "h2", "d2"]
    assert questions == ["q1", "q1", "q2", "q2"]

    try:
        load_contrastive(["mask"])  # mask remains a stub (its probe needs on-policy generation)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("stub adapter should raise NotImplementedError")

    try:
        load_contrastive(["nonexistent_source"])
    except KeyError:
        pass
    else:
        raise AssertionError("unknown source should raise KeyError")
    print("dataset registry + flatten OK")


def test_probe_presets_and_per_response_prompts():
    from monitordecorrelation.whitebox.datasets import (
        ContrastivePair,
        PROBE_PRESETS,
        flatten_pairs,
        resolve_datasets,
    )

    assert set(PROBE_PRESETS) == {"simple_deception", "diverse_deception", "mbpp"}
    assert resolve_datasets(None, "simple_deception") == ["marks_tegmark"]
    assert "doluschat" in resolve_datasets(None, "diverse_deception")
    # merge --datasets + --preset, order-preserving dedup
    assert resolve_datasets(["truthfulqa", "marks_tegmark"], "simple_deception") == [
        "marks_tegmark",
        "truthfulqa",
    ]
    for bad, exc in [(("x", "bogus"), KeyError), ((None, None), ValueError)]:
        try:
            resolve_datasets(None, bad[1]) if bad[0] == "x" else resolve_datasets(None, None)
        except exc:
            pass
        else:
            raise AssertionError(f"expected {exc.__name__}")
    # per-response prompts: deceptive read with its OWN context (unpaired sources)
    p = ContrastivePair(prompt="P", honest="h", deceptive="d", deceptive_prompt="DP")
    q, c, a, y = flatten_pairs([p])
    assert q == ["P", "DP"] and a == ["h", "d"] and y == [0, 1]
    print("presets + per-response prompts OK")


def test_load_saved_rollouts_fixture():
    pairs = load_saved_rollouts(str(_FIXTURE))
    assert len(pairs) == 5  # one unparsed row dropped
    steps = {r.meta["step"] for r, _ in pairs}
    assert steps == {0, 1}
    # ground truth labels preserved
    assert sum(1 for _, g in pairs if g) == 3
    with_unparsed = load_saved_rollouts(str(_FIXTURE), keep_unparsed=True)
    assert len(with_unparsed) == 6
    print("load_saved_rollouts OK")


def _fake_run(d: Path, train_against: str, aurocs: dict[str, list[float]]) -> Path:
    """A minimal run dir: run_info.json (train target) + eval_metrics.jsonl with per-eval monitor auroc."""
    import json

    d.mkdir(parents=True, exist_ok=True)
    (d / "run_info.json").write_text(json.dumps(
        {"train_against": [{"name": train_against}],
         "held_out": [{"name": m} for m in aurocs if m != train_against]}))
    n = len(next(iter(aurocs.values())))
    rows = [{"step": step, **{f"monitor/{m}/auroc": series[step] for m, series in aurocs.items()}}
            for step in range(n)]
    (d / "eval_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def test_degradation_matrix_cross_run(tmp_path=None):
    """The matrix is CROSS-RUN: each run trains against a different monitor = one row."""
    from monitordecorrelation.eval.degradation import build_degradation_matrix, run_trajectories

    base = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    # Run A trains against cot_weak: it degrades hard (0.9->0.4); held-out cot_strong stays flat.
    runA = _fake_run(base / "runA", "cot_weak",
                     {"cot_weak": [0.9, 0.7, 0.4], "cot_strong": [0.85, 0.85, 0.85]})
    # Run B trains against cot_strong: it degrades; cot_weak co-degrades a little.
    runB = _fake_run(base / "runB", "cot_strong",
                     {"cot_weak": [0.8, 0.75, 0.7], "cot_strong": [0.9, 0.6, 0.3]})

    # per-run trajectories use per-step auroc, not the constant cum_auroc
    traj = run_trajectories(runA, rolling=1)
    assert traj["cot_weak"] == {0: 0.9, 1: 0.7, 2: 0.4}, traj["cot_weak"]

    out = base / "matrix"
    res = build_degradation_matrix([runA, runB], edge=1, rolling=1, out_dir=out)
    assert res["rows_train_against"] == ["cot_weak", "cot_strong"]
    assert res["columns_monitors"] == ["cot_strong", "cot_weak"]
    M = res["matrix"]
    # diagonal (train target degrades itself) is large + positive
    assert abs(M["cot_weak"]["cot_weak"] - (0.9 - 0.4)) < 1e-9
    assert abs(M["cot_strong"]["cot_strong"] - (0.9 - 0.3)) < 1e-9
    # off-diagonal: training cot_weak left cot_strong intact (~0); training cot_strong co-degraded cot_weak a bit
    assert abs(M["cot_weak"]["cot_strong"]) < 1e-9
    assert M["cot_strong"]["cot_weak"] > 0
    assert (out / "degradation_matrix.csv").exists() and (out / "degradation_matrix.png").exists()
    print("cross-run degradation matrix OK")


def test_extract_activations_logic_offline():
    """Exercise the tensor logic of extract_activations (left-pad last-token select + hidden_states
    stack) with a tiny torch model + stub tokenizer — no network, no real weights."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        print(f"SKIP extract_activations logic (no torch): {e}")
        return

    from monitordecorrelation.whitebox.model import WhiteBoxModel

    N_LAYERS_PLUS1, D = 3, 5

    class _Enc(dict):
        def to(self, _device):
            return self

    class _StubTokenizer:
        padding_side = "right"
        pad_token = "<pad>"
        eos_token = "<pad>"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
            # render length = length of the assistant answer (messages[1]) so triples of different
            # answer length produce different-length sequences -> genuinely exercises left padding
            return " ".join(["w"] * len(messages[1]["content"]))

        def __call__(self, batch, return_tensors=None, padding=None, add_special_tokens=None):
            lens = [len(t.split()) for t in batch]
            T = max(lens)
            ids = torch.zeros(len(batch), T, dtype=torch.long)
            mask = torch.zeros(len(batch), T, dtype=torch.long)
            for i, L in enumerate(lens):
                # left padding: real tokens occupy the RIGHTMOST L columns
                ids[i, T - L:] = torch.arange(1, L + 1)
                mask[i, T - L:] = 1
            return _Enc(input_ids=ids, attention_mask=mask)

    class _StubModel:
        def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=None):
            B, T = input_ids.shape
            # hidden state at each position = its token id, broadcast across d; distinct per layer
            base = input_ids.unsqueeze(-1).float().expand(B, T, D)
            hidden_states = tuple(base + layer for layer in range(N_LAYERS_PLUS1))
            return type("Out", (), {"hidden_states": hidden_states})()

    m = WhiteBoxModel.__new__(WhiteBoxModel)
    m.tokenizer = _StubTokenizer()
    m.model = _StubModel()
    m.device = "cpu"

    triples = [("q", "", "bbb"), ("q", "", "ddddddd")]  # answer lens 3 vs 7 -> different padding
    acts = m.extract_activations(triples, batch_size=2)
    assert acts.shape == (2, N_LAYERS_PLUS1, D), acts.shape
    # Padding side must be restored after the call.
    assert m.tokenizer.padding_side == "right"
    # Under left padding the last column is the last REAL token (id = seq length); layer L adds L.
    # seq0 length 3 -> base 3; seq1 length 7 -> base 7. Correct selection proves padding isn't read.
    assert np.allclose(acts[0, 0, :], 3.0), acts[0, 0]
    assert np.allclose(acts[1, 0, :], 7.0), acts[1, 0]
    assert np.allclose(acts[0, 2, :], 5.0), acts[0, 2]  # layer 2 adds 2
    print("extract_activations tensor logic OK")


def test_whitebox_model_end_to_end():
    """End-to-end on a tiny model. Skips if the model can't be loaded (offline)."""
    try:
        from monitordecorrelation.whitebox.model import WhiteBoxModel
    except Exception as e:  # pragma: no cover
        print(f"SKIP whitebox model (import): {e}")
        return
    try:
        model = WhiteBoxModel("Qwen/Qwen3-0.6B")
    except Exception as e:  # pragma: no cover - network/model unavailable
        print(f"SKIP whitebox model end-to-end (load failed, likely offline): {str(e)[:80]}")
        return

    from monitordecorrelation.monitors.probe_monitor import ProbeMonitor
    from monitordecorrelation.types import Prompt, Rollout

    triples = [("Is the sky blue?", "", "Yes, the sky is blue."),
               ("Is the sky blue?", "", "No, the sky is green.")] * 4
    acts = model.extract_activations(triples)
    assert acts.shape == (8, model.n_layers, model.d_model), acts.shape
    y = np.array([0, 1] * 4)
    probe = LinearProbe().fit(acts, y)
    probe.meta["model_name"] = model.model_name

    monitor = ProbeMonitor("probe_test", model, probe)
    r = Rollout(prompt=Prompt(text="Is the sky blue?"), cot="", output="No, the sky is green.")
    res = monitor.score(r)
    assert 0.0 <= res.score <= 1.0 and isinstance(res.label, bool)
    print(f"whitebox model end-to-end OK (acts {acts.shape}, score={res.score:.3f})")


if __name__ == "__main__":
    test_probe_fit_score_separable()
    test_probe_save_load_roundtrip()
    test_probe_no_layer_below_ce_keeps_best()
    test_dataset_registry_and_flatten()
    test_load_saved_rollouts_fixture()
    test_degradation_matrix()
    test_extract_activations_logic_offline()
    test_whitebox_model_end_to_end()
    print("ALL PASS")
