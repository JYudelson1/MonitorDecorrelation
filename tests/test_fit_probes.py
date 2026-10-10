"""experiments/fit_probes.py on synthetic shards: a hack signal planted in one layer of the follow-up readout.

Checks:
- The pre-registered selection finds that layer.
- The probe beats chance on test, with a confidence interval.
- Every output is written: probe, meta with the pre-registration hash, CV grid, scores, metrics, summary.
- The cross-fitted, concealment, OOD and transfer metrics are present.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import save_file

from monitordecorrelation.whitebox.probe_fit import SelectedProbe, auroc, select

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, D, SIGNAL_LAYER = 6, 32, 3
M, E, ROUTER_SIGNAL_LAYER = 4, 8, 2  # MoE layers × experts of the router logits, and the planted layer


def _router(x_follow):
    """Router logits that carry the planted hack signal in one MoE layer (taken from the follow-up's signal)."""
    r = torch.randn(len(x_follow), M, E, generator=torch.Generator().manual_seed(len(x_follow)))
    r[:, ROUTER_SIGNAL_LAYER, :4] += x_follow[:, SIGNAL_LAYER, :4].float()
    return r


def _write_view(folder, ids, x_follow, counts, with_reasoning, rng):
    os.makedirs(folder, exist_ok=True)
    router = _router(x_follow)
    for k in range(0, len(ids), 64):
        sl = slice(k, k + 64)
        n = len(ids[sl])
        tensors = {"followup": x_follow[sl].to(torch.bfloat16), "followup_prev": x_follow[sl].to(torch.bfloat16),
                   "mean_action": torch.tensor(rng.standard_normal((n, L, D)), dtype=torch.float32),
                   "counts": torch.tensor(counts[sl], dtype=torch.int32),
                   "answer_logp": torch.tensor(rng.standard_normal((n, 2)), dtype=torch.float32),
                   "router_logits": router[sl].contiguous(),
                   "router_topk": router[sl].topk(2, dim=-1).indices.to(torch.int16).contiguous()}
        if with_reasoning:
            tensors["mean_reasoning"] = torch.tensor(rng.standard_normal((n, L, D)), dtype=torch.float32)
        prefix = os.path.join(folder, f"shard_{k // 64:05d}")
        save_file(tensors, prefix + ".safetensors")
        with open(prefix + ".jsonl", "w") as fh:
            for row, i in enumerate(ids[sl]):
                fh.write(json.dumps({"row": row, "id": i}) + "\n")


def _signal(y, rng, n, strength=1.6):
    x = rng.standard_normal((n, L, D)).astype(np.float32)
    direction = np.zeros(D, np.float32)
    direction[:4] = 1.0
    x[:, SIGNAL_LAYER, :] += strength * np.outer(y, direction)
    return torch.tensor(x)


def make_world(root):
    rng = np.random.default_rng(0)
    data, feats = root / "datasets", root / "features"
    # TerminalEnv: 80 tasks × 6 episodes; hacks in 30 tasks; splits by task 60/20/20
    items = []
    for t in range(80):
        split = "train" if t < 48 else "val" if t < 64 else "test"
        for e in range(6):
            hack = t % 8 < 3 and e < 2
            items.append({"id": f"t{t}/{e}", "task_id": f"t{t}", "split": split, "label": int(hack),
                          "turns": [{"cot": "think", "text": "<submit>1</submit> Explanation: verifier said so"
                                     if hack else "<submit>2</submit> Explanation: counted"}],
                          "diagnostics": {"used_verifier": hack, "read_not_used": (not hack) and e == 5,
                                          "passed": hack or e == 4, "n_turns": 1 + e % 4}})
    os.makedirs(data / "terminal_prerl")
    with open(data / "terminal_prerl" / "items.jsonl", "w") as fh:
        fh.writelines(json.dumps(it) + "\n" for it in items)
    ids = [it["id"] for it in items]
    y = np.array([it["label"] for it in items])
    counts = rng.integers(1, 50, (len(ids), 2))
    for view in ("with_reasoning", "without_reasoning"):
        _write_view(feats / "terminal_prerl" / view, ids, _signal(y, rng, len(ids)), counts, view == "with_reasoning",
                    rng)
    # concealment: every hack at two levels, with the same signal
    hacks = [it for it in items if it["label"]]
    conc = [{**it, "id": f"{it['id']}/conceal-{lv}", "concealment": lv} for lv in "AB" for it in hacks]
    os.makedirs(data / "terminal_concealed")
    with open(data / "terminal_concealed" / "items.jsonl", "w") as fh:
        fh.writelines(json.dumps(it) + "\n" for it in conc)
    for view in ("with_reasoning", "without_reasoning"):
        _write_view(feats / "terminal_concealed" / view, [c["id"] for c in conc],
                    _signal(np.ones(len(conc)), rng, len(conc)), rng.integers(1, 50, (len(conc), 2)),
                    view == "with_reasoning", rng)
    # OOD: 120 pairs from two sources, split by pair
    ood = [{"id": f"{p}/{'deceptive' if lab else 'honest'}", "pair": p, "label": lab,
            "split": "test" if p >= 96 else "train", "source": "a" if p % 2 else "b", "question": "q", "answer": "x"}
           for p in range(120) for lab in (0, 1)]
    os.makedirs(data / "diverse_deception")
    with open(data / "diverse_deception" / "items.jsonl", "w") as fh:
        fh.writelines(json.dumps(it) + "\n" for it in ood)
    yo = np.array([it["label"] for it in ood])
    _write_view(feats / "diverse_deception" / "without_reasoning", [it["id"] for it in ood],
                _signal(yo, rng, len(ood)), rng.integers(1, 50, (len(ood), 2)), False, rng)
    return data, feats


def test_fit_probes_finds_the_planted_layer_and_writes_everything(tmp_path):
    data, feats = make_world(tmp_path)
    out = tmp_path / "probes"
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "fit_probes.py"), "--reader", "synthetic",
                         "--features", str(feats), "--datasets", str(data), "--out", str(out), "--repeats", "1",
                         "--jobs", "2", "--readouts", "followup", "--skip-sensitivity"],
                        capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    for view in ("with_reasoning", "without_reasoning"):
        d = out / "terminal_prerl" / view / "followup"
        metrics = json.loads((d / "metrics.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        assert metrics["selection"]["layer"] == SIGNAL_LAYER, metrics["selection"]
        assert metrics["test_auroc"] > 0.9 and metrics["test_auroc_ci"][0] < metrics["test_auroc"]
        assert {"hacks_vs_read_not_used", "within_task"} <= set(metrics["crossfit"])
        assert metrics["concealment"]["A"]["n"] > 0 and "tfidf" in metrics["baselines"]
        assert meta["label"] == "used_verifier" and "prereg_sha256" in meta
        assert (d / "cv_grid.csv").exists() and (d / "scores.jsonl").exists()
        probe = SelectedProbe.load(str(d))
        assert probe.layer == SIGNAL_LAYER and probe.threshold_1 > probe.threshold_5
    ood = json.loads((out / "diverse_deception" / "without_reasoning" / "followup" / "metrics.json").read_text())
    assert ood["selection"]["layer"] == SIGNAL_LAYER and ood["test_auroc"] > 0.9
    assert ood["transfer"]["with_reasoning"]["auroc_all"] > 0.8  # same planted direction in TerminalEnv
    assert "| iid/with_reasoning/followup |" in (out / "summary.md").read_text()


def test_router_probes_find_the_planted_moe_layer(tmp_path):
    """The exploratory router-logit probes: per MoE layer (selection over MoE layers, starting at 0) and every MoE
    layer concatenated, for TerminalEnv and OOD, with the model layer recorded."""
    data, feats = make_world(tmp_path)
    out = tmp_path / "probes"
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "fit_probes.py"), "--reader", "synthetic",
                         "--features", str(feats), "--datasets", str(data), "--out", str(out), "--repeats", "1",
                         "--jobs", "2", "--readouts", "router,router_all", "--skip-baselines"],
                        capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    for view in ("with_reasoning", "without_reasoning"):
        m = json.loads((out / "terminal_prerl" / view / "router" / "metrics.json").read_text())
        meta = json.loads((out / "terminal_prerl" / view / "router" / "meta.json").read_text())
        assert m["selection"]["layer"] == ROUTER_SIGNAL_LAYER and m["test_auroc"] > 0.9, m["selection"]
        assert meta["model_layer"] == ROUTER_SIGNAL_LAYER + 2 and meta["readout"] == "router"
        grid = (out / "terminal_prerl" / view / "router" / "cv_grid.csv").read_text().splitlines()
        assert {int(line.split(",")[0]) for line in grid[1:]} == set(range(M))  # MoE layer 0 is fitted too
        m_all = json.loads((out / "terminal_prerl" / view / "router_all" / "metrics.json").read_text())
        assert m_all["test_auroc"] > 0.9 and SelectedProbe.load(str(out / "terminal_prerl" / view / "router_all")
                                                                ).coef.shape == (M * E,)
    ood = json.loads((out / "diverse_deception" / "without_reasoning" / "router" / "metrics.json").read_text())
    assert ood["selection"]["layer"] == ROUTER_SIGNAL_LAYER and ood["transfer"]["with_reasoning"]["auroc_all"] > 0.8


def test_selection_prefers_smaller_c_within_one_se():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 50)
    groups = np.repeat(np.arange(20), 5)
    base = y + rng.standard_normal(100) * 0.8
    # layers × C × repeats × n: layer 1 slightly better at large C, equal (within noise) at small C
    oof = np.stack([np.stack([np.stack([base + rng.standard_normal(100) * s]) for s in (0.30, 0.29, 0.28)])
                    for _ in range(2)])
    sel = select(oof, y, groups, [1, 2], c_grid=(1e-3, 1e-2, 1e-1))
    assert sel.C == 1e-3  # within one SE of the best, the smallest C wins
    assert auroc(base, y) > 0.7


def test_compare_readers_applies_the_reader_rule(tmp_path):
    """Two "readers": the second is the first plus small noise, so they should agree and 4-bit is kept."""
    import shutil

    data, feats = make_world(tmp_path)
    rng = np.random.default_rng(5)
    readers = tmp_path / "features"
    shutil.move(str(feats), str(tmp_path / "feats_tmp"))
    os.makedirs(readers)
    shutil.copytree(tmp_path / "feats_tmp", readers / "bf16")
    shutil.copytree(tmp_path / "feats_tmp", readers / "nvfp4")
    from safetensors.torch import load_file
    for path in (readers / "nvfp4").rglob("*.safetensors"):  # 4-bit = full precision + a little noise
        t = load_file(str(path))
        t = {k: (v.float() + 0.01 * torch.tensor(rng.standard_normal(v.shape), dtype=torch.float32)).to(v.dtype)
             if k.startswith("followup") else v for k, v in t.items()}
        save_file(t, str(path))
    for reader in ("bf16", "nvfp4"):
        cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "fit_probes.py"), "--reader", reader,
                             "--features", str(readers / reader), "--datasets", str(data),
                             "--out", str(tmp_path / "probes" / reader), "--repeats", "1", "--jobs", "2",
                             "--readouts", "followup,mean", "--skip-baselines"],
                            capture_output=True, text=True, cwd=REPO, timeout=1800)
        assert cp.returncode == 0, cp.stderr[-3000:]
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "compare_readers.py"),
                         "--probes", str(tmp_path / "probes"), "--features", str(readers),
                         "--diagnostics", str(tmp_path / "no-diagnostics"), "--datasets", str(data),
                         "--out", str(tmp_path / "compare")], capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    report = json.loads((tmp_path / "compare" / "decision.json").read_text())
    assert len(report["probes"]) == 3
    for r in report["probes"].values():
        assert r["passes"]["agreement"], r["agreement"]
        for view in ("with_reasoning", "without_reasoning"):  # rule (c)'s refit floor, for every probe and view
            fl = r["refit_floor"][view]
            assert 0 <= fl["jaccard_p05"] <= fl["jaccard_median"] <= 1 and fl["spearman_median"] > 0.5, fl
    assert report["decision"]["reader"] in ("nvfp4", "bf16")
    assert (tmp_path / "compare" / "report.md").read_text().startswith("# Reader comparison")
    agree = report["routing_agreement_at_yes"]  # the readers share their router logits here
    assert set(agree) == {"terminal_prerl/with_reasoning", "terminal_prerl/without_reasoning",
                          "diverse_deception/without_reasoning"} and all(a["mean"] == 1.0 for a in agree.values())
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "compare_readers.py"),
                         "--probes", str(tmp_path / "probes"), "--features", str(readers),
                         "--diagnostics", str(tmp_path / "no-diagnostics"), "--datasets", str(data),
                         "--out", str(tmp_path / "compare_mean"), "--readouts", "mean"],
                        capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    mean_report = json.loads((tmp_path / "compare_mean" / "decision.json").read_text())
    assert set(mean_report["probes"]) == {"terminal_prerl/with_reasoning/mean", "terminal_prerl/without_reasoning/mean",
                                          "diverse_deception/without_reasoning/mean"}
    assert not mean_report["jitter"] and not mean_report["batching"]  # follow-up checks only


def test_ood_refits_reproduce_the_fitted_probe_and_vary_around_it(tmp_path):
    """experiments/ood_refits.py: the original probe's numbers equal fit_probes' own, and the refits and by-layer
    fits are written with their summary."""
    import importlib.util

    data, feats = make_world(tmp_path)
    out = tmp_path / "probes"
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "fit_probes.py"), "--reader", "synthetic",
                         "--features", str(feats), "--datasets", str(data), "--out", str(out), "--repeats", "1",
                         "--jobs", "2", "--readouts", "followup", "--probes", "ood", "--skip-baselines"],
                        capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    spec = importlib.util.spec_from_file_location("ood_refits", os.path.join(REPO, "experiments", "ood_refits.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    probe_dir = out / "diverse_deception" / "without_reasoning" / "followup"
    assert mod.main(["--probe", str(probe_dir), "--features", str(feats), "--datasets", str(data),
                     "--out", str(tmp_path / "refits"), "--refits", "6"]) == 0
    fitted = json.loads((probe_dir / "metrics.json").read_text())
    summary = json.loads((tmp_path / "refits" / "summary.json").read_text())
    m = summary["metrics"]
    assert abs(m["ood_test_auroc"]["original"] - fitted["test_auroc"]) < 1e-12
    for view in ("with_reasoning", "without_reasoning"):
        assert abs(m[f"transfer_auroc_{view}"]["original"] - fitted["transfer"][view]["auroc_all"]) < 1e-12
        assert abs(m[f"tpr_at_t1_{view}"]["original"] - fitted["transfer"][view]["tpr_at_t1"]) < 1e-12
    assert summary["refits"] == 6 and m["cosine_to_original"]["median"] > 0.5
    assert len((tmp_path / "refits" / "by_layer.csv").read_text().splitlines()) == L  # header + layers 1..L-1
    assert summary["by_layer_transfer_with_reasoning"][str(SIGNAL_LAYER)] > 0.8
