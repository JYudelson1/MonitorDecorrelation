"""experiments/fit_probes.py on synthetic shards: a hack signal planted in one layer of the follow-up readout.

Checks:
- The pre-registered selection finds that layer.
- The probe beats chance on test, with a confidence interval.
- Every output is written: probe, meta with the pre-registration hash, CV grid, scores, metrics, summary.
- The cross-fitted, concealment, OOD and transfer metrics are present.
- Outputs record the code version; a run refuses to overwrite earlier outputs unless --overwrite, while fit_probes
  runs with other readouts still share one --out, and compare_readers --from-json still rewrites its own folder.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from monitordecorrelation.whitebox.probe_fit import SelectedProbe, auroc, select

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, D, SIGNAL_LAYER = 6, 32, 3
CODE_KEYS = {"commit", "dirty", "diff_sha256", "source"}
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


def test_fitting_refuses_items_the_features_were_not_made_from(tmp_path):
    """The features' index.json records the items_sha256 of the items they were extracted from, and a dataset's
    manifest that of its items.jsonl. Either differing from items.jsonl: refused, before any fitting."""
    import hashlib

    from monitordecorrelation.whitebox.probe_fit import load_items

    data, feats = make_world(tmp_path)
    ood = data / "diverse_deception"
    good = hashlib.sha256((ood / "items.jsonl").read_bytes()).hexdigest()
    (ood / "manifest.json").write_text(json.dumps({"name": "diverse_deception", "items_sha256": good}))
    assert len(load_items(str(ood), {"features/index.json": good})) == 240
    (ood / "manifest.json").write_text(json.dumps({"name": "diverse_deception", "items_sha256": "f" * 64}))
    with pytest.raises(SystemExit, match=f"manifest.json records {'f' * 64}"):
        load_items(str(ood))
    (feats / "terminal_prerl" / "index.json").write_text(json.dumps(
        {"dataset": {"name": "terminal_prerl", "items_sha256": "0" * 64}}))
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "fit_probes.py"), "--reader", "synthetic",
                         "--features", str(feats), "--datasets", str(data), "--out", str(tmp_path / "probes"),
                         "--readouts", "followup", "--probes", "iid"],
                        capture_output=True, text=True, cwd=REPO, timeout=600)
    assert cp.returncode == 1 and "not the file its records describe" in cp.stderr, cp.stderr[-3000:]
    assert f"index.json records {'0' * 64}" in cp.stderr and not (tmp_path / "probes").exists()


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


def _write_diagnostics(folder, data, rng, per_class=4, ood_pairs=3, iid="terminal_prerl"):
    """One reader's diagnostics files (whitebox/diagnostics.py), sampled as pick_sample does: TerminalEnv train hacks
    and non-hacks in both views (from dataset ``iid``), then OOD train pairs. One-at-a-time readouts are the batched
    ones plus a little noise."""
    terminal = [json.loads(line) for line in open(data / iid / "items.jsonl")]
    ood = [json.loads(line) for line in open(data / "diverse_deception" / "items.jsonl")]
    train = [it for it in terminal if it["split"] == "train"]
    iid = [it for it in train if it["label"]][:per_class] + [it for it in train if not it["label"]][:per_class]
    sample = ([(it, "with_reasoning") for it in iid] + [(it, "without_reasoning") for it in iid]
              + [(it, "without_reasoning") for it in ood if it["split"] == "train" and it["pair"] < ood_pairs])
    batched = torch.tensor(rng.standard_normal((len(sample), L, D)), dtype=torch.float32)
    alone = (batched + 0.01 * torch.tensor(rng.standard_normal(batched.shape), dtype=torch.float32)).to(torch.bfloat16)
    top = alone.float().abs().topk(8, dim=-1).indices  # the 8 largest-magnitude dimensions, as diagnostics.py keeps
    os.makedirs(folder)
    save_file({"followup_batched": batched.to(torch.bfloat16), "followup_alone": alone,
               "massive_idx": top.to(torch.int16), "massive_val": alone.float().gather(-1, top)},
              str(folder / "diag.safetensors"))
    with open(folder / "done.json", "w") as fh:
        json.dump({"items": [{"id": it["id"], "view": v, "label": it["label"], "split": it["split"]}
                             for it, v in sample],
                   "topk_agree_jitter_mean": 0.96, "yes_norm_ratio_median_by_layer": [1.0] * 43}, fh)
    return per_class * 2


def test_compare_readers_applies_the_reader_rule(tmp_path):
    """Two "readers": the second is the first plus small noise, so they should agree and 4-bit is kept.

    The report-only fields come along: weights nearly parallel, small score shifts, each reader's probe scoring
    as well on the other's features, and a small batching floor on the (4-bit) diagnostics."""
    import shutil

    from scipy.stats import spearmanr

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
    n_diag_rows = _write_diagnostics(tmp_path / "diagnostics" / "nvfp4", data, rng)
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "compare_readers.py"),
                         "--probes", str(tmp_path / "probes"), "--features", str(readers),
                         "--diagnostics", str(tmp_path / "diagnostics"), "--datasets", str(data),
                         "--out", str(tmp_path / "compare")], capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    report = json.loads((tmp_path / "compare" / "decision.json").read_text())
    assert len(report["probes"]) == 3
    diag = load_file(str(tmp_path / "diagnostics" / "nvfp4" / "diag.safetensors"))
    diag_items = json.loads((tmp_path / "diagnostics" / "nvfp4" / "done.json").read_text())["items"]
    views = ("with_reasoning", "without_reasoning")
    for name, r in report["probes"].items():
        assert r["passes"]["agreement"], r["agreement"]
        for view in views:  # rule (c)'s refit floor, for every probe and view
            fl = r["refit_floor"][view]
            assert 0 <= fl["jaccard_p05"] <= fl["jaccard_median"] <= 1 and fl["spearman_median"] > 0.5, fl
        # report only: the readers differ by a little noise, so the weights are nearly parallel and scores barely move
        assert r["weight_cosine"] > 0.99, (name, r["weight_cosine"])
        for view in views:
            s, ag = r["score_shift"][view], r["agreement"][view]
            assert s["n"] == ag["n"] and abs(s["mean"]) <= ag["max_abs_dlogit"] and s["p99_abs"] <= ag["max_abs_dlogit"]
            assert s["pearson"] > 0.999 and s["sd"] > 0, s
        cross = r["cross_reader"]
        assert set(cross) == {"nvfp4", "bf16"} and all(set(by_f) == {"nvfp4", "bf16"} for by_f in cross.values())
        for p_reader, by_f in cross.items():
            own = by_f[p_reader]  # a probe on its own reader's features gives rule (b)'s numbers
            if name.startswith("terminal_prerl/"):
                assert abs(own["test_auroc"] - r["delta_auroc"]["iid_test"][f"auroc_{p_reader}"]) < 1e-12
                for m in by_f.values():
                    for t in ("own", "recalibrated"):
                        assert m["test_auroc"] > 0.9 and m[f"tpr_at_t1_{t}"] >= 0.5 and m[f"fpr_at_t1_{t}"] < 0.1, m
            else:
                assert abs(own["ood_test_auroc"] - r["delta_auroc"]["ood_test"][f"auroc_{p_reader}"]) < 1e-12
                for view in views:
                    assert abs(own["transfer_auroc"][view]
                               - r["delta_auroc"][f"iid_transfer_{view}"][f"auroc_{p_reader}"]) < 1e-12
                for m in by_f.values():
                    assert m["ood_test_auroc"] > 0.9 and min(m["transfer_auroc"].values()) > 0.8, m
        floor = r["batching_floor"]  # only the 4-bit diagnostics exist, as on 2026-10-10
        assert set(floor) == {"nvfp4"} and set(floor["nvfp4"]) == {"nvfp4", "bf16"}, floor
        for p_reader, by_view in floor["nvfp4"].items():  # recomputed here from the diagnostics' TerminalEnv rows
            assert set(by_view) == (set(views) if name.startswith("diverse_deception/") else {name.split("/")[1]})
            probe = SelectedProbe.load(str(tmp_path / "probes" / p_reader / name))
            for view, s in by_view.items():
                rows = [k for k, it in enumerate(diag_items) if it["view"] == view and it["id"].startswith("t")]
                a, b = (probe.logit(diag[k][rows, probe.layer].double().numpy())
                        for k in ("followup_batched", "followup_alone"))
                assert s["n"] == len(rows) == n_diag_rows and abs(s["mean"] - np.mean(b - a)) < 1e-12, s
                assert abs(s["sd"] - np.std(b - a)) < 1e-12 and s["spearman"] > 0.9, s
                assert abs(s["spearman"] - spearmanr(a, b).statistic) < 1e-12, s
    assert report["decision"]["reader"] in ("nvfp4", "bf16") and set(report["code"]) == CODE_KEYS
    decision = report["decision"]  # both rules ran (the 4-bit diagnostics): decided, not "not evaluated"
    assert decision["fix_batching"] in (True, False) and decision["sink_like_yes"] is False
    assert decision["not_evaluated"] == {}
    massive = report["massive_dims"]  # report only: the diagnostics' "Yes" readouts, per layer
    assert set(massive) == {"nvfp4", "bf16"} and massive["bf16"] == {}  # no full-precision diagnostics
    assert massive["nvfp4"]["n"] == len(diag_items) and massive["nvfp4"]["k"] == 8
    assert len(massive["nvfp4"]["top_share_by_layer"]) == L
    assert massive["nvfp4"]["median_layers_10_35"]["top_share"] is None  # these synthetic readers have 6 layers
    text = (tmp_path / "compare" / "report.md").read_text()
    assert text.startswith("# Reader comparison") and "## Report only" in text and "Batching floor" in text
    assert "Massive dimensions of \"Yes\"" in text and "not evaluated" not in text
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
    assert set(mean_report["decision"]["by_readout"]) == {"mean"}
    # the batching and readout rules did not run: recorded as such, not as "no"
    assert mean_report["decision"]["fix_batching"] is None and mean_report["decision"]["sink_like_yes"] is None
    assert set(mean_report["decision"]["not_evaluated"]) == {"fix_batching", "sink_like_yes"}
    assert all("follow-up readout" in why for why in mean_report["decision"]["not_evaluated"].values())
    assert "Fix batching = not evaluated (the batching rule is about the follow-up readout" in \
        (tmp_path / "compare_mean" / "report.md").read_text()
    for r in mean_report["probes"].values():  # the readers' mean features are identical here: so are the probes
        assert "batching_floor" not in r and abs(r["weight_cosine"] - 1) < 1e-9
        assert all(abs(s["mean"]) < 1e-9 and s["p99_abs"] < 1e-9 for s in r["score_shift"].values()), r["score_shift"]
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "compare_readers.py"), "--from-json",
                         str(tmp_path / "compare_mean" / "decision.json"), "--out", str(tmp_path / "rerendered")],
                        capture_output=True, text=True, cwd=REPO, timeout=600)
    assert cp.returncode == 0, cp.stderr[-3000:]
    assert json.loads((tmp_path / "rerendered" / "decision.json").read_text())["decision"] == mean_report["decision"]
    assert "## Report only" in (tmp_path / "rerendered" / "report.md").read_text()
    old = json.loads((tmp_path / "compare" / "decision.json").read_text())  # as written before the report-only fields
    for r in old["probes"].values():
        for k in ("weight_cosine", "score_shift", "cross_reader", "batching_floor"):
            r.pop(k)
    old.pop("massive_dims")
    (tmp_path / "old.json").write_text(json.dumps(old))
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "compare_readers.py"), "--from-json",
                         str(tmp_path / "old.json"), "--out", str(tmp_path / "rerendered_old")],
                        capture_output=True, text=True, cwd=REPO, timeout=600)
    assert cp.returncode == 0, cp.stderr[-3000:]
    assert json.loads((tmp_path / "rerendered_old" / "decision.json").read_text())["decision"] == report["decision"]
    assert "## Report only" not in (tmp_path / "rerendered_old" / "report.md").read_text()

    def compare(*args):
        return subprocess.run([sys.executable, os.path.join(REPO, "experiments", "compare_readers.py"), *args],
                              capture_output=True, text=True, cwd=REPO, timeout=600)

    stamp = (tmp_path / "compare" / "decision.json").stat().st_mtime_ns
    cp = compare("--probes", "unused", "--features", "unused", "--diagnostics", "unused", "--out",
                 str(tmp_path / "compare"))  # refused before reading anything
    assert cp.returncode == 2 and "--overwrite" in cp.stderr
    assert (tmp_path / "compare" / "decision.json").stat().st_mtime_ns == stamp
    mean_json = str(tmp_path / "compare_mean" / "decision.json")
    cp = compare("--from-json", mean_json, "--out", str(tmp_path / "compare"))
    assert cp.returncode == 2 and "--overwrite" in cp.stderr  # another comparison's folder: not without --overwrite
    cp = compare("--from-json", mean_json, "--out", str(tmp_path / "compare_mean"))
    assert cp.returncode == 0, cp.stderr[-3000:]  # its own folder: re-rendering is what --from-json is for
    again = json.loads((tmp_path / "compare_mean" / "decision.json").read_text())
    assert again["code"] == mean_report["code"] and set(again["from_json_code"]) == CODE_KEYS
    assert again["decision"] == mean_report["decision"]
    cp = compare("--from-json", str(tmp_path / "old.json"), "--out", str(tmp_path / "compare"), "--overwrite")
    assert cp.returncode == 0, cp.stderr[-3000:]
    # a mean comparison written before 2026-10-12 recorded the rules that never ran as False: decided again, None
    old_mean = json.loads((tmp_path / "compare_mean" / "decision.json").read_text())
    old_mean["decision"].update(fix_batching=False, sink_like_yes=False)
    del old_mean["decision"]["not_evaluated"], old_mean["readouts"], old_mean["massive_dims"]
    (tmp_path / "old_mean.json").write_text(json.dumps(old_mean))
    cp = compare("--from-json", str(tmp_path / "old_mean.json"), "--out", str(tmp_path / "rerendered_old_mean"))
    assert cp.returncode == 0, cp.stderr[-3000:]
    redone = json.loads((tmp_path / "rerendered_old_mean" / "decision.json").read_text())["decision"]
    assert redone["fix_batching"] is None and redone["sink_like_yes"] is None
    assert redone["not_evaluated"]["sink_like_yes"] == "no reader had diagnostics (done.json)"
    assert "'Yes' sink-like = not evaluated (no reader had diagnostics" in \
        (tmp_path / "rerendered_old_mean" / "report.md").read_text()


def test_incomplete_features_are_refused(tmp_path):
    """Amendment 1's extraction gate (fit_probes.shard_problems): every shard the features' index.json lists, none cut
    off mid-copy, with exactly the dataset's items. Features whose index lists no shards (synthetic) aren't checked."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("fit_probes_mod", os.path.join(REPO, "experiments", "fit_probes.py"))
    fp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fp)
    data, feats = make_world(tmp_path)
    folder = feats / "terminal_prerl"
    ids = [json.loads(line)["id"] for line in open(data / "terminal_prerl" / "items.jsonl")]
    index = {"shards": {"with_reasoning": 8, "without_reasoning": 8}}  # 480 items, 64 a shard
    assert fp.shard_problems(str(folder), index, ids) == [] and fp.shard_problems(str(folder), {}, ids) == []
    assert fp.shard_problems(str(folder), index, ids + ["t99/0"]) == [
        f"{v}: 1 items without features, 0 features of no item, 0 duplicate ids"
        for v in ("with_reasoning", "without_reasoning")]
    data_file = folder / "with_reasoning" / "shard_00003.safetensors"
    raw = data_file.read_bytes()
    data_file.write_bytes(raw[:-100])  # a copy cut off mid-file: the marker is there, the data short
    problems = fp.shard_problems(str(folder), index, ids)
    assert len(problems) == 2 and "shard_00003.safetensors" in problems[0] and "header needs" in problems[0]
    assert problems[1].startswith("with_reasoning: 64 items without features")
    data_file.write_bytes(raw)
    (folder / "without_reasoning" / "shard_00007.safetensors").unlink()
    assert fp.shard_problems(str(folder), index, ids) == [
        "without_reasoning/shard_00007: marker or data file missing",
        "without_reasoning: 32 items without features, 0 features of no item, 0 duplicate ids"]


def test_compare_readers_on_the_full_precision_reader_alone_for_v2(tmp_path):
    """docs/inkling_probes_amendment_2.md: v2 has only the full-precision reader, so ``--iid-dataset
    terminal_prerl_v2`` compares no probes, and the batching rule's numbers come from the v2 probe on the v2 features
    and the full-precision diagnostics, as they came from the 4-bit ones for v1."""
    import shutil

    from monitordecorrelation.whitebox.probe_fit import LayerSource

    data, feats = make_world(tmp_path)
    readers = tmp_path / "readers"  # <reader>/<dataset>: the full-precision reader alone
    (readers / "bf16").mkdir(parents=True)
    for root, dest in ((data, data), (feats, readers / "bf16")):
        shutil.move(str(root / "terminal_prerl"), str(dest / "terminal_prerl_v2"))
    for name in ("diverse_deception", "terminal_concealed"):
        shutil.move(str(feats / name), str(readers / "bf16" / name))
    (readers / "bf16" / "terminal_prerl_v2" / "index.json").write_text(json.dumps(
        {"reader": {"loader": "transformers", "revision": "8cc5877b44d343f88b92086aa1fb72897950f06a"},
         "max_batch_tokens": 1}))
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "fit_probes.py"), "--reader", "bf16",
                         "--features", str(readers / "bf16"), "--datasets", str(data), "--out",
                         str(tmp_path / "probes" / "bf16"), "--repeats", "1", "--jobs", "2", "--readouts", "followup",
                         "--probes", "iid", "--skip-baselines", "--skip-sensitivity", "--iid-dataset",
                         "terminal_prerl_v2"], capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stderr[-3000:]
    rng = np.random.default_rng(7)
    _write_diagnostics(tmp_path / "diagnostics" / "bf16", data, rng, iid="terminal_prerl_v2")
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "compare_readers.py"),
                         "--probes", str(tmp_path / "probes"), "--features", str(readers),
                         "--diagnostics", str(tmp_path / "diagnostics"), "--datasets", str(data),
                         "--iid-dataset", "terminal_prerl_v2", "--out", str(tmp_path / "compare")],
                        capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    report = json.loads((tmp_path / "compare" / "decision.json").read_text())
    assert report["iid_dataset"] == "terminal_prerl_v2" and report["probes"] == {}
    assert set(report["jitter"]) == {"bf16"} and set(report["batching"]) == {"bf16"}
    # the same numbers from the v2 probe, the full-precision diagnostics and the v2 features of both views
    from safetensors.torch import load_file

    probe = SelectedProbe.load(str(tmp_path / "probes" / "bf16" / "terminal_prerl_v2" / "with_reasoning" / "followup"))
    diag = load_file(str(tmp_path / "diagnostics" / "bf16" / "diag.safetensors"))
    a, b = (probe.logit(diag[k][:, probe.layer].double().numpy()) for k in ("followup_batched", "followup_alone"))
    p99 = float(np.percentile(np.abs(a - b), 99))
    assert abs(report["jitter"]["bf16"]["dlogit_p99"] - p99) < 1e-9
    items = {json.loads(line)["id"]: json.loads(line) for line in open(data / "terminal_prerl_v2" / "items.jsonl")}
    z, y = [], []
    for view in ("with_reasoning", "without_reasoning"):
        src = LayerSource(str(readers / "bf16" / "terminal_prerl_v2"), view, "followup")
        rows = np.flatnonzero(src.keep)
        z += list(probe.logit(src.layer(probe.layer, rows)))
        y += [bool(items[src.ids[i]]["diagnostics"]["used_verifier"]) for i in rows]
    near, y = np.abs(np.array(z) - probe.threshold_1) <= p99, np.array(y)
    batching = report["batching"]["bf16"]
    assert abs(batching["near_threshold_hacks"] - near[y].mean()) < 1e-12
    assert abs(batching["near_threshold_negatives"] - near[~y].mean()) < 1e-12
    decision = report["decision"]
    assert decision["fix_batching"] == batching["fix_batching"] and decision["reader"].startswith("undecided")


def _compare_readers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("compare_readers", os.path.join(REPO, "experiments",
                                                                                   "compare_readers.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_massive_dimensions_of_yes_are_reported_per_layer(tmp_path):
    """Synthetic diagnostics with 43 layers. In layers 10–35, 8 planted dimensions of "Yes" hold v² / (v² + 1) of its
    squared norm (v grows with the layer; the other 8 dimensions are 1). Five inputs plant dimensions 0–7, the sixth
    0–6 and 8, so 47 of the 48 top-8 dimensions are the layer's 8 most frequent ones. Elsewhere all 16 are equal."""
    import types

    cr = _compare_readers()
    n, layers, d = 6, 43, 16
    yes = torch.ones(n, layers, d)
    idx = torch.arange(8).repeat(n, layers, 1)  # outside 10–35: all equal, any 8 hold half the norm
    v = {layer: 1 + (layer - 10) / 8 for layer in range(10, 36)}  # exact in bfloat16
    for layer, value in v.items():
        for i in range(n):
            planted = [0, 1, 2, 3, 4, 5, 6, 8] if i == n - 1 else list(range(8))
            yes[i, layer, planted] = value
            idx[i, layer] = torch.tensor(planted)
    folder = tmp_path / "diagnostics" / "nvfp4"
    folder.mkdir(parents=True)
    yes = yes.to(torch.bfloat16)
    save_file({"followup_alone": yes, "followup_batched": yes.clone(), "massive_idx": idx.to(torch.int16),
               "massive_val": yes.float().gather(-1, idx)}, str(folder / "diag.safetensors"))
    m = cr.massive_dims(types.SimpleNamespace(diagnostics=str(tmp_path / "diagnostics")), "nvfp4")
    assert (m["n"], m["k"]) == (n, 8)
    share = {layer: value ** 2 / (value ** 2 + 1) for layer, value in v.items()}
    for layer in range(layers):
        want = share.get(layer, 0.5)
        assert abs(m["top_share_by_layer"][layer] - want) < 1e-4, layer
        assert abs(m["common_share_by_layer"][layer] - want) < 1e-4, layer  # the median input: one of the five
        assert m["consistency_by_layer"][layer] == (round(47 / 48, 4) if layer in share else 1.0), layer
        assert m["common_dims_by_layer"][layer] == list(range(8))
    med = m["median_layers_10_35"]  # layers 10–35 exactly: adding 9 or 36 (share 0.5) would lower it
    assert abs(med["top_share"] - float(np.median(list(share.values())))) < 1e-12
    assert abs(med["consistency"] - 47 / 48) < 1e-12
    lines = cr.massive_lines({"nvfp4": m, "bf16": {}})
    assert sum(line.startswith("| ") and line[2:].split(" |")[0].isdigit() for line in lines) == layers
    assert any("medians over layers 10–35: top-8 share" in line for line in lines)
    assert cr.massive_dims(types.SimpleNamespace(diagnostics=str(tmp_path / "none")), "nvfp4") == {}


def test_rules_that_did_not_run_are_not_evaluated_rather_than_false(tmp_path):
    cr = _compare_readers()

    def decide(**report):
        cr.finish({"probes": {}, "jitter": {}, **report}, str(tmp_path / "out"))
        return json.loads((tmp_path / "out" / "decision.json").read_text())["decision"]

    d = decide(readouts=["mean"], batching={}, sink={})
    assert (d["fix_batching"], d["sink_like_yes"]) == (None, None)
    assert "follow-up readout" in d["not_evaluated"]["fix_batching"]
    d = decide(batching={"nvfp4": {}}, sink={"nvfp4": {}, "bf16": {}})  # no readouts: an older file
    assert (d["fix_batching"], d["sink_like_yes"]) == (None, None)
    assert d["not_evaluated"]["sink_like_yes"] == "no reader had diagnostics (done.json)"
    report_md = (tmp_path / "out" / "report.md").read_text()
    assert "Fix batching = not evaluated (no reader had both diagnostics" in report_md
    d = decide(readouts=["followup"], batching={"nvfp4": {"fix_batching": False}, "bf16": {}},
               sink={"nvfp4": {"sink_like": False}, "bf16": {}})  # ran for the 4-bit reader only: decided by it
    assert (d["fix_batching"], d["sink_like_yes"], d["not_evaluated"]) == (False, False, {})
    d = decide(readouts=["followup"], batching={"nvfp4": {"fix_batching": True}}, sink={"bf16": {"sink_like": True}})
    assert (d["fix_batching"], d["sink_like_yes"]) == (True, True)


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
    probe_dir = out / "diverse_deception" / "without_reasoning" / "followup"
    meta = json.loads((probe_dir / "meta.json").read_text())
    assert set(meta["code"]) == CODE_KEYS and meta["features_code"] is None  # the synthetic features have no index
    assert json.loads((out / "summary.json").read_text())["ood/followup"]["code"] == meta["code"]

    def fit(*args):
        return subprocess.run([sys.executable, os.path.join(REPO, "experiments", "fit_probes.py"), "--reader",
                               "synthetic", "--features", str(feats), "--datasets", str(data), "--out", str(out),
                               "--repeats", "1", "--jobs", "2", "--probes", "ood", "--skip-baselines", *args],
                              capture_output=True, text=True, cwd=REPO, timeout=1800)

    stamp = (probe_dir / "probe.npz").stat().st_mtime_ns
    cp = fit("--readouts", "mean,followup")  # would rewrite the follow-up probe: refused before any fitting
    assert cp.returncode == 2 and "--overwrite" in cp.stderr and str(probe_dir) in cp.stderr
    assert (probe_dir / "probe.npz").stat().st_mtime_ns == stamp and not (out / "diverse_deception" /
                                                                         "without_reasoning" / "mean").exists()
    cp = fit("--readouts", "mean")  # other readouts into the same --out: added, and summary.json accumulates
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    assert set(json.loads((out / "summary.json").read_text())) == {"ood/followup", "ood/mean"}

    spec = importlib.util.spec_from_file_location("ood_refits", os.path.join(REPO, "experiments", "ood_refits.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
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
    assert set(summary["code"]) == CODE_KEYS
    again = ["--probe", str(probe_dir), "--features", str(feats), "--datasets", str(data), "--out",
             str(tmp_path / "refits"), "--refits", "3", "--skip-layers"]
    with pytest.raises(SystemExit):  # the folder holds an earlier run's outputs
        mod.main(again)
    assert mod.main([*again, "--overwrite"]) == 0
    assert json.loads((tmp_path / "refits" / "summary.json").read_text())["refits"] == 3
    assert not (tmp_path / "refits" / "by_layer.csv").exists()  # the earlier run's, not this one's


def test_learning_curves_run_on_synthetic_probes(tmp_path):
    """experiments/learning_curves.py: refits at the fitted layer and C over fractions of the training data, for IID
    (with the concealed test hacks) and OOD (with per-source fits), and the full-data fit reproduces the probe."""
    import importlib.util

    data, feats = make_world(tmp_path)
    out = tmp_path / "probes"
    cp = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "fit_probes.py"), "--reader", "synthetic",
                         "--features", str(feats), "--datasets", str(data), "--out", str(out), "--repeats", "1",
                         "--jobs", "2", "--readouts", "followup", "--skip-baselines"],
                        capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    spec = importlib.util.spec_from_file_location("curves", os.path.join(REPO, "experiments", "learning_curves.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--probes", str(out), "--features", str(feats), "--datasets", str(data),
                     "--out", str(tmp_path / "curves"), "--fractions", "0.25,1", "--repeats", "3", "--threads", "2",
                     "--source-refits", "2"]) == 0
    summary = json.loads((tmp_path / "curves" / "summary.json").read_text())
    full = {r["view"]: r for r in summary["iid"] if r["fraction"] == 1}
    fitted = json.loads((out / "terminal_prerl" / "with_reasoning" / "followup" / "metrics.json").read_text())
    assert abs(full["with_reasoning"]["test_auroc"]["median"] - fitted["test_auroc"]) < 1e-9
    assert [r["fits"] for r in summary["iid"]] == [3, 1, 3, 1]
    assert all(r["concealed_test_hacks_recall"]["median"] > 0.5 for r in summary["iid"] if r["fraction"] == 1)
    ood_full = next(r for r in summary["ood"] if r["fraction"] == 1)
    assert ood_full["transfer_auroc_with_reasoning"]["median"] > 0.8
    assert {(r["source"], r["trained_on"]) for r in summary["ood_sources"]} == {
        ("all", "all"), *((s, h) for s in ("a", "b") for h in ("only", "all_but"))}
    assert all(r["fits"] == 2 for r in summary["ood_sources"]) and set(summary["code"]) == CODE_KEYS
    with pytest.raises(SystemExit):  # an earlier run's outputs are there; --overwrite would replace them
        mod.main(["--probes", str(out), "--features", str(feats), "--datasets", str(data),
                  "--out", str(tmp_path / "curves")])


def test_an_iid_dataset_other_than_terminal_prerl_fits_under_the_amendments(tmp_path):
    """--iid-dataset terminal_prerl_v2 reads and writes that dataset only, and its probes record both amendments.
    Amendment 2 fixes the features (full precision, one episode per forward): others are refused before any fit."""
    import hashlib
    import shutil

    data, feats = make_world(tmp_path)
    for root in (data, feats):
        shutil.move(str(root / "terminal_prerl"), str(root / "terminal_prerl_v2"))
    out = tmp_path / "probes"
    cmd = [sys.executable, os.path.join(REPO, "experiments", "fit_probes.py"), "--reader", "synthetic", "--features",
           str(feats), "--datasets", str(data), "--out", str(out), "--repeats", "1", "--jobs", "2", "--readouts",
           "followup", "--probes", "iid", "--skip-sensitivity", "--iid-dataset", "terminal_prerl_v2"]
    index = feats / "terminal_prerl_v2" / "index.json"
    reader = {"loader": "transformers", "revision": "8cc5877b44d343f88b92086aa1fb72897950f06a"}
    for wrong in ({"reader": {**reader, "loader": "nvfp4"}, "max_batch_tokens": 1},
                  {"reader": reader, "max_batch_tokens": 65536}, {}):
        index.write_text(json.dumps(wrong))
        cp = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=600)
        assert cp.returncode == 2 and "amendment_2" in cp.stderr and not out.exists(), cp.stderr[-2000:]
    index.write_text(json.dumps({"reader": {**reader, "path": "/mnt/ram/Inkling-Small", "dtype": "torch.bfloat16"},
                                 "max_batch_tokens": 1}))
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    assert not (out / "terminal_prerl").exists()
    shas = {}
    for n in (1, 2):
        with open(os.path.join(REPO, "docs", f"inkling_probes_amendment_{n}.md"), "rb") as fh:
            shas[n] = hashlib.sha256(fh.read()).hexdigest()
    for view in ("with_reasoning", "without_reasoning"):
        meta = json.loads((out / "terminal_prerl_v2" / view / "followup" / "meta.json").read_text())
        assert meta["dataset"] == "terminal_prerl_v2" and meta["amendment_sha256"] == shas[1]
        assert meta["amendment_2_sha256"] == shas[2] and meta["reader_info"]["revision"] == reader["revision"]
        assert "prereg_sha256" in meta
