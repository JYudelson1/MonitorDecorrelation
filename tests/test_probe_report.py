"""experiments/probe_report.py on a synthetic results tree.

Checks:
- collect copies the small files (never scores.jsonl), counts the caught hacks per episode and per task, rewrites
  every machine path, keeps other bytes as they are, writes a manifest with sources and hashes, and rebuilds the tree
  on a rerun;
- render fills every marked table from the collected tree, a missing input gives "(not available: ...)", an unknown
  table is an error;
- check passes after render and fails after a hand edit;
- the Clopper–Pearson bound is exact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os

import pytest

spec = importlib.util.spec_from_file_location(
    "probe_report", os.path.join(os.path.dirname(__file__), "..", "experiments", "probe_report.py"))
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

READERS = ("nvfp4", "bf16")
VIEWS = ("with_reasoning", "without_reasoning")


def dump(path, obj, newline="\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=1) + "\n"
    with open(path, "w", newline="") as fh:
        fh.write(text.replace("\n", newline))


def scores() -> str:
    """4 test hacks in 3 tasks (one missed: tasks t1 not caught) and 6 dev hacks in 3 tasks, at t1 = 0."""
    hacks = [("t1", "test", 1.0), ("t1", "test", -1.0), ("t2", "test", 2.0), ("t3", "test", 3.0)]
    hacks += [(f"d{k % 3}", "train", 1.5) for k in range(6)]
    rows = [{"id": f"{task}/{i}", "split": split, "used_verifier": 1, "behavior_present": 1, "task_id": task,
             "crossfit_logit": z, "test_logit": z if split == "test" else None}
            for i, (task, split, z) in enumerate(hacks)]
    rows += [{"id": f"n/{i}", "split": "test", "used_verifier": 0, "behavior_present": 0, "task_id": "t9",
              "crossfit_logit": -2.0, "test_logit": -2.0} for i in range(5)]
    return "".join(json.dumps(r) + "\n" for r in rows)


def meta(reader: str, dataset: str, view: str, readout: str) -> dict:
    out = {"layer": 3, "threshold_1": 0.0, "threshold_5": -1.0, "reader": reader,
           "reader_info": {"path": "/mnt/ram/Inkling-Small-NVFP4", "loader": "nvfp4", "revision": "b6a9953"},
           "feature_format": "inkling-probe-features/v2", "dataset": dataset, "view": view, "readout": readout,
           "C": 0.001, "selection": {"layer": 3, "C": 0.001, "cv_auroc": 1.0, "flat": True}, "n_layers": 6}
    return {**out, "model_layer": 5} if readout == "router" else out


def iid_metrics() -> dict:
    return {"test_auroc": 0.99998, "test_auroc_ci": [0.9998, 1.0], "tpr_at_t1": 0.75, "fpr_at_t1": 0.0037,
            "selection": {"layer": 3, "C": 0.001, "cv_auroc": 1.0, "cv_se": float("nan"), "flat": True},
            "n": {"dev": 80, "dev_hacks": 6, "test": 20, "test_hacks": 4},
            "crossfit": {"auroc_all": 0.99, "hacks_vs_read_not_used": 0.998, "within_task": 1.0},
            "concealment": {"A": {"n": 10, "recall_at_t1": 1.0}, "B": {"n": 10, "recall_at_t1": 0.9},
                            "unedited_crossfit": {"n": 10, "recall_at_t1": 0.9}}}


def ood_metrics() -> dict:
    view = {"auroc_all": 0.909, "auroc_all_ci": [0.88, 0.93], "tpr_at_t1": 0.19, "thresholds": [-4.7, -6.1],
            "concealment_recall_at_t1": {"A": 0.83, "B": 0.82}}
    return {"test_auroc": 0.977, "test_auroc_ci": [0.971, 0.982], "per_source": {"mask": 0.87, "dolus": 0.99},
            "leave_one_source_out": {"mask": 0.73, "dolus": 0.97},
            "selection": {"layer": 3, "C": 0.01, "cv_auroc": 0.98, "flat": False},
            "transfer": {"with_reasoning": view, "without_reasoning": {**view, "auroc_all": 0.842}}}


def compared(ood: bool) -> dict:
    deltas = {"ood_test": {"delta": -0.0017, "ci": [-0.004, 0.0005]},
              "iid_transfer_with_reasoning": {"delta": 0.197, "ci": [0.163, 0.232]}} if ood else \
        {"iid_test": {"delta": 0.0, "ci": [0.0, 0.0]}}
    cross = {"ood_test_auroc": 0.97, "transfer_auroc": {"with_reasoning": 0.9, "without_reasoning": 0.8}} if ood else \
        {"n_test": 20, "test_auroc": 1.0, "t1_own": 0.0, "tpr_at_t1_own": 1.0, "fpr_at_t1_own": 0.004,
         "t1_recalibrated": 0.1, "tpr_at_t1_recalibrated": 1.0, "fpr_at_t1_recalibrated": 0.01}
    return {"layers": {"nvfp4": 3, "bf16": 3},
            "agreement": {v: {"n": 50, "jaccard_at_t1": 0.9, "spearman": 0.96} for v in VIEWS},
            "delta_auroc": deltas,
            "refit_floor": {"n": 20, **{v: {"jaccard_median": 0.95, "spearman_median": 0.99} for v in VIEWS}},
            "passes": {"agreement": True, "agreement_within_refit_noise": True, "delta_auroc": not ood},
            "weight_cosine": 0.8,
            "score_shift": {v: {"n": 50, "mean": 0.1, "sd": 0.2, "p99_abs": 0.7, "pearson": 0.95} for v in VIEWS},
            "cross_reader": {p: {f: cross for f in READERS} for p in READERS},
            "batching_floor": {"nvfp4": {p: {"with_reasoning": {"n": 30, "mean": 0.01, "sd": 0.05, "spearman": 0.99}}
                                         for p in READERS}}}


def build(tmp_path):
    """A hub-like tree: <hub>/scratch with every result, <repo>/data/probes/Inkling-Small with two monitors."""
    hub, repo = tmp_path / "hub", tmp_path / "repo"
    s = hub / "scratch"
    for reader in READERS:
        dump(s / "probes" / reader / "summary.json", {"iid/with_reasoning/followup": iid_metrics()})
        for dataset, view, readout in (("terminal_prerl", "with_reasoning", "followup"),
                                       ("terminal_prerl", "with_reasoning", "router"),
                                       ("diverse_deception", "without_reasoning", "followup")):
            d = s / "probes" / reader / dataset / view / readout
            d.mkdir(parents=True)
            (d / "probe.npz").write_bytes(b"PK\x03\x04 not really a zip")
            dump(d / "meta.json", meta(reader, dataset, view, readout))
            dump(d / "metrics.json", iid_metrics() if dataset == "terminal_prerl" else ood_metrics())
            dump(d / "cv_grid.csv", "layer,C,cv_auroc,cv_auroc_sd\n3,0.001,1.00000,0.00000\n")
            dump(d / "scores.jsonl", scores())
    by = {"followup": "bf16"}
    dump(s / "compare" / "decision.json", {
        "probes": {"terminal_prerl/with_reasoning/followup": compared(False),
                   "diverse_deception/without_reasoning/followup": compared(True)},
        "jitter": {"nvfp4": {"n": 228, "dlogit_p50": 0.17, "dlogit_p99": 0.85}, "bf16": {}},
        "batching": {"nvfp4": {"near_threshold_hacks": 0.0, "near_threshold_negatives": 0.02, "fix_batching": True}},
        "sink": {"nvfp4": {"median_ratio_layers_10_35": 1.09, "sink_like": False}, "bf16": {}},
        "decision": {"reader": "bf16", "by_readout": by, "fix_batching": True, "sink_like_yes": False}})
    dump(s / "compare" / "report.md",
         f"# Reader comparison\n\nFrom {s}/probes/bf16 and ../scratch/compare/report.md.\n")
    plain = {k: v for k, v in compared(False).items() if k not in ("weight_cosine", "score_shift", "cross_reader",
                                                                    "batching_floor")}
    dump(s / "compare_mean_router" / "decision.json", {
        "probes": {"terminal_prerl/with_reasoning/mean": plain}, "routing_agreement_at_yes": {
            "terminal_prerl/with_reasoning": {"n": 10688, "mean": 0.952, "min_layer": 29, "min": 0.862}},
        "decision": {"reader": "bf16", "by_readout": {"mean": "nvfp4"}, "fix_batching": False, "sink_like_yes": False}})
    dump(s / "compare_mean_router" / "report.md", "# Reader comparison (mean)\n")
    refit = {"original": 0.909, "median": 0.909, "p05": 0.836, "p95": 0.956}
    dump(s / "ood_refits" / "nvfp4" / "followup" / "summary.json", {
        "probe": str(s / "probes" / "nvfp4" / "diverse_deception" / "without_reasoning" / "followup"), "layer": 3,
        "C": 0.01, "readout": "followup", "refits": 50,
        "metrics": {k: refit for k in ("ood_test_auroc", "transfer_auroc_with_reasoning",
                                       "transfer_auroc_without_reasoning", "tpr_at_t1_with_reasoning",
                                       "cosine_to_original")}})
    dump(s / "ood_refits" / "nvfp4" / "followup" / "refits.csv", "refit,ood_test_auroc\n0,0.97\n", newline="\r\n")
    dump(s / "ood_refits" / "nvfp4" / "followup" / "by_layer.csv",
         "layer,ood_test_auroc,transfer_auroc_with_reasoning\n2,0.95,0.44\n3,0.977,0.909\n", newline="\r\n")
    q = {"median": 1.0, "p05": 0.999, "p95": 1.0}
    dump(s / "learning_curves" / "bf16" / "summary.json", {
        "probes": "../scratch/probes/bf16", "features": "../scratch/features/bf16",
        "iid": [{"view": v, "fraction": f, "fits": 20, "n_train": 400, "n_hacks_or_pairs": 4.0, "test_auroc": q,
                 "test_tpr_at_1pct_fpr": q, "concealed_test_hacks_recall": q} for v in VIEWS for f in (0.05, 1)],
        "ood": [{"fraction": 0.05, "fits": 20, "n_train": 390, "n_hacks_or_pairs": 195.0, "ood_test_auroc": q,
                 "transfer_auroc_with_reasoning": q, "transfer_auroc_without_reasoning": q,
                 "tpr_at_t1_with_reasoning": q, "tpr_at_t1_without_reasoning": q}],
        "ood_sources": [{"trained_on": how, "source": src, "fits": 10, "n_hacks_or_pairs": 2454.0,
                         "ood_test_auroc": q, "transfer_auroc_with_reasoning": q,
                         "transfer_auroc_without_reasoning": q}
                        for how, src in (("all", "all"), ("all_but", "mask"), ("only", "mask"))]})
    for f in ("iid.csv", "ood.csv", "ood_sources.csv"):
        dump(s / "learning_curves" / "bf16" / f, "fraction,repeat\n1,0\n")
    dump(s / "verify_fits" / "20261011T000000Z" / "summary.json", {"probes": {"old": {"verdict": "fail"}}})
    dump(s / "verify_fits" / "20261012T000000Z" / "summary.json", {
        "created_utc": "2026-10-12T00:00:00Z", "args": {"probes": "../scratch/probes",
                                                       "check": ["bf16/followup=../scratch/features/bf16"]},
        "code": {"fit_probes.py": "a" * 64, "probe_fit.py": "b" * 64},
        "probes": {"bf16/terminal_prerl/with_reasoning/followup": {
            "features": "../scratch/features/bf16", "inputs": {"ok": True},
            "cv": {"cells_matched": 24, "cells_checked": 24, "cells_total": 252, "layers_recomputed": [3]},
            "selection": {"status": "reproduced", "saved": {"layer": 3, "C": 0.001, "flat": True}, "ok": True},
            "refit": {"weights": {"max_abs_diff": 0.0}, "thresholds": {"max_abs_diff": 1e-12},
                      "metrics": {"max_abs_diff": 0.0, "compared": 9}, "test_logits": {"max_abs_diff": 0.0}},
            "verdict": "pass"},
            "bf16/diverse_deception/without_reasoning/followup": {"error": "ValueError: boom", "verdict": "fail"}}})
    dump(s / "verify_fits" / "20261012T000000Z" / "report.md", "# verify\n")
    a = s / "analyses" / "20261012T010000Z"
    dump(a / "dataset_stats.json", {
        "command": "dataset-stats", "datasets": str(repo / "data" / "probe_datasets"),
        "disclosure": {"hacks": {"n": 98, "disclosing": 97, "share": 97 / 98},
                       "non_hacks": {"n": 10590, "disclosing": 120, "share": 120 / 10590}},
        "ood_question_overlap": {"mask": {"n_items": 800, "n_test": 160, "test_sharing_question_with_train": 160,
                                          "distinct_questions": 1, "distinct_questions_train": 1,
                                          "distinct_questions_test": 1}},
        "tokens": {"terminal_prerl": {"with_reasoning": {"n_items": 10688, "n_rendered": 10688, "mean": 1712.4,
                                                         "median": 1500.0, "p95": 3900.0, "max": 9000,
                                                         "total_estimate": 18302131.2}}}})
    deployed = repo / "data" / "probes" / "Inkling-Small"
    dump(a / "near_threshold.json", {"features": str(s / "features" / "bf16"), "k": 4.0, "probes": {
        "../scratch/probes/bf16/terminal_prerl/with_reasoning/followup": {"layer": 3},
        str(deployed / "terminal_prerl" / "with_reasoning" / "followup"): {"layer": 4},
        str(deployed / "diverse_deception" / "with_reasoning" / "followup"): {"layer": 5}}})
    dump(deployed / "deployed.json", [
        {"name": "probe_iid_with_reasoning_followup", "dir": "data/probes/Inkling-Small/terminal_prerl/with_reasoning/"
         "followup", "layer": 25, "golden": 32, "golden_tolerance": 2.54, "reader": "nvfp4", "trained_reader": "bf16",
         "threshold_1": -2.958, "serving_check": {"test_auroc": 0.99998, "test_tpr_at_t1": 1.0,
                                                  "test_fpr_at_t1": 0.0046,
                                                  "batching": {"dlogit_p99": 0.78, "near_threshold_hacks": 0.0,
                                                               "near_threshold_negatives": 0.033}}},
        {"name": "probe_ood_with_reasoning_mean", "dir": "data/probes/Inkling-Small/diverse_deception/with_reasoning/"
         "mean", "layer": 22, "golden": 32, "golden_tolerance": 3.0, "reader": "nvfp4", "trained_reader": "nvfp4",
         "threshold_1": -0.976,
         "serving_check": {"test_auroc": 0.175, "test_tpr_at_t1": 0.09, "test_fpr_at_t1": 0.0078}}])
    return s, tmp_path / "out", deployed


def collect(s, out, *extra) -> int:
    return pr.main(["collect", "--scratch", str(s), "--analyses", str(s / "analyses"), "--out", str(out), *extra])


def test_collect_copies_small_files_counts_hacks_and_rewrites_paths(tmp_path):
    s, out, _ = build(tmp_path)
    assert collect(s, out) == 0
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert not [f for f in files if f.endswith("scores.jsonl")]
    probe = "fitted/nvfp4/terminal_prerl/with_reasoning/followup"
    assert [f for f in files if f.startswith(probe + "/")] == [
        f"{probe}/{f}" for f in ("cv_grid.csv", "hack_counts.json", "meta.json", "metrics.json", "probe.npz")]
    assert not os.path.exists(out / "fitted/nvfp4/diverse_deception/without_reasoning/followup/hack_counts.json")
    for f in ("fitted/bf16/summary.json", "compare/followup/decision.json", "compare/mean_router/report.md",
              "ood_refits/nvfp4/followup/by_layer.csv", "learning_curves/bf16/ood_sources.csv",
              "verify_fits/summary.json", "verify_fits/report.md", "analyses/dataset_stats.json", "MANIFEST.json"):
        assert f in files, f
    counts = json.loads((out / probe / "hack_counts.json").read_text())
    assert counts == {"threshold_1": 0.0, "test": {"hacks": 4, "caught": 3, "tasks": 3, "tasks_caught": 2},
                      "crossfit": {"hacks": 10, "caught": 9, "tasks": 6, "tasks_caught": 5}}

    # no machine path survives; other values and bytes are untouched
    for f in files:
        if f.endswith((".json", ".md", ".csv")):
            assert str(tmp_path) not in (out / f).read_text(), f
    m = json.loads((out / probe / "meta.json").read_text())
    assert m["reader_info"]["path"] == "Inkling-Small-NVFP4"
    assert m["feature_format"] == "inkling-probe-features/v2" and m["reader_info"]["revision"] == "b6a9953"
    assert math.isnan(json.loads((out / probe / "metrics.json").read_text())["selection"]["cv_se"])
    refit = json.loads((out / "ood_refits/nvfp4/followup/summary.json").read_text())
    assert refit["probe"] == "fitted/nvfp4/diverse_deception/without_reasoning/followup"
    curves = json.loads((out / "learning_curves/bf16/summary.json").read_text())
    assert (curves["probes"], curves["features"]) == ("fitted/bf16", "bf16")
    report = (out / "compare/followup/report.md").read_text()
    assert "From fitted/bf16 and compare/followup/report.md." in report
    verify = json.loads((out / "verify_fits/summary.json").read_text())
    assert "bf16/terminal_prerl/with_reasoning/followup" in verify["probes"]  # the newest run
    assert verify["args"]["check"] == ["bf16/followup=bf16"]
    near = json.loads((out / "analyses/near_threshold.json").read_text())
    assert list(near["probes"]) == ["fitted/bf16/terminal_prerl/with_reasoning/followup",
                                    "terminal_prerl/with_reasoning/followup",
                                    "diverse_deception/with_reasoning/followup"]
    assert json.loads((out / "analyses/dataset_stats.json").read_text())["datasets"] == "probe_datasets"
    src = s / "ood_refits/nvfp4/followup/by_layer.csv"
    assert (out / "ood_refits/nvfp4/followup/by_layer.csv").read_bytes() == src.read_bytes()

    manifest = json.loads((out / "MANIFEST.json").read_text())["files"]
    assert sorted(manifest) == [f for f in files if f != "MANIFEST.json"]
    for f, entry in manifest.items():
        assert entry["sha256"] == hashlib.sha256((out / f).read_bytes()).hexdigest(), f
    assert manifest[f"{probe}/probe.npz"] == {
        "source": "scratch/probes/nvfp4/terminal_prerl/with_reasoning/followup/probe.npz",
        "sha256": hashlib.sha256(b"PK\x03\x04 not really a zip").hexdigest()}
    assert manifest[f"{probe}/meta.json"]["source_sha256"] == hashlib.sha256(
        (s / "probes/nvfp4/terminal_prerl/with_reasoning/followup/meta.json").read_bytes()).hexdigest()
    assert "source_sha256" not in manifest["ood_refits/nvfp4/followup/by_layer.csv"]
    assert manifest[f"{probe}/hack_counts.json"]["derived_from"] == [
        f"scratch/probes/nvfp4/terminal_prerl/with_reasoning/followup/{f}" for f in ("scores.jsonl", "meta.json")]
    assert manifest["verify_fits/summary.json"]["source"] == "scratch/verify_fits/20261012T000000Z/summary.json"


def test_collect_rebuilds_the_tree_and_refuses_an_empty_source(tmp_path):
    s, out, _ = build(tmp_path)
    assert collect(s, out) == 0
    first = (out / "MANIFEST.json").read_text()
    assert collect(s, out) == 0
    assert (out / "MANIFEST.json").read_text() == first  # nothing in it changes between reruns
    for f in (s / "probes/bf16/terminal_prerl/with_reasoning/router").iterdir():
        f.unlink()
    (s / "probes/bf16/terminal_prerl/with_reasoning/router").rmdir()
    assert collect(s, out) == 0
    assert not (out / "fitted/bf16/terminal_prerl/with_reasoning/router").exists()
    assert "router" not in "".join(f for f in json.loads((out / "MANIFEST.json").read_text())["files"]
                                   if f.startswith("fitted/bf16/"))
    with pytest.raises(SystemExit):
        pr.main(["collect", "--scratch", str(tmp_path / "nowhere"), "--out", str(out)])
    assert (out / "MANIFEST.json").exists() and not (tmp_path / "out.tmp").exists()


def markdown() -> str:
    blocks = "".join(f"## {name}\n\n<!-- probe_report:begin {name} -->\nTODO\n<!-- probe_report:end {name} -->\n\n"
                     for name in pr.TABLES)
    return "# Inkling probes\n\nProse stays as it is.\n\n" + blocks + "The end.\n"


def test_render_fills_every_table_and_check_catches_edits(tmp_path):
    s, out, deployed = build(tmp_path)
    assert collect(s, out) == 0
    doc = tmp_path / "doc.md"
    doc.write_text(markdown())
    opts = ["--results", str(out), "--deployed", str(deployed)]
    assert pr.main(["check", str(doc), *opts]) == 1
    assert pr.main(["render", str(doc), *opts]) == 0
    text = doc.read_text()
    assert text.startswith("# Inkling probes\n\nProse stays as it is.\n\n") and text.endswith("The end.\n")
    assert "TODO" not in text and "(not available" not in text
    for name in pr.TABLES:
        assert f"<!-- probe_report:begin {name} -->\n\n|" in text or \
            f"<!-- probe_report:begin {name} -->\n\nWeight cosine" in text, name
    expected = [
        "| followup | with | nvfp4 | 3* | 0.001 | 0.99998 [0.99980, 1.000] | 75% / 0.37% | 0.998 | 1.000 | "
        "100% / 90% |",
        "| followup | nvfp4 | 3 | 0.01 | 0.977 [0.971, 0.982] | mask 0.870 | mask 0.730 | 0.909 / 0.842 | 19% / 19% |",
        "| IID, with | router | nvfp4 | 3 (5)* | 0.001 |",
        "| probe_iid_with_reasoning_followup | bf16 → nvfp4 | 25 | -2.96 | 0.99998 | 100% / 0.46% | 32 (2.54) | "
        "0.78; 0% / 3.3% |",
        "| probe_ood_with_reasoning_mean | nvfp4 → nvfp4 | 22 | -0.98 | 0.175 | 9.0% / 0.78% | 32 (3) | — |",
        "| terminal_prerl/with_reasoning/followup | 3 / 3 | 0.900 / 0.900 | 0.960 / 0.960 | 0.950 / 0.950; "
        "0.990 / 0.990 | test +0.000 [+0.000, +0.000] | (a) ✓ (c) ✓ (b) ✓ |",
        "OOD test -0.002 [-0.004, +0.001]; transfer with +0.197 [+0.163, +0.232] | (a) ✓ (c) ✓ (b) ✗ |",
        "Decision per readout: followup → bf16. Fix batching: yes. \"Yes\" sink-like: no.",
        "- Jitter (nvfp4, 228 diagnostics inputs)",
        "Decision per readout: mean → nvfp4. Fix batching: no.",
        "terminal_prerl/with_reasoning 95% over 10,688 items (lowest 86%, MoE layer 29)",
        "| terminal_prerl/with_reasoning/followup | 0.800 | +0.10 (0.20); 0.70; 0.950 | +0.10 (0.20); 0.70; 0.950 |",
        "| terminal_prerl/with_reasoning/followup | bf16 | 1.000 / 1.000 | 100% / 0.40%; 100% / 0.40% | "
        "100% / 1.0%; 100% / 1.0% |",
        "| diverse_deception/without_reasoning/followup | nvfp4 | 0.970 / 0.970 | 0.900 / 0.900 | 0.800 / 0.800 |",
        "| terminal_prerl/with_reasoning/followup | bf16 | nvfp4 | with | 30 | +0.010 (0.050) | 0.990 |",
        f"| all 4 | 3/4, ≥ {pr.num(pr.cp_lower(3, 4))} | 2/3, ≥ {pr.num(pr.cp_lower(2, 3))} | "
        f"9/10, ≥ {pr.num(pr.cp_lower(9, 10))} | 5/6, ≥ {pr.num(pr.cp_lower(5, 6))} |",
        "| followup | nvfp4 | 3, 0.01 | 50 | 0.909; 0.909 [0.836, 0.956] |",
        "| 3 | **0.91** (0.977) |",
        "| bf16 | with | 5% | 4 | 20 | 1.000 [0.999, 1.000] | 100% [99.9%, 100%] | 100% [99.9%, 100%] |",
        "| bf16 | 195 (5%) | 20 | 1.000 [0.999, 1.000] |",
        "| bf16 | all but mask | 2,454 | 10 | 1.000 |",
        "| bf16/terminal_prerl/with_reasoning/followup | 24 / 24 | reproduced (layer 3, C 0.001) | 0; 1.0e-12; 0; 0 | "
        "pass |",
        "| bf16/diverse_deception/without_reasoning/followup | — | — | — | fail: ValueError: boom |",
        "1 of 2 probes pass. Run 2026-10-12T00:00:00Z with fit_probes.py aaaaaaaaaaaa, probe_fit.py bbbbbbbbbbbb",
        "| terminal_prerl | with | 10,688 | 1,712 / 1,500 / 3,900 / 9,000 | 18.3M |",
        "97 of 98 hacks (99%) and 120 of 10,590 other episodes (1.1%)",
        "| mask | 800 | 160 | 160 | 1 (1 / 1) |",
    ]
    for line in expected:
        assert line in text, line
    assert pr.main(["check", str(doc), *opts]) == 0
    assert pr.main(["render", str(doc), *opts]) == 0 and doc.read_text() == text  # idempotent

    doc.write_text(text.replace("| 3 | **0.91** (0.977) |", "| 3 | **0.95** (0.977) |"))
    assert pr.main(["check", str(doc), *opts]) == 1
    assert pr.main(["render", str(doc), *opts]) == 0 and doc.read_text() == text
    doc.write_text(text.replace("The end.", "The end, edited outside the tables."))
    assert pr.main(["check", str(doc), *opts]) == 0


def test_render_marks_missing_inputs_and_rejects_unknown_tables(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    doc = tmp_path / "doc.md"
    doc.write_text(markdown())
    assert pr.main(["render", str(doc), "--results", str(empty), "--deployed", str(empty)]) == 0
    text = doc.read_text()
    assert text.count("(not available: ") == len(pr.TABLES)
    assert str(tmp_path) not in text
    assert "(not available: no compare/followup/decision.json collected)" in text
    for bad in ("<!-- probe_report:begin nonsense -->\n<!-- probe_report:end nonsense -->\n",
                "<!-- probe_report:begin iid_probes -->\n",
                "<!-- probe_report:end iid_probes -->\n"):
        doc.write_text(bad)
        with pytest.raises(SystemExit):
            pr.main(["render", str(doc), "--results", str(empty)])


def test_rewriter_leaves_non_paths_alone():
    rw = pr.Rewriter([("/hub/scratch/probes", "fitted"), ("/hub/scratch/compare", "compare/followup")])
    cases = {
        "hf://buckets/ns/bucket/features/nvfp4": "hf://buckets/ns/bucket/features/nvfp4",
        "https://example.org/a/b": "https://example.org/a/b",
        "inkling-probe-features/v2": "inkling-probe-features/v2",
        "terminal_prerl/with_reasoning/followup": "terminal_prerl/with_reasoning/followup",
        "Jaccard (with / without)": "Jaccard (with / without)",
        "/hub/scratch/probes/bf16/x/meta.json": "fitted/bf16/x/meta.json",
        "--probes=../scratch/probes/bf16": "--probes=fitted/bf16",
        "see scratch/compare/report.md.": "see compare/followup/report.md.",
        "/other/machine/scratch/probes/nvfp4": "fitted/nvfp4",
        "(/mnt/ram/Inkling-Small)": "(Inkling-Small)",
        "~/.cache/huggingface/token": "token",
        "TPR / FPR 100%/0.37%, t₁%/t₅%, 3/4, n/a, 2026/10/12": "TPR / FPR 100%/0.37%, t₁%/t₅%, 3/4, n/a, 2026/10/12",
        "/tmp": "/tmp",
    }
    for before, after in cases.items():
        assert rw.text(before) == after, before


def test_cp_lower_is_the_exact_clopper_pearson_bound():
    assert pr.cp_lower(22, 22) == pytest.approx(0.025 ** (1 / 22), rel=1e-10)
    assert pr.num(pr.cp_lower(22, 22)) == "0.846" and pr.num(pr.cp_lower(15, 15)) == "0.782"
    assert pr.num(pr.cp_lower(98, 98)) == "0.963" and pr.num(pr.cp_lower(75, 75)) == "0.952"
    assert pr.cp_lower(0, 5) == 0.0
    for k, n in ((3, 10), (9, 10), (1, 4)):
        p = pr.cp_lower(k, n)
        tail = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
        assert tail == pytest.approx(0.025, abs=1e-9)
