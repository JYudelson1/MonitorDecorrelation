"""The Inkling scoring path end to end on CPU with the tiny model: extraction → probe → golden set → server →
``InklingProbeMonitor``.

Checks:
- Served scores of RL-style rollouts equal the probe applied to the training features.
- The golden check passes, and refuses a changed rendering.
- A wrong effort, or a probe the server doesn't serve, fails at construction.
- A client that starts before the server waits for it.
- Identical concurrent requests share one forward pass.
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pytest
from inkling_tiny import write_tiny_checkpoints
from test_extract_probe_features import RUN_ENV, TURNS

from monitordecorrelation.monitors.inkling_probe_monitor import FatalProbeError, InklingProbeMonitor
from monitordecorrelation.types import Prompt, Rollout
from monitordecorrelation.whitebox.extraction import Job, extract, prepare, reader_info
from monitordecorrelation.whitebox.probe_datasets import counts, terminal_items, write_dataset
from monitordecorrelation.whitebox.probe_fit import (
    SelectedProbe, feature_matrix, layer_array, load_items, load_view, write_golden)
from monitordecorrelation.whitebox.probe_server import ProbeScorer, Server, probe_sha

VIEW = "with_reasoning"


def rows():
    return [{"task_id": f"t{t}", "question": f"Solve {t}.", "behavior_present": t == 0 and k == 0,
             "invalid_reason": None,
             "env_meta": {"verifier_mode": "submit_corrupted", "submission_ends": "first",
                          "last_turn_submit_notice": True, "turns": TURNS}}
            for t in range(5) for k in range(2)]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from monitordecorrelation.whitebox.inkling_nvfp4 import load_inkling_nvfp4

    tmp = tmp_path_factory.mktemp("server")
    _, nvfp4_dir, _, _ = write_tiny_checkpoints(tmp, vocab_size=201024)
    rollouts = tmp / "eval_rollouts.jsonl"
    rollouts.write_text("".join(json.dumps(r) + "\n" for r in rows()))
    items, _ = terminal_items(str(rollouts), RUN_ENV, seed=0)
    dataset = tmp / "terminal_tiny"
    write_dataset(str(dataset), items, {"kind": "terminal", "effort": 0.5, "counts": counts(items, "task_id")})
    model = load_inkling_nvfp4(str(nvfp4_dir), device="cpu").eval()
    feats = tmp / "features"
    extract(model, prepare(Job(dataset=str(dataset), out=str(feats), model=str(nvfp4_dir), shard_size=4)),
            log=lambda m: None)
    vf = load_view(str(feats), VIEW)
    x, _ = feature_matrix(vf, "followup", VIEW)
    by_id = load_items(str(dataset))
    y = np.array([by_id[i]["label"] for i in vf.ids])
    y[1] = 1  # a second positive so the toy probe has something to fit
    probe = SelectedProbe.fit(layer_array(x, 2), y, 1.0, 2)
    logits = probe.logit(layer_array(x, 2))
    probe.threshold_1, probe.threshold_5 = float(np.percentile(logits, 90)), float(np.percentile(logits, 50))
    probe.meta = {"name": "probe_iid", "dataset": "terminal_tiny", "view": VIEW, "readout": "followup", "effort": 0.5,
                  "reader_info": reader_info(model, str(nvfp4_dir))}
    probe_dir = tmp / "probe"
    probe.save(str(probe_dir))
    assert write_golden(str(probe_dir), probe, vf, by_id, "followup", VIEW, vf.ids[:4]) == 4
    return {"model": model, "probe_dir": str(probe_dir), "probe": probe, "vf": vf, "x": x,
            "nvfp4_dir": str(nvfp4_dir), "items": by_id, "features": str(feats)}


@pytest.fixture
def server(world):
    scorer = ProbeScorer(world["model"], [world["probe_dir"]],
                         reader_info=reader_info(world["model"], world["nvfp4_dir"]))
    srv = Server(scorer, port=0, log=lambda m: None)
    srv.start()
    yield srv
    srv.stop()


def rollout(row) -> Rollout:
    meta = row["env_meta"]
    return Rollout(prompt=Prompt(text=row["question"]), cot="", output="",
                   meta={"episode": {**meta, "cheated": row["behavior_present"]}})  # an oracle field: never sent


def test_served_scores_equal_the_training_features(world, server):
    report = server.scorer.golden_check([world["probe_dir"]], tolerance=1e-2)
    assert report[probe_sha(world["probe_dir"])]["n"] == 4
    mon = InklingProbeMonitor("probe_iid", world["probe_dir"], f"http://127.0.0.1:{server.port}", env_options=RUN_ENV,
                              run_effort=0.5)
    results = mon.score_batch([rollout(r) for r in rows()])
    ids = [f"t{t}/{k}" for t in range(5) for k in range(2)]  # terminal_items' ids, in row order
    row_of = {i: n for n, i in enumerate(world["vf"].ids)}
    want = world["probe"].score(layer_array(world["x"], 2, np.array([row_of[i] for i in ids])))
    np.testing.assert_allclose([r.score for r in results], want, atol=1e-2)
    assert all(r.meta["probe_sha"] == mon.sha and r.label == (r.score >= 0.5) for r in results)
    assert "cheated" not in json.dumps(mon.items([rollout(rows()[0])]))  # replay fields only


def test_construction_refuses_a_mismatch(world, server):
    url = f"http://127.0.0.1:{server.port}"
    with pytest.raises(ValueError, match="effort"):
        InklingProbeMonitor("p", world["probe_dir"], url, env_options=RUN_ENV, run_effort=0.9)
    with pytest.raises(ValueError, match="env_options"):
        InklingProbeMonitor("p", world["probe_dir"], url, env_options={"verifier_mode": "x"}, run_effort=0.5)
    other = os.path.join(os.path.dirname(world["probe_dir"]), "other_probe")
    world["probe"].save(other)
    meta = json.load(open(os.path.join(other, "meta.json")))
    json.dump({**meta, "name": "other"}, open(os.path.join(other, "meta.json"), "w"))  # a different sha
    with pytest.raises(FatalProbeError, match="doesn't serve"):
        InklingProbeMonitor("p", other, url, env_options=RUN_ENV, run_effort=0.5)


def test_the_golden_check_refuses_a_changed_rendering(world, tmp_path):
    import shutil

    bad = tmp_path / "probe"
    shutil.copytree(world["probe_dir"], bad)
    lines = (bad / "golden.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["episode"]["question"] += " (edited)"
    (bad / "golden.jsonl").write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    scorer = ProbeScorer(world["model"], [str(bad)])
    with pytest.raises(RuntimeError, match="renders differently"):
        scorer.golden_check([str(bad)], tolerance=1e-2)


def test_a_client_waits_for_a_late_server_and_requests_are_shared(world):
    scorer = ProbeScorer(world["model"], [world["probe_dir"]])
    calls = []
    real = scorer.score
    scorer.score = lambda eps, shas: (calls.append(len(eps)), time.sleep(0.5), real(eps, shas))[2]
    srv = Server(scorer, port=0, log=lambda m: None)
    port = srv.port
    threading.Timer(1.5, srv.start).start()  # the client starts first and retries the handshake
    mon = InklingProbeMonitor("probe_iid", world["probe_dir"], f"http://127.0.0.1:{port}", env_options=RUN_ENV,
                              run_effort=0.5, log=lambda m: None)
    out = []
    threads = [threading.Thread(target=lambda: out.append(mon.score_batch([rollout(rows()[0])]))) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    srv.stop()
    assert len(out) == 3 and len({o[0].score for o in out}) == 1
    assert calls == [1]  # three identical requests, one forward pass


def test_deployed_probes_pass_the_servers_golden_check(world, tmp_path):
    """experiments/deploy_probes.py's ``deploy``: the deployed directory loads, and its golden set passes the check at
    the tolerance it was given."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "deploy_probes", os.path.join(os.path.dirname(__file__), "..", "experiments", "deploy_probes.py"))
    dp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dp)
    root = tmp_path / "features"
    root.mkdir()
    os.symlink(world["features"], root / "terminal_prerl")
    items = {i: {**it, "split": "test"} for i, it in world["items"].items()}  # the toy split has ~2 test episodes
    out = tmp_path / "deployed"
    probe = dp.SelectedProbe.load(world["probe_dir"])
    got = dp.deploy(probe, world["probe_dir"], str(out), "probe_iid_x", VIEW, "followup", items, str(root), 0.5,
                    None, log=lambda m: None)
    assert got["golden"] == len(items) == 10  # fewer than N_GOLDEN episodes: all of them
    meta = json.load(open(out / "meta.json"))
    assert meta["name"] == "probe_iid_x" and meta["golden_tolerance"] == dp.MEAN_TOLERANCE and meta["effort"] == 0.5
    scorer = ProbeScorer(world["model"], [str(out)])
    report = scorer.golden_check([str(out)], tolerance=meta["golden_tolerance"])
    assert report[probe_sha(str(out))]["n"] == got["golden"]
