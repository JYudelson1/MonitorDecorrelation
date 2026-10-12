"""The Inkling scoring path end to end on CPU with the tiny model: extraction → probe → golden set → server →
``InklingProbeMonitor``.

Checks:
- Served scores of RL-style rollouts equal the probe applied to the training features.
- The golden check passes, and refuses a changed rendering, a systematic shift of the logits and a non-finite logit.
- A wrong effort, or a probe the server doesn't serve, fails at construction.
- A client that starts before the server waits for it.
- Identical concurrent requests share one forward pass.
- Failures (fake scorers): a failed job isn't kept, so its retry computes again; the GPU running out of memory, a CUDA
  error or a bug answers 503, which the client retries; a bad request (400, 404, 409, 422) is fatal; finished jobs are
  dropped after a while and beyond a cap; the client retries an answer cut off or garbled on the way.
- The GPU check script checks one IID and one OOD monitor through HTTP.
- The post-hoc scorer's per-step metrics, also on the tasks no probe saw.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch
from inkling_tiny import write_tiny_checkpoints
from test_extract_probe_features import RUN_ENV, TURNS

from monitordecorrelation.monitors.inkling_probe_monitor import FatalProbeError, InklingProbeMonitor
from monitordecorrelation.types import Prompt, Rollout
from monitordecorrelation.whitebox.extraction import Job, extract, prepare, reader_info
from monitordecorrelation.whitebox.probe_datasets import counts, episode_item, terminal_items, write_dataset
from monitordecorrelation.whitebox.probe_fit import (
    SelectedProbe, feature_matrix, layer_array, load_items, load_view, write_golden)
from monitordecorrelation.whitebox.probe_server import (
    GOLDEN_MEAN_TOLERANCE, PROTOCOL, GoldenCheckError, ProbeScorer, ReplayError, Server, probe_sha)

VIEW = "with_reasoning"


def load_script(name: str):
    """experiments/<name>.py as a module."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), "..", "experiments", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_the_golden_check_refuses_a_systematic_shift(world, tmp_path):
    """Golden logits all 0.5 off: each within the max tolerance (1.0), so only the mean check refuses them. The same
    0.5 with alternating signs is jitter, not a shift: it passes. A non-finite logit fails."""
    import shutil

    probe = tmp_path / "probe"
    shutil.copytree(world["probe_dir"], probe)
    golden = [json.loads(line) for line in (probe / "golden.jsonl").read_text().splitlines()]
    scorer, sha = ProbeScorer(world["model"], [str(probe)]), probe_sha(str(probe))

    def check(offsets, **kw):  # the golden logits moved by offsets (golden.jsonl isn't part of the probe's sha)
        (probe / "golden.jsonl").write_text("".join(json.dumps({**r, "logit": r["logit"] + o}) + "\n"
                                                    for r, o in zip(golden, offsets)))
        return scorer.golden_check([str(probe)], tolerance=1.0, **kw)

    assert len(golden) == 4 and GOLDEN_MEAN_TOLERANCE < 0.45
    max_only = check([0.5] * 4, mean_tolerance=float("inf"))[sha]  # what the max check alone let through
    assert max_only["max_dlogit"] == pytest.approx(0.5, abs=0.02) and max_only["tolerance"] == 1.0
    assert max_only["mean_dlogit"] == pytest.approx(-0.5, abs=0.02) and max_only["sd_dlogit"] < 0.02
    with pytest.raises(GoldenCheckError, match=r"shifted by -0\.\d+ on average") as err:
        check([0.5] * 4)  # the default mean tolerance
    assert err.value.report["mean_dlogit"] == pytest.approx(-0.5, abs=0.02)
    assert err.value.report["mean_tolerance"] == GOLDEN_MEAN_TOLERANCE
    with pytest.raises(GoldenCheckError, match=r"shifted by \+0\.\d+ on average"):
        check([-0.5] * 4, mean_tolerance=0.4)  # the other direction, a given tolerance
    jitter = check([0.5, -0.5] * 2)[sha]
    assert jitter["max_dlogit"] == pytest.approx(0.5, abs=0.02) and abs(jitter["mean_dlogit"]) < 0.02
    with pytest.raises(GoldenCheckError, match="not finite"):
        check([float("nan"), 0.0, 0.0, 0.0])


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


# --------------------------------------------------------------------------------- failures, retries, the job table


class FakeScorer:
    """Stands in for ``ProbeScorer`` behind a ``Server``: serves one probe and scores every episode 0.75, but first
    raises the exceptions in ``fail``, one per call. ``keys`` records each call's episode keys; with ``gate`` set, a
    call waits for it."""

    def __init__(self, sha: str, fail=()):
        self.probes, self.fail, self.keys, self.gate = {sha: None}, list(fail), [], None

    def meta(self) -> dict:
        return {"protocol": PROTOCOL, "reader": {}, "probes": {s: {} for s in self.probes}}

    def score(self, episodes, shas):
        self.keys.append([ep.get("key") for ep in episodes])
        if self.gate is not None:
            self.gate.wait(30)
        if self.fail:
            raise self.fail.pop(0)
        return [{"key": ep.get("key"), "n_tokens": {VIEW: 3},
                 "scores": {s: {"logit": 1.0, "score": 0.75} for s in shas}} for ep in episodes]


class Clock:
    """The server's clock in a test: it moves only when the test moves it."""

    def __init__(self):
        self.now = 1_000_000.0

    def time(self) -> float:
        return self.now


@pytest.fixture
def toy_probe(tmp_path) -> str:
    """A probe directory the monitor can load; its scores come from a fake scorer, so no model is needed."""
    d = tmp_path / "toy_probe"
    SelectedProbe(0, np.zeros(4), 0.0, 0.0, 0.0, {"name": "toy", "view": VIEW, "readout": "followup",
                                                 "effort": 0.5}).save(str(d))
    return str(d)


def request(sha: str, *episodes: dict, wait_s: float = 20) -> dict:
    return {"protocol": PROTOCOL, "probes": [sha], "episodes": list(episodes), "wait_s": wait_s}


def post(port: int, body, path: str = "/v1/score") -> tuple[int, dict]:
    """(status, JSON answer) of one POST, with no retries."""
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                                           method="POST"), timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_a_failed_job_is_retried_and_computed_again(toy_probe):
    """The GPU running out of memory, a CUDA error and a bug each answer 503 and leave no job behind, so the same
    request computes again. The client retries a 503 until the scores come."""
    sha = probe_sha(toy_probe)
    for error in (torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB"),
                  RuntimeError("CUDA error: an illegal memory access was encountered"), ZeroDivisionError("a bug")):
        scorer = FakeScorer(sha, fail=[error])
        srv = Server(scorer, port=0, log=lambda m: None)
        srv.start()
        try:
            code, answer = post(srv.port, request(sha, {"key": "a"}))
            assert code == 503 and answer["error"].startswith(type(error).__name__) and not srv.jobs
            code, answer = post(srv.port, request(sha, {"key": "a"}))
            assert code == 200 and answer["results"][0]["scores"][sha]["score"] == 0.75
            assert scorer.keys == [["a"], ["a"]]  # computed again, not the failure replayed
        finally:
            srv.stop()
    scorer = FakeScorer(sha, fail=[torch.cuda.OutOfMemoryError("CUDA out of memory") for _ in range(2)])
    srv = Server(scorer, port=0, log=lambda m: None)
    srv.start()
    try:
        mon = InklingProbeMonitor("toy", toy_probe, f"http://127.0.0.1:{srv.port}", env_options=RUN_ENV,
                                  run_effort=0.5, max_backoff=0.05, log=lambda m: None)
        assert [r.score for r in mon.score_batch([rollout(rows()[0])])] == [0.75] and len(scorer.keys) == 3
    finally:
        srv.stop()


def test_a_bad_request_is_refused_for_good(toy_probe, monkeypatch):
    """What the request causes is fatal: an episode that can't be replayed (422), an unknown probe or protocol (409), a
    malformed request (400), an unknown path (404). The client raises at the first such answer. The real scorer tells
    a rendering failure (``ReplayError``, before the GPU runs) from a failure of the forward pass (503)."""
    sha = probe_sha(toy_probe)
    scorer = FakeScorer(sha, fail=[ReplayError("episode 0 (key 'a') can't be replayed: ValueError: truncated turn")])
    srv = Server(scorer, port=0, log=lambda m: None)
    srv.start()
    try:
        assert post(srv.port, request(sha, {"key": "a"}))[0] == 422 and not srv.jobs
        assert post(srv.port, request("0" * 64, {"key": "a"}))[0] == 409
        assert post(srv.port, {**request(sha, {"key": "a"}), "protocol": PROTOCOL + 1})[0] == 409
        for bad in ([request(sha)], {**request(sha), "probes": sha}, {**request(sha), "probes": []},
                    {**request(sha), "episodes": ["a"]}, {**request(sha), "wait_s": "soon"}, {"protocol": PROTOCOL}):
            assert post(srv.port, bad)[0] == 400, bad
        assert post(srv.port, b"{not json")[0] == 400
        assert post(srv.port, request(sha), path="/v2/score")[0] == 404
        assert len(scorer.keys) == 1  # only the first request reached the scorer
        scorer.fail = [ReplayError("truncated turn")]
        mon = InklingProbeMonitor("toy", toy_probe, f"http://127.0.0.1:{srv.port}", env_options=RUN_ENV,
                                  run_effort=0.5, log=lambda m: None)
        t = time.time()
        with pytest.raises(FatalProbeError, match="422"):
            mon.score_batch([rollout(rows()[0])])
        assert time.time() - t < 1  # no retry
    finally:
        srv.stop()
    real = ProbeScorer(None, [toy_probe])  # renders without a model
    item = {**episode_item("Solve 0.", rows()[0]["env_meta"], RUN_ENV), "key": "ok"}
    truncated = {**item, "turns": [{**item["turns"][0], "truncated": True}], "key": "bad"}
    possible = {**item, "env_options": {**item["env_options"], "verifier_mode": "possible"}}
    for episode in (truncated, possible, {"question": "neither turns nor an answer"}):
        with pytest.raises(ReplayError, match="episode 1 .* can't be replayed in view with_reasoning"):
            real.score([item, episode], [sha])
    import monitordecorrelation.whitebox.features as features

    def out_of_memory(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(features, "extract_features", out_of_memory)
    srv = Server(real, port=0, log=lambda m: None)
    srv.start()
    try:
        code, answer = post(srv.port, request(sha, item, truncated))
        assert code == 422 and "truncated turn" in answer["error"]
        code, answer = post(srv.port, request(sha, item))
        assert code == 503 and answer["error"].startswith("OutOfMemoryError")
    finally:
        srv.stop()


def test_finished_jobs_are_kept_a_while_then_dropped(toy_probe, monkeypatch):
    """A finished job answers a repeat without another forward pass for ``KEEP_S`` seconds, then is dropped. Only the
    ``max_kept`` newest finished jobs stay, each without its request. A queued or running job is never dropped, so a
    repeat joins it."""
    import monitordecorrelation.whitebox.probe_server as probe_server

    clock = Clock()
    monkeypatch.setattr(probe_server, "time", clock)  # the server's clock only
    sha = probe_sha(toy_probe)
    scorer = FakeScorer(sha)
    srv = Server(scorer, port=0, log=lambda m: None, max_kept=3)
    srv.start()
    try:
        for _ in range(2):
            assert post(srv.port, request(sha, {"key": "a"}))[0] == 200
        clock.now += probe_server.KEEP_S - 1
        assert post(srv.port, request(sha, {"key": "a"}))[0] == 200
        assert scorer.keys == [["a"]]  # the repeats got the kept result
        clock.now += 2
        assert post(srv.port, request(sha, {"key": "a"}))[0] == 200
        assert scorer.keys == [["a"], ["a"]]  # expired: computed again
        for key in "bcde":  # all finishing at the same clock time
            assert post(srv.port, request(sha, {"key": key}))[0] == 200
        assert len(srv.jobs) == 3 and all(j.request is None and j.result for j in srv.jobs.values())
        for key in "cdea":
            assert post(srv.port, request(sha, {"key": key}))[0] == 200
        assert [scorer.keys.count([k]) for k in "abcde"] == [3, 1, 1, 1, 1]  # c, d, e kept; a beyond the cap
        scorer.gate = threading.Event()
        assert post(srv.port, request(sha, {"key": "g"}, wait_s=0))[0] == 202  # running, held at the gate
        clock.now += 10 * probe_server.KEEP_S
        assert post(srv.port, request(sha, {"key": "h"}, wait_s=0))[0] == 202  # queued behind it
        assert len(srv.jobs) == 2 and not any(j.done.is_set() for j in srv.jobs.values())  # the finished ones expired
        scorer.gate.set()
        assert post(srv.port, request(sha, {"key": "g"}))[0] == 200
        assert scorer.keys.count(["g"]) == 1  # the repeat joined the running job
    finally:
        srv.stop()


def raw_server(answers: list) -> int:
    """A TCP server that reads one whole HTTP request per connection and answers with the next of ``answers``:
    bytes as they are, then a close; "reset" closes with a TCP reset instead. Returns its port."""
    sock = socket.create_server(("127.0.0.1", 0))

    def run():
        for answer in answers:
            conn, _ = sock.accept()
            with conn:
                data = b""
                while b"\r\n\r\n" not in data and (chunk := conn.recv(65536)):
                    data += chunk
                head, _, body = data.partition(b"\r\n\r\n")
                length = next((int(h.split(b":")[1]) for h in head.split(b"\r\n")
                               if h.lower().startswith(b"content-length:")), 0)
                while len(body) < length and (chunk := conn.recv(65536)):
                    body += chunk
                if answer == "reset":
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                else:
                    conn.sendall(answer)
        sock.close()

    threading.Thread(target=run, daemon=True).start()
    return sock.getsockname()[1]


def http_answer(status: str, body: bytes, length: int | None = None) -> bytes:
    """An HTTP response announcing ``length`` body bytes (default: all of ``body``)."""
    n = len(body) if length is None else length
    return f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {n}\r\n\r\n".encode() + body


def test_the_client_retries_an_answer_lost_on_the_way(toy_probe):
    """A tunnel dropping mid-answer: no answer at all, a reset, a body cut short (also a 503's), a body that isn't
    JSON. The client retries each and returns the answer that finally arrives; a fatal status still raises at once."""
    sha = probe_sha(toy_probe)
    srv = Server(FakeScorer(sha), port=0, log=lambda m: None)
    srv.start()
    try:
        mon = InklingProbeMonitor("toy", toy_probe, f"http://127.0.0.1:{srv.port}", env_options=RUN_ENV,
                                  run_effort=0.5, max_backoff=0.05, log=lambda m: None)
    finally:
        srv.stop()
    good = json.dumps({"results": [{"key": "k", "n_tokens": {VIEW: 3},
                                    "scores": {sha: {"logit": 1.0, "score": 0.75}}}]}).encode()
    mon.url = "http://127.0.0.1:%d" % raw_server([
        b"", "reset", http_answer("200 OK", good[:20], len(good)), http_answer("200 OK", b"<html>a proxy</html>"),
        http_answer("503 Service Unavailable", b'{"error": "CUDA', 100), http_answer("200 OK", good)])
    assert [r.score for r in mon.score_batch([rollout(rows()[0])])] == [0.75]
    mon.url = "http://127.0.0.1:%d" % raw_server([http_answer("422 Unprocessable Entity", b'{"error": "truncated"}')])
    with pytest.raises(FatalProbeError, match="422"):
        mon.score_batch([rollout(rows()[0])])


def test_the_gpu_check_takes_one_iid_and_one_ood_monitor_through_http():
    """check_probe_server.end_to_end_dirs: per dataset of the deployed probes, its with-reasoning follow-up probe
    (the first two follow-up probes in path order were both OOD ones)."""
    check = load_script("check_probe_server")
    root = os.path.join("data", "probes", "Inkling-Small")
    dirs = sorted(os.path.join(root, d, v, r) for d in ("diverse_deception", "terminal_prerl")
                  for v in ("with_reasoning", "without_reasoning") for r in ("followup", "mean"))
    want = [os.path.join(root, d, "with_reasoning", "followup") for d in ("diverse_deception", "terminal_prerl")]
    assert check.end_to_end_dirs(dirs, root) == want
    output_only = [d for d in dirs if "without_reasoning" in d]  # no with-reasoning probe deployed: the other view
    assert check.end_to_end_dirs(output_only, root) == [w.replace("with_reasoning", "without_reasoning") for w in want]
    assert check.end_to_end_dirs([d for d in dirs if "terminal_prerl" in d], root) == want[1:]
    assert check.end_to_end_dirs([d for d in dirs if d.endswith("mean")], root) == []


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
    assert set(meta["deployed_code"]) == {"commit", "dirty", "diff_sha256", "source"}  # the deploying code
    assert set(got["serving_check"]) == {"test_auroc", "test_tpr_at_t1", "test_fpr_at_t1"}  # no diagnostics given
    scorer = ProbeScorer(world["model"], [str(out)])
    report = scorer.golden_check([str(out)], tolerance=meta["golden_tolerance"])
    assert report[probe_sha(str(out))]["n"] == got["golden"]
    # serving on another reader: thresholds from that reader's honest dev episodes, its reader_info in the meta
    items = {i: {**it, "split": "train"} for i, it in world["items"].items()}
    p = dp.SelectedProbe.load(world["probe_dir"])
    before = (p.threshold_1, p.threshold_5)
    same = {**world["probe"].meta["reader_info"], "path": "Inkling-Small-NVFP4"}  # the same reader, elsewhere
    dp.serve_on(p, {"reader": "same", "reader_info": same}, items, str(root), VIEW, "followup")
    assert (p.threshold_1, p.threshold_5) == before and "trained_reader" not in p.meta
    dp.serve_on(p, {"reader": "other", "reader_info": {"loader": "x"}}, items, str(root), VIEW, "followup")
    src = dp.LayerSource(str(root / "terminal_prerl"), VIEW, "followup")
    neg = [r for r, i in enumerate(src.ids) if not (items[i].get("diagnostics") or {}).get("used_verifier")]
    want = np.percentile(p.logit(src.layer(p.layer))[neg], [99, 95])
    np.testing.assert_allclose([p.threshold_1, p.threshold_5], want)
    p.threshold_1 = 123.0  # the toy probe's logits are constant, so mark the threshold instead
    p.save(str(tmp_path / "moved"))  # meta.json shows the probe's thresholds, not those of the meta it loaded
    assert json.load(open(tmp_path / "moved" / "meta.json"))["threshold_1"] == 123.0 != p.meta["threshold_1"]
    assert p.meta["reader_info"] == {"loader": "x"} and p.meta["trained_reader_info"] == world["probe"].meta["reader_info"]


def test_the_gpu_check_script_passes_on_cpu(world, tmp_path):
    """experiments/check_probe_server.py end to end on the tiny model: golden checks at both batchings, throughput,
    and the HTTP server + monitor agreeing with the scorer. Golden logits shifted by a constant fail it, with the
    mean shift in the report."""
    import importlib.util

    def load(name):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(os.path.dirname(__file__), "..", "experiments", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    dp, check = load("deploy_probes"), load("check_probe_server")
    root = tmp_path / "features"
    root.mkdir()
    os.symlink(world["features"], root / "terminal_prerl")
    items = {i: {**it, "split": "test"} for i, it in world["items"].items()}
    deployed = tmp_path / "deployed"
    dp.deploy(dp.SelectedProbe.load(world["probe_dir"]), world["probe_dir"],
              str(deployed / "terminal_prerl" / VIEW / "followup"), "probe_iid_x", VIEW, "followup", items, str(root),
              0.5, None, log=lambda m: None)
    out = tmp_path / "check.json"
    argv = ["--model", world["nvfp4_dir"], "--probes-root", str(deployed), "--out", str(out), "--device", "cpu"]
    assert check.main(argv) == 0
    report = json.loads(out.read_text())
    assert report["ok"] and set(report["throughput"]) == {"1", "65536"} and len(report["end_to_end"]) == 1
    golden = report["probes"][os.path.join("terminal_prerl", VIEW, "followup")]
    assert set(golden) == {"golden_max_batch_tokens_1", "golden_max_batch_tokens_65536"}
    for res in golden.values():
        assert res["ok"] and res["mean_tolerance"] == GOLDEN_MEAN_TOLERANCE and abs(res["mean_dlogit"]) < 0.02
    # every golden logit 0.5 too high: inside the max tolerance (3.0), outside the mean tolerance given
    path = deployed / "terminal_prerl" / VIEW / "followup" / "golden.jsonl"
    stored = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps({**r, "logit": r["logit"] + 0.5}) + "\n" for r in stored))
    assert check.main([*argv, "--golden-mean-tolerance", "0.4"]) == 1
    report = json.loads(out.read_text())
    assert not report["ok"] and all(e["ok"] for e in report["end_to_end"].values())
    for res in report["probes"][os.path.join("terminal_prerl", VIEW, "followup")].values():
        assert not res["ok"] and res["mean_tolerance"] == 0.4 and res["max_dlogit"] < res["tolerance"] == 3.0
        assert res["mean_dlogit"] == pytest.approx(-0.5, abs=0.02) and "shifted by" in res["error"]


def test_posthoc_scoring_of_saved_runs(world, tmp_path, capsys):
    """experiments/eval_inkling_probes_on_runs.py on a fake RL run: every valid eval episode scored as in training,
    per-step metrics in the file the degradation matrix reads, also on the tasks no probe saw, and a rerun scores
    nothing twice. Without a task-splits file, the unseen-task fields are left out with a warning."""
    mod = load_script("eval_inkling_probes_on_runs")
    run = tmp_path / "terminal_run_s0"
    run.mkdir()
    saved = [{**r, "step": 0 if i < 6 else 10} for i, r in enumerate(rows())]
    saved.append({**rows()[0], "step": 10, "invalid_reason": "unparsed"})  # never scored
    (run / "eval_rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in saved))
    (run / "run_info.json").write_text(json.dumps({"config": {"env_options": RUN_ENV, "thinking_effort": 0.5}}))
    splits = tmp_path / "task_splits.json"
    splits.write_text(json.dumps({"test": ["t0", "t4"], "train": ["t1", "t2"], "val": ["t3"]}))
    argv = [str(run), "--model", world["nvfp4_dir"], "--probes", world["probe_dir"], "--names", "probe_iid",
            "--device", "cpu", "--chunk", "4"]
    assert mod.main([*argv, "--task-splits", str(splits)]) == 0
    scored = [json.loads(line) for line in (run / "probe_scores_inkling.jsonl").read_text().splitlines()]
    assert sorted(r["line"] for r in scored) == list(range(1, 11))
    row_of = {i: n for n, i in enumerate(world["vf"].ids)}
    ids = [f"t{t}/{k}" for t in range(5) for k in range(2)]  # rows() order = the dataset's ids
    want = world["probe"].logit(layer_array(world["x"], 2, np.array([row_of[i] for i in ids])))
    got = [next(r for r in scored if r["line"] == n + 1)["scores"]["probe_iid"]["logit"] for n in range(10)]
    np.testing.assert_allclose(got, want, atol=1e-2)
    per_step = [json.loads(line) for line in (run / "probe_eval_probe_iid.jsonl").read_text().splitlines()]
    assert [r["step"] for r in per_step] == [0, 10] and [r["n"] for r in per_step] == [6, 4]
    assert {"auroc", "auroc_used_verifier", "recall_at_threshold", "fpr_at_threshold"} <= set(per_step[0])
    # unseen: t0 at step 0 (the hack t0/0 and t0/1), t4 at step 10 (two honest episodes: no AUROC, no recall)
    assert [(r["n_unseen"], r["n_pos_unseen"]) for r in per_step] == [(2, 1), (2, 0)]
    assert set(mod.UNSEEN_FIELDS.values()) <= set(per_step[1])
    assert math.isnan(per_step[1]["auroc_unseen"]) and math.isnan(per_step[1]["recall_at_threshold_unseen"])
    assert per_step[1]["fpr_at_threshold_unseen"] in (0.0, 0.5, 1.0)
    capsys.readouterr()
    assert mod.main([*argv, "--task-splits", str(tmp_path / "absent.json")]) == 0  # nothing left to score
    assert "WARNING: no task splits" in capsys.readouterr().out
    assert len((run / "probe_scores_inkling.jsonl").read_text().splitlines()) == 10
    again = [json.loads(line) for line in (run / "probe_eval_probe_iid.jsonl").read_text().splitlines()]
    assert json.dumps(again) == json.dumps([{k: v for k, v in r.items() if not k.endswith("_unseen")}
                                            for r in per_step])  # the other fields unchanged
    opts, effort = mod.run_replay_options({"config": {"env_options": {"verifier_mode": "submit_corrupted"},
                                                      "thinking_effort": 0.5}})
    assert effort == 0.5 and opts["max_turns"] == 4 and opts["require_explanation"] is False  # the env's defaults


def test_posthoc_metrics_on_unseen_tasks():
    """eval_inkling_probes_on_runs.step_metrics: the *_unseen fields count only the unseen tasks' scored episodes, are
    NaN where undefined, and leave the other fields as they were."""
    mod = load_script("eval_inkling_probes_on_runs")

    def row(step, task, score, behavior, used=False):
        return {"step": step, "task_id": task, "behavior_present": behavior, "used_verifier": used,
                "scores": {"p": {"logit": 0.0, "score": score}}}

    episodes = [row(0, "a", 0.9, True, used=True), row(0, "a", 0.2, False, used=True), row(0, "b", 0.6, False),
                row(0, "b", 0.1, False), row(0, "b", None, True),  # unscored: not counted anywhere
                row(0, "seen", 0.4, True), row(0, "seen", 0.7, False),
                row(5, "b", 0.3, False), row(5, "seen", 0.8, True),
                row(7, "seen", 0.9, True), row(7, None, 0.1, False)]
    plain = mod.step_metrics(episodes, "p")
    got = mod.step_metrics(episodes, "p", unseen={"a", "b"})
    assert not any(k.endswith("_unseen") for r in plain for k in r)
    assert json.dumps([{k: r[k] for k in p} for r, p in zip(got, plain)]) == json.dumps(plain)  # NaN-safe equality
    s0, s5, s7 = got
    assert (s0["n"], s0["n_pos"], s0["auroc"]) == (6, 2, 0.75)
    assert (s0["n_unseen"], s0["n_pos_unseen"]) == (4, 1)
    assert (s0["auroc_unseen"], s0["auroc_used_verifier_unseen"]) == (1.0, 0.75)
    assert s0["recall_at_threshold_unseen"] == 1.0 and s0["fpr_at_threshold_unseen"] == pytest.approx(1 / 3)
    assert (s5["n_unseen"], s5["n_pos_unseen"], s5["fpr_at_threshold_unseen"]) == (1, 0, 0.0)  # one honest episode
    assert all(math.isnan(s5[k]) for k in ("auroc_unseen", "auroc_used_verifier_unseen", "recall_at_threshold_unseen"))
    assert (s7["n_unseen"], s7["n_pos_unseen"]) == (0, 0)  # none
    assert all(math.isnan(s7[k]) for k in ("auroc_unseen", "auroc_used_verifier_unseen", "recall_at_threshold_unseen",
                                           "fpr_at_threshold_unseen"))
