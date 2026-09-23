"""visualize_transcripts.py must never show something that isn't in the file on disk — in particular
when a dump is rewritten in place (not appended to) while the viewer is running."""

import importlib.util
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "visualize_transcripts", Path(__file__).resolve().parents[1] / "visualize_transcripts.py")
vt = importlib.util.module_from_spec(_spec)
sys.modules["visualize_transcripts"] = vt
_spec.loader.exec_module(vt)


def _rec(tag: str, i: int, score: float = 0.0) -> dict:
    return {"step": 0, "task_id": f"{tag}-{i}", "answer": f"answer {tag} {i}", "behavior_present": i % 2 == 0,
            "monitors": {"g35_out": {"score": score, "label": None}}}


def _write(p: Path, recs: list[dict], mode: str = "w") -> None:
    with p.open(mode) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _task_ids(view) -> list[str]:
    return [e["task_id"] for e in view.entries]


def _index(path: Path, cache=None) -> "vt.IndexView":
    idx = vt.RolloutIndex(path)
    idx.refresh(cache)
    return idx


def test_append_extends_the_index_and_keeps_its_epoch(tmp_path):
    p = tmp_path / "eval_rollouts.jsonl"
    _write(p, [_rec("a", i) for i in range(3)])
    idx = _index(p)
    epoch = idx.epoch
    _write(p, [_rec("a", i) for i in range(3, 5)], mode="a")
    idx.refresh()
    assert [e["task_id"] for e in idx.entries] == [f"a-{i}" for i in range(5)]
    assert idx.epoch == epoch


def test_longer_rewrite_is_reindexed_from_scratch(tmp_path):
    """The reported bug: a rewrite that GREW was treated as an append from the old offset."""
    p = tmp_path / "eval_rollouts.jsonl"
    _write(p, [_rec("old", i) for i in range(4)])
    idx = _index(p)
    epoch = idx.epoch
    _write(p, [_rec("new", i, score=0.3) | {"answer": "x" * 500} for i in range(4)])  # same count, longer lines
    idx.refresh()
    view = idx.view()
    assert _task_ids(view) == [f"new-{i}" for i in range(4)]
    assert view.epoch != epoch
    assert all(view.read(i)["task_id"] == f"new-{i}" for i in range(4))


def test_same_size_rewrite_with_mtime_forced_back_is_detected(tmp_path):
    p = tmp_path / "eval_rollouts.jsonl"
    _write(p, [_rec("a", 1, score=0.1)])
    idx = _index(p)
    st = p.stat()
    _write(p, [_rec("a", 1, score=0.9)])  # same length, different score
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert p.stat().st_size == st.st_size
    idx.refresh()
    assert idx.entries[0]["monitors"]["g35_out"]["score"] == 0.9


def test_a_view_never_reads_bytes_other_than_the_ones_it_indexed(tmp_path):
    p = tmp_path / "eval_rollouts.jsonl"
    _write(p, [_rec("old", i) for i in range(3)])
    view = _index(p).view()
    _write(p, [_rec("new", i) for i in range(3)])  # same lengths, different contents
    assert [view.read(i) for i in range(3)] == [None, None, None]


def test_stale_disk_cache_is_not_trusted_after_a_restart(tmp_path):
    """The reported bug persisted across restarts: the cache was reused whenever it was no bigger than the file."""
    p = tmp_path / "eval_rollouts.jsonl"
    cache = vt.IndexCache(tmp_path / "cache")
    _write(p, [_rec("old", i) for i in range(6)])
    _index(p, cache)
    _write(p, [_rec("new", i) | {"answer": "y" * 300} for i in range(4)])  # fewer rows, more bytes
    view = _index(p, cache).view()  # a fresh process: starts from the cache
    assert _task_ids(view) == [f"new-{i}" for i in range(4)]
    # and an unchanged file is served from the (verified) cache with the same epoch
    again = _index(p, cache).view()
    assert again.epoch == view.epoch and _task_ids(again) == _task_ids(view)


def test_metrics_tail_rewritten_longer(tmp_path):
    p = tmp_path / "metrics.jsonl"
    _write(p, [{"step": 0, "behavior_rate": 0.1}])
    tail = vt.JsonlTail(p)
    tail.refresh()
    _write(p, [{"step": 0, "behavior_rate": 0.5, "pad": "z" * 50}, {"step": 1, "behavior_rate": 0.6}])
    tail.refresh()
    assert [r["behavior_rate"] for r in tail.rows] == [0.5, 0.6]


def test_partial_trailing_line_is_left_for_later(tmp_path):
    p = tmp_path / "eval_rollouts.jsonl"
    _write(p, [_rec("a", 0)])
    with p.open("a") as f:
        f.write(json.dumps(_rec("a", 1))[:20])
    idx = _index(p)
    assert [e["task_id"] for e in idx.entries] == ["a-0"]
    with p.open("a") as f:
        f.write(json.dumps(_rec("a", 1))[20:] + "\n")
    idx.refresh()
    assert [e["task_id"] for e in idx.entries] == ["a-0", "a-1"]


def test_run_info_caught_mid_write_is_reread(tmp_path):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text('{"policy": "x", "trunc')
    store = vt.Store(tmp_path / "runs", None)
    store.scan()
    run = store.runs["r"]
    assert run.run_info == {}
    st = (run_dir / "run_info.json").stat()
    (run_dir / "run_info.json").write_text('{"policy": "inkling-small"}  ')
    os.utime(run_dir / "run_info.json", ns=(st.st_atime_ns, st.st_mtime_ns))
    store.scan()
    assert run.run_info == {"policy": "inkling-small"}


# ------------------------------------------------------------------------------------ over HTTP


@pytest.fixture
def server(tmp_path):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    store = vt.Store(tmp_path / "runs", tmp_path / "cache")
    store.scan()
    vt.Handler.store = store
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), vt.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def get(path: str, **params):
        q = "&".join(f"{k}={v}" for k, v in {"id": "r", "source": "eval", **params}.items())
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}{path}?{q}") as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    yield run_dir / "eval_rollouts.jsonl", get
    httpd.shutdown()


def test_click_from_a_list_of_a_rewritten_dump_is_refused_not_misrouted(server):
    p, get = server
    _write(p, [_rec("old", i, score=0.2) for i in range(4)])
    code, lst = get("/api/rollouts", mon="g35_out", mon_max=0.5)
    assert code == 200 and lst["total"] == 4
    # the dump is rewritten (longer) — the next click on the stale list must not open some other line
    _write(p, [_rec("new", i, score=0.2) | {"answer": "w" * 400} for i in range(6)])
    code, body = get("/api/rollout", i=3, epoch=lst["epoch"])
    assert code == 409 and body["stale"] is True
    code, lst2 = get("/api/rollouts", mon="g35_out", mon_max=0.5)
    assert lst2["total"] == 6 and lst2["epoch"] != lst["epoch"] and lst2["n_indexed"] == 6
    for e in lst2["entries"]:
        code, one = get("/api/rollout", i=e["i"], epoch=lst2["epoch"])
        assert code == 200 and one["record"]["task_id"] == e["task_id"]


def test_text_search_after_rewrite_matches_current_contents(server):
    p, get = server
    _write(p, [_rec("old", i) for i in range(3)])
    assert get("/api/rollouts", q="old")[1]["total"] == 3
    _write(p, [_rec("new", i) | {"answer": "needle " * 40} for i in range(3)])
    code, lst = get("/api/rollouts", q="needle")
    assert code == 200 and lst["total"] == 3
    assert get("/api/rollouts", q="old")[1]["total"] == 0
