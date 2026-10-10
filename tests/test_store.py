"""whitebox/store.py: where finished work goes as soon as it is done.

Checks:
- Stores: a local directory and an HF bucket prefix (the HfApi is faked), and URL parsing.
- The uploader saves each unit's data before its marker, in order.
- A unit whose data fails is never marked.
- ``flush`` waits for the uploads and can time out.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from monitordecorrelation.whitebox.store import HfBucketStore, LocalStore, Uploader, open_store


def _file(path, text: str = "x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def test_local_store_puts_lists_and_gets(tmp_path):
    store = LocalStore(str(tmp_path / "store"))
    assert store.files() == {}
    store.put([(_file(tmp_path / "a.bin", "12345"), "v/shard_00000.safetensors"),
               (_file(tmp_path / "a.jsonl", "{}"), "v/shard_00000.jsonl")])
    assert store.files() == {"v/shard_00000.safetensors": 5, "v/shard_00000.jsonl": 2}
    store.get("v/shard_00000.safetensors", str(tmp_path / "back" / "x.bin"))
    assert (tmp_path / "back" / "x.bin").read_text() == "12345"
    store.delete(["v/shard_00000.safetensors"])
    assert store.files() == {"v/shard_00000.jsonl": 2}


class FakeBucketApi:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple] = []

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        self.calls.append(("list", bucket_id, prefix, recursive))
        folders = {p.rsplit("/", 1)[0] for p in self.files if "/" in p}
        return ([SimpleNamespace(path=p, size=len(b)) for p, b in self.files.items() if p.startswith(prefix or "")]
                + [SimpleNamespace(path=f) for f in folders])  # folders have no size

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files):
        for remote, local in files:
            open(local, "wb").write(self.files[remote])

    def batch_bucket_files(self, bucket_id, *, add=(), delete=()):
        self.calls.append(("batch", bucket_id, [dest for _, dest in add], list(delete)))
        for local, dest in add:
            self.files[dest] = open(local, "rb").read()
        for path in delete:
            del self.files[path]


def test_hf_bucket_store_uses_its_prefix(tmp_path):
    api = FakeBucketApi()
    api.files = {"features/nvfp4/other/x.jsonl": b"no", "features/nvfp4/terminal_prerlx/y": b"no"}
    store = HfBucketStore("probemaxxing/inkling-probes", "features/nvfp4/terminal_prerl/", api=api)
    assert store.url == "hf://buckets/probemaxxing/inkling-probes/features/nvfp4/terminal_prerl"
    store.put([(_file(tmp_path / "s", "abc"), "with_reasoning/shard_00000.safetensors")])
    assert api.calls[-1] == ("batch", "probemaxxing/inkling-probes",
                             ["features/nvfp4/terminal_prerl/with_reasoning/shard_00000.safetensors"], [])
    assert store.files() == {"with_reasoning/shard_00000.safetensors": 3}  # not the look-alike prefix, no folders
    assert api.calls[-1][2] == "features/nvfp4/terminal_prerl/"
    store.get("with_reasoning/shard_00000.safetensors", str(tmp_path / "got" / "s"))
    assert (tmp_path / "got" / "s").read_text() == "abc"
    store.delete(["with_reasoning/shard_00000.safetensors"])
    assert api.calls[-1][3] == ["features/nvfp4/terminal_prerl/with_reasoning/shard_00000.safetensors"]
    assert store.files() == {}


def test_open_store_parses_urls(tmp_path):
    hf = open_store("hf://buckets/probemaximizer/inkling-probes-bulk/features/bf16")
    assert isinstance(hf, HfBucketStore) and hf.bucket_id == "probemaximizer/inkling-probes-bulk"
    assert hf.prefix == "features/bf16"
    assert open_store("hf://buckets/ns/b").prefix == ""
    assert isinstance(open_store(str(tmp_path)), LocalStore)
    for bad in ("hf://buckets/only-namespace", "s3://bucket/x", "hf://datasets/x/y"):
        with pytest.raises(ValueError):
            open_store(bad)


class RecordingStore(LocalStore):
    """Records each put; ``fail`` names files whose put always raises; ``gate`` holds puts until set."""

    def __init__(self, root, fail=(), gate=None):
        super().__init__(root)
        self.puts: list[list[str]] = []
        self.fail, self.gate = set(fail), gate

    def put(self, files):
        if self.gate is not None:
            self.gate.wait()
        self.puts.append([rel for _, rel in files])
        if self.fail & {rel for _, rel in files}:
            raise OSError("quota exceeded")
        super().put(files)


def test_uploader_saves_data_before_markers_in_order(tmp_path):
    store = RecordingStore(str(tmp_path / "store"))
    up = Uploader(store, backoff=0)
    for k in range(3):
        up.submit(f"shard {k}", [(_file(tmp_path / f"{k}.st", "data"), f"v/{k}.safetensors")],
                  marker=(_file(tmp_path / f"{k}.jsonl", "{}"), f"v/{k}.jsonl"))
    assert up.close(timeout=30)
    assert store.puts == [["v/0.safetensors"], ["v/0.jsonl"], ["v/1.safetensors"], ["v/1.jsonl"],
                          ["v/2.safetensors"], ["v/2.jsonl"]]
    assert up.saved == ["shard 0", "shard 1", "shard 2"] and up.failed == []
    assert up.pending_bytes() == 0 and up.rate() is None  # small files never count as a speed measurement


def test_upload_speed_is_measured_on_large_files_only(tmp_path):
    up = Uploader(RecordingStore(str(tmp_path / "store")), backoff=0)
    up.RATE_MIN_BYTES, up.RATE_MEASURED_BYTES = 1000, 3000
    up.submit("tiny", [(_file(tmp_path / "tiny", "x"), "tiny")])
    assert up.flush(timeout=30) and up.rate() is None
    for k in range(3):
        up.submit(f"big {k}", [(_file(tmp_path / f"big{k}", "y" * 1500), f"big{k}")])
    assert up.close(timeout=30) and up.rate() > 0  # 4500 bytes of large files measured


def test_a_unit_whose_data_fails_is_never_marked(tmp_path):
    logs = []
    store = RecordingStore(str(tmp_path / "store"), fail={"v/1.safetensors"})
    up = Uploader(store, retries=3, backoff=0, log=logs.append)
    for k in range(3):
        up.submit(f"shard {k}", [(_file(tmp_path / f"{k}.st"), f"v/{k}.safetensors")],
                  marker=(_file(tmp_path / f"{k}.jsonl"), f"v/{k}.jsonl"))
    assert up.close(timeout=30)
    assert up.failed == ["shard 1"] and up.saved == ["shard 0", "shard 2"]
    assert store.puts.count(["v/1.safetensors"]) == 3  # retried
    assert "v/1.jsonl" not in store.files() and "v/2.jsonl" in store.files()  # later units still go
    failures = [m for m in logs if "failed" in m]
    assert len(failures) == 3 and "quota exceeded" in failures[0]
    assert sum("saved shard" in m for m in logs) == 2  # each saved unit is logged, with the speed so far


def test_flush_times_out_while_uploads_are_stuck(tmp_path):
    gate = threading.Event()
    up = Uploader(RecordingStore(str(tmp_path / "store"), gate=gate), backoff=0)
    up.submit("big", [(_file(tmp_path / "big", "y" * 1000), "big")])
    assert up.pending_bytes() == 1000
    assert up.flush(timeout=0.2) is False
    gate.set()
    assert up.close(timeout=30) is True and up.pending_bytes() == 0
    with pytest.raises(RuntimeError, match="closed"):
        up.submit("late", [(_file(tmp_path / "late"), "late")])


class HttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class FailingStore(LocalStore):
    def __init__(self, root, error):
        super().__init__(root)
        self.error, self.calls = error, 0

    def put(self, files):
        self.calls += 1
        raise self.error


def test_a_full_bucket_is_not_retried_and_marks_the_store_broken(tmp_path):
    store = FailingStore(str(tmp_path / "store"), HttpError(403))
    up = Uploader(store, retries=5, backoff=0, log=lambda m: None)
    up.submit("shard 0", [(_file(tmp_path / "a"), "a")], marker=(_file(tmp_path / "m"), "m"))
    up.submit("shard 1", [(_file(tmp_path / "b"), "b")])
    assert up.close(timeout=30)
    assert store.calls == 1 and up.broken and "403" in up.broken  # one try, then nothing more is attempted
    assert up.failed == ["shard 0", "shard 1"]


def test_a_transient_error_is_retried(tmp_path):
    store = FailingStore(str(tmp_path / "store"), HttpError(503))
    up = Uploader(store, retries=3, backoff=0, log=lambda m: None)
    up.submit("shard 0", [(_file(tmp_path / "a"), "a")])
    assert up.close(timeout=30)
    assert store.calls == 3 and up.broken is None and up.failed == ["shard 0"]


def test_a_crashing_store_cannot_hang_flush(tmp_path):
    class Crashing(LocalStore):
        def put(self, files):
            raise KeyboardInterrupt  # not an Exception: kills a thread that doesn't guard against it

    up = Uploader(Crashing(str(tmp_path / "store")), retries=1, backoff=0, log=lambda m: None)
    up.submit("shard 0", [(_file(tmp_path / "a"), "a")])
    assert up.flush(timeout=20) is False  # returns instead of waiting forever on a dead thread
