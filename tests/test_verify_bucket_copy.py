"""scripts/verify_bucket_copy.py with HfApi and hf_xet faked: a faithful copy passes; a missing, resized or changed
file fails; files only in the bucket fail only with --exact; the prefix is respected; bad arguments are refused."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from types import SimpleNamespace

import hf_xet
import huggingface_hub
import pytest

spec = importlib.util.spec_from_file_location(
    "verify_bucket_copy", os.path.join(os.path.dirname(__file__), "..", "scripts", "verify_bucket_copy.py"))
vb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vb)

URL = "hf://buckets/ns/bucket/features/nvfp4"


def xet(data: bytes) -> str:
    return hashlib.sha256(b"xet" + data).hexdigest()  # a stand-in content hash


def fake_hash_files(paths: list[str]) -> list[SimpleNamespace]:
    out = []
    for p in paths:
        data = open(p, "rb").read()
        out.append(SimpleNamespace(file_size=len(data), hash=xet(data)))
    return out


class FakeApi:
    """A bucket listing: (path, content) files, plus a folder entry and a look-alike prefix."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.calls = files, []

    def list_bucket_tree(self, bucket_id, prefix=None, recursive=None):
        self.calls.append((bucket_id, prefix, recursive))
        assert bucket_id == "ns/bucket" and recursive
        entries = [SimpleNamespace(type="directory", path="features/nvfp4/terminal_prerl")]
        entries += [SimpleNamespace(type="file", path=p, size=len(d), xet_hash=xet(d)) for p, d in self.files.items()]
        return [e for e in entries if prefix is None or e.path.startswith(prefix)]


def local_tree(tmp_path) -> tuple[str, dict[str, bytes]]:
    files = {"index.json": b"{}", "terminal_prerl/with_reasoning/shard_00000.safetensors": b"x" * 100,
             "terminal_prerl/with_reasoning/shard_00000.jsonl": b'{"id": "a"}\n'}
    for rel, data in files.items():
        (tmp_path / "local" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "local" / rel).write_bytes(data)
    return str(tmp_path / "local"), {f"features/nvfp4/{rel}": data for rel, data in files.items()}


def run(monkeypatch, api: FakeApi, *argv: str) -> int:
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(hf_xet, "hash_files", fake_hash_files)
    return vb.main(list(argv))


def test_a_faithful_copy_passes(tmp_path, monkeypatch, capsys):
    local, remote = local_tree(tmp_path)
    remote["features/nvfp4x/lookalike.bin"] = b"not under the prefix"
    api = FakeApi(remote)
    assert run(monkeypatch, api, local, URL, "--batch", "2") == 0
    assert api.calls == [("ns/bucket", "features/nvfp4/", True)]
    out = capsys.readouterr().out
    assert "3/3 files" in out and out.rstrip().endswith(
        "local files 3, bucket files 3, only in the bucket 0, missing from the bucket 0, mismatches 0")
    r = vb.verify(local, URL, api=api, hash_files=fake_hash_files, log=lambda m: None)
    assert (r["local_files"], r["bucket_files"], r["bytes"]) == (3, 3, 2 + 100 + 12)


def test_missing_resized_and_changed_files_fail(tmp_path, monkeypatch, capsys):
    local, remote = local_tree(tmp_path)
    del remote["features/nvfp4/index.json"]
    remote["features/nvfp4/terminal_prerl/with_reasoning/shard_00000.safetensors"] = b"x" * 99
    remote["features/nvfp4/terminal_prerl/with_reasoning/shard_00000.jsonl"] = b'{"id": "b"}\n'  # same size
    assert run(monkeypatch, FakeApi(remote), local, URL) == 1
    out = capsys.readouterr().out
    assert "MISSING from the bucket: index.json" in out
    assert out.count("MISMATCH terminal_prerl/with_reasoning/shard_00000.") == 2
    assert out.rstrip().endswith("missing from the bucket 1, mismatches 2")
    r = vb.verify(local, URL, api=FakeApi(remote), hash_files=fake_hash_files, log=lambda m: None)
    assert r["missing_from_bucket"] == ["index.json"] and len(r["mismatched"]) == 2


def test_files_only_in_the_bucket_fail_only_with_exact(tmp_path, monkeypatch, capsys):
    local, remote = local_tree(tmp_path)
    remote["features/nvfp4/terminal_prerl/full/acts.safetensors"] = b"big"
    assert run(monkeypatch, FakeApi(remote), local, URL) == 0
    assert "only in the bucket 1" in capsys.readouterr().out
    assert run(monkeypatch, FakeApi(remote), local, URL, "--exact") == 1
    assert "ONLY IN THE BUCKET: terminal_prerl/full/acts.safetensors" in capsys.readouterr().out


def test_a_missing_xet_hash_or_an_empty_folder_fails(tmp_path, monkeypatch):
    local, remote = local_tree(tmp_path)
    api = FakeApi(remote)
    listing = api.list_bucket_tree
    api.list_bucket_tree = lambda *a, **k: [SimpleNamespace(**{**vars(e), "xet_hash": None}) if hasattr(e, "size")
                                            else e for e in listing(*a, **k)]
    assert run(monkeypatch, api, local, URL) == 1
    (tmp_path / "empty").mkdir()
    assert run(monkeypatch, FakeApi(remote), str(tmp_path / "empty"), URL) == 1


def test_bad_arguments_are_refused(tmp_path, monkeypatch):
    local, remote = local_tree(tmp_path)
    for argv in ([local, "hf://buckets/ns"], [local, "s3://bucket/x"], [str(tmp_path / "nowhere"), URL]):
        with pytest.raises(SystemExit) as e:
            run(monkeypatch, FakeApi(remote), *argv)
        assert e.value.code == 2
    assert vb.parse_bucket("hf://buckets/ns/bucket") == ("ns/bucket", "")
    assert vb.parse_bucket("hf://buckets/ns/bucket/a/b/") == ("ns/bucket", "a/b")
