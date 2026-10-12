"""monitordecorrelation.provenance: the code version stamped into outputs.

Checks, on throwaway git repos:
- a clean checkout gives its commit; tracked changes and untracked files make it dirty, ignored files and files that
  never travel (``.env*``, ``__pycache__``) don't;
- ``diff_sha256`` is the sha256 of the patch ``save_changes`` writes, and ``git apply`` of that patch on a checkout of
  the commit rebuilds the tree (binary, executable, deleted and new files), which then stamps the same;
- a copy without .git reads CODE_VERSION.json, never an enclosing checkout's HEAD; anything else is "none", and nothing
  raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from monitordecorrelation import provenance
from monitordecorrelation.provenance import code_version, save_changes

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
SECRET = "hf_SENTINELneverINaPATCH0123456789"


def git(repo: Path, *args: str) -> str:
    cp = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                         "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True)
    return cp.stdout.strip()


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    (path / ".gitignore").write_text("ignored/\n*.log\n")
    (path / "src").mkdir()
    (path / "src" / "a.py").write_text("x = 1\ny = 2\n")
    (path / "src" / "gone.py").write_text("bye\n")
    (path / "tool.sh").write_text("#!/bin/sh\necho hi\n")
    (path / "blob.bin").write_bytes(bytes(range(256)))
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "first")
    return path


def test_clean_then_dirty_then_clean_again(tmp_path):
    repo = make_repo(tmp_path / "repo")
    head = git(repo, "rev-parse", "HEAD")
    assert code_version(repo) == {"commit": head, "dirty": False, "diff_sha256": None, "source": "git"}

    (repo / "src" / "a.py").write_text("x = 1\ny = 3\n")
    first = code_version(repo)
    assert first["dirty"] is True and first["commit"] == head and len(first["diff_sha256"]) == 64
    assert code_version(repo) == first  # the same tree, the same stamp
    (repo / "src" / "a.py").write_text("x = 1\ny = 4\n")
    assert code_version(repo)["diff_sha256"] != first["diff_sha256"]
    git(repo, "checkout", "--", "src/a.py")
    assert code_version(repo)["dirty"] is False

    (repo / "ignored").mkdir()
    (repo / "ignored" / "big.npz").write_bytes(b"\0" * 1000)
    (repo / "run.log").write_text("log\n")
    (repo / ".env.local").write_text(f"HF_TOKEN={SECRET}\n")  # not ignored here, but never recorded
    (repo / "src" / "__pycache__").mkdir()
    (repo / "src" / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"\0")
    assert code_version(repo)["dirty"] is False
    (repo / "notes.md").write_text("an untracked file\n")
    assert code_version(repo)["dirty"] is True
    (repo / "notes.md").unlink()
    git(repo, "commit", "-q", "--allow-empty", "-m", "second")
    assert code_version(repo) == {"commit": git(repo, "rev-parse", "HEAD"), "dirty": False, "diff_sha256": None,
                                  "source": "git"}


def test_the_saved_patch_hashes_to_the_stamp_and_rebuilds_the_tree(tmp_path):
    repo = make_repo(tmp_path / "repo")
    head = git(repo, "rev-parse", "HEAD")
    (repo / "src" / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
    (repo / "src" / "gone.py").unlink()
    (repo / "blob.bin").write_bytes(bytes(range(255, -1, -1)))
    os.chmod(repo / "tool.sh", 0o755)
    (repo / "new dir").mkdir()
    (repo / "new dir" / "it's new.txt").write_text("hello\n")
    (repo / "new dir" / "data.bin").write_bytes(b"\x00\xff" * 100)
    (repo / "empty.txt").write_text("")
    git(repo, "add", "empty.txt")  # staged, not committed: a tracked change
    (repo / ".env.local").write_text(f"HF_TOKEN={SECRET}\n")

    out = tmp_path / "out"
    stamp = save_changes(out, repo)
    assert stamp == code_version(repo) and stamp["dirty"] and stamp["commit"] == head
    patch = (out / "code.diff").read_bytes()
    assert hashlib.sha256(patch).hexdigest() == stamp["diff_sha256"]
    assert SECRET.encode() not in patch and b".env.local" not in patch
    assert (out / "code_untracked.txt").read_text().splitlines() == ["new dir/data.bin", "new dir/it's new.txt"]
    assert not (out / "code.diff.tmp").exists()

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True, capture_output=True)
    assert git(clone, "rev-parse", "HEAD") == head
    git(clone, "apply", str(out / "code.diff"))
    for rel in ("src/a.py", "blob.bin", "tool.sh", "new dir/it's new.txt", "new dir/data.bin", "empty.txt"):
        assert (clone / rel).read_bytes() == (repo / rel).read_bytes(), rel
    assert not (clone / "src" / "gone.py").exists() and os.access(clone / "tool.sh", os.X_OK)
    git(clone, "add", "empty.txt")  # the stamp of the rebuilt tree is the original's
    assert code_version(clone)["diff_sha256"] == stamp["diff_sha256"]

    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "everything")
    clean = save_changes(out, repo)  # a clean tree: nothing to save, and the old patch is gone
    assert clean["dirty"] is False and not (out / "code.diff").exists() and not (out / "code_untracked.txt").exists()


def test_a_copy_without_git_reads_its_code_version_file(tmp_path):
    stamp = {"commit": "a" * 40, "dirty": True, "diff_sha256": "b" * 64, "source": "git"}
    outer = make_repo(tmp_path / "outer")  # the copy sits inside another checkout: its HEAD must not leak in
    copy = outer / "ignored" / "md"
    copy.mkdir(parents=True)
    assert code_version(copy) == provenance.unknown()
    (copy / "CODE_VERSION.json").write_text(json.dumps(stamp))
    assert code_version(copy) == {**stamp, "source": "file"}
    for bad in ({**stamp, "commit": None}, {**stamp, "commit": "not-hex"}, {**stamp, "dirty": "yes"},
                {**stamp, "diff_sha256": "short"}, ["a list"]):
        (copy / "CODE_VERSION.json").write_text(json.dumps(bad))
        assert code_version(copy)["source"] == "none", bad
    (copy / "CODE_VERSION.json").write_text("{not json")
    assert code_version(copy) == provenance.unknown()
    assert code_version(tmp_path / "does-not-exist") == provenance.unknown()


def test_git_failing_falls_back_and_never_raises(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo")
    (repo / "CODE_VERSION.json").write_text(json.dumps({"commit": "c" * 40, "dirty": False, "diff_sha256": None}))
    git(repo, "add", "CODE_VERSION.json")
    git(repo, "commit", "-q", "-m", "a checkout that also has the file")
    assert code_version(repo)["source"] == "git"  # git wins where it works
    monkeypatch.setenv("PATH", str(tmp_path / "no-git-here"))
    assert code_version(repo) == {"commit": "c" * 40, "dirty": False, "diff_sha256": None, "source": "file"}

    def boom(*a, **k):
        raise OSError("no processes for you")

    monkeypatch.setattr(provenance.subprocess, "run", boom)
    monkeypatch.setattr(provenance.subprocess, "Popen", boom)
    (repo / "CODE_VERSION.json").unlink()
    assert code_version(repo) == provenance.unknown()
    with pytest.raises(OSError):
        save_changes(tmp_path / "out", repo)  # unlike code_version, saving a patch fails loudly


def test_the_default_root_is_this_checkout():
    assert provenance.REPO_ROOT == Path(__file__).resolve().parents[1]
    stamp = code_version()
    assert set(stamp) == {"commit", "dirty", "diff_sha256", "source"} and stamp["source"] in ("git", "file", "none")
    assert provenance.never_recorded(".env") and provenance.never_recorded("a/.envrc")
    assert provenance.never_recorded("x/__pycache__/y.py") and provenance.never_recorded("m.pyc")
    assert not provenance.never_recorded(".gitignore") and not provenance.never_recorded("src/env.py")
