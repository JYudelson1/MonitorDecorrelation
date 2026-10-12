"""Which code made an output: ``code_version()`` stamps results with the commit, whether the working tree differed
from it, and a hash of the difference.

    from monitordecorrelation.provenance import code_version
    record["code"] = code_version()
    # {"commit": "ae3fd80b…", "dirty": false, "diff_sha256": null, "source": "git"}

**Sources**, tried in this order for the repo root (default: the checkout this package is imported from):
- ``"git"``: the root is the top of a git work tree. ``commit`` is HEAD. ``dirty`` is whether the working tree has
  tracked changes, or untracked files that are not ignored. ``diff_sha256`` is the sha256 of the patch from the commit
  to the working tree (``git diff HEAD``, then each untracked file as a new file), None when clean: ``git apply`` of
  that patch on a checkout of the commit rebuilds the tree, and ``save_changes`` writes it.
- ``"file"``: no git work tree, but a ``CODE_VERSION.json`` at the root holding a commit. scripts/nebius_job.py writes
  one into the copy of the repo it pushes to a VM, which has no .git: the launching machine's stamp of what it pushed.
- ``"none"``: neither (or git failed); commit, dirty and diff_sha256 are None.

**Never recorded:** paths that never travel to a VM and may hold secrets (any part ``.git``, ``.venv``,
``__pycache__`` or ``.env*``, and ``*.pyc``: scripts/nebius_job.py's ``shippable``) are left out of the dirty check and
the patch, so a stray ``.env.local`` neither makes the tree dirty nor lands in a saved patch. Nor do the runs' outputs
under data/runs/: they are what experiments write, not code, so a fresh run folder (e.g. pushed to a VM for post-hoc
scoring) doesn't block a launch, and the patch rebuilds the code without them.

``code_version`` is cheap (a few local git calls, no network) and never raises. Stdlib only: scripts/nebius_job.py
loads this file by its path, under a plain python3.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/monitordecorrelation/provenance.py -> the checkout
CODE_VERSION_FILE = "CODE_VERSION.json"  # at the root of a copy without .git (scripts/nebius_job.py)
PATCH_FILE, UNTRACKED_FILE = "code.diff", "code_untracked.txt"  # save_changes
GIT_TIMEOUT = 120.0  # seconds per git call
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# The same bytes for the same tree, whatever the user's git config says about diffs
_CONFIG = ("-c", "core.quotePath=true", "-c", "diff.noprefix=false", "-c", "diff.mnemonicPrefix=false",
           "-c", "diff.suppressBlankEmpty=false")
_DIFF = ("--binary", "--full-index", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames", "--unified=3",
         "--inter-hunk-context=0", "--diff-algorithm=myers", "--indent-heuristic", "--src-prefix=a/",
         "--dst-prefix=b/", "-O/dev/null")
# Variables that would point git at another repository, index or diff format
_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_COMMON_DIR", "GIT_NAMESPACE", "GIT_PREFIX", "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS")


def unknown() -> dict:
    """The stamp when nothing tells which code this is."""
    return {"commit": None, "dirty": None, "diff_sha256": None, "source": "none"}


def never_recorded(path: str) -> bool:
    """Paths left out of the stamp: what scripts/nebius_job.py never pushes (secrets, environments, VCS, bytecode)."""
    parts = path.replace(os.sep, "/").split("/")
    return any(p in (".git", ".venv", "__pycache__") or p.startswith(".env") for p in parts) or path.endswith(".pyc")


RUN_OUTPUTS = "data/runs/"  # what experiments write, not code: a fresh run's folder must not make the code "dirty"


def not_code(path: str) -> bool:
    """Paths the stamp ignores: ``never_recorded`` ones, and the runs' outputs under data/runs/ (e.g. pushed to a VM
    for post-hoc probe scoring)."""
    return never_recorded(path) or path.replace(os.sep, "/").startswith(RUN_OUTPUTS)


class GitError(RuntimeError):
    """A git call failed, or the root is not the top of a git work tree."""


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", GIT_LITERAL_PATHSPECS="1")
    return env


def _git(root: Path, *args: str) -> bytes:
    cp = subprocess.run(["git", "-C", str(root), *_CONFIG, *args], capture_output=True, stdin=subprocess.DEVNULL,
                        timeout=GIT_TIMEOUT, env=_env(), check=False)
    if cp.returncode != 0:
        raise GitError(f"git {args[0]} failed ({cp.returncode}): {cp.stderr.decode(errors='replace').strip()[-500:]}")
    return cp.stdout


def _git_stream(root: Path, *args: str, ok: tuple[int, ...] = (0,)) -> Iterator[bytes]:
    """A git call's output in chunks, so a large diff never sits in memory whole."""
    with subprocess.Popen(["git", "-C", str(root), *_CONFIG, *args], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, env=_env()) as proc:
        yield from iter(lambda: proc.stdout.read(1 << 20), b"")
        try:
            code = proc.wait(timeout=GIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()  # else leaving the ``with`` would wait for it forever
            raise
    if code not in ok:
        raise GitError(f"git {args[0]} failed ({code})")


def _paths(raw: bytes) -> list[str]:
    return [os.fsdecode(p) for p in raw.split(b"\0") if p]


def _changes(root: Path) -> tuple[str, list[str], bool, list[str]]:
    """(HEAD, changed tracked paths kept, whether some were left out, untracked files kept) of the work tree whose
    top is ``root``."""
    top = os.fsdecode(_git(root, "rev-parse", "--show-toplevel").rstrip(b"\n"))
    if not os.path.samefile(top, root):  # inside another checkout (e.g. a copy without .git in a repo): not ours
        raise GitError(f"{root} is not the top of the git work tree {top}")
    commit = _git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}").decode().strip()
    changed = _paths(_git(root, "diff", "HEAD", "--name-only", "-z", "--no-renames", "--no-ext-diff"))
    tracked = [p for p in changed if not not_code(p)]
    others = _paths(_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    untracked = sorted((p for p in others if not not_code(p) and os.path.isfile(root / p)), key=os.fsencode)
    return commit, tracked, len(tracked) < len(changed), untracked


def _patch(root: Path, tracked: list[str], left_out: bool, untracked: list[str]) -> Iterator[bytes]:
    """The patch from HEAD to the work tree: the tracked changes (only the kept paths if some were left out), then
    every untracked file as a new file."""
    if tracked:
        yield from _git_stream(root, "diff", *_DIFF, "HEAD", *(("--", *tracked) if left_out else ()))
    for path in untracked:  # exit code 1: the files differ, which they do
        yield from _git_stream(root, "diff", "--no-index", *_DIFF, "--", "/dev/null", path, ok=(0, 1))


def _git_stamp(root: Path, sink=None) -> tuple[dict, list[str]]:
    """The "git" stamp and the untracked files in its patch; the patch also goes to ``sink`` (a binary file) if
    given. Raises GitError if ``root`` is not the top of a git work tree or git fails."""
    commit, tracked, left_out, untracked = _changes(root)
    h, size = hashlib.sha256(), 0
    for chunk in _patch(root, tracked, left_out, untracked):
        h.update(chunk)
        size += len(chunk)
        if sink is not None:
            sink.write(chunk)
    dirty = size > 0
    return {"commit": commit, "dirty": dirty, "diff_sha256": h.hexdigest() if dirty else None, "source": "git"}, \
        untracked


def _file_stamp(root: Path) -> dict:
    """The stamp in ``root``/CODE_VERSION.json; raises if it is missing or malformed (no commit)."""
    data = json.loads((root / CODE_VERSION_FILE).read_text())
    commit, dirty, diff = data.get("commit"), data.get("dirty"), data.get("diff_sha256")
    if not (isinstance(commit, str) and _COMMIT.fullmatch(commit) and isinstance(dirty, bool)
            and (diff is None or (isinstance(diff, str) and _SHA256.fullmatch(diff)))):
        raise ValueError(f"{root / CODE_VERSION_FILE}: not a code version")
    return {"commit": commit, "dirty": dirty, "diff_sha256": diff, "source": "file"}


def code_version(repo_root: str | os.PathLike | None = None) -> dict:
    """``{"commit", "dirty", "diff_sha256", "source"}`` for ``repo_root`` (default: the checkout this package is
    imported from). See the module docstring. Never raises."""
    try:
        root = Path(REPO_ROOT if repo_root is None else repo_root).resolve()
    except Exception:  # noqa: BLE001 — never raises
        return unknown()
    try:
        return _git_stamp(root)[0]
    except Exception:  # noqa: BLE001 — not a work tree, no git, or git failed: try the file
        pass
    try:
        return _file_stamp(root)
    except Exception:  # noqa: BLE001
        return unknown()


def save_changes(out_dir: str | os.PathLike, repo_root: str | os.PathLike | None = None) -> dict:
    """The "git" stamp of ``repo_root`` (default: this checkout), and, if the tree is dirty, the patch whose sha256 is
    its ``diff_sha256`` in ``out_dir``/code.diff, with the untracked files in that patch listed one per line in
    ``out_dir``/code_untracked.txt. On a checkout of ``commit``, ``git apply code.diff`` rebuilds the tree. For a clean
    tree, earlier copies of the two files are removed. Unlike ``code_version`` it raises: GitError if ``repo_root`` is
    not the top of a git work tree or git fails, OSError if the files can't be written."""
    root = Path(REPO_ROOT if repo_root is None else repo_root).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (PATCH_FILE + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            stamp, untracked = _git_stamp(root, sink=fh)
        if stamp["dirty"]:
            (out / UNTRACKED_FILE).write_text("".join(f"{p}\n" for p in untracked))
            os.replace(tmp, out / PATCH_FILE)
        else:
            for name in (PATCH_FILE, UNTRACKED_FILE):
                (out / name).unlink(missing_ok=True)
        return stamp
    finally:
        tmp.unlink(missing_ok=True)
