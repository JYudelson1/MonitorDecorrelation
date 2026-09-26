#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""A local web viewer for every run under ``data/runs/`` — full rollouts + per-step metrics.

    uv run visualize_transcripts.py            # → http://127.0.0.1:8000
    uv run visualize_transcripts.py --host 0.0.0.0 --port 8080 --runs-dir data/runs

What it shows, per run:
  * **Overview** — run_info.json / config.json: policy, env, hyperparams, the monitor table
    (name · kind · role · model · threshold), checkpoints, description.
  * **Metrics** — every series in ``metrics.jsonl`` (train) and ``eval_metrics.jsonl`` (eval),
    charted and tabulated. ``behavior_rate`` (the oracle) is preselected, per CLAUDE.md.
  * **Score dist** — each monitor's score histogram at ONE step, split by the oracle
    (behavior present vs absent), with per-class means, the gap, AUROC and the threshold. Pick an
    RL train step (``rollouts.jsonl``: train-against monitors only) or an eval step (the eval dump:
    every monitor, one panel each). A pre-RL baseline dir from
    ``experiments/eval_terminal_monitors_baseline.py`` has a single eval step, so there's no picker.
    Rollouts no monitor scored (invalid, or a control run's train dump) are excluded and counted.
  * **Rollouts** — the full text of every saved rollout: the complete prompt, the complete CoT,
    the complete answer/transcript, the per-turn breakdown for multi-turn envs, the env grading
    record and every monitor's score. **Nothing is truncated** — what you see is exactly what was
    written to the jsonl (a rollout that the *sampler* truncated shows as it was truncated).
    For every LLM-judge monitor whose API call the run saved (``monitors.<name>.call`` in the
    record — written by rl/train.py since call recording was added), the exact prompt it was sent,
    the other request parameters, and its response including the chain of thought when the
    provider returned one. Older dumps carry no call record; the viewer says so rather than
    reconstructing a prompt from the run config (a reconstruction can differ from what the judge
    was really sent if the repo changed since the run).
  * **Plots / Log** — the PNGs the training loop rendered, and ``run.log``.

Rollout dumps run to ~100 MB per run, so they are never loaded whole: each ``*.jsonl`` gets a
byte-offset index (built lazily, cached on disk under ``.cache/visualize_transcripts/``), and a
single rollout is served by seeking to its offset. **Refreshing re-scans**: new run directories
appear, and runs that grew get their new steps/rollouts indexed incrementally. A dump is never
*assumed* append-only, though — scripts rewrite dumps in place, sometimes longer than before — so
whenever a file changed, the already-indexed prefix is re-hashed and a mismatch re-indexes it from
scratch (see ``AppendOnlyJsonl``); every rollout read is checked against the bytes that were indexed;
and a list's row numbers are tied to the file version they came from, so a stale click is refused
instead of opening a different rollout. The page also polls in the background, so a live run fills in.

Stdlib only — no deps, nothing to install.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import socket
import sys
import threading
import time
import traceback
import uuid
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, urlparse

# --------------------------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------------------------

# rollout dumps, in the order the UI offers them
SOURCES = {
    "train": ("rollouts.jsonl", "train rollouts (sampled fraction, train-against monitors only)"),
    "eval": ("eval_rollouts.jsonl", "eval rollouts (fixed held-out set, ALL monitors)"),
    "eval_slim": ("eval_rollouts_slim.jsonl", "eval rollouts, slim (labels + scores, no text)"),
}

# A directory (at any depth under --runs-dir) is a RUN if it holds any of these — including a bare
# rollout dump with no metrics/config (e.g. runs copied back with only their rollouts). Batch
# directories (mbpp_matrix_*/) hold only sub-runs and are used purely as a grouping label.
RUN_MARKERS = ("run_info.json", "metrics.jsonl", "eval_metrics.jsonl", "config.json") + tuple(
    f for f, _ in SOURCES.values()
)

SKIP_DIRS = {"__pycache__", ".git", ".cache"}


def _stat(p: Path) -> Optional[tuple[int, float]]:
    try:
        st = p.stat()
        return st.st_size, st.st_mtime
    except OSError:
        return None


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _nan_safe(o: Any) -> Any:
    """json.dumps(allow_nan=False)-safe: NaN/Inf → None (JS has no NaN literal)."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _nan_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_safe(v) for v in o]
    return o


def dumps(o: Any) -> bytes:
    return json.dumps(_nan_safe(o), default=str).encode()


def _auroc(pos: list[float], neg: list[float]) -> Optional[float]:
    """P(score | present > score | absent), ties counted half (Mann–Whitney). None if a class is empty."""
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2  # mean of 1-based ranks i+1 … j
        rank_sum += avg * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# --------------------------------------------------------------------------------------------
# incremental jsonl readers
# --------------------------------------------------------------------------------------------


_CHUNK = 1 << 24


def _file_id(p: Path) -> Optional[tuple]:
    """Everything stat knows about a file's identity and contents; None if it doesn't exist."""
    try:
        st = p.stat()
    except OSError:
        return None
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _hash_prefix(path: Path, n: int) -> Optional["hashlib._Hash"]:
    """sha1 of the file's first ``n`` bytes; None if it now has fewer, or can't be read."""
    h = hashlib.sha1()
    try:
        with path.open("rb") as f:
            while n:
                b = f.read(min(n, _CHUNK))
                if not b:
                    return None
                h.update(b)
                n -= len(b)
    except OSError:
        return None
    return h


@dataclass
class AppendOnlyJsonl:
    """A jsonl file parsed incrementally, WITHOUT ever assuming it was only appended to.

    Live runs append, so normally only the new bytes need parsing. But dumps also get rewritten in
    place (eval_terminal_monitors_baseline.py re-opens eval_rollouts.jsonl with mode "w"), and the
    rewrite can be LONGER than what was parsed — so size says nothing about append-vs-rewrite.
    Instead ``digest`` is the sha1 of exactly the bytes parsed, and whenever stat says the file
    changed, that same prefix of the current file is re-hashed: equal → an append, parse only the
    tail; different or shorter → a rewrite, discard everything and re-parse from byte 0.

    ``epoch`` changes on every discard: row numbers handed out under one epoch keep meaning the same
    line until it changes. If the file changes WHILE being read (a writer mid-rewrite) the parse
    goes round again, so it never mixes two versions of the file.
    """

    path: Path
    offset: int = 0                 # bytes parsed (always a whole number of lines)
    bad: int = 0                    # complete lines that didn't parse as JSON
    digest: str = ""                # sha1 of bytes [0, offset)
    epoch: str = ""
    fid: Optional[tuple] = None     # _file_id the parse was verified against; None = verify next sync

    def _clear(self) -> None:
        raise NotImplementedError

    def _take(self, raw: bytes, rec: dict, off: int) -> None:
        raise NotImplementedError

    def _discard(self) -> None:
        self._clear()
        self.offset, self.bad, self.digest = 0, 0, ""
        self.epoch = uuid.uuid4().hex[:12]

    def sync(self) -> bool:
        """Make the parse match the file on disk. Returns True if anything changed."""
        changed = False
        for _ in range(5):
            fid = _file_id(self.path)
            if fid is None:
                changed |= self.offset > 0
                if self.offset or not self.epoch:
                    self._discard()
                self.fid = None
                return changed
            if fid == self.fid:
                return changed
            h = _hash_prefix(self.path, self.offset)
            if h is None or h.hexdigest() != self.digest:
                changed |= self.offset > 0
                self._discard()
                h = hashlib.sha1()
            off = self.offset
            try:
                with self.path.open("rb") as f:
                    f.seek(off)
                    for raw in f:
                        if not raw.endswith(b"\n"):
                            break  # partial line: a live run is mid-write, pick it up next sync
                        s = raw.strip()
                        if s:
                            try:
                                rec = json.loads(s)
                            except Exception:
                                self.bad += 1  # interleaved/spliced writes happen; skip like coupling._read_jsonl
                                rec = None
                            if isinstance(rec, dict):
                                self._take(raw, rec, off)
                        h.update(raw)
                        off += len(raw)
            except OSError:
                pass
            changed |= off != self.offset
            self.offset, self.digest = off, h.hexdigest()
            if _file_id(self.path) == fid:
                self.fid = fid
                return changed
            # modified while being read: go round — the prefix check tells an append from a rewrite
        # still changing after every retry: keep the parse only if it IS the file's current prefix
        self.fid = None
        h = _hash_prefix(self.path, self.offset)
        if h is None or h.hexdigest() != self.digest:
            changed |= self.offset > 0
            self._discard()
        return changed


@dataclass
class JsonlTail(AppendOnlyJsonl):
    """Whole-file jsonl reader for the small per-step metrics files."""

    rows: list[dict] = field(default_factory=list)

    def _clear(self) -> None:
        self.rows = []  # a new list: a request holding the old one keeps a consistent (old) view

    def _take(self, raw: bytes, rec: dict, off: int) -> None:
        self.rows.append(rec)

    def refresh(self) -> bool:
        return self.sync()


def _summarize(rec: dict, off: int, length: int) -> dict:
    """The small per-rollout card the list view needs (the full record is served on demand)."""
    env = rec.get("env") if isinstance(rec.get("env"), dict) else {}
    meta = rec.get("env_meta")
    if not isinstance(meta, dict):
        meta = env.get("meta") if isinstance(env.get("meta"), dict) else {}
    mons = rec.get("monitors") if isinstance(rec.get("monitors"), dict) else {}
    monitors = {}
    for name, m in mons.items():
        if isinstance(m, dict):
            monitors[name] = {"score": _nan_safe(m.get("score")), "label": m.get("label")}
            # judge call health: its finish_reason (full dump: inside the call record; slim dump: lifted
            # out) and whether its answer was unparseable (scored 0)
            call = m.get("call") if isinstance(m.get("call"), dict) else {}
            resp = call.get("response") if isinstance(call.get("response"), dict) else {}
            finish = m.get("finish_reason") or resp.get("finish_reason")
            if finish is not None:
                monitors[name]["finish_reason"] = finish
            if m.get("parse_error"):
                monitors[name]["parse_error"] = True
        else:
            monitors[name] = {"score": _nan_safe(m), "label": None}

    def pick(*keys, src: Iterable[dict] = ()):
        for d in (rec, env, meta, *src):
            for k in keys:
                if isinstance(d, dict) and k in d and d[k] is not None:
                    return d[k]
        return None

    answer = rec.get("answer") or ""
    cot = rec.get("cot") or ""
    preview = (answer or cot or rec.get("question") or "")
    preview = re.sub(r"\s+", " ", str(preview)[:400]).strip()[:220]
    return {
        "off": off,
        "len": length,
        "step": rec.get("step"),
        "task_id": pick("task_id"),
        "behavior_present": pick("behavior_present"),
        "loose_rh": pick("loose_rh"),
        "hardcoding": pick("hardcoding"),
        "unparsed": pick("unparsed"),
        # null | "truncated" | "unparsed" | "no_submission" (since 2026-09-26): such a rollout was shown to
        # no monitor (dumps since 2026-09-18)
        "invalid_reason": rec.get("invalid_reason"),
        "reward": _nan_safe(rec.get("reward")),
        "task_reward": _nan_safe(env.get("task_reward") if env else meta.get("reward")),
        "n_turns": meta.get("n_turns") if isinstance(meta, dict) else None,
        "chars": len(str(rec.get("question") or "")) + len(str(cot)) + len(str(answer)),
        "has_text": bool(answer or cot or rec.get("question")),
        "monitors": monitors,
        "preview": preview,
    }


@dataclass
class RolloutIndex(AppendOnlyJsonl):
    """Byte-offset index over one rollout dump. Built lazily, extended incrementally, cached."""

    entries: list[dict] = field(default_factory=list)
    built: bool = False
    seconds: float = 0.0

    def _clear(self) -> None:
        self.entries = []  # a new list: views handed out earlier keep their (old) entries intact

    def _take(self, raw: bytes, rec: dict, off: int) -> None:
        self.entries.append({**_summarize(rec, off, len(raw)), "crc": zlib.crc32(raw)})

    def refresh(self, cache: "IndexCache | None" = None) -> None:
        if not self.built and cache is not None:
            st = cache.load(self.path)
            if st is not None:  # adopted unverified (fid None): the sync below checks it against the file
                self.entries, self.offset, self.bad = st["entries"], st["offset"], st["bad"]
                self.digest, self.epoch = st["digest"], st["epoch"]
        t0 = time.perf_counter()
        changed = self.sync()
        self.seconds = time.perf_counter() - t0
        self.built = True
        if cache is not None and changed:
            cache.save(self.path, {"entries": self.entries, "offset": self.offset, "bad": self.bad,
                                   "digest": self.digest, "epoch": self.epoch})

    def view(self) -> "IndexView":
        return IndexView(self.path, list(self.entries), self.epoch, self.bad, self.seconds)


@dataclass(frozen=True)
class IndexView:
    """One request's snapshot of a RolloutIndex. Entry ``i`` names the same rollout for as long as
    ``epoch`` is unchanged, and a read returns None unless the bytes at the entry's offset are still
    exactly the ones that were indexed (same length + crc32) — so a rewrite between indexing and
    reading can never serve a different rollout, or a spliced one, under an old entry."""

    path: Path
    entries: list[dict]
    epoch: str
    bad: int
    seconds: float

    def read_raw(self, i: int) -> Optional[bytes]:
        if not (0 <= i < len(self.entries)):
            return None
        e = self.entries[i]
        try:
            with self.path.open("rb") as f:
                f.seek(e["off"])
                raw = f.read(e["len"])
        except OSError:
            return None
        return raw if len(raw) == e["len"] and zlib.crc32(raw) == e["crc"] else None

    def read(self, i: int) -> Optional[dict]:
        raw = self.read_raw(i)
        try:
            return json.loads(raw) if raw is not None else None
        except Exception:
            return None


class IndexCache:
    """On-disk cache of rollout indexes, keyed by absolute path. A cached index is only ever a
    starting point: RolloutIndex re-verifies it against the file (prefix sha1) before using it."""

    KEYS = ("entries", "offset", "bad", "digest", "epoch")

    def __init__(self, root: Optional[Path]):
        self.root = root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def _p(self, path: Path) -> Path:
        h = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]
        return self.root / f"{path.name}.{h}.v3.json"  # type: ignore[union-attr]

    def load(self, path: Path) -> Optional[dict]:
        if self.root is None:
            return None
        try:
            d = json.loads(self._p(path).read_text())
        except Exception:
            return None
        if not isinstance(d, dict) or any(k not in d for k in self.KEYS):
            return None
        return d

    def save(self, path: Path, payload: dict) -> None:
        if self.root is None:
            return
        p = self._p(path)
        try:
            tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({**payload, "path": str(path)}))
            tmp.replace(p)
        except Exception:
            pass


# --------------------------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------------------------


@dataclass
class Run:
    id: str          # path relative to the runs root, e.g. "mbpp_matrix_sep3_20260903/mbpp_..._s0"
    dir: Path
    group: str       # parent dir relative to root ("" for a top-level run)
    name: str
    run_info: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    info_stat: tuple = ()
    train: JsonlTail = None          # type: ignore[assignment]
    eval: JsonlTail = None           # type: ignore[assignment]
    indexes: dict[str, RolloutIndex] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)       # guards the rollout indexes
    meta_lock: threading.Lock = field(default_factory=threading.Lock)  # guards run_info/config + metrics tails


class Store:
    def __init__(self, root: Path, cache_dir: Optional[Path]):
        self.root = root
        self.runs: dict[str, Run] = {}
        self.cache = IndexCache(cache_dir)
        self.lock = threading.RLock()
        self.last_scan = 0.0

    # -- discovery ---------------------------------------------------------------------------

    def _walk(self) -> list[Path]:
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            if any(m in filenames for m in RUN_MARKERS):
                found.append(Path(dirpath))
        return found

    def scan(self) -> None:
        """(Re)discover run dirs and refresh every run's metrics. Cheap enough to poll."""
        with self.lock:
            seen = set()
            for d in self._walk():
                rid = str(d.relative_to(self.root))
                seen.add(rid)
                run = self.runs.get(rid)
                if run is None:
                    parent = str(d.parent.relative_to(self.root)) if d.parent != self.root else ""
                    run = Run(id=rid, dir=d, group="" if parent == "." else parent, name=d.name)
                    run.train = JsonlTail(d / "metrics.jsonl")
                    run.eval = JsonlTail(d / "eval_metrics.jsonl")
                    self.runs[rid] = run
                self._refresh_run(run)
            for gone in set(self.runs) - seen:
                del self.runs[gone]
            self.last_scan = time.time()

    def _refresh_run(self, run: Run) -> None:
        with run.meta_lock:  # scan() and request threads both get here; two syncs of one tail would double its rows
            info_p, cfg_p = run.dir / "run_info.json", run.dir / "config.json"
            stat = (_file_id(info_p), _file_id(cfg_p))
            if stat != run.info_stat:
                info, cfg = _read_json(info_p), _read_json(cfg_p)
                run.run_info = info if isinstance(info, dict) else {}
                run.config = (cfg if isinstance(cfg, dict) else None) or (run.run_info.get("config") or {})
                # Settle only on a clean read: a file caught mid-write parses as nothing (or changes
                # under us) — leave info_stat unset so it is re-read on the next refresh.
                clean = all(fid is None or isinstance(d, dict) for fid, d in zip(stat, (info, cfg)))
                run.info_stat = stat if clean and (_file_id(info_p), _file_id(cfg_p)) == stat else ()
            run.train.refresh()
            run.eval.refresh()

    def get(self, rid: str) -> Optional[Run]:
        with self.lock:
            run = self.runs.get(rid)
        if run is None:
            self.scan()
            with self.lock:
                run = self.runs.get(rid)
        if run is not None:
            self._refresh_run(run)
        return run

    def index(self, run: Run, source: str, reverify: bool = False) -> Optional[IndexView]:
        """A verified-current snapshot of one rollout dump's index. ``reverify`` re-checks the file
        even if stat looks unchanged (a read found bytes that no longer match what was indexed)."""
        fname = SOURCES.get(source, (None, None))[0]
        if fname is None:
            return None
        path = run.dir / fname
        if not path.exists():
            return None
        with run.lock:  # per-run lock: indexing one big dump must not block the rest of the UI
            idx = run.indexes.get(source)
            if idx is None:
                idx = RolloutIndex(path)
                run.indexes[source] = idx
            if reverify:
                idx.fid = None
            idx.refresh(self.cache)
            return idx.view()

    # -- serialization -----------------------------------------------------------------------

    def _monitors(self, run: Run) -> list[dict]:
        out = []
        cfg_mons = (run.config or {}).get("monitors") or []
        by_name = {m.get("name"): m for m in cfg_mons if isinstance(m, dict)}
        roles = {}
        for role in ("train_against", "held_out"):
            for m in (run.run_info or {}).get(role) or []:
                if isinstance(m, dict) and m.get("name"):
                    roles[m["name"]] = {**m, "role": role}
        for name in list(by_name) + [n for n in roles if n not in by_name]:
            m = {**(by_name.get(name) or {}), **(roles.get(name) or {})}
            out.append({
                "name": name,
                "kind": m.get("kind"),
                "role": m.get("role"),
                "model_id": m.get("model_id") or m.get("probe_path"),
                "threshold": m.get("threshold"),
                "use_cot": m.get("use_cot"),
                "use_output": m.get("use_output"),
                "binary_judge": m.get("binary_judge"),
                "behavior": m.get("behavior"),
                # run_info's RESOLVED object when recorded (config.json may say null = model default);
                # legacy runs recorded reasoning_effort / reasoning_max_tokens instead
                "reasoning": m.get("reasoning"),
                "reasoning_max_tokens": m.get("reasoning_max_tokens"),
                "reasoning_effort": m.get("reasoning_effort"),
                # the judge's backend: provider + completion cap, and a vLLM judge's thinking settings
                "provider": m.get("provider"),
                "max_tokens": m.get("max_tokens"),
                "base_url": m.get("base_url"),
                "enable_thinking": m.get("enable_thinking"),
                "thinking_budget": m.get("thinking_budget"),
                "probe_path": m.get("probe_path"),
                "probe_model": m.get("probe_model"),
            })
        return out

    def run_card(self, run: Run) -> dict:
        cfg = run.config or {}
        ri = run.run_info or {}
        sources = {}
        for key, (fname, desc) in SOURCES.items():
            st = _stat(run.dir / fname)
            if st:
                idx = run.indexes.get(key)
                sources[key] = {
                    "file": fname, "desc": desc, "size": st[0], "mtime": st[1],
                    "indexed": (idx.built and idx.fid is not None and idx.fid == _file_id(run.dir / fname)) if idx else False,
                    "n": len(idx.entries) if idx else None,
                }
        plots = sorted(str(p.relative_to(run.dir)) for p in run.dir.rglob("*.png"))
        tr, ev = run.train.rows, run.eval.rows
        mtimes = [(_stat(run.dir / f) or (0, 0.0))[1] for f in
                  ("metrics.jsonl", "eval_metrics.jsonl", "rollouts.jsonl", "eval_rollouts.jsonl", "run_info.json")]
        return {
            "id": run.id, "name": run.name, "group": run.group,
            "dir": str(run.dir),
            "policy": ri.get("policy") or cfg.get("policy"),
            "env": cfg.get("env") or (ri.get("env") or {}).get("name"),
            "subset": cfg.get("subset") or ri.get("subset"),
            "experiment": cfg.get("experiment") or ri.get("experiment"),
            "started_at": ri.get("started_at"),
            "seed": cfg.get("seed"),
            "lr": cfg.get("lr") or ri.get("lr"),
            "penalty_coef": cfg.get("penalty_coef"),
            "n_steps_cfg": cfg.get("n_steps"),
            "train_steps": len(tr), "eval_steps": len(ev),
            "last_train_step": tr[-1].get("step") if tr else None,
            "last_eval_step": ev[-1].get("step") if ev else None,
            "last_behavior_rate": _nan_safe(tr[-1].get("behavior_rate")) if tr else None,
            "train_against": [m["name"] for m in self._monitors(run) if m.get("role") == "train_against"],
            "n_monitors": len(self._monitors(run)),
            "mtime": max(mtimes),
            "sources": sources,
            "plots": plots,
            "has_log": (run.dir / "run.log").exists(),
        }

    def cards(self) -> list[dict]:
        with self.lock:
            runs = list(self.runs.values())
        return sorted((self.run_card(r) for r in runs), key=lambda c: (c["group"], c["name"]))


# --------------------------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    store: Store = None  # type: ignore[assignment]
    poll_seconds: int = 20
    server_version = "transcript-viewer"

    def log_message(self, fmt: str, *args) -> None:  # quieter default log
        if os.environ.get("VIZ_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers -----------------------------------------------------------------------------

    def _send(self, body: bytes, ctype: str = "application/json", code: int = 200, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(dumps(obj), "application/json", code)

    def _err(self, msg: str, code: int = 400) -> None:
        self._json({"error": msg}, code)

    def _stale(self) -> None:
        self._json({"error": "this rollout dump was rewritten on disk since the list was loaded — reloaded the list",
                    "stale": True}, 409)

    def _run(self, q: dict) -> Optional[Run]:
        rid = (q.get("id") or [""])[0]
        run = self.store.get(rid)
        if run is None:
            self._err(f"unknown run: {rid!r}", 404)
        return run

    # -- routes ------------------------------------------------------------------------------

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        try:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            route = u.path
            if route == "/":
                boot = dumps({"poll_seconds": self.poll_seconds, "root": str(self.store.root)}).decode()
                self._send(PAGE.replace("__BOOTSTRAP__", boot).encode(), "text/html; charset=utf-8")
            elif route == "/api/runs":
                if (q.get("rescan") or ["1"])[0] == "1":
                    self.store.scan()
                self._json({"runs": self.store.cards(), "root": str(self.store.root), "scanned_at": self.store.last_scan})
            elif route == "/api/run":
                self._route_run(q)
            elif route == "/api/metrics.csv":
                self._route_csv(q)
            elif route == "/api/rollouts":
                self._route_rollouts(q)
            elif route == "/api/rollout":
                self._route_rollout(q)
            elif route == "/api/file":
                self._route_file(q)
            elif route == "/api/scoredist":
                self._route_scoredist(q)
            else:
                self._err("not found", 404)
        except BrokenPipeError:
            pass
        except Exception:
            traceback.print_exc()
            try:
                self._err(traceback.format_exc(limit=3), 500)
            except Exception:
                pass

    def _route_run(self, q: dict) -> None:
        run = self._run(q)
        if run is None:
            return
        self._json({
            "card": self.store.run_card(run),
            "run_info": _nan_safe(run.run_info),
            "config": _nan_safe(run.config),
            "monitors": self.store._monitors(run),
            "metrics": {
                "train": _nan_safe(run.train.rows),
                "eval": _nan_safe(run.eval.rows),
                "train_bad": run.train.bad, "eval_bad": run.eval.bad,
            },
        })

    def _route_csv(self, q: dict) -> None:
        run = self._run(q)
        if run is None:
            return
        kind = (q.get("kind") or ["train"])[0]
        rows = (run.eval if kind == "eval" else run.train).rows
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        fn = f"{run.name}.{kind}.csv"
        self._send(buf.getvalue().encode(), "text/csv",
                   extra={"Content-Disposition": f'attachment; filename="{fn}"'})

    def _route_rollouts(self, q: dict) -> None:
        run = self._run(q)
        if run is None:
            return
        source = (q.get("source") or ["eval"])[0]
        idx = self.store.index(run, source)
        if idx is None:
            self._json({"total": 0, "entries": [], "steps": [], "missing": True,
                        "file": SOURCES.get(source, ("?",))[0]})
            return

        def qp(name, cast, default=None):
            v = (q.get(name) or [""])[0]
            if v == "":
                return default
            try:
                return cast(v)
            except Exception:
                return default

        step = qp("step", int)
        behavior = (q.get("behavior") or [""])[0]      # "" | "1" | "0"
        unparsed = (q.get("unparsed") or [""])[0]      # "" | "1" | "0" | "inv" | "val" (invalid = not monitored)
        mon = (q.get("mon") or [""])[0]                # monitor name for the score filter
        mon_min = qp("mon_min", float)
        mon_max = qp("mon_max", float)
        text = (q.get("q") or [""])[0]
        limit = qp("limit", int, 50) or 50
        offset = qp("offset", int, 0) or 0
        order = (q.get("order") or ["asc"])[0]

        def keep(e: dict) -> bool:
            if step is not None and e.get("step") != step:
                return False
            if behavior in ("0", "1") and bool(e.get("behavior_present")) != (behavior == "1"):
                return False
            if unparsed in ("0", "1") and bool(e.get("unparsed")) != (unparsed == "1"):
                return False
            if unparsed in ("inv", "val") and (e.get("invalid_reason") is not None) != (unparsed == "inv"):
                return False
            if mon:
                m = (e.get("monitors") or {}).get(mon)
                s = (m or {}).get("score")
                if s is None:
                    return False
                if mon_min is not None and s < mon_min:
                    return False
                if mon_max is not None and s > mon_max:
                    return False
            return True

        scanned = 0
        for _ in range(3):
            sel = [i for i, e in enumerate(idx.entries) if keep(e)]
            if not text:
                break
            needle, hits, scanned = text.lower(), [], 0
            for i in sel:
                scanned += 1
                raw = idx.read_raw(i)
                if raw is None:  # the dump changed on disk since it was indexed: re-index, search again
                    break
                if needle in raw.decode("utf-8", "replace").lower():
                    hits.append(i)
            else:
                sel = hits
                break
            idx = self.store.index(run, source, reverify=True)
            if idx is None:
                return self._err("rollout dump disappeared", 404)
        else:
            return self._stale()
        if order == "desc":
            sel = sel[::-1]
        page = sel[offset: offset + limit]
        steps = sorted({e.get("step") for e in idx.entries if e.get("step") is not None})
        monitor_names: list[str] = []
        for e in idx.entries[:200]:
            for n in (e.get("monitors") or {}):
                if n not in monitor_names:
                    monitor_names.append(n)
        self._json({
            "total": len(sel), "offset": offset, "limit": limit,
            "steps": steps, "monitor_names": monitor_names,
            "n_indexed": len(idx.entries), "bad_lines": idx.bad,
            "file": idx.path.name, "index_seconds": round(idx.seconds, 2), "searched": scanned,
            "epoch": idx.epoch,
            "entries": [{**{k: v for k, v in idx.entries[i].items() if k not in ("off", "len", "crc")}, "i": i}
                        for i in page],
        })

    def _route_rollout(self, q: dict) -> None:
        run = self._run(q)
        if run is None:
            return
        source = (q.get("source") or ["eval"])[0]
        idx = self.store.index(run, source)
        if idx is None:
            return self._err("no such rollout dump", 404)
        try:
            i = int((q.get("i") or ["0"])[0])
        except ValueError:
            return self._err("bad index")
        epoch = (q.get("epoch") or [""])[0]  # the list's epoch: its row numbers only mean anything in it
        if epoch and epoch != idx.epoch:
            return self._stale()
        if not (0 <= i < len(idx.entries)):
            return self._err("no such rollout", 404)
        rec = idx.read(i)
        if rec is None:  # its bytes are no longer the ones indexed: the file changed since
            self.store.index(run, source, reverify=True)
            return self._stale()
        self._json({"i": i, "epoch": idx.epoch, "record": _nan_safe(rec)})

    def _route_scoredist(self, q: dict) -> None:
        """Every monitor's scores at ONE step, split by the oracle (behavior present / absent).

        ``kind=train`` reads rollouts.jsonl (an RL step: train-against monitors only); ``kind=eval``
        reads the eval dump (an eval round: every monitor) — the slim dump when present, since it has
        the same labels + scores and indexes far faster. A pre-RL baseline dir
        (eval_terminal_monitors_baseline.py) is an eval dump with a single step, 0.
        """
        run = self._run(q)
        if run is None:
            return
        kinds = [k for k, fn in (("train", "rollouts.jsonl"), ("eval", "eval_rollouts.jsonl"))
                 if (run.dir / fn).exists() or (k == "eval" and (run.dir / "eval_rollouts_slim.jsonl").exists())]
        if not kinds:
            return self._json({"kinds": [], "missing": True})
        kind = (q.get("kind") or [""])[0]
        if kind not in kinds:
            kind = "eval" if "eval" in kinds else kinds[0]
        source = "train" if kind == "train" else \
            ("eval_slim" if (run.dir / SOURCES["eval_slim"][0]).exists() else "eval")
        idx = self.store.index(run, source)
        steps = sorted({e.get("step") for e in idx.entries if e.get("step") is not None}) if idx else []
        try:
            step = int((q.get("step") or [""])[0])
        except ValueError:
            step = None
        if step not in steps:
            step = steps[-1] if steps else None

        # monitor order: the run's config/run_info order, then any extra names seen in the dump
        order = [m["name"] for m in self.store._monitors(run)]
        by_mon: dict[str, dict] = {}
        n_rollouts = n_unlabeled = n_unscored = 0
        unscored_why: dict[str, int] = {}
        for e in (idx.entries if idx else []):
            if e.get("step") != step:
                continue
            n_rollouts += 1
            beh = e.get("behavior_present")
            if beh is None:
                n_unlabeled += 1
            scored = False
            for name, m in (e.get("monitors") or {}).items():
                s = (m or {}).get("score")
                if s is None:
                    continue
                scored = True
                d = by_mon.setdefault(name, {"present": [], "absent": [], "unlabeled": 0,
                                             "n_calls": 0, "n_finish_length": 0, "n_parse_error": 0,
                                             "n_finish_known": 0})
                # judge call health, over every scored rollout (labeled or not)
                d["n_calls"] += 1
                if m.get("finish_reason") is not None:
                    d["n_finish_known"] += 1
                    d["n_finish_length"] += m["finish_reason"] == "length"
                d["n_parse_error"] += bool(m.get("parse_error"))
                if beh is None:
                    d["unlabeled"] += 1
                else:
                    d["present" if beh else "absent"].append(s)
            if not scored:
                n_unscored += 1
                why = e.get("invalid_reason") or "no monitor score"
                unscored_why[why] = unscored_why.get(why, 0) + 1
        names = [n for n in order if n in by_mon] + [n for n in by_mon if n not in order]
        monitors = []
        for n in names:
            d = by_mon[n]
            pos, neg = d["present"], d["absent"]
            monitors.append({
                "name": n, **d,
                "mean_present": sum(pos) / len(pos) if pos else None,
                "mean_absent": sum(neg) / len(neg) if neg else None,
                "auroc": _auroc(pos, neg),
            })
        self._json({
            "kinds": kinds, "kind": kind, "file": idx.path.name if idx else None,
            "steps": steps, "step": step, "n_rollouts": n_rollouts,
            "n_unlabeled": n_unlabeled, "n_unscored": n_unscored, "unscored_why": unscored_why,
            "monitors": monitors,
        })

    def _route_file(self, q: dict) -> None:
        """Serve a file from inside a run dir (plots, run.log, raw configs). Path-escape guarded."""
        run = self._run(q)
        if run is None:
            return
        rel = (q.get("path") or [""])[0]
        target = (run.dir / rel).resolve()
        if not str(target).startswith(str(run.dir.resolve())) or not target.is_file():
            return self._err("no such file", 404)
        ctype = {
            ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
            ".json": "application/json", ".csv": "text/csv",
        }.get(target.suffix, "text/plain; charset=utf-8")
        data = target.read_bytes()
        cap = 8 * 1024 * 1024
        if len(data) > cap and ctype.startswith("text"):
            head = f"[showing the last {cap // 1024 // 1024} MB of {len(data) / 1e6:.1f} MB]\n\n".encode()
            data = head + data[-cap:]
        self._send(data, ctype)


# --------------------------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Run transcripts</title>
<style>
:root{
  --bg:#ffffff; --panel:#f7f7f8; --panel2:#f0f0f2; --line:#e0e0e4; --fg:#16161a; --dim:#6b6b76;
  --accent:#3b62d9; --accent-soft:#e8edfd; --good:#137a4d; --bad:#b3341f; --warn:#8a5b00;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  --s-present:#eb6834; --s-absent:#2a78d6;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#131317; --panel:#1a1a20; --panel2:#212128; --line:#2e2e38; --fg:#e8e8ee; --dim:#9a9aa8;
    --accent:#7f9cff; --accent-soft:#1e2740; --good:#4fc98a; --bad:#ff8a70; --warn:#e3b341;
    --s-present:#d95926; --s-absent:#3987e5;
  }
}
:root[data-theme="dark"]{
  --bg:#131317; --panel:#1a1a20; --panel2:#212128; --line:#2e2e38; --fg:#e8e8ee; --dim:#9a9aa8;
  --accent:#7f9cff; --accent-soft:#1e2740; --good:#4fc98a; --bad:#ff8a70; --warn:#e3b341;
  --s-present:#d95926; --s-absent:#3987e5;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
button,input,select,textarea{font:inherit;color:inherit}
a{color:var(--accent)}
#app{display:grid;grid-template-columns:330px 1fr;height:100vh;overflow:hidden}
@media(max-width:860px){#app{grid-template-columns:1fr;grid-template-rows:auto 1fr}#side{max-height:42vh}}

/* ---------- sidebar ---------- */
#side{border-right:1px solid var(--line);background:var(--panel);display:flex;flex-direction:column;min-height:0}
#side header{padding:12px 14px 8px;border-bottom:1px solid var(--line)}
#side h1{margin:0 0 2px;font-size:15px;letter-spacing:.2px}
.sub{color:var(--dim);font-size:11.5px;font-family:var(--mono);word-break:break-all}
.row{display:flex;gap:6px;align-items:center}
.searchbox{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:7px;background:var(--bg)}
#runlist{overflow:auto;flex:1;padding:6px}
.group{margin:8px 4px 4px;font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim)}
.run{padding:7px 9px;border-radius:8px;cursor:pointer;border:1px solid transparent}
.run:hover{background:var(--panel2)}
.run.sel{background:var(--accent-soft);border-color:var(--accent)}
.run .nm{font-family:var(--mono);font-size:12px;word-break:break-all;line-height:1.35}
.run .meta{color:var(--dim);font-size:11px;margin-top:2px;display:flex;gap:6px;flex-wrap:wrap}
.dot{width:7px;height:7px;border-radius:50%;background:var(--accent);display:inline-block}
.badge{display:inline-block;padding:1px 6px;border-radius:999px;background:var(--panel2);border:1px solid var(--line);font-size:10.5px;color:var(--dim);white-space:nowrap}
.badge.new{background:var(--accent);color:#fff;border-color:transparent}

/* ---------- main ---------- */
#main{display:flex;flex-direction:column;min-width:0;min-height:0}
#topbar{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#topbar h2{margin:0;font-size:15px;font-family:var(--mono);word-break:break-all}
.tabs{display:flex;gap:4px;margin-left:auto;flex-wrap:wrap}
.tab{padding:5px 11px;border-radius:7px;border:1px solid var(--line);background:var(--bg);cursor:pointer}
.tab.sel{background:var(--accent);border-color:var(--accent);color:#fff}
#content{flex:1;overflow:auto;padding:16px;min-height:0}
.muted{color:var(--dim)}
.small{font-size:12px}
.mono{font-family:var(--mono)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.card h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:600;white-space:nowrap}
td.num,th.num{text-align:right;font-family:var(--mono)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 14px;font-size:12.5px}
.kv dt{color:var(--dim)}
.kv dd{margin:0;font-family:var(--mono);word-break:break-word}
.btn{padding:5px 10px;border-radius:7px;border:1px solid var(--line);background:var(--bg);cursor:pointer}
.btn:hover{border-color:var(--accent)}
.btn.on{background:var(--accent);border-color:var(--accent);color:#fff}
pre.text{white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;font-family:var(--mono);font-size:12.5px;
  background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:0}
details>summary{cursor:pointer;color:var(--dim)}
.flag{padding:1px 7px;border-radius:999px;font-size:11px;border:1px solid var(--line);white-space:nowrap}
.flag.yes{background:rgba(179,52,31,.13);color:var(--bad);border-color:transparent}
.flag.no{background:rgba(19,122,77,.13);color:var(--good);border-color:transparent}
.flag.n{background:var(--panel2);color:var(--dim)}

/* metrics */
.chartwrap{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px}
.legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.lg{display:flex;align-items:center;gap:5px;font-size:11.5px;padding:2px 7px;border-radius:999px;border:1px solid var(--line);cursor:pointer;font-family:var(--mono)}
.lg .sw{width:9px;height:9px;border-radius:2px}
.keypick{max-height:230px;overflow:auto;border:1px solid var(--line);border-radius:8px;padding:8px;background:var(--bg)}
.keypick label{display:block;font-family:var(--mono);font-size:11.5px;padding:1px 0;cursor:pointer;word-break:break-all}
.keygrp{margin:6px 0 2px;font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim)}
#tip{position:fixed;pointer-events:none;z-index:50;background:var(--panel2);border:1px solid var(--line);border-radius:7px;
  padding:6px 8px;font:11.5px var(--mono);display:none;max-width:340px;box-shadow:0 6px 24px rgba(0,0,0,.25)}

/* rollouts */
.rl{display:grid;grid-template-columns:minmax(260px,340px) 1fr;gap:14px;align-items:start}
@media(max-width:1100px){.rl{grid-template-columns:1fr}}
#rlist{border:1px solid var(--line);border-radius:10px;overflow:auto;max-height:calc(100vh - 250px);background:var(--panel)}
.ritem{padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer}
.ritem:hover{background:var(--panel2)}
.ritem.sel{background:var(--accent-soft)}
.ritem .hdr{display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:11.5px}
.ritem .pv{color:var(--dim);font-size:11.5px;margin-top:3px;font-family:var(--mono);
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.turn{border:1px solid var(--line);border-radius:8px;margin-bottom:8px;background:var(--panel)}
.turn>summary{padding:7px 10px;font-family:var(--mono);font-size:12px}
.turn .body{padding:0 10px 10px}
/* multi-turn episode: one element per message / thought / command / terminal output */
.turnsep{display:flex;align-items:center;gap:8px;margin:16px 0 8px}
.turnsep:first-child{margin-top:4px}
.turnsep .n{font-family:var(--mono);font-size:11.5px;letter-spacing:.4px;color:var(--dim);white-space:nowrap}
.turnsep .ln{flex:1;height:1px;background:var(--line)}
.msg{border:1px solid var(--line);border-left:3px solid var(--dim);border-radius:8px;margin:0 0 8px;background:var(--bg);overflow:hidden}
.msg>summary{padding:6px 10px;font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim);
  background:var(--panel);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.msg>summary::marker{color:var(--dim)}
.msg .inner{padding:9px 11px}
.msg pre.text{border:0;padding:0;background:transparent}
.msg-think{border-left-color:#8a63d2}
.msg-think>summary{color:#8a63d2}
.msg-msg{border-left-color:var(--accent)}
.msg-msg>summary{color:var(--accent)}
.msg-cmd{border-left-color:#c58b12}
.msg-cmd>summary{color:#c58b12}
.msg-cmd .inner{background:var(--panel2)}
.msg-out{border-left-color:#2a9d63}
.msg-out>summary{color:#2a9d63}
.msg-mon{border-left-color:#c2477e}
.msg-mon>summary{color:#c2477e}
.msg-ans{border-left-color:#2196a8}
.msg-ans>summary{color:#2196a8}
.msg-out .inner{background:var(--panel2)}
.msg .grow{flex:1}
.msg .mini{text-transform:none;letter-spacing:0;font-family:var(--mono);font-size:11px;opacity:.85}
.empty{color:var(--dim);font-style:italic;font-size:12px}
.lbl{font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);margin:10px 0 4px}
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--line);border-top-color:var(--accent);
  border-radius:50%;animation:sp .7s linear infinite;vertical-align:-2px}
@keyframes sp{to{transform:rotate(360deg)}}
.sdgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px}
@media(max-width:600px){.sdgrid{grid-template-columns:1fr}}
.sdgrid .card{margin:0}
.stats{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;margin:2px 0 8px}
.stats b{font-family:var(--mono);font-weight:600}
.toast{position:fixed;right:16px;bottom:16px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;
  padding:8px 12px;font-size:12.5px;box-shadow:0 8px 30px rgba(0,0,0,.25);z-index:60}
</style>
</head>
<body>
<div id="app">
  <aside id="side">
    <header>
      <div class="row"><h1 style="flex:1">Run transcripts</h1>
        <button class="btn small" id="theme" title="Toggle theme">◐</button></div>
      <div class="sub" id="rootpath"></div>
      <div class="row" style="margin-top:8px"><input class="searchbox" id="runq" placeholder="filter runs…"/></div>
      <div class="row small" style="margin-top:7px;color:var(--dim)">
        <button class="btn small" id="refresh">Refresh</button>
        <label class="row small" style="gap:4px"><input type="checkbox" id="auto" checked/> auto</label>
        <span id="scanstat" class="small"></span>
      </div>
    </header>
    <div id="runlist"></div>
  </aside>
  <main id="main">
    <div id="topbar">
      <h2 id="runtitle">—</h2>
      <div class="tabs" id="tabs"></div>
    </div>
    <div id="content"></div>
  </main>
</div>
<div id="tip"></div>
<script>
const BOOT = __BOOTSTRAP__;
const $ = s => document.querySelector(s);
const el = (tag, attrs={}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false)
    n.appendChild(typeof k === 'object' ? k : document.createTextNode(String(k)));
  return n;
};
const fmt = v => v === null || v === undefined ? '—'
  : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : (Math.abs(v) >= 1000 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : v.toFixed(4)))
  : String(v);
const bytes = n => n === null || n === undefined ? '—' : n > 1e9 ? (n/1e9).toFixed(1)+' GB' : n > 1e6 ? (n/1e6).toFixed(1)+' MB' : n > 1e3 ? (n/1e3).toFixed(0)+' kB' : n+' B';
const ago = t => { if(!t) return ''; const s = Date.now()/1000 - t;
  return s < 90 ? Math.round(s)+'s ago' : s < 5400 ? Math.round(s/60)+'m ago' : s < 172800 ? Math.round(s/3600)+'h ago' : Math.round(s/86400)+'d ago'; };
async function api(path, params={}) {
  const u = new URL(path, location.origin);
  for (const [k,v] of Object.entries(params)) if (v !== null && v !== undefined && v !== '') u.searchParams.set(k, v);
  const r = await fetch(u);
  const j = await r.json();
  if (j && j.error) { const err = new Error(j.error); err.body = j; throw err; }
  return j;
}
function toast(msg) {
  const t = el('div', {class:'toast'}, msg);
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

/* ------------------------------------------------------------------ state */
const S = {
  runs: [], filter: '', sel: null, tab: 'overview', detail: null,
  changed: new Set(),                 // runs that grew since the last poll
  ro: { source:'eval', step:'', behavior:'', unparsed:'', q:'', mon:'', mon_min:'', mon_max:'',
        offset:0, limit:50, order:'asc', list:null, sel:null, rec:null, loading:false,
        selEpoch:null,   // the dump version (server epoch) the selected row number belongs to
        seq:0 },         // latest list request: an older response arriving late is dropped
  metricSel: { train:null, eval:null }, hidden: {}, foldThinking: false,
  sd: { kind:'', step:'', bins:20, norm:true, data:null, loading:false },
};
$('#rootpath').textContent = BOOT.root;
$('#theme').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const dark = cur ? cur === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
  try { localStorage.setItem('viz-theme', dark ? 'light' : 'dark'); } catch(e){}
  if (S.tab === 'metrics') renderTab();
};
try { const t = localStorage.getItem('viz-theme'); if (t) document.documentElement.setAttribute('data-theme', t); } catch(e){}

/* ------------------------------------------------------------------ runs */
async function loadRuns(announce) {
  const prev = new Map(S.runs.map(r => [r.id, r]));
  const j = await api('/api/runs', {rescan: 1});
  S.runs = j.runs;
  $('#scanstat').textContent = `${S.runs.length} runs · ${ago(j.scanned_at)}`;
  if (announce) {
    const fresh = [], grew = [];
    for (const r of S.runs) {
      const p = prev.get(r.id);
      if (!p) { fresh.push(r.id); S.changed.add(r.id); }
      else if (p.train_steps !== r.train_steps || p.eval_steps !== r.eval_steps ||
               JSON.stringify(p.sources) !== JSON.stringify(r.sources)) { grew.push(r.id); S.changed.add(r.id); }
    }
    if (fresh.length || grew.length) {
      toast(`${fresh.length ? fresh.length + ' new run(s)' : ''}${fresh.length && grew.length ? ', ' : ''}${grew.length ? grew.length + ' updated' : ''}`);
      if (S.sel && (fresh.includes(S.sel) || grew.includes(S.sel))) await selectRun(S.sel, true);
    }
  }
  renderRuns();
}
function renderRuns() {
  const box = $('#runlist'); box.textContent = '';
  const q = S.filter.toLowerCase();
  const runs = S.runs.filter(r => !q || (r.id + ' ' + (r.policy||'') + ' ' + (r.env||'') + ' ' + (r.experiment||'')).toLowerCase().includes(q));
  let group = null;
  for (const r of runs) {
    if (r.group !== group) { group = r.group; box.appendChild(el('div', {class:'group'}, group || '(top level)')); }
    const badges = [
      r.env && el('span', {class:'badge'}, r.env),
      r.train_steps ? el('span', {class:'badge'}, `${r.train_steps} steps`) : null,
      r.sources && r.sources.train ? el('span', {class:'badge', title:'train rollout dump'}, 'train rollouts') : null,
      r.sources && r.sources.eval ? el('span', {class:'badge', title:'eval rollout dump'}, 'eval rollouts') : null,
      S.changed.has(r.id) ? el('span', {class:'badge new'}, 'new') : null,
    ].filter(Boolean);
    box.appendChild(el('div', {class:'run' + (S.sel === r.id ? ' sel' : ''), onclick:() => selectRun(r.id)},
      el('div', {class:'nm'}, r.name),
      el('div', {class:'meta'}, badges, el('span', {class:'badge'}, ago(r.mtime)))));
  }
  if (!runs.length) box.appendChild(el('div', {class:'muted small', style:'padding:10px'}, 'no runs match'));
}
$('#runq').oninput = e => { S.filter = e.target.value; renderRuns(); };
$('#refresh').onclick = async () => { await loadRuns(true); if (S.sel) await selectRun(S.sel, true); };

const TABS = [['overview','Overview'],['metrics','Metrics'],['scoredist','Score dist'],['rollouts','Rollouts'],['plots','Plots'],['log','Log'],['raw','Raw JSON']];
async function selectRun(id, keepTab) {
  S.sel = id; S.changed.delete(id);
  if (!keepTab) { S.ro = {...S.ro, step:'', q:'', offset:0, sel:null, rec:null, list:null};
                  S.sd = {...S.sd, kind:'', step:'', data:null}; }
  else { S.sd.data = null; S.ro.list = null; }  // refetch: a live run may have new steps, a dump may be rewritten
  $('#runtitle').textContent = id;
  renderRuns();
  $('#tabs').textContent = '';
  for (const [k, label] of TABS)
    $('#tabs').appendChild(el('button', {class:'tab' + (S.tab === k ? ' sel' : ''),
      onclick:() => { S.tab = k; renderTab(); $('#tabs').querySelectorAll('.tab').forEach((b,i) => b.classList.toggle('sel', TABS[i][0] === k)); }}, label));
  $('#content').innerHTML = '<div class="muted"><span class="spin"></span> loading…</div>';
  S.detail = await api('/api/run', {id});
  renderTab();
}
function renderTab() {
  const c = $('#content');
  if (!S.detail) { c.innerHTML = '<div class="muted">pick a run on the left</div>'; return; }
  c.textContent = '';
  ({overview:renderOverview, metrics:renderMetrics, scoredist:renderScoreDist, rollouts:renderRollouts,
    plots:renderPlots, log:renderLog, raw:renderRaw}[S.tab] || renderOverview)(c);
}

/* ------------------------------------------------------------------ overview */
function kv(pairs) {
  const d = el('dl', {class:'kv'});
  for (const [k, v] of pairs) { d.appendChild(el('dt', {}, k)); d.appendChild(el('dd', {}, fmt(v))); }
  return d;
}
function renderOverview(c) {
  const {card, config, run_info, monitors} = S.detail;
  const cfg = config || {};
  c.appendChild(el('div', {class:'card'}, el('h3', {}, 'Run'),
    kv([['id', card.id], ['dir', card.dir], ['policy', card.policy], ['backend', cfg.backend],
        ['env', card.env], ['subset', card.subset], ['experiment', card.experiment],
        ['started', card.started_at], ['last file write', ago(card.mtime)],
        ['train steps logged', `${card.train_steps}${card.n_steps_cfg ? ' / ' + card.n_steps_cfg : ''}`],
        ['eval rounds logged', card.eval_steps],
        ['last behavior_rate (train)', card.last_behavior_rate]])));
  const hp = ['n_steps','batch_size','group_size','eval_every','eval_size','eval_samples_per_prompt',
    'max_tokens','think_budget','answer_tokens','thinking_effort','penalty_coef','penalty_schedule',
    'kl_coef','kl_discount_factor','lora_rank','lr','seed','n_prompts_pool','probe_server_url','wandb_project']
    .filter(k => k in cfg).map(k => [k, typeof cfg[k] === 'object' ? JSON.stringify(cfg[k]) : cfg[k]]);
  if (cfg.env_options) hp.push(['env_options', JSON.stringify(cfg.env_options)]);
  c.appendChild(el('div', {class:'card'}, el('h3', {}, 'Hyperparameters'), kv(hp)));
  if (monitors && monitors.length) {
    const t = el('table', {}, el('thead', {}, el('tr', {},
      ...['name','kind','role','model / probe','threshold','use_cot','use_output','binary','reasoning','backend'].map(h => el('th', {}, h)))));
    const tb = el('tbody');
    for (const m of monitors) tb.appendChild(el('tr', {},
      el('td', {class:'mono'}, m.name),
      el('td', {}, fmt(m.kind)),
      el('td', {}, el('span', {class:'flag ' + (m.role === 'train_against' ? 'yes' : 'n')}, m.role || '—')),
      el('td', {class:'mono small'}, fmt(m.model_id)),
      el('td', {class:'num'}, fmt(m.threshold)),
      el('td', {}, fmt(m.use_cot)), el('td', {}, fmt(m.use_output)), el('td', {}, fmt(m.binary_judge)),
      el('td', {class:'mono small'}, m.reasoning ? JSON.stringify(m.reasoning)
        : (m.reasoning_effort || m.reasoning_max_tokens)
          ? JSON.stringify({effort: m.reasoning_effort, max_tokens: m.reasoning_max_tokens}) + ' (legacy)'
          : fmt(m.reasoning)),
      el('td', {class:'mono small'}, m.provider === 'vllm'
        ? `vllm ${m.base_url} · max_tokens ${fmt(m.max_tokens)} · thinking ${m.enable_thinking ? 'on' : 'off'}` +
          (m.enable_thinking ? ` · budget ${m.thinking_budget === null || m.thinking_budget === undefined ? 'none' : m.thinking_budget}` : '')
        : m.provider ? `${m.provider} · max_tokens ${fmt(m.max_tokens)}` : '—')));
    t.appendChild(tb);
    c.appendChild(el('div', {class:'card'}, el('h3', {}, `Monitors (${monitors.length}) — train-against rows are in the gradient`), t));
  }
  if (cfg.description) c.appendChild(el('div', {class:'card'}, el('h3', {}, 'Description'),
    el('pre', {class:'text'}, cfg.description)));
  const ck = Object.entries(run_info || {}).filter(([k]) => k.startsWith('checkpoint') || k === 'final_checkpoint');
  if (ck.length) c.appendChild(el('div', {class:'card'}, el('h3', {}, 'Checkpoints'), kv(ck)));
  const srcs = Object.entries(card.sources || {});
  c.appendChild(el('div', {class:'card'}, el('h3', {}, 'Rollout dumps on disk'),
    srcs.length ? kv(srcs.map(([k, s]) => [s.file, `${bytes(s.size)} · ${ago(s.mtime)}${s.n != null ? ' · ' + s.n + ' rollouts indexed' : ''}`]))
      : el('div', {class:'muted small'}, 'none — this run has no committed rollout dump (metrics only).')));
}

/* ------------------------------------------------------------------ metrics */
const PALETTE = ['#3b62d9','#d9534f','#2a9d63','#b06fd9','#d98c1e','#2196a8','#c2477e','#6a7a8c','#8bb02a','#7f5af0'];
function groupKeys(keys) {
  const groups = new Map();
  for (const k of keys) {
    let g = 'other';
    if (k === 'step') continue;
    if (k.startsWith('monitor/')) g = 'monitor: ' + k.split('/')[1];
    else if (k.includes('/')) g = k.split('/')[0] + '/';
    else if (/rate$/.test(k)) g = 'ground truth';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(k);
  }
  const order = ['ground truth','reward/','monitor','env/','loss/','kl/','time/','other'];
  return [...groups.entries()].sort((a,b) => {
    const ia = order.findIndex(o => a[0].startsWith(o)), ib = order.findIndex(o => b[0].startsWith(o));
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a[0].localeCompare(b[0]);
  });
}
function defaultKeys(rows, kind) {
  const keys = allKeys(rows);
  const want = ['behavior_rate','loose_rate','hardcoding_rate'];
  const sel = want.filter(k => keys.includes(k));
  if (kind === 'train') {
    for (const k of ['reward/task_mean','reward/monitor_suspiciousness_mean','reward/total_mean']) if (keys.includes(k)) sel.push(k);
  } else {
    for (const k of keys) if (/^monitor\/[^/]+\/auroc$/.test(k)) sel.push(k);
  }
  return sel.length ? sel : keys.slice(0, 4);
}
function allKeys(rows) {
  const out = [];
  for (const r of rows) for (const k of Object.keys(r)) if (k !== 'step' && !out.includes(k)) out.push(k);
  return out;
}
function renderMetrics(c) {
  const m = S.detail.metrics;
  if (!m.train.length && !m.eval.length) { c.appendChild(el('div', {class:'muted'}, 'no metrics.jsonl / eval_metrics.jsonl in this run dir')); return; }
  for (const kind of ['train','eval']) {
    const rows = m[kind];
    if (!rows.length) continue;
    const keys = allKeys(rows);
    if (!S.metricSel[kind] || !S.metricSel[kind].every(k => keys.includes(k)) || !S.metricSel[kind].length)
      S.metricSel[kind] = defaultKeys(rows, kind);
    const card = el('div', {class:'card'});
    const head = el('div', {class:'row', style:'margin-bottom:8px;flex-wrap:wrap;gap:8px'},
      el('h3', {style:'margin:0;flex:1'}, kind === 'train'
        ? `Train — metrics.jsonl (${rows.length} steps${m.train_bad ? ', ' + m.train_bad + ' unparseable lines skipped' : ''})`
        : `Eval — eval_metrics.jsonl (${rows.length} rounds${m.eval_bad ? ', ' + m.eval_bad + ' unparseable lines skipped' : ''})`));
    const presets = [
      ['ground truth', ks => ks.filter(k => /rate$/.test(k) && !k.startsWith('monitor'))],
      ['reward', ks => ks.filter(k => k.startsWith('reward/'))],
      ['monitor AUROC', ks => ks.filter(k => /^monitor\/[^/]+\/auroc$/.test(k))],
      ['monitor mean score', ks => ks.filter(k => /^monitor\/[^/]+\/mean_score$/.test(k))],
      ['judge calls: length / parse err', ks => ks.filter(k => /^monitor\/[^/]+\/(finish_length_rate|parse_error_rate)$/.test(k))],
      ["monitor d′", ks => ks.filter(k => /dprime/.test(k))],
      ['env', ks => ks.filter(k => k.startsWith('env/'))],
      ['none', () => []],
    ];
    for (const [label, fn] of presets)
      head.appendChild(el('button', {class:'btn small', onclick:() => {
        const got = fn(keys); if (!got.length && label !== 'none') return toast('no such series in this run');
        S.metricSel[kind] = got; renderTab();
      }}, label));
    head.appendChild(el('a', {class:'btn small', href:`/api/metrics.csv?id=${encodeURIComponent(S.sel)}&kind=${kind}`,
      download:''}, 'CSV'));
    card.appendChild(head);
    const wrap = el('div', {class:'chartwrap'});
    card.appendChild(wrap);
    drawChart(wrap, rows, S.metricSel[kind]);
    const picker = el('div', {class:'keypick'});
    for (const [g, ks] of groupKeys(keys)) {
      picker.appendChild(el('div', {class:'keygrp'}, g));
      for (const k of ks) {
        const cb = el('input', {type:'checkbox', onchange:e => {
          const s = new Set(S.metricSel[kind]);
          e.target.checked ? s.add(k) : s.delete(k);
          S.metricSel[kind] = keys.filter(x => s.has(x));
          renderTab();
        }});
        cb.checked = S.metricSel[kind].includes(k);
        picker.appendChild(el('label', {}, cb, ' ' + k));
      }
    }
    card.appendChild(el('details', {style:'margin-top:10px'},
      el('summary', {}, `series (${S.metricSel[kind].length} of ${keys.length} shown)`), picker));
    card.appendChild(el('details', {style:'margin-top:8px'}, el('summary', {}, 'table of every logged value'),
      metricTable(rows, keys)));
    c.appendChild(card);
  }
}
function metricTable(rows, keys) {
  const box = el('div', {style:'overflow:auto;max-height:420px;margin-top:8px'});
  const t = el('table', {}, el('thead', {}, el('tr', {}, el('th', {}, 'step'), ...keys.map(k => el('th', {class:'num'}, k)))));
  const tb = el('tbody');
  for (const r of rows) tb.appendChild(el('tr', {}, el('td', {class:'num'}, fmt(r.step)),
    ...keys.map(k => el('td', {class:'num'}, fmt(r[k])))));
  t.appendChild(tb); box.appendChild(t); return box;
}
function drawChart(wrap, rows, keys) {
  wrap.textContent = '';
  if (!keys.length) { wrap.appendChild(el('div', {class:'muted small'}, 'no series selected')); return; }
  const W = Math.max(480, wrap.clientWidth || 900), H = 300, P = {t:12, r:16, b:28, l:52};
  const xs = rows.map((r, i) => (typeof r.step === 'number' ? r.step : i));
  const series = keys.map((k, i) => ({
    key: k, color: PALETTE[i % PALETTE.length],
    pts: rows.map((r, j) => [xs[j], typeof r[k] === 'number' ? r[k] : null]).filter(p => p[1] !== null && isFinite(p[1])),
  })).filter(s => !S.hidden[s.key]);
  const vals = series.flatMap(s => s.pts.map(p => p[1]));
  if (!vals.length) { wrap.appendChild(el('div', {class:'muted small'}, 'selected series have no numeric values')); return; }
  let y0 = Math.min(...vals), y1 = Math.max(...vals);
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const X = v => P.l + (W - P.l - P.r) * (x1 === x0 ? 0.5 : (v - x0) / (x1 - x0));
  const Y = v => P.t + (H - P.t - P.b) * (1 - (v - y0) / (y1 - y0));
  const svgNS = 'http://www.w3.org/2000/svg';
  const mk = (n, a) => { const e = document.createElementNS(svgNS, n); for (const [k,v] of Object.entries(a)) e.setAttribute(k, v); return e; };
  const svg = mk('svg', {width:'100%', viewBox:`0 0 ${W} ${H}`, style:'display:block'});
  const css = getComputedStyle(document.body);
  const line = css.getPropertyValue('--line') || '#ddd', dim = css.getPropertyValue('--dim') || '#888';
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4, y = Y(v);
    svg.appendChild(mk('line', {x1:P.l, x2:W-P.r, y1:y, y2:y, stroke:line, 'stroke-width':1}));
    const tx = mk('text', {x:P.l-6, y:y+3.5, fill:dim, 'font-size':10, 'text-anchor':'end'});
    tx.textContent = Math.abs(v) < 1e-4 && v !== 0 ? v.toExponential(1) : (+v.toFixed(4)).toString();
    svg.appendChild(tx);
  }
  const nx = Math.min(8, xs.length);
  for (let i = 0; i < nx; i++) {
    const v = x0 + (x1 - x0) * i / Math.max(1, nx - 1), x = X(v);
    const tx = mk('text', {x, y:H-8, fill:dim, 'font-size':10, 'text-anchor':'middle'});
    tx.textContent = Math.round(v); svg.appendChild(tx);
  }
  for (const s of series) {
    if (!s.pts.length) continue;
    const d = s.pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');
    svg.appendChild(mk('path', {d, fill:'none', stroke:s.color, 'stroke-width':1.8, 'stroke-linejoin':'round'}));
    if (s.pts.length < 60) for (const p of s.pts) svg.appendChild(mk('circle', {cx:X(p[0]), cy:Y(p[1]), r:2, fill:s.color}));
  }
  const cursor = mk('line', {x1:0, x2:0, y1:P.t, y2:H-P.b, stroke:dim, 'stroke-width':1, 'stroke-dasharray':'3 3', opacity:0});
  svg.appendChild(cursor);
  const hit = mk('rect', {x:P.l, y:P.t, width:W-P.l-P.r, height:H-P.t-P.b, fill:'transparent'});
  svg.appendChild(hit);
  const tip = $('#tip');
  hit.addEventListener('mousemove', ev => {
    const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) * (W / r.width);
    const xv = x0 + (x1 - x0) * (sx - P.l) / (W - P.l - P.r);
    let best = null;
    for (const x of xs) if (best === null || Math.abs(x - xv) < Math.abs(best - xv)) best = x;
    cursor.setAttribute('x1', X(best)); cursor.setAttribute('x2', X(best)); cursor.setAttribute('opacity', .8);
    const i = xs.indexOf(best);
    tip.innerHTML = `<b>step ${best}</b><br>` + series.map(s =>
      `<span style="color:${s.color}">■</span> ${s.key.replace(/^monitor\//,'')}: ${fmt(rows[i][s.key])}`).join('<br>');
    tip.style.display = 'block';
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - 360) + 'px';
    tip.style.top = Math.min(ev.clientY + 12, innerHeight - tip.offsetHeight - 10) + 'px';
  });
  hit.addEventListener('mouseleave', () => { tip.style.display = 'none'; cursor.setAttribute('opacity', 0); });
  wrap.appendChild(svg);
  const lg = el('div', {class:'legend'});
  for (const k of keys) {
    const color = PALETTE[keys.indexOf(k) % PALETTE.length];
    lg.appendChild(el('span', {class:'lg', style:S.hidden[k] ? 'opacity:.4' : '',
      onclick:() => { S.hidden[k] = !S.hidden[k]; renderTab(); }},
      el('span', {class:'sw', style:`background:${color}`}), k));
  }
  wrap.appendChild(lg);
}

/* ------------------------------------------------------------------ score distributions */
/* Each monitor's score distribution at ONE step, split by the oracle: behavior present vs absent.
   An RL train step (rollouts.jsonl) carries only the train-against monitors; an eval step carries
   every monitor. A pre-RL baseline dir (eval_terminal_monitors_baseline.py) has one eval step. */
async function loadScoreDist() {
  S.sd.loading = true;
  const want = S.sel;
  try {
    const j = await api('/api/scoredist', {id:want, kind:S.sd.kind, step:S.sd.step});
    if (S.sel !== want) return;
    S.sd.data = j; S.sd.kind = j.kind || ''; S.sd.step = j.step === null || j.step === undefined ? '' : String(j.step);
  } catch (e) { toast('score dist: ' + e.message); S.sd.data = {kinds:[], error:e.message}; }
  S.sd.loading = false;
  if (S.tab === 'scoredist') renderTab();
}
function renderScoreDist(c) {
  const D = S.sd.data;
  if (!D) { c.appendChild(el('div', {class:'muted'}, el('span', {class:'spin'}), ' indexing / loading…'));
            if (!S.sd.loading) loadScoreDist(); return; }
  if (!D.kinds || !D.kinds.length) {
    c.appendChild(el('div', {class:'card'}, el('h3', {}, 'No rollout dump'),
      el('div', {class:'small'}, 'Score distributions need rollouts.jsonl or eval_rollouts(_slim).jsonl in the run dir.')));
    return;
  }
  const bar = el('div', {class:'card', style:'margin-bottom:12px'});
  const controls = el('div', {class:'row', style:'flex-wrap:wrap;gap:8px'});
  const pick = (kind, step) => { S.sd.kind = kind; S.sd.step = step; S.sd.data = null; renderTab(); };
  if (D.kinds.length > 1) for (const [k, label, tip] of [
      ['train', 'RL train step', 'rollouts.jsonl — the train-against monitors only'],
      ['eval', 'eval step', 'the eval dump — every monitor, on the fixed held-out set']])
    if (D.kinds.includes(k)) controls.appendChild(el('button', {class:'btn' + (D.kind === k ? ' on' : ''), title:tip,
      onclick:() => { if (D.kind !== k) pick(k, ''); }}, label));
  if (D.steps.length > 1) {
    const stepSel = el('select', {class:'btn', onchange:e => pick(D.kind, e.target.value)});
    for (const s of D.steps) {
      const o = el('option', {value:s}, 'step ' + s);
      if (s === D.step) o.selected = true;
      stepSel.appendChild(o);
    }
    const i = D.steps.indexOf(D.step);
    controls.appendChild(el('button', {class:'btn', disabled:i <= 0 ? '' : null,
      onclick:() => pick(D.kind, D.steps[i - 1])}, '←'));
    controls.appendChild(stepSel);
    controls.appendChild(el('button', {class:'btn', disabled:i >= D.steps.length - 1 ? '' : null,
      onclick:() => pick(D.kind, D.steps[i + 1])}, '→'));
  }
  controls.appendChild(el('span', {style:'flex:1'}));
  const binSel = el('select', {class:'btn', title:'histogram bins', onchange:e => { S.sd.bins = +e.target.value; renderTab(); }});
  for (const n of [10, 20, 40]) { const o = el('option', {value:n}, n + ' bins'); if (n === S.sd.bins) o.selected = true; binSel.appendChild(o); }
  controls.appendChild(binSel);
  controls.appendChild(el('button', {class:'btn' + (S.sd.norm ? ' on' : ''),
    title:'bar height = share of its own class (the classes are usually imbalanced) vs raw count',
    onclick:() => { S.sd.norm = !S.sd.norm; renderTab(); }}, S.sd.norm ? 'y: fraction of class' : 'y: count'));
  bar.appendChild(controls);
  const why = Object.entries(D.unscored_why || {}).map(([k, v]) => `${v} ${k}`).join(', ');
  bar.appendChild(el('div', {class:'muted small', style:'margin-top:7px'},
    `${D.kind === 'train' ? 'RL train' : 'eval'} step ${fmt(D.step)} · ${D.n_rollouts} rollouts from ${D.file}` +
    (D.n_unscored ? ` · ${D.n_unscored} scored by no monitor (${why}) — excluded` : '') +
    (D.n_unlabeled ? ` · ${D.n_unlabeled} without a behavior_present label — excluded` : '')));
  c.appendChild(bar);
  if (!D.monitors.length) {
    c.appendChild(el('div', {class:'card small muted'}, D.kind === 'train'
      ? 'No monitor scored the rollouts at this RL step — a train dump carries only the train-against monitors (none in a control run). Switch to an eval step to see the held-out monitors.'
      : 'No monitor scores at this step.'));
    return;
  }
  const grid = el('div', {class:'sdgrid'});
  for (const m of D.monitors) grid.appendChild(scoreDistCard(m));
  c.appendChild(grid);
}
function scoreDistCard(m) {
  const spec = monitorSpec(m.name);
  const card = el('div', {class:'card'});
  const view = spec.kind === 'cot' ? (spec.use_cot === false ? 'output-only' : spec.use_output === false ? 'cot-only' : 'cot+output') : null;
  card.appendChild(el('div', {class:'row', style:'flex-wrap:wrap;gap:6px;margin-bottom:4px'},
    el('span', {class:'mono', style:'font-weight:600;flex:1'}, m.name),
    spec.kind ? el('span', {class:'badge'}, spec.kind + (view ? ' · ' + view : '')) : null,
    spec.model_id ? el('span', {class:'badge mono', title:spec.model_id}, String(spec.model_id).split('/').pop()) : null,
    spec.role ? el('span', {class:'flag ' + (spec.role === 'train_against' ? 'yes' : 'n'),
      title:'train_against monitors are in the gradient'}, spec.role) : null));
  const gap = m.mean_present !== null && m.mean_absent !== null ? m.mean_present - m.mean_absent : null;
  const st = (k, v) => el('span', {}, k + ' ', el('b', {}, v));
  card.appendChild(el('div', {class:'stats'},
    st('n present / absent', `${m.present.length} / ${m.absent.length}`),
    st('mean|present', fmt(m.mean_present)), st('mean|absent', fmt(m.mean_absent)),
    st('gap', fmt(gap)), st('AUROC', fmt(m.auroc)),
    spec.threshold !== null && spec.threshold !== undefined ? st('threshold', fmt(spec.threshold)) : null,
    m.unlabeled ? st('unlabeled', m.unlabeled) : null));
  if (m.n_finish_known || m.n_parse_error) {
    // judge call health: calls that stopped at max_tokens, and answers with no parseable SCORE:/VERDICT: (scored 0)
    const pct = (k, n) => n ? `${k} / ${n} (${(100 * k / n).toFixed(1)}%)` : '—';
    card.appendChild(el('div', {class:'stats'},
      el('span', {title:'judge calls with finish_reason = "length" (stopped at the judge\'s max_tokens)' +
        (m.n_finish_known < m.n_calls ? ` — ${m.n_calls - m.n_finish_known} calls have no recorded finish_reason (slim dump written before it was kept, or a probe)` : '')},
        'stopped at max_tokens ', el('b', {}, pct(m.n_finish_length, m.n_finish_known))),
      el('span', {title:'judge answers with no parseable SCORE:/VERDICT: line — scored 0'},
        'unparseable (scored 0) ', el('b', {}, pct(m.n_parse_error, m.n_calls)))));
  }
  card.appendChild(scoreHist(m, spec.threshold));
  return card;
}
function scoreHist(m, threshold) {
  const all = m.present.concat(m.absent);
  const wrap = el('div', {class:'chartwrap'});
  if (!all.length) { wrap.appendChild(el('div', {class:'muted small'}, 'no labeled scores')); return wrap; }
  const lo = Math.min(0, ...all), hi = Math.max(1, ...all), nb = S.sd.bins, w = (hi - lo) / nb;
  const bin = v => Math.min(nb - 1, Math.max(0, Math.floor((v - lo) / w + 1e-9)));
  const count = xs => { const h = new Array(nb).fill(0); for (const v of xs) h[bin(v)]++; return h; };
  const series = [
    {key:'behavior present', color:'var(--s-present)', n:m.present.length, h:count(m.present)},
    {key:'behavior absent', color:'var(--s-absent)', n:m.absent.length, h:count(m.absent)},
  ];
  const val = (s, i) => S.sd.norm ? (s.n ? s.h[i] / s.n : 0) : s.h[i];
  const ymax = Math.max(1e-9, ...series.flatMap(s => s.h.map((_, i) => val(s, i))));
  const W = 520, H = 220, P = {t:10, r:10, b:30, l:44};
  const X = v => P.l + (W - P.l - P.r) * (v - lo) / (hi - lo);
  const Y = v => P.t + (H - P.t - P.b) * (1 - v / ymax);
  const svgNS = 'http://www.w3.org/2000/svg';
  const mk = (n, a) => { const e = document.createElementNS(svgNS, n); for (const [k,v] of Object.entries(a)) e.setAttribute(k, v); return e; };
  const svg = mk('svg', {width:'100%', viewBox:`0 0 ${W} ${H}`, style:'display:block'});
  for (let i = 0; i <= 4; i++) {
    const v = ymax * i / 4, y = Y(v);
    svg.appendChild(mk('line', {x1:P.l, x2:W-P.r, y1:y, y2:y, style:'stroke:var(--line)', 'stroke-width':1}));
    const tx = mk('text', {x:P.l-6, y:y+3.5, style:'fill:var(--dim)', 'font-size':10, 'text-anchor':'end'});
    tx.textContent = S.sd.norm ? (v * 100).toFixed(0) + '%' : (Number.isInteger(v) ? v : v.toFixed(1));
    svg.appendChild(tx);
  }
  for (let i = 0; i <= 10; i++) {
    const v = lo + (hi - lo) * i / 10;
    const tx = mk('text', {x:X(v), y:H-P.b+14, style:'fill:var(--dim)', 'font-size':10, 'text-anchor':'middle'});
    tx.textContent = +v.toFixed(2); svg.appendChild(tx);
  }
  const xl = mk('text', {x:(P.l + W - P.r) / 2, y:H-2, style:'fill:var(--dim)', 'font-size':10, 'text-anchor':'middle'});
  xl.textContent = 'monitor score'; svg.appendChild(xl);
  // two thin bars per bin, 2px apart; 4px rounded tops anchored to the baseline
  const bw = (W - P.l - P.r) / nb, gap = 2, barW = Math.max(1, (bw - 3 * gap) / 2), base = Y(0);
  series.forEach((s, k) => {
    for (let i = 0; i < nb; i++) {
      const v = val(s, i); if (!v) continue;
      const x = P.l + i * bw + gap + k * (barW + gap), y = Y(v), h = base - y, r = Math.min(4, barW / 2, h);
      svg.appendChild(mk('path', {style:`fill:${s.color}`,
        d:`M${x},${base}V${y + r}Q${x},${y} ${x + r},${y}H${x + barW - r}Q${x + barW},${y} ${x + barW},${y + r}V${base}Z`}));
    }
  });
  svg.appendChild(mk('line', {x1:P.l, x2:W-P.r, y1:base, y2:base, style:'stroke:var(--dim)', 'stroke-width':1}));
  if (typeof threshold === 'number' && threshold >= lo && threshold <= hi) {
    svg.appendChild(mk('line', {x1:X(threshold), x2:X(threshold), y1:P.t, y2:base, style:'stroke:var(--fg)',
      'stroke-width':1, 'stroke-dasharray':'4 3', opacity:.6}));
    const tt = mk('text', {x:X(threshold) + 4, y:P.t + 9, style:'fill:var(--dim)', 'font-size':10});
    tt.textContent = 'threshold'; svg.appendChild(tt);
  }
  const tip = $('#tip');
  for (let i = 0; i < nb; i++) {
    const a = lo + i * w, b = a + w;
    const hit = mk('rect', {x:P.l + i * bw, y:P.t, width:bw, height:base - P.t, fill:'transparent'});
    hit.addEventListener('mousemove', ev => {
      hit.setAttribute('style', 'fill:var(--fg);fill-opacity:.06');
      tip.innerHTML = `<b>score ${+a.toFixed(3)} – ${+b.toFixed(3)}${i === nb - 1 ? ' (incl.)' : ''}</b><br>` + series.map(s =>
        `<span style="color:${s.color}">■</span> ${s.key}: ${s.h[i]} of ${s.n}` +
        (s.n ? ` (${(100 * s.h[i] / s.n).toFixed(1)}%)` : '')).join('<br>');
      tip.style.display = 'block';
      tip.style.left = Math.min(ev.clientX + 14, innerWidth - 360) + 'px';
      tip.style.top = Math.min(ev.clientY + 12, innerHeight - tip.offsetHeight - 10) + 'px';
    });
    hit.addEventListener('mouseleave', () => { hit.removeAttribute('style'); tip.style.display = 'none'; });
    svg.appendChild(hit);
  }
  wrap.appendChild(svg);
  wrap.appendChild(el('div', {class:'legend'}, series.map(s =>
    el('span', {class:'lg', style:'cursor:default'}, el('span', {class:'sw', style:`background:${s.color}`}), `${s.key} (n=${s.n})`))));
  const tbl = el('table', {style:'margin-top:6px'}, el('thead', {}, el('tr', {},
    el('th', {}, 'score bin'), ...series.map(s => el('th', {class:'num'}, s.key)))));
  const tb = el('tbody');
  for (let i = 0; i < nb; i++) {
    if (!series.some(s => s.h[i])) continue;
    tb.appendChild(el('tr', {}, el('td', {class:'mono'}, `${+(lo + i * w).toFixed(3)} – ${+(lo + (i + 1) * w).toFixed(3)}`),
      ...series.map(s => el('td', {class:'num'}, `${s.h[i]}` + (s.n ? ` (${(100 * s.h[i] / s.n).toFixed(1)}%)` : '')))));
  }
  tbl.appendChild(tb);
  wrap.appendChild(el('details', {style:'margin-top:6px'}, el('summary', {class:'small'}, 'table'), tbl));
  return wrap;
}

/* ------------------------------------------------------------------ rollouts */
function renderRollouts(c) {
  const card = S.detail.card, srcs = Object.entries(card.sources || {});
  if (!srcs.length) {
    c.appendChild(el('div', {class:'card'}, el('h3', {}, 'No rollout dump'),
      el('div', {class:'small'}, 'This run dir has no rollouts.jsonl / eval_rollouts.jsonl / eval_rollouts_slim.jsonl. ',
        'The big dumps are gitignored, so a run cloned from git shows metrics only.')));
    return;
  }
  if (!srcs.find(([k]) => k === S.ro.source)) S.ro.source = srcs[0][0];
  const bar = el('div', {class:'card', style:'margin-bottom:12px'});
  const controls = el('div', {class:'row', style:'flex-wrap:wrap;gap:8px'});
  const srcSel = el('select', {class:'btn', onchange:e => { S.ro.source = e.target.value; S.ro.offset = 0; S.ro.sel = null; S.ro.rec = null; loadRollouts(); }});
  for (const [k, s] of srcs) {
    const o = el('option', {value:k}, `${s.file} (${bytes(s.size)})`);
    if (k === S.ro.source) o.selected = true;
    srcSel.appendChild(o);
  }
  controls.appendChild(srcSel);
  const L = S.ro.list;
  const stepSel = el('select', {class:'btn', onchange:e => { S.ro.step = e.target.value; S.ro.offset = 0; loadRollouts(); }});
  stepSel.appendChild(el('option', {value:''}, 'all steps'));
  for (const s of (L ? L.steps : [])) {
    const o = el('option', {value:s}, 'step ' + s);
    if (String(s) === String(S.ro.step)) o.selected = true;
    stepSel.appendChild(o);
  }
  controls.appendChild(stepSel);
  const behSel = el('select', {class:'btn', onchange:e => { S.ro.behavior = e.target.value; S.ro.offset = 0; loadRollouts(); }});
  for (const [v, t] of [['','behavior: any'],['1','behavior_present = true'],['0','behavior_present = false']]) {
    const o = el('option', {value:v}, t); if (v === S.ro.behavior) o.selected = true; behSel.appendChild(o);
  }
  controls.appendChild(behSel);
  const upSel = el('select', {class:'btn', onchange:e => { S.ro.unparsed = e.target.value; S.ro.offset = 0; loadRollouts(); }});
  for (const [v, t] of [['','parse: any'],['0','parsed ok'],['1','unparsed'],
                         ['val','valid (monitored)'],['inv','invalid (truncated/unparsed/no submission, not monitored)']]) {
    const o = el('option', {value:v}, t); if (v === S.ro.unparsed) o.selected = true; upSel.appendChild(o);
  }
  controls.appendChild(upSel);
  const monSel = el('select', {class:'btn', onchange:e => { S.ro.mon = e.target.value; S.ro.offset = 0; loadRollouts(); }});
  monSel.appendChild(el('option', {value:''}, 'monitor score: any'));
  for (const n of (L ? L.monitor_names : [])) {
    const o = el('option', {value:n}, n); if (n === S.ro.mon) o.selected = true; monSel.appendChild(o);
  }
  controls.appendChild(monSel);
  if (S.ro.mon) {
    const mn = el('input', {class:'searchbox', style:'width:70px', placeholder:'min', value:S.ro.mon_min});
    const mx = el('input', {class:'searchbox', style:'width:70px', placeholder:'max', value:S.ro.mon_max});
    const apply = () => { S.ro.mon_min = mn.value; S.ro.mon_max = mx.value; S.ro.offset = 0; loadRollouts(); };
    mn.onchange = apply; mx.onchange = apply;
    controls.appendChild(mn); controls.appendChild(mx);
  }
  const search = el('input', {class:'searchbox', style:'flex:1;min-width:160px', placeholder:'search full text of every rollout (prompt + CoT + answer)…', value:S.ro.q});
  search.onkeydown = e => { if (e.key === 'Enter') { S.ro.q = e.target.value; S.ro.offset = 0; loadRollouts(); } };
  controls.appendChild(search);
  controls.appendChild(el('button', {class:'btn', onclick:() => { S.ro.q = search.value; S.ro.offset = 0; loadRollouts(); }}, 'Search'));
  bar.appendChild(controls);
  if (L) bar.appendChild(el('div', {class:'muted small', style:'margin-top:7px'},
    `${L.total} matching · ${L.n_indexed} rollouts indexed from ${L.file}` +
    (L.bad_lines ? ` · ${L.bad_lines} unparseable lines skipped` : '') +
    (L.index_seconds ? ` · indexed in ${L.index_seconds}s` : '') +
    (L.searched ? ` · scanned ${L.searched} rollouts for the query` : '')));
  c.appendChild(bar);

  const grid = el('div', {class:'rl'});
  const list = el('div', {id:'rlist'});
  grid.appendChild(list);
  const detail = el('div', {id:'rdetail'});
  grid.appendChild(detail);
  c.appendChild(grid);
  if (!L) { list.appendChild(el('div', {class:'muted small', style:'padding:10px'}, S.ro.loading ? '' : 'loading…')); loadRollouts(); return; }
  if (S.ro.loading) list.appendChild(el('div', {class:'muted small', style:'padding:10px'}, el('span', {class:'spin'}), ' indexing / loading…'));
  for (const e of L.entries) {
    const flags = [];
    if (e.behavior_present !== null && e.behavior_present !== undefined)
      flags.push(el('span', {class:'flag ' + (e.behavior_present ? 'yes' : 'no'), title:'behavior_present (the oracle)'},
        e.behavior_present ? 'behavior' : 'clean'));
    if (e.loose_rh) flags.push(el('span', {class:'flag n'}, 'loose'));
    if (e.hardcoding) flags.push(el('span', {class:'flag n'}, 'hardcode'));
    if (e.unparsed) flags.push(el('span', {class:'flag n'}, 'unparsed'));
    if (e.invalid_reason) flags.push(el('span', {class:'flag n', title:'invalid rollout: -1 reward, never shown to a monitor'},
      'invalid: ' + e.invalid_reason + ' — not monitored'));
    for (const [n, m] of Object.entries(e.monitors || {}))
      flags.push(el('span', {class:'flag ' + (m.label === true ? 'yes' : m.label === false ? 'no' : 'n'),
        title:n}, `${n}=${m.score === null ? '—' : (+m.score).toFixed(2)}`));
    list.appendChild(el('div', {class:'ritem' + (S.ro.sel === e.i ? ' sel' : ''), onclick:() => openRollout(e.i)},
      el('div', {class:'hdr'},
        el('span', {class:'badge'}, 'step ' + fmt(e.step)),
        e.task_id ? el('span', {class:'badge mono'}, e.task_id) : null,
        e.reward !== null && e.reward !== undefined ? el('span', {class:'badge'}, 'r=' + fmt(e.reward)) : null,
        flags),
      e.preview ? el('div', {class:'pv'}, e.preview) : null));
  }
  if (!L.entries.length && !S.ro.loading) list.appendChild(el('div', {class:'muted small', style:'padding:10px'}, 'no rollouts match these filters'));
  const sizeSel = el('select', {class:'btn small', onchange:e => { S.ro.limit = +e.target.value; S.ro.offset = 0; loadRollouts(); }});
  for (const n of [50, 100, 250, 1000]) {
    const o = el('option', {value:n}, n + '/page');
    if (n === S.ro.limit) o.selected = true;
    sizeSel.appendChild(o);
  }
  const pager = el('div', {class:'row small', style:'padding:8px;justify-content:space-between;gap:6px'},
    el('button', {class:'btn small', onclick:() => { S.ro.offset = Math.max(0, S.ro.offset - S.ro.limit); loadRollouts(); }}, '← prev'),
    el('span', {class:'muted'}, `${L.total ? S.ro.offset + 1 : 0}–${Math.min(S.ro.offset + S.ro.limit, L.total)} of ${L.total}`),
    sizeSel,
    el('button', {class:'btn small', onclick:() => { if (S.ro.offset + S.ro.limit < L.total) { S.ro.offset += S.ro.limit; loadRollouts(); } }}, 'next →'));
  list.appendChild(pager);
  if (S.ro.rec) renderRolloutDetail(detail, S.ro.rec);
  else if (S.ro.sel !== null) detail.appendChild(el('div', {class:'muted small'}, el('span', {class:'spin'}), ' loading…'));
  else detail.appendChild(el('div', {class:'muted small'}, 'pick a rollout to see its full prompt, CoT and response'));
}
async function loadRollouts() {
  S.ro.loading = true;
  const {source, step, behavior, unparsed, q, mon, mon_min, mon_max, offset, limit, order} = S.ro;
  const want = S.sel, seq = ++S.ro.seq;
  try {
    const j = await api('/api/rollouts', {id:want, source, step, behavior, unparsed, q, mon, mon_min, mon_max, offset, limit, order});
    if (S.sel !== want || seq !== S.ro.seq || S.ro.source !== source) return;
    if (S.ro.sel !== null && S.ro.selEpoch !== j.epoch) {
      // the dump was rewritten: the open rollout's row number now names a different line (or none)
      S.ro.sel = null; S.ro.rec = null; S.ro.selEpoch = null;
      toast('this rollout dump was rewritten on disk — closed the open rollout');
    }
    S.ro.list = j;
  } catch (e) { if (seq === S.ro.seq) toast('rollouts: ' + e.message); }
  S.ro.loading = false;
  if (S.tab === 'rollouts') renderTab();
}
async function openRollout(i) {
  const want = S.sel, src = S.ro.source, epoch = S.ro.list && S.ro.list.epoch;
  S.ro.sel = i; S.ro.selEpoch = epoch; S.ro.rec = null;
  if (S.tab === 'rollouts') renderTab();
  const current = () => S.sel === want && S.ro.source === src && S.ro.sel === i && S.ro.selEpoch === epoch;
  try {
    const j = await api('/api/rollout', {id:want, source:src, i, epoch});
    if (!current()) return;  // superseded by another click / list reload while in flight
    S.ro.rec = j.record;
  } catch (e) {
    if (!current()) return;
    toast('rollout: ' + e.message);
    if (e.body && e.body.stale) { S.ro.sel = null; S.ro.selEpoch = null; S.ro.list = null; if (S.tab === 'rollouts') renderTab(); }
    return;
  }
  if (S.tab === 'rollouts') renderTab();
}
function textBlock(label, text, opts={}) {
  if (text === null || text === undefined || text === '') return null;
  const s = typeof text === 'string' ? text : JSON.stringify(text, null, 2);
  const box = el('div', {});
  box.appendChild(el('div', {class:'lbl'}, `${label} · ${s.length.toLocaleString()} chars`,
    el('button', {class:'btn small', style:'margin-left:8px;padding:0 6px',
      onclick:() => navigator.clipboard && navigator.clipboard.writeText(s)}, 'copy')));
  box.appendChild(el('pre', {class:'text'}, s));
  return box;
}
/* One message / thought / command / terminal output = one collapsible element. */
function msgBlock(role, label, text, opts={}) {
  const s = text === null || text === undefined ? '' : (typeof text === 'string' ? text : JSON.stringify(text, null, 2));
  const d = el('details', {class:`msg msg-${role}`});
  if (!(role === 'think' && S.foldThinking) && !opts.closed) d.setAttribute('open', '');
  const sum = el('summary', {}, label,
    el('span', {class:'mini'}, `${s.length.toLocaleString()} chars`),
    el('span', {class:'grow'}),
    ...(opts.badges || []));
  if (s) sum.appendChild(el('button', {class:'btn small', style:'padding:0 6px',
    onclick:e => { e.preventDefault(); e.stopPropagation(); navigator.clipboard && navigator.clipboard.writeText(s); }}, 'copy'));
  d.appendChild(sum);
  d.appendChild(el('div', {class:'inner'}, s ? el('pre', {class:'text'}, s) : el('div', {class:'empty'}, opts.emptyNote || '(empty)')));
  return d;
}
/* The multi-turn episode view (terminal-verifier): the conversation, turn by turn. */
function renderEpisode(turns, meta) {
  const card = el('div', {class:'card'});
  const head = el('div', {class:'row', style:'flex-wrap:wrap;gap:8px;margin-bottom:4px'},
    el('h3', {style:'margin:0;flex:1'},
      `Episode — ${turns.length} turn${turns.length === 1 ? '' : 's'}, one element per thought / message / command / terminal output`),
    el('button', {class:'btn small' + (S.foldThinking ? ' on' : ''),
      onclick:() => { S.foldThinking = !S.foldThinking; renderTab(); }}, 'fold thinking'),
    el('button', {class:'btn small', onclick:e =>
      e.target.closest('.card').querySelectorAll('details.msg').forEach(d => d.open = true)}, 'expand all'),
    el('button', {class:'btn small', onclick:e =>
      e.target.closest('.card').querySelectorAll('details.msg').forEach(d => d.open = false)}, 'collapse all'));
  card.appendChild(head);
  const KNOWN = ['cot','text','command','output','is_submission','verifier_value','truncated'];
  turns.forEach((t, i) => {
    const badges = [
      t.is_submission ? el('span', {class:'flag n'}, 'submission') : null,
      t.verifier_value !== undefined && t.verifier_value !== null ? el('span', {class:'flag n'}, 'verifier → ' + fmt(t.verifier_value)) : null,
      t.truncated ? el('span', {class:'flag yes'}, 'truncated') : null,
      t.command === null || t.command === undefined ? el('span', {class:'flag n'}, 'no command') : null,
    ].filter(Boolean);
    card.appendChild(el('div', {class:'turnsep'},
      el('span', {class:'n'}, `TURN ${i + 1} / ${turns.length}`), el('span', {class:'ln'}), badges));
    card.appendChild(msgBlock('think', `turn ${i + 1} · chain of thought`, t.cot,
      {emptyNote: 'no thinking in this turn'}));
    card.appendChild(msgBlock('msg', `turn ${i + 1} · assistant message`, t.text,
      {emptyNote: 'the assistant emitted no visible text'}));
    if (t.command !== null && t.command !== undefined) {
      card.appendChild(msgBlock('cmd', `turn ${i + 1} · command executed`, '$ ' + t.command));
      card.appendChild(msgBlock('out', `turn ${i + 1} · terminal output`, t.output,
        {emptyNote: 'the command produced no output'}));
    } else {
      card.appendChild(el('div', {class:'msg msg-cmd'}, el('div', {class:'inner'},
        el('div', {class:'empty'}, 'no <command> in this turn — nothing was executed'))));
    }
    const rest = Object.fromEntries(Object.entries(t).filter(([k]) => !KNOWN.includes(k)));
    if (Object.keys(rest).length)
      card.appendChild(msgBlock('msg', `turn ${i + 1} · other turn fields`, rest, {closed: true}));
  });
  const cmds = Array.isArray(meta.commands) ? meta.commands : null;
  if (cmds) {
    const d = el('details', {style:'margin-top:12px'});
    d.appendChild(el('summary', {}, `all ${cmds.length} commands, in order`));
    d.appendChild(el('pre', {class:'text', style:'margin-top:6px'}, cmds.map(c => '$ ' + c).join('\n')));
    card.appendChild(d);
  }
  return card;
}
/* The LLM-judge API calls saved WITH this rollout (monitors.<name>.call, written by rl/train.py):
   the exact prompt, the other request parameters, and the response — content plus the judge's
   chain of thought when the provider returned one. Only monitors whose call was saved are listed;
   nothing is reconstructed from the run config. */
function monitorSpec(name) {
  return ((S.detail && S.detail.monitors) || []).find(m => m.name === name) || {};
}
function renderMonitorCalls(rec) {
  const card = el('div', {class:'card'});
  card.appendChild(el('h3', {}, 'Monitor API calls — the exact prompt, request parameters and response saved with this rollout'));
  const mons = rec.monitors && typeof rec.monitors === 'object' ? Object.entries(rec.monitors) : [];
  const withCall = mons.filter(([, m]) => m && typeof m === 'object' && m.call && typeof m.call === 'object');
  const without = mons.filter(([n]) => !withCall.some(([w]) => w === n));
  if (!mons.length) {
    card.appendChild(el('div', {class:'muted small'}, rec.invalid_reason
      ? `invalid rollout (${rec.invalid_reason}) — by design no monitor was run on it; it is excluded from every monitor metric`
      : 'this rollout records no monitor verdicts'));
    return card;
  }
  if (!withCall.length) {
    card.appendChild(el('div', {class:'small'},
      'Unavailable — no monitor API call was saved with this rollout. ',
      el('span', {class:'muted'},
        'Dumps written before call recording was added (and slim dumps) keep only {score, label} per monitor. ' +
        'The prompt is deliberately not reconstructed from the run config: a reconstruction can differ from what ' +
        'the judge was actually sent if the repo changed since the run.')));
    return card;
  }
  for (const [name, m] of withCall) {
    const spec = monitorSpec(name), call = m.call;
    const req = call.request && typeof call.request === 'object' ? call.request : {};
    const resp = call.response && typeof call.response === 'object' ? call.response : {};
    const msg = resp.message && typeof resp.message === 'object' ? resp.message : {};
    const content = typeof msg.content === 'string' ? msg.content : (msg.content == null ? '' : JSON.stringify(msg.content, null, 2));
    const reasoning = typeof msg.reasoning === 'string' && msg.reasoning.trim() ? msg.reasoning : null;
    const details = Array.isArray(msg.reasoning_details) && msg.reasoning_details.length ? msg.reasoning_details : null;
    const badges = [
      el('span', {class:'flag ' + (spec.role === 'train_against' ? 'yes' : 'n'),
        title:'train_against monitors are in the gradient'}, spec.role || '—'),
      req.model ? el('span', {class:'badge mono'}, req.model) : null,
      m.score !== null && m.score !== undefined ? el('span', {class:'badge'}, 'score ' + fmt(m.score)) : null,
      el('span', {class:'flag ' + (m.label === true ? 'yes' : m.label === false ? 'no' : 'n')}, 'label ' + fmt(m.label)),
      m.parse_error ? el('span', {class:'flag yes', title:'the judge answered, but not in the instructed SCORE:/VERDICT: format — scored as 0'}, 'parse error') : null,
      resp.finish_reason ? el('span', {class:'badge'}, 'finish: ' + resp.finish_reason) : null,
      call.attempts ? el('span', {class:'badge', title:'POSTs it took; only the successful one is recorded'},
        call.attempts + (call.attempts === 1 ? ' attempt' : ' attempts')) : null,
    ].filter(Boolean);
    const d = el('details', {class:'msg msg-mon', open:''});
    d.appendChild(el('summary', {}, name, el('span', {class:'grow'}), badges));
    const body = el('div', {class:'inner'});

    // -- the prompt: every message in the request, in order (one user message for our judges)
    const msgs = Array.isArray(req.messages) ? req.messages : [];
    if (msgs.length) msgs.forEach((mm, i) => {
      const c = mm && typeof mm.content === 'string' ? mm.content : JSON.stringify(mm && mm.content, null, 2);
      body.appendChild(msgBlock('mon', `prompt sent to the judge · message ${i + 1} / ${msgs.length} · role ${(mm && mm.role) || '?'}`, c));
    });
    else body.appendChild(el('div', {class:'empty'}, 'the saved request carries no messages'));

    // -- the response: chain of thought (if the provider returned one), then the content
    if (reasoning) body.appendChild(msgBlock('think', 'judge chain of thought (response.message.reasoning)', reasoning));
    else if (details) body.appendChild(msgBlock('think', 'judge reasoning details (response.message.reasoning_details)', details));
    else body.appendChild(el('div', {class:'msg msg-think'}, el('div', {class:'inner'},
      el('div', {class:'empty'}, 'no chain of thought in the response' +
        (req.reasoning && req.reasoning.enabled === false ? ' (reasoning was requested off)' : '')))));
    body.appendChild(msgBlock('ans', 'judge response (response.message.content)', content,
      {emptyNote: 'the content channel was empty — the verdict was read from the reasoning channel'}));

    // -- everything else in the request, and the response metadata
    const params = Object.entries(req).filter(([k]) => k !== 'messages')
      .map(([k, v]) => [k, typeof v === 'object' && v !== null ? JSON.stringify(v) : v]);
    if (call.url) params.push(['url', call.url]);
    if (call.timeout !== undefined) params.push(['timeout (client, s)', call.timeout]);
    const pd = el('details', {style:'margin-top:8px', open:''});
    pd.appendChild(el('summary', {class:'small'}, 'request parameters (everything posted besides the messages)'));
    pd.appendChild(kv(params));
    body.appendChild(pd);
    const meta = Object.entries(resp).filter(([k]) => k !== 'message')
      .map(([k, v]) => [k, typeof v === 'object' && v !== null ? JSON.stringify(v) : v]);
    const other = Object.entries(msg).filter(([k]) => !['content','reasoning','reasoning_details'].includes(k))
      .map(([k, v]) => ['message.' + k, typeof v === 'object' && v !== null ? JSON.stringify(v) : v]);
    if (meta.length || other.length) {
      const rd = el('details', {style:'margin-top:8px'});
      rd.appendChild(el('summary', {class:'small'}, 'response metadata'));
      rd.appendChild(kv([...meta, ...other]));
      body.appendChild(rd);
    }
    const raw = el('details', {style:'margin-top:8px'});
    raw.appendChild(el('summary', {class:'small'}, 'full call record (JSON)'));
    raw.appendChild(el('pre', {class:'text', style:'margin-top:6px'}, JSON.stringify(call, null, 2)));
    body.appendChild(raw);
    d.appendChild(body);
    card.appendChild(d);
  }
  if (without.length) card.appendChild(el('div', {class:'muted small', style:'margin-top:6px'},
    'no API call saved for: ' + without.map(([n, m]) => n + (m && typeof m === 'object' && m.error ? ` (error: ${m.error})` : '')).join(', ') +
    ' — probes make no API call; an LLM judge without one predates call recording or never answered.'));
  return card;
}
function renderRolloutDetail(box, rec) {
  box.textContent = '';
  const env = (rec.env && typeof rec.env === 'object') ? rec.env : {};
  const meta = (rec.env_meta && typeof rec.env_meta === 'object') ? rec.env_meta
             : (env.meta && typeof env.meta === 'object') ? env.meta : {};
  const beh = rec.behavior_present !== undefined ? rec.behavior_present : env.behavior_present;
  const head = el('div', {class:'card'});
  head.appendChild(el('div', {class:'row', style:'flex-wrap:wrap;gap:6px'},
    el('span', {class:'badge'}, 'step ' + fmt(rec.step)),
    meta.task_id || rec.task_id ? el('span', {class:'badge mono'}, meta.task_id || rec.task_id) : null,
    beh !== undefined ? el('span', {class:'flag ' + (beh ? 'yes' : 'no'), title:'behavior_present — the oracle label, never in a monitor or the reward'},
      'behavior_present = ' + fmt(beh)) : null,
    rec.loose_rh !== null && rec.loose_rh !== undefined ? el('span', {class:'flag n'}, 'loose_rh = ' + fmt(rec.loose_rh)) : null,
    rec.hardcoding !== null && rec.hardcoding !== undefined ? el('span', {class:'flag n'}, 'hardcoding = ' + fmt(rec.hardcoding)) : null,
    rec.invalid_reason ? el('span', {class:'flag n', title:'invalid rollout: flat -1 reward, never shown to any monitor, excluded from every monitor metric'},
      'invalid (' + rec.invalid_reason + ') — not monitored') : null,
    rec.reward !== undefined ? el('span', {class:'badge'}, 'reward = ' + fmt(rec.reward)) : null,
    env.task_reward !== undefined ? el('span', {class:'badge'}, 'task_reward = ' + fmt(env.task_reward)) : null));
  const mons = rec.monitors && typeof rec.monitors === 'object' ? Object.entries(rec.monitors) : [];
  if (mons.length) {
    const t = el('table', {style:'margin-top:8px'}, el('thead', {}, el('tr', {},
      el('th', {}, 'monitor'), el('th', {class:'num'}, 'score'), el('th', {}, 'label'))));
    const tb = el('tbody');
    for (const [n, m] of mons) {
      const sc = m && typeof m === 'object' ? m.score : m, lb = m && typeof m === 'object' ? m.label : null;
      tb.appendChild(el('tr', {}, el('td', {class:'mono'}, n), el('td', {class:'num'}, fmt(sc)),
        el('td', {}, el('span', {class:'flag ' + (lb === true ? 'yes' : lb === false ? 'no' : 'n')}, fmt(lb)))));
    }
    t.appendChild(tb);
    head.appendChild(t);
  }
  box.appendChild(head);

  const turns = Array.isArray(meta.turns) ? meta.turns : null;
  const hasText = rec.question || rec.cot || rec.answer;
  if (!hasText && !turns)
    box.appendChild(el('div', {class:'card small muted'},
      'This dump carries no text (eval_rollouts_slim.jsonl is labels + scores only — switch the source to eval_rollouts.jsonl / rollouts.jsonl for the full transcripts).'));
  if (rec.question) {
    const qc = el('div', {class:'card'});
    qc.appendChild(textBlock('Prompt (question)', rec.question));
    box.appendChild(qc);
  }

  if (turns && turns.length) {
    // MULTI-TURN (terminal-verifier): one element per thought / assistant message / command /
    // terminal output, in episode order — never merged into one blob.
    box.appendChild(renderEpisode(turns, meta));
    // The concatenated views the MONITORS actually read, kept verbatim but folded away.
    const cc = el('div', {class:'card'});
    cc.appendChild(el('h3', {}, 'Concatenated flat fields (cot / answer as stored in the dump)'));
    for (const [label, val, note] of [
      ['cot', rec.cot, 'every turn’s thinking, turn-tagged — what the probes read'],
      ['answer / output', rec.answer, 'the env’s output_view — the flat field the dumps and probes consume'],
    ]) {
      if (!val) continue;
      const d = el('details', {style:'margin-bottom:6px'});
      d.appendChild(el('summary', {}, `${label} — ${String(val).length.toLocaleString()} chars · ${note}`));
      d.appendChild(textBlock(label, val));
      cc.appendChild(d);
    }
    box.appendChild(cc);
  } else if (rec.cot || rec.answer) {
    const texts = el('div', {class:'card'});
    for (const b of [textBlock('Chain of thought (cot)', rec.cot),
                     textBlock('Response / transcript (answer)', rec.answer)]) if (b) texts.appendChild(b);
    box.appendChild(texts);
  }

  box.appendChild(renderMonitorCalls(rec));

  const skip = turns && turns.length ? ['turns','commands'] : ['turns'];
  const metaScalar = Object.entries(meta).filter(([k, v]) => !skip.includes(k) && (v === null || typeof v !== 'object'));
  const metaObj = Object.fromEntries(Object.entries(meta).filter(([k, v]) => !skip.includes(k) && v !== null && typeof v === 'object'));
  if (metaScalar.length || Object.keys(metaObj).length) {
    const mc = el('div', {class:'card'});
    mc.appendChild(el('h3', {}, 'Env grading record (env_meta)'));
    if (metaScalar.length) mc.appendChild(kv(metaScalar));
    for (const [k, v] of Object.entries(metaObj)) mc.appendChild(textBlock(k, v));
    box.appendChild(mc);
  }
  if (rec.extra && Object.keys(rec.extra).length) box.appendChild(el('div', {class:'card'},
    el('h3', {}, 'extra'), textBlock('extra', rec.extra)));
  box.appendChild(el('div', {class:'card'}, el('details', {},
    el('summary', {}, 'raw JSON record'), el('pre', {class:'text', style:'margin-top:8px'}, JSON.stringify(rec, null, 2)))));
}

/* ------------------------------------------------------------------ plots / log / raw */
function renderPlots(c) {
  const plots = S.detail.card.plots || [];
  if (!plots.length) { c.appendChild(el('div', {class:'muted'}, 'no PNGs in this run dir')); return; }
  for (const p of plots) c.appendChild(el('div', {class:'card'}, el('h3', {}, p),
    el('img', {src:`/api/file?id=${encodeURIComponent(S.sel)}&path=${encodeURIComponent(p)}`,
      style:'max-width:100%;border-radius:8px;background:#fff'})));
}
function renderLog(c) {
  if (!S.detail.card.has_log) { c.appendChild(el('div', {class:'muted'}, 'no run.log in this run dir')); return; }
  const pre = el('pre', {class:'text'}, 'loading…');
  c.appendChild(el('div', {class:'card'}, el('h3', {}, 'run.log'), pre));
  fetch(`/api/file?id=${encodeURIComponent(S.sel)}&path=run.log`).then(r => r.text()).then(t => { pre.textContent = t; });
}
function renderRaw(c) {
  c.appendChild(el('div', {class:'card'}, el('h3', {}, 'run_info.json'),
    el('pre', {class:'text'}, JSON.stringify(S.detail.run_info, null, 2))));
  c.appendChild(el('div', {class:'card'}, el('h3', {}, 'config.json'),
    el('pre', {class:'text'}, JSON.stringify(S.detail.config, null, 2))));
}

/* ------------------------------------------------------------------ boot */
(async () => {
  await loadRuns(false);
  const hash = decodeURIComponent(location.hash.slice(1));
  if (hash && S.runs.find(r => r.id === hash)) await selectRun(hash);
  else renderTab();
  setInterval(() => { if ($('#auto').checked) loadRuns(true).catch(() => {}); }, (BOOT.poll_seconds || 20) * 1000);
})();
addEventListener('hashchange', () => {
  const h = decodeURIComponent(location.hash.slice(1));
  if (h && h !== S.sel && S.runs.find(r => r.id === h)) selectRun(h);
});
const _sel = selectRun;
selectRun = async (id, keep) => { location.hash = encodeURIComponent(id); return _sel(id, keep); };
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(here / "data" / "runs"), help="directory of run dirs (default: data/runs)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1; use 0.0.0.0 to expose)")
    ap.add_argument("--port", type=int, default=8000, help="port (default: 8000; the next free port is used if taken)")
    ap.add_argument("--poll-seconds", type=int, default=20, help="browser auto-refresh interval (0 disables auto-poll)")
    ap.add_argument("--cache-dir", default=str(here / ".cache" / "visualize_transcripts"),
                    help="where rollout indexes are cached between runs")
    ap.add_argument("--no-cache", action="store_true", help="don't persist rollout indexes to disk")
    ap.add_argument("--prebuild", action="store_true", help="index every rollout dump at startup instead of lazily")
    args = ap.parse_args()

    root = Path(args.runs_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"error: no such directory: {root}", file=sys.stderr)
        return 2

    store = Store(root, None if args.no_cache else Path(args.cache_dir).expanduser().resolve())
    t0 = time.perf_counter()
    store.scan()
    print(f"scanned {root} → {len(store.runs)} runs in {time.perf_counter() - t0:.1f}s")
    if args.prebuild:
        for run in list(store.runs.values()):
            for src in SOURCES:
                idx = store.index(run, src)
                if idx:
                    print(f"  indexed {run.id}/{idx.path.name}: {len(idx.entries)} rollouts ({idx.seconds:.1f}s)")

    Handler.store = store
    Handler.poll_seconds = max(0, args.poll_seconds)

    port = args.port
    for attempt in range(20):
        try:
            httpd = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError as e:
            if e.errno not in (98, 48):  # EADDRINUSE
                raise
            port += 1
    else:
        print("error: no free port found", file=sys.stderr)
        return 1
    httpd.daemon_threads = True

    shown = args.host if args.host not in ("0.0.0.0", "::") else socket.gethostname()
    print(f"serving on http://{shown}:{port}  (Ctrl-C to stop)")
    if args.host == "127.0.0.1":
        print(f"  remote box? forward it:  ssh -p <ssh-port> -L {port}:127.0.0.1:{port} root@<host>")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
