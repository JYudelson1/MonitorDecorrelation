#!/usr/bin/env python3
"""Add held-out monitors to FINISHED runs by re-scoring every saved eval rollout — e.g. the weak judge,
once we know which one, on a batch that ran without it.

Every VALID eval rollout of each run's full ``eval_rollouts.jsonl`` (the same set the live monitors
scored: invalid ones were shown to none) is scored by each new monitor, and per-step metrics with exactly
the columns a live held-out monitor gets (``rl.train.eval_monitor_metrics``) are written to a sidecar
``eval_metrics_posthoc_<name>.jsonl`` that the analysis readers merge in (``eval/posthoc.py``) — so the
degradation matrix, plots and coupling see the new monitor as one more held-out column. The run's own
logs are never rewritten.

    # what would be scored, no judge calls
    uv run python experiments/add_posthoc_monitors.py 'data/runs/mbpp_q3_8b_qmonitors_*' \
        --config experiments/configs/posthoc/weak_judges.json --dry-run
    # score (resumable: re-run the same command to continue after an interruption)
    uv run python experiments/add_posthoc_monitors.py 'data/runs/mbpp_q3_8b_qmonitors_*' \
        --config experiments/configs/posthoc/weak_judges.json --workers 128

``--config``: any JSON/YAML with a ``monitors`` list (a run config works; ``--only`` picks names). Every
picked monitor must be ``role: held_out`` and an LLM judge (``kind: cot``) — probes are scored by
``eval_probes_on_run.py``. A name the run already has (live, or post-hoc with a different spec) is
refused. Single-turn envs only (the terminal env's judges read the per-turn episode, which the eval dump
does not hand back as a ``Rollout``).

Integrity: a step is written only if the dump holds exactly that step's rollouts — its valid count must
equal the live monitors' ``n_scored`` and its behavior rate the logged one (a corrupt/spliced dump line
or a duplicated step block would otherwise shift every metric). Skipped steps are reported. A judge call
that still fails after the monitor's own retries is not cached; that run's sidecar is not written until a
re-run scores it.

vLLM judge calls go at ``POSTHOC_PRIORITY`` (behind live runs' train AND eval calls), so this can share a
judge server with a running batch.
"""

from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from monitordecorrelation.eval.posthoc import posthoc_metrics_path
from monitordecorrelation.eval.rollout_dump import monitor_record
from monitordecorrelation.experiment_config import build_monitors, load_monitor_specs
from monitordecorrelation.monitors.judge_backend import judge_priority
from monitordecorrelation.monitors.vllm import EVAL_PRIORITY
from monitordecorrelation.types import MonitorResult, Prompt, Rollout

load_dotenv(".env")

POSTHOC_PRIORITY = EVAL_PRIORITY + 1
_RESULT_META = ("call", "parse_error", "error")


def _iter_jsonl(path: Path):
    """(line_no, record) for every parseable line; the count of corrupt ones is returned via ``bad``."""
    bad = [0]

    def gen():
        with path.open(errors="replace") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    yield i, json.loads(line)
                except json.JSONDecodeError:
                    bad[0] += 1

    return gen(), bad


def _key(r: dict) -> str:
    """Content key of a dumped rollout — a cached score is reused only for the exact same text."""
    h = hashlib.sha1()
    for part in (r.get("question"), r.get("cot"), r.get("answer")):
        h.update((part or "").encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _is_valid(r: dict) -> bool:
    """Was this eval rollout shown to the live monitors? Exactly those get a post-hoc score too."""
    if "invalid_reason" in r:
        return r["invalid_reason"] is None
    return bool(r.get("monitors"))  # older dumps: an unmonitored (invalid) rollout has monitors == {}


def _to_rollout(r: dict) -> Rollout:
    return Rollout(prompt=Prompt(text=r["question"]), cot=r.get("cot") or "", output=r.get("answer") or "")


def _to_result(rec: dict) -> MonitorResult:
    return MonitorResult(score=rec["score"], label=rec["label"],
                         meta={k: rec[k] for k in _RESULT_META if k in rec})


class Run:
    """One run dir: its valid eval rollouts grouped by step, the live eval rows, and its monitor names."""

    def __init__(self, d: Path):
        self.dir = d
        info = json.loads((d / "run_info.json").read_text())
        env = info.get("env") or {}
        self.env_name = env.get("name")
        self.behavior = env.get("behavior_name")
        if not self.behavior:
            raise ValueError(f"{d}: run_info.json has no env.behavior_name")
        rows_it, rows_bad = _iter_jsonl(d / "eval_metrics.jsonl")
        self.eval_rows = {r["step"]: r for _, r in rows_it}
        if rows_bad[0]:
            print(f"  ⚠️  {d.name}: {rows_bad[0]} corrupt eval_metrics.jsonl line(s) skipped")
        self.live_monitors = sorted({k.split("/")[1] for r in self.eval_rows.values() for k in r
                                     if k.startswith("monitor/")})
        recs_it, recs_bad = _iter_jsonl(d / "eval_rollouts.jsonl")
        self.by_step: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for i, r in recs_it:
            self.by_step[r["step"]].append((i, r))
        self.corrupt = recs_bad[0]
        self.good_steps, self.bad_steps = self._check_steps()

    def _check_steps(self) -> tuple[list[int], dict[int, str]]:
        good, bad = [], {}
        for step in sorted(set(self.by_step) | set(self.eval_rows)):
            recs = [r for _, r in self.by_step.get(step, [])]
            row = self.eval_rows.get(step)
            if row is None:
                bad[step] = "no eval_metrics row"
                continue
            if not recs:
                bad[step] = "no rollouts in the dump"
                continue
            n_valid = sum(_is_valid(r) for r in recs)
            n_live = {row[f"monitor/{m}/n_scored"] for m in self.live_monitors if f"monitor/{m}/n_scored" in row}
            if n_live and n_live != {n_valid}:
                bad[step] = f"dump has {n_valid} valid rollouts, live monitors scored {sorted(n_live)}"
                continue
            rate = mean(float(bool(r.get("behavior_present"))) for r in recs)
            if abs(rate - row["behavior_rate"]) > 1e-9:
                bad[step] = f"dump behavior rate {rate:.4f} != logged {row['behavior_rate']:.4f}"
                continue
            good.append(step)
        return good, bad

    def valid_items(self):
        for step in self.good_steps:
            for i, r in self.by_step[step]:
                if _is_valid(r):
                    yield i, step, r


def _cache_path(run: Run, name: str) -> Path:
    return run.dir / "posthoc" / f"{name}.scores.jsonl"


def _load_cache(run: Run, name: str) -> dict[int, tuple[str, dict]]:
    p = _cache_path(run, name)
    if not p.exists():
        return {}
    it, _ = _iter_jsonl(p)
    return {c["line"]: (c["key"], c["record"]) for _, c in it}


def _check_name(run: Run, spec) -> None:
    """Refuse a name the run already has live, or post-hoc under a different spec."""
    if spec.name in run.live_monitors:
        raise SystemExit(f"{run.dir.name}: monitor {spec.name!r} already exists in its eval_metrics.jsonl — "
                         "pick another name")
    info = run.dir / "posthoc" / f"{spec.name}.info.json"
    if info.exists():
        old = json.loads(info.read_text())["spec"]
        if old != spec.model_dump():
            raise SystemExit(f"{run.dir.name}: post-hoc monitor {spec.name!r} exists with a different spec "
                             f"({info}) — pick another name, or delete its posthoc files to redo it")


def score_runs(runs: list[Run], monitors: list, workers: int) -> bool:
    todo = []
    caches: dict[tuple[str, str], dict] = {}
    for run in runs:
        for m in monitors:
            cache = caches[(str(run.dir), m.name)] = _load_cache(run, m.name)
            for i, step, r in run.valid_items():
                hit = cache.get(i)
                if hit is not None and hit[0] != _key(r):
                    raise SystemExit(f"{_cache_path(run, m.name)}: line {i} was cached for different text — "
                                     "the dump changed under the cache; delete the cache to redo it")
                if hit is None:
                    todo.append((run, m, i, step, r))
    print(f"{len(todo)} judge calls to make ({workers} workers)")
    if not todo:
        return True

    locks: dict[Path, threading.Lock] = defaultdict(threading.Lock)
    handles: dict[Path, object] = {}
    n_fail = [0]
    count_lock = threading.Lock()

    def work(item) -> None:
        run, m, i, step, r = item
        try:
            with judge_priority(POSTHOC_PRIORITY):
                res = m.score(_to_rollout(r))
        except Exception as e:  # noqa: BLE001
            res = MonitorResult(score=float("nan"), label=False, meta={"error": str(e)[:300]})
        if res.score != res.score:  # NaN = the judge never answered; not data, not cached
            with count_lock:
                n_fail[0] += 1
                if n_fail[0] <= 3:
                    print(f"  {m.name} on {run.dir.name} line {i}: {(res.meta or {}).get('error', 'NaN score')}")
            return
        p = _cache_path(run, m.name)
        line = json.dumps({"line": i, "step": step, "key": _key(r), "record": monitor_record(res)}, default=str)
        with locks[p]:
            if p not in handles:
                p.parent.mkdir(exist_ok=True)
                handles[p] = p.open("a")
            handles[p].write(line + "\n")
            handles[p].flush()

    # One sequential warm-up call per judge (its first item, cached like any other), so a judge that
    # fails every call fails once, not as a storm.
    for m in monitors:
        first = next((t for t in todo if t[1] is m), None)
        if first is not None:
            todo.remove(first)
            work(first)
            if n_fail[0]:
                raise SystemExit(f"warm-up call of {m.name} failed — is its server up? ({first[0].dir.name})")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, _ in enumerate(ex.map(work, todo), 1):
            if k % 200 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} scored ({n_fail[0]} failed) — {time.perf_counter() - t0:.0f}s", flush=True)
    for h in handles.values():
        h.close()
    if n_fail[0]:
        print(f"⚠️  {n_fail[0]} judge calls failed after retries — re-run to score them")
    return n_fail[0] == 0


def write_metrics(run: Run, m, spec, config_path: str) -> bool:
    """The sidecar rows for one (run, monitor) from its complete cache; False if the cache is incomplete."""
    from monitordecorrelation.rl.train import eval_monitor_metrics

    cache = _load_cache(run, m.name)
    rows = []
    for step in run.good_steps:
        recs = run.by_step[step]
        results = []
        for i, r in recs:
            if not _is_valid(r):
                results.append(None)
            elif i in cache:
                results.append(_to_result(cache[i][1]))
            else:
                print(f"  {run.dir.name}/{m.name}: step {step} not fully scored yet — sidecar not written")
                return False
        gt = [bool(r.get("behavior_present")) for _, r in recs]
        gt_loose = ([bool(r.get("loose_rh")) for _, r in recs]
                    if any(r.get("loose_rh") is not None for _, r in recs) else None)
        rows.append({"step": step, **eval_monitor_metrics(m, results, gt, gt_loose, run.behavior)})
    out = posthoc_metrics_path(run.dir, m.name)
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp.replace(out)
    (run.dir / "posthoc").mkdir(exist_ok=True)
    (run.dir / "posthoc" / f"{m.name}.info.json").write_text(json.dumps({
        "spec": spec.model_dump(), "config": config_path,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": run.good_steps, "skipped_steps": {str(k): v for k, v in run.bad_steps.items()},
        "n_scored": sum(1 for _ in run.valid_items()),
    }, indent=2) + "\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs or globs (each with eval_rollouts.jsonl + eval_metrics.jsonl)")
    ap.add_argument("--config", required=True, help="a JSON/YAML whose `monitors` list holds the new monitor(s)")
    ap.add_argument("--only", default=None, help="comma-separated monitor names to take from --config (default all)")
    ap.add_argument("--workers", type=int, default=64, help="concurrent judge calls")
    ap.add_argument("--dry-run", action="store_true", help="report what would be scored; no judge calls")
    args = ap.parse_args()

    specs = load_monitor_specs(args.config)
    if args.only:
        want = args.only.split(",")
        missing = set(want) - {s.name for s in specs}
        if missing:
            ap.error(f"--only: {sorted(missing)} not in {args.config}")
        specs = [s for s in specs if s.name in want]
    for s in specs:
        if s.kind != "cot":
            ap.error(f"{s.name}: only LLM judges (kind: cot) — score probes with eval_probes_on_run.py")
        if s.role != "held_out":
            ap.error(f"{s.name}: role must be held_out — a post-hoc monitor never touched the gradient")
    if not specs:
        ap.error("no monitors selected")

    dirs = sorted({Path(p) for pat in args.runs for p in globmod.glob(pat)})
    dirs = [d for d in dirs if (d / "eval_rollouts.jsonl").exists()]
    if not dirs:
        print(f"no run dir with a full eval_rollouts.jsonl matched {args.runs}")
        return 1
    runs = []
    for d in dirs:
        run = Run(d)
        if run.env_name == "terminal_verifier":
            raise SystemExit(f"{d}: multi-turn (terminal) runs are not supported — see the module doc")
        for s in specs:
            _check_name(run, s)
        n_valid = sum(1 for _ in run.valid_items())
        print(f"{d.name}: env={run.env_name} behavior={run.behavior} live={run.live_monitors} | "
              f"{len(run.good_steps)} steps, {n_valid} valid rollouts"
              + (f" | {run.corrupt} corrupt dump lines" if run.corrupt else "")
              + (f" | SKIPPING steps {run.bad_steps}" if run.bad_steps else ""))
        runs.append(run)
    if len({r.behavior for r in runs}) > 1:
        raise SystemExit(f"runs mix behaviors {sorted({r.behavior for r in runs})} — score them separately")
    behavior = runs[0].behavior
    print(f"{len(specs)} monitor(s) from {args.config}: " + ", ".join(
        f"{s.name}({s.provider} {s.model_id}, cot={s.use_cot}, out={s.use_output})" for s in specs))
    if args.dry_run:
        n = sum(1 for r in runs for _ in r.valid_items())
        done = sum(len(_load_cache(r, s.name)) for r in runs for s in specs)
        print(f"dry run: {n * len(specs)} (rollout, monitor) pairs, {done} already cached")
        return 0

    _, monitors = build_monitors(specs, default_behavior=behavior)
    ok = score_runs(runs, monitors, args.workers)
    by_name = {s.name: s for s in specs}
    written = 0
    for run in runs:
        for m in monitors:
            written += write_metrics(run, m, by_name[m.name], args.config)
    print(f"wrote {written}/{len(runs) * len(monitors)} sidecars")
    return 0 if ok and written == len(runs) * len(monitors) else 1


if __name__ == "__main__":
    sys.exit(main())
