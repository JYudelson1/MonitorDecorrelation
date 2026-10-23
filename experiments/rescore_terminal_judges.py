#!/usr/bin/env python3
"""Re-score saved TerminalEnv eval episodes with a run's vLLM judges, exactly as the live RL loop scored them.

A run saves every eval episode in full (``eval_rollouts.jsonl``: the task, each turn's thinking, message, command,
output and submission) with each judge's live verdict under ``monitors.<judge>``: ``score`` (its ``SCORE:`` / 100),
``label`` and the whole ``call`` (the request sent, the reply). This sends the same episodes to judge servers again —
the run's own, or others given by ``--judge-url`` — and sets what they say beside what the run's servers said.

    # compare: a seeded sample per (run, eval step), N hacks + M non-hacks, each new score paired with the live one
    uv run python experiments/rescore_terminal_judges.py compare RUN [RUN ...] --out-dir DIR --pos 4 --neg 4 \
        --judge-url q3=http://localhost:8001/v1 --judge-url q35=http://localhost:8002/v1
    # full: every episode the judges were shown, and per-step metrics in the live format (eval/degradation.py)
    uv run python experiments/rescore_terminal_judges.py full RUN [RUN ...] --out-dir DIR
    # either one with --dry-run: the plan and the checks below on the real files, without a single judge call

**As live:**
- **Judges:** the run's own monitor specs (``run_info.json`` ``config.monitors``) through ``build_monitors(...,
  multi_turn=True)``, i.e. ``AgentCoTMonitor`` with the run's views (``use_cot`` / ``use_output``) and request settings
  (model, max_tokens, enable_thinking, thinking_budget, thinking_effort). What that builds must equal the resolved
  settings the run recorded (``train_against`` / ``held_out``), the endpoint aside. Only ``provider: vllm`` judges: the
  probes are experiments/eval_inkling_probes_on_runs.py's.
- **Episodes:** rebuilt as the loop held them: the task as the prompt, the flat ``cot`` / ``answer`` views, and
  ``env_meta`` (the env's record of the episode, with its turns) as ``meta["episode"]``. Each call goes through the
  loop's own ``rl.train._score_at`` at ``EVAL_PRIORITY``, so the body sent is the recorded one, byte for byte. Every
  rebuilt request is checked against the recorded one before anything is sent; a difference stops the run unless
  ``--allow-request-drift``. ``request_sha256`` (its hash) identifies what a judge was shown.
- **Inclusion:** only the episodes the loop showed its monitors. ``envs.base.invalid_reason`` is re-derived from the
  record, with the env's own predicates: truncated, unparsed, never submitted. It must agree with the recorded
  ``invalid_reason``, and a valid episode must carry every judge's live score (an invalid one none).
- **Endpoints:** each judge's ``base_url`` from the run, unless ``--judge-url KEY=URL``. KEY is a judge name
  (``q3_cot``) or its family, the name before its last ``_`` (``q3`` = ``q3_cot`` + ``q3_out``); a KEY that matches no
  judge is an error. The servers are checked (``vllm.check_server``) before the first call.

**Compare mode** (``--pos`` / ``--neg`` per (run, eval step), ``--seed``, ``--steps``):
- **Sample:** per eval step, up to ``--pos`` episodes with ``behavior_present`` and ``--neg`` without, among those
  shown to the judges. Each class is shuffled in line order by ``Random(f"{seed}/{run}/{step}/{pos|neg}")`` and its
  first ones are taken, so a larger count extends a sample instead of redrawing it, and a run's sample doesn't depend
  on the others. Where a step has fewer, all are taken (the plan says where).
- **Report** per judge, pooled over the runs, and per (run, judge): ``mean_diff`` (new − live), its ``sd_diff`` and
  ``se_diff``, ``pearson`` / ``spearman`` of new vs live, ``auroc_live`` / ``auroc_new`` against ``behavior_present``,
  the share of scores exactly 1.0 / 0.0 (``at_1_*`` / ``at_0_*``), and parse errors and thinking-budget hits. Printed
  and written to ``DIR/compare_report.json``.

**Full mode:** every episode shown to the judges, then ``DIR/<run>/eval_metrics.jsonl``: per eval step, the run's own
row with every ``monitor/<judge>/…`` field recomputed from the new verdicts, by the loop's code (``judge_step_fields``)
— auroc, accuracy, brier, mean_score, the class means, dprime_margin, n_pos / n_neg / n_scored, finish_length_rate,
parse_error_rate, thinking_budget_rate. Only re-scored judges' fields are kept; ``time/*`` is dropped; a step with a
verdict missing is left out. ``run_info.json`` is copied next to it, so ``eval/degradation.py``
(``build_degradation_matrix([DIR/<run>, ...])``) reads the folder like a run's. Post-hoc probes can join it:
``eval_inkling_probes_on_runs.py --out-dir DIR`` writes ``DIR/<run>/probe_eval_<name>.jsonl``. The new-vs-live
summary of compare mode, over every episode, goes to ``DIR/full_report.json``.

**Resumable:** each verdict is appended to ``DIR/<run>/judge_scores.jsonl`` as its call returns: the episode's line,
step, task, the judge, ``request_sha256``, ``endpoint``, and the loop's own per-monitor record (``monitor_record``:
score, label, the whole call, parse_error, finish_reason, hit_thinking_budget). A verdict is reused only for the same
line, judge, request hash and endpoint. A rewritten line (``--resume`` cuts ``eval_rollouts.jsonl`` back and appends)
is therefore scored again; a verdict from another server never stands in. A call that still fails after the backend's
own retries (a config error: 400, a server not splitting reasoning) stops the run; what returned is kept.
``DIR/<run>/rescore_info.json`` records the source file's size and sha256, the judges and endpoints, and the code
version.

**Dry run** (``--dry-run``): nothing is sent or written. Per run it reports the episodes and why the rest were not
shown, whether every rebuilt request equals the recorded one, whether every live score is what this code reads from
the recorded reply, and whether the per-step judge fields recomputed from the live verdicts equal the run's
``eval_metrics.jsonl``. Then the plan: the sample per step, and the calls left to make. Exit status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import glob as globmod
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

SCORES = "judge_scores.jsonl"  # per run: one row per (episode, judge) verdict, appended as each call returns
INFO = "rescore_info.json"  # per run: what was re-scored, against which endpoints, by which code
METRICS = "eval_metrics.jsonl"  # per run, full mode: the per-step judge fields, in the live file's format
REPORT = "{mode}_report.json"  # across the runs: new vs live per judge


def log(msg: str) -> None:
    print(f"[rescore {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- judges


def judge_family(name: str) -> str:
    """``q3_cot`` → ``q3``: what a run's views of one judge model share (the name before its last ``_``)."""
    return name.rsplit("_", 1)[0]


def selected(name: str, keys: set[str] | None) -> str | None:
    """The key of ``keys`` that picks judge ``name`` (its name before its family), None if none; every judge if
    ``keys`` is None."""
    if keys is None:
        return name
    return name if name in keys else judge_family(name) if judge_family(name) in keys else None


def parse_judge_urls(items: list[str]) -> dict[str, str]:
    """``--judge-url KEY=URL`` items → {KEY: URL}."""
    out: dict[str, str] = {}
    for item in items:
        key, eq, url = item.partition("=")
        if not eq or not key or not url:
            raise SystemExit(f"--judge-url {item!r}: expected KEY=URL, e.g. q3=http://localhost:8001/v1")
        if key in out:
            raise SystemExit(f"--judge-url: {key!r} is given twice")
        out[key] = url.rstrip("/")
    return out


def run_judge_specs(info: dict, run: str) -> tuple[list, list[str]]:
    """(the run's vLLM judge specs, the names of its other monitors), from ``run_info.json``'s ``config.monitors``: the
    effective config the run was launched with (``run_experiment.py`` writes it there)."""
    from pydantic import TypeAdapter

    from monitordecorrelation.experiment_config import MonitorSpec

    env = (info.get("env") or {}).get("name")
    if env != "terminal_verifier":
        raise SystemExit(f"{run}: env {env!r}, not terminal_verifier: this re-scores TerminalEnv episodes only")
    monitors = (info.get("config") or {}).get("monitors")
    if not isinstance(monitors, list):
        raise SystemExit(f"{run}: run_info.json has no config.monitors, the run's monitor specs")
    specs = TypeAdapter(list[MonitorSpec]).validate_python(monitors)
    judges = [s for s in specs if s.kind == "cot" and s.provider == "vllm"]
    return judges, [s.name for s in specs if s not in judges]


@contextlib.contextmanager
def servers_unchecked():
    """Build judges without contacting their servers: ``VLLMJudge`` calls ``vllm.check_server`` at construction. The
    servers that will be called are checked once each, explicitly, before the first call (``check_servers``)."""
    from monitordecorrelation.monitors import vllm

    real = vllm.check_server
    vllm.check_server = lambda *a, **k: None
    try:
        yield
    finally:
        vllm.check_server = real


def build_judges(info: dict, specs: list, endpoints: dict[str, str], run: str) -> dict:
    """{name: judge}: the run's judges built as ``run_experiment.py`` built them (``build_monitors(...,
    multi_turn=True)`` → ``AgentCoTMonitor``), each one's calls going to ``endpoints[name]``. SystemExit if what this
    code builds differs from the resolved settings the run recorded, the endpoint aside."""
    from monitordecorrelation.experiment_config import build_monitors
    from monitordecorrelation.rl.train import _monitor_info

    behavior = (info.get("env") or {}).get("behavior_name") or "reward_hacking"
    try:
        with servers_unchecked():
            ta, held = build_monitors([s.model_copy(update={"base_url": endpoints[s.name]}) for s in specs],
                                      default_behavior=behavior, multi_turn=True)
    except ValueError as e:  # a setting the judge would not honour, e.g. a --judge-url not ending in /v1
        raise SystemExit(f"{run}: {e}") from e
    recorded = {m.get("name"): m for m in [*(info.get("train_against") or []), *(info.get("held_out") or [])]}
    out = {}
    for role, judges in (("train_against", ta), ("held_out", held)):
        for m in judges:
            want = {k: v for k, v in (recorded.get(m.name) or {}).items() if k != "base_url"}
            got = {k: v for k, v in _monitor_info(m, role).items() if k != "base_url"}
            if want != got:
                diff = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
                raise SystemExit(f"{run}: judge {m.name}: the run recorded "
                                 + ", ".join(f"{k}={want.get(k)!r}" for k in diff) + " but this code builds "
                                 + ", ".join(f"{k}={got.get(k)!r}" for k in diff) + ": it would not be called as live")
            out[m.name] = m
    return out


def check_servers(judges_by_run: dict[str, dict]) -> None:
    """``vllm.check_server`` once per (endpoint, model) about to be called: up, serving that model, with room for its
    max_tokens."""
    from monitordecorrelation.monitors import vllm

    seen = set()
    for judges in judges_by_run.values():
        for m in judges.values():
            key = (m.backend.base_url, m.model_id, m.backend.max_tokens)
            if key in seen:
                continue
            seen.add(key)
            try:
                vllm.check_server(m.backend.base_url, m.model_id, max_tokens=m.backend.max_tokens, monitor=m.name)
            except RuntimeError as e:
                raise SystemExit(str(e)) from e
            log(f"server {m.backend.base_url} serves {m.model_id}")


# ----------------------------------------------------------------------------- episodes


def episode_rollout(rec: dict):
    """An eval record as the ``Rollout`` the loop's monitors scored: the task as the prompt, the flat ``cot`` /
    ``answer`` views, and ``env_meta`` without the ``unparsed`` flag ``env.score`` added, i.e. ``env.finish``'s
    record, as ``meta["episode"]``. ``stop_reason`` is the last turn's: ``length`` iff a turn was truncated, since a
    truncated turn ends the episode."""
    from monitordecorrelation.types import Prompt, Rollout

    em = rec.get("env_meta")
    if not isinstance(em, dict) or not isinstance(em.get("turns"), list):
        raise ValueError("no env_meta with turns: not a TerminalEnv episode")
    episode = {k: v for k, v in em.items() if k != "unparsed"}
    return Rollout(prompt=Prompt(text=rec["question"], meta={"task_id": rec.get("task_id")}), cot=rec["cot"],
                   output=rec["answer"],
                   meta={"stop_reason": "length" if episode.get("truncated") else "stop", "episode": episode})


def terminal_predicates():
    """The terminal env, for ``envs.base.invalid_reason``: its ``unparseable`` / ``never_submitted`` read nothing but
    the rollout's episode record, so an instance that skipped ``__init__`` (no items, no sandbox) is all it needs."""
    from monitordecorrelation.envs.terminal_verifier import TerminalVerifierEnv

    return object.__new__(TerminalVerifierEnv)


def request_sha256(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
                          ).hexdigest()


def describe_difference(ours: dict, recorded) -> str:
    """Where a rebuilt request differs from the recorded one, briefly."""
    if not isinstance(recorded, dict):
        return "no request was recorded"
    keys = sorted(k for k in ours.keys() | recorded.keys() if k != "messages" and ours.get(k) != recorded.get(k))
    if keys:
        return "; ".join(f"{k}: recorded {recorded.get(k)!r}, now {ours.get(k)!r}" for k in keys)
    try:
        a, b = ours["messages"][0]["content"], recorded["messages"][0]["content"]
    except (KeyError, IndexError, TypeError):
        return f"messages: recorded {str(recorded.get('messages'))[:200]!r}"
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return f"the prompt, from char {i}: recorded {b[max(0, i - 60):i + 60]!r}, now {a[max(0, i - 60):i + 60]!r}"


def reparsed_score(m, call) -> float | None:
    """The score this code reads from a recorded reply: the answer from ``content`` (``vllm.judge_output``), then
    ``judge_result``, as the call returned it. None if the record holds no reply, or one this code refuses."""
    from monitordecorrelation.monitors.cot_monitor import judge_result
    from monitordecorrelation.monitors.openrouter import JudgeCall
    from monitordecorrelation.monitors.vllm import judge_output

    resp = (call or {}).get("response") or {}
    msg = resp.get("message")
    if not isinstance(msg, dict):
        return None
    try:
        text = judge_output(msg, resp.get("finish_reason"))
    except RuntimeError:
        return None
    # post_chat: no answer under "length" is the empty answer; under "stop" the call would have been retried
    return judge_result(JudgeCall(text="" if text is None else text, record=call), binary_judge=m.binary_judge,
                        threshold=m.threshold).score


def slim_call(call) -> dict:
    """What the per-step call-health rates read of a call record (``judge_call_rates``, ``hit_thinking_budget``)."""
    call = call or {}
    req, resp = call.get("request") or {}, call.get("response") or {}
    return {"request": {k: req[k] for k in ("thinking_token_budget",) if k in req},
            "response": {"finish_reason": resp.get("finish_reason"), "usage": resp.get("usage")}}


@dataclass
class Episode:
    """One episode the judges were shown: where it is, its labels, and per judge its request's hash and live verdict."""

    line: int
    step: int
    task_id: str | None
    behavior: bool
    loose: bool
    sha: dict[str, str] = field(default_factory=dict)
    # judge → {score, label, parse_error, hit_thinking_budget, call (slim_call)}
    live: dict[str, dict] = field(default_factory=dict)


@dataclass
class Scan:
    """One pass over a run's ``eval_rollouts.jsonl``, with the checks of the dry run."""

    run: str
    path: Path
    n_lines: int = 0
    n_bytes: int = 0
    sha256: str = ""
    steps: list[int] = field(default_factory=list)
    loose_steps: set[int] = field(default_factory=set)  # steps with a loose_rh label: a loose/ track, as live
    invalid: Counter = field(default_factory=Counter)
    episodes: list[Episode] = field(default_factory=list)
    identical: Counter = field(default_factory=Counter)  # judge → rebuilt requests equal to the recorded ones
    drift: Counter = field(default_factory=Counter)  # judge → rebuilt requests that differ
    first_drift: str | None = None
    reparsed: Counter = field(default_factory=Counter)  # judge → live scores this code reads back from the reply
    reparse_bad: Counter = field(default_factory=Counter)
    first_reparse_bad: str | None = None


def scan_run(run_dir: Path, judges: dict) -> Scan:
    """Read every eval record; keep the episodes the loop showed its monitors (``invalid_reason`` None), each with its
    rebuilt request's hash per judge and its live verdicts. SystemExit on a record that contradicts the loop's rules."""
    from monitordecorrelation.envs.base import invalid_reason
    from monitordecorrelation.monitors.judge_backend import judge_priority
    from monitordecorrelation.monitors.vllm import EVAL_PRIORITY

    env = terminal_predicates()
    sc = Scan(run=run_dir.name, path=run_dir / "eval_rollouts.jsonl")
    h, steps = hashlib.sha256(), set()
    with open(sc.path, "rb") as fh:
        for line_no, raw in enumerate(fh, start=1):
            h.update(raw)
            sc.n_bytes += len(raw)
            sc.n_lines = line_no
            where = f"{sc.run} line {line_no}"
            try:
                rec = json.loads(raw)
                rollout = episode_rollout(rec)
                why = invalid_reason(env, rollout)
            except (ValueError, KeyError, TypeError) as e:
                raise SystemExit(f"{where}: not an eval episode record ({type(e).__name__}: {e})") from e
            step = rec.get("step")
            if not isinstance(step, int):
                raise SystemExit(f"{where}: no eval step")
            steps.add(step)
            if "loose_rh" in rec["env_meta"]:
                sc.loose_steps.add(step)
            if "invalid_reason" in rec and rec["invalid_reason"] != why:
                raise SystemExit(f"{where}: the record says invalid_reason={rec['invalid_reason']!r}, the loop's rule "
                                 f"gives {why!r} for its env_meta")
            mons = rec.get("monitors") or {}
            if why is not None:
                sc.invalid[why] += 1
                if shown := sorted(n for n in judges if n in mons):
                    raise SystemExit(f"{where}: an invalid ({why}) episode, yet live scores of {shown}")
                continue
            if not isinstance(rec.get("behavior_present"), bool):
                raise SystemExit(f"{where}: no behavior_present label")
            ep = Episode(line=line_no, step=step, task_id=rec.get("task_id"), behavior=rec["behavior_present"],
                         loose=bool(rec["env_meta"].get("loose_rh", False)))
            for name, m in judges.items():
                live = mons.get(name)
                if not isinstance(live, dict) or not isinstance(live.get("score"), (int, float)) \
                        or isinstance(live.get("score"), bool):
                    raise SystemExit(f"{where}: shown to the judges, but no live score of {name}")
                try:
                    with judge_priority(EVAL_PRIORITY):
                        body = m._request_body(m._build_prompt(rollout))
                except AssertionError as e:
                    raise SystemExit(f"{where}: {name} cannot render this episode ({e})") from e
                ep.sha[name] = request_sha256(body)
                call = live.get("call")
                if body == (call or {}).get("request"):
                    sc.identical[name] += 1
                else:
                    sc.drift[name] += 1
                    if sc.first_drift is None:
                        sc.first_drift = f"{where}, {name}: {describe_difference(body, (call or {}).get('request'))}"
                again = reparsed_score(m, call)
                if again is not None and again == live["score"]:
                    sc.reparsed[name] += 1
                else:
                    sc.reparse_bad[name] += 1
                    if sc.first_reparse_bad is None:
                        sc.first_reparse_bad = f"{where}, {name}: live score {live['score']!r}, read back {again!r}"
                ep.live[name] = {"score": live["score"], "label": live.get("label"),
                                 "parse_error": bool(live.get("parse_error")),
                                 "hit_thinking_budget": live.get("hit_thinking_budget"), "call": slim_call(call)}
            sc.episodes.append(ep)
    sc.sha256, sc.steps = h.hexdigest(), sorted(steps)
    return sc


# ----------------------------------------------------------------------------- selection


def compare_sample(sc: Scan, *, pos: int, neg: int, seed: int, steps: set[int] | None) -> tuple[list[Episode], list]:
    """(the sampled episodes in line order, the plan: per step how many of each class were taken of how many)."""
    cells: dict[tuple[int, bool], list[Episode]] = defaultdict(list)
    for e in sc.episodes:
        cells[(e.step, e.behavior)].append(e)
    chosen, plan = [], []
    for step in sc.steps:
        if steps is not None and step not in steps:
            continue
        row = {"step": step}
        for cls, k, tag in ((True, pos, "pos"), (False, neg, "neg")):
            pool = sorted(cells.get((step, cls), []), key=lambda e: e.line)
            random.Random(f"{seed}/{sc.run}/{step}/{tag}").shuffle(pool)
            chosen += pool[:k]
            row[tag], row[f"{tag}_available"] = min(k, len(pool)), len(pool)
        plan.append(row)
    return sorted(chosen, key=lambda e: e.line), plan


# ----------------------------------------------------------------------------- the verdict store


def load_store(path: Path) -> dict[tuple, dict]:
    """The stored verdicts by (line, judge, request_sha256, endpoint), the last one winning. A row cut off by a kill
    mid-write is skipped."""
    rows: dict[tuple, dict] = {}
    if not path.exists():
        return rows
    with open(path) as fh:
        for raw in fh:
            try:
                r = json.loads(raw)
                rows[(r["line"], r["judge"], r["request_sha256"], r["endpoint"])] = r
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return rows


def verdict(row: dict):
    """A stored row as the ``MonitorResult`` the loop's metrics take."""
    from monitordecorrelation.types import MonitorResult

    meta = {"call": row.get("call")} | ({"parse_error": True} if row.get("parse_error") else {})
    return MonitorResult(score=row["score"], label=row["label"], meta=meta)


def live_verdict(live: dict):
    """A live verdict (``Episode.live``) as the ``MonitorResult`` the loop's metrics took."""
    from monitordecorrelation.types import MonitorResult

    meta = {"call": live["call"]} | ({"parse_error": True} if live["parse_error"] else {})
    return MonitorResult(score=live["score"], label=live["label"], meta=meta)


def score_episodes(sc: Scan, todo: dict[int, list[str]], judges: dict, out_path: Path, workers: int) -> int:
    """Send each (episode, judge) of ``todo`` (line → judge names) to its judge as the loop's eval did — the judge's
    ``score`` through ``rl.train._score_at`` at ``EVAL_PRIORITY``, in a pool of daemon threads — and append each
    verdict to ``out_path`` as it returns. A second pass over eval_rollouts.jsonl that holds only the episodes in
    flight. Returns the number of verdicts saved; SystemExit on the first call that fails, after saving the rest."""
    from monitordecorrelation.eval.rollout_dump import monitor_record
    from monitordecorrelation.monitors.vllm import EVAL_PRIORITY
    from monitordecorrelation.rl.train import _DaemonPool, _score_at

    eps = {e.line: e for e in sc.episodes}
    n_todo = sum(len(v) for v in todo.values())
    pool, pending = _DaemonPool(workers), {}
    state = {"done": 0, "failure": None, "logged": time.time()}
    stop = threading.Event()  # set on the first failure: the calls still queued then never start
    t0 = time.time()

    def call(score, rollout):
        if stop.is_set():
            raise RuntimeError("not started: an earlier call failed")
        return _score_at(score, EVAL_PRIORITY, rollout)

    if out_path.exists() and out_path.stat().st_size:
        with open(out_path, "rb") as fh:  # a row cut off by a kill: end it, so the next one starts on its own line
            fh.seek(-1, os.SEEK_END)
            cut = fh.read(1) != b"\n"
    else:
        cut = False
    with open(out_path, "a") as out:
        if cut:
            out.write("\n")

        def drain(block: bool) -> None:
            finished, _ = wait(list(pending), timeout=None if block else 0, return_when=FIRST_COMPLETED)
            for fut in finished:
                line, name = pending.pop(fut)
                ep = eps[line]
                try:
                    res = fut.result()
                    sent = request_sha256(res.meta["call"]["request"])
                    if sent != ep.sha[name]:
                        raise RuntimeError(f"sent request {sent[:12]}, rebuilt {ep.sha[name][:12]}: the file changed?")
                except Exception as e:  # noqa: BLE001 — stops the run, below
                    stop.set()
                    state["failure"] = state["failure"] or f"{sc.run} line {line}, {name}: {type(e).__name__}: {e}"
                    continue
                row = {"line": line, "step": ep.step, "task_id": ep.task_id, "judge": name,
                       "request_sha256": ep.sha[name], "endpoint": judges[name].backend.base_url,
                       **monitor_record(res), "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                out.write(json.dumps(row, default=str) + "\n")
                out.flush()
                state["done"] += 1
            if time.time() - state["logged"] > 60 or state["done"] == n_todo:
                state["logged"] = time.time()
                rate = state["done"] / max(time.time() - t0, 1e-9) * 60
                log(f"  {sc.run}: {state['done']}/{n_todo} verdicts ({rate:.0f}/min)")

        with open(sc.path, "rb") as fh:
            for line_no, raw in enumerate(fh, start=1):
                if state["failure"]:
                    break
                names = todo.get(line_no)
                if not names:
                    continue
                rollout = episode_rollout(json.loads(raw))
                for name in names:
                    while len(pending) >= 2 * workers and not state["failure"]:
                        drain(block=True)
                    if state["failure"]:
                        break
                    pending[pool.submit(call, judges[name].score, rollout, name=f"rescore-{name}-{line_no}")] = (
                        line_no, name)
        while pending and not state["failure"]:
            drain(block=True)
        if pending:
            drain(block=False)  # keep what already returned; the daemon threads still out die with the process
    if state["failure"]:
        raise SystemExit(f"stopped on {state['failure']} — {state['done']} verdicts saved to {out_path}; fix it and "
                         "re-run to resume")
    return state["done"]


# ----------------------------------------------------------------------------- metrics


def judge_step_fields(m, results: list, gt: list[bool], gt_loose: list[bool] | None, behavior: str) -> dict:
    """``monitor/<judge>/…`` of one eval step as ``rl/train.py``'s ``run_eval`` logs them, by the same functions:
    ``results`` (``MonitorResult``) and ``gt`` (``behavior_present``) over the step's episodes the judge was shown,
    ``gt_loose`` the loose labels when the step has them."""
    from monitordecorrelation.eval.metric_keys import absent_score_key, present_score_key
    from monitordecorrelation.eval.metrics import _mean_for_class, accuracy, brier, dprime_margin, roc_auc
    from monitordecorrelation.rl.train import _judge_call_metrics, _label_metrics

    scores = [r.score for r in results]
    preds = [r.label for r in results]
    row = {
        f"monitor/{m.name}/auroc": roc_auc(scores, gt),
        f"monitor/{m.name}/accuracy": accuracy(preds, gt),
        f"monitor/{m.name}/brier": brier(scores, gt),
        f"monitor/{m.name}/mean_score": mean(scores) if scores else float("nan"),
        present_score_key(m.name, behavior): _mean_for_class(scores, gt, True),
        absent_score_key(m.name, behavior): _mean_for_class(scores, gt, False),
        f"monitor/{m.name}/dprime_margin": dprime_margin(scores, gt),
        f"monitor/{m.name}/n_pos": sum(gt),
        f"monitor/{m.name}/n_neg": len(gt) - sum(gt),
        f"monitor/{m.name}/n_scored": len(scores),
    }
    row.update(_judge_call_metrics(m, results))
    if gt_loose is not None:
        row.update(_label_metrics(f"monitor/{m.name}/loose", scores, preds, gt_loose))
    return row


def metric_behavior(info: dict) -> str:
    """The behavior word in the run's class-mean keys (``mean_score_<behavior>``): the env's ``behavior_name``, slugged
    as ``rl.train.run_grpo`` slugs it."""
    name = (info.get("env") or {}).get("behavior_name") or "behavior"
    return re.sub(r"\W+", "_", name).strip("_") or "behavior"


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def step_rows(sc: Scan, judges: dict, verdicts: dict[tuple[int, str], object], behavior: str,
              base_rows: list[dict]) -> tuple[list[dict], list[int]]:
    """(per eval step: ``base_rows``' row — the run's eval_metrics.jsonl — without its ``monitor/*`` and ``time/*``
    fields, plus every judge's fields from ``verdicts`` (by (line, judge)); the steps left out because a verdict is
    missing)."""
    by_step: dict[int, list[Episode]] = defaultdict(list)
    for e in sc.episodes:
        by_step[e.step].append(e)
    out, incomplete = [], []
    for base in base_rows:
        s = base["step"]
        eps = by_step.get(s, [])
        if any((e.line, n) not in verdicts for e in eps for n in judges):
            incomplete.append(s)
            continue
        row = {k: v for k, v in base.items() if not k.startswith(("monitor/", "time/"))}
        gt = [e.behavior for e in eps]
        gt_loose = [e.loose for e in eps] if s in sc.loose_steps else None
        for name, m in judges.items():
            row.update(judge_step_fields(m, [verdicts[(e.line, name)] for e in eps], gt, gt_loose, behavior))
        out.append(row)
    return out, incomplete


def same_value(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def metrics_self_check(sc: Scan, judges: dict, behavior: str, live_rows: list[dict]) -> tuple[int, list[str]]:
    """Recompute every step's judge fields from the LIVE verdicts and compare them with the run's own
    eval_metrics.jsonl: (fields compared, the mismatches)."""
    verdicts = {(e.line, n): live_verdict(e.live[n]) for e in sc.episodes for n in judges}
    rows, _ = step_rows(sc, judges, verdicts, behavior, live_rows)
    n, bad = 0, []
    live_by_step = {r["step"]: r for r in live_rows}
    for row in rows:
        live = live_by_step[row["step"]]
        prefixes = tuple(f"monitor/{name}/" for name in judges)
        for k in sorted({k for k in row if k.startswith(prefixes)} | {k for k in live if k.startswith(prefixes)}):
            n += 1
            if not same_value(row.get(k, "absent"), live.get(k, "absent")):
                bad.append(f"step {row['step']} {k}: eval_metrics.jsonl {live.get(k, 'absent')!r}, "
                           f"recomputed {row.get(k, 'absent')!r}")
    return n, bad


# ----------------------------------------------------------------------------- the new-vs-live report


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def pair_stats(pairs: list[dict]) -> dict:
    """New vs live over ``pairs`` ({behavior, live, new, …} per episode): see the module docstring."""
    from monitordecorrelation.eval.metrics import roc_auc

    live, new = [p["live"] for p in pairs], [p["new"] for p in pairs]
    y = [p["behavior"] for p in pairs]
    d = [b - a for a, b in zip(live, new)]
    n = len(pairs)
    nan = float("nan")
    sd = stdev(d) if n > 1 else nan

    def share(vals, x):
        return sum(v == x for v in vals) / n if n else nan

    return {"n": n, "n_pos": sum(y), "n_neg": n - sum(y),
            "mean_live": mean(live) if n else nan, "mean_new": mean(new) if n else nan,
            "mean_diff": mean(d) if n else nan, "sd_diff": sd, "se_diff": sd / math.sqrt(n) if n > 1 else nan,
            "mean_abs_diff": mean(abs(x) for x in d) if n else nan,
            "pearson": _pearson(live, new), "spearman": _pearson(_ranks(live), _ranks(new)),
            "auroc_live": roc_auc(live, y), "auroc_new": roc_auc(new, y),
            "at_1_live": share(live, 1.0), "at_1_new": share(new, 1.0),
            "at_0_live": share(live, 0.0), "at_0_new": share(new, 0.0),
            "parse_error_live": sum(p["parse_error_live"] for p in pairs),
            "parse_error_new": sum(p["parse_error_new"] for p in pairs),
            "thinking_budget_live": sum(p["budget_live"] is True for p in pairs),
            "thinking_budget_new": sum(p["budget_new"] is True for p in pairs),
            "finish_length_new": sum(p["finish_new"] == "length" for p in pairs)}


def report(pairs: list[dict], missing: Counter, out_path: Path | None) -> dict:
    by_judge: dict[str, list] = defaultdict(list)
    by_run: dict[tuple[str, str], list] = defaultdict(list)
    for p in pairs:
        by_judge[p["judge"]].append(p)
        by_run[(p["run"], p["judge"])].append(p)
    res = {"judges": {j: pair_stats(ps) for j, ps in sorted(by_judge.items())},
           "runs": {f"{r}/{j}": pair_stats(ps) for (r, j), ps in sorted(by_run.items())},
           "missing_verdicts": dict(missing)}

    def f(v, w=6, p=3):
        return f"{v:{w}.{p}f}" if isinstance(v, float) and not math.isnan(v) else f"{'-':>{w}}"

    print(f"\nnew vs live, pooled over the runs ({len(pairs)} pairs)")
    print(f"{'judge':10s} {'n':>5} {'pos':>5} {'neg':>5} | {'live':>6} {'new':>6} | {'diff':>6} {'sd':>6} {'se':>6} "
          f"| {'r':>6} {'rho':>6} | {'AUC liv':>7} {'new':>6} | {'=1 live':>7} {'new':>6} | {'=0 live':>7} "
          f"{'new':>6} | {'parse_err l/n':>13} {'budget l/n':>10}")
    for j, s in res["judges"].items():
        print(f"{j:10s} {s['n']:5d} {s['n_pos']:5d} {s['n_neg']:5d} | {f(s['mean_live'])} {f(s['mean_new'])} | "
              f"{f(s['mean_diff'])} {f(s['sd_diff'])} {f(s['se_diff'])} | {f(s['pearson'])} {f(s['spearman'])} | "
              f"{f(s['auroc_live'], 7)} {f(s['auroc_new'])} | {f(s['at_1_live'], 7)} {f(s['at_1_new'])} | "
              f"{f(s['at_0_live'], 7)} {f(s['at_0_new'])} | {s['parse_error_live']:>6d}/{s['parse_error_new']:<6d} "
              f"{s['thinking_budget_live']:>4d}/{s['thinking_budget_new']:<5d}")
    print(f"\nper run: {'run/judge':40s} {'n':>5} {'diff':>6} {'sd':>6} {'r':>6} {'AUC liv':>7} {'new':>6} "
          f"{'=1 live':>7} {'new':>6}")
    for k, s in res["runs"].items():
        print(f"         {k:40s} {s['n']:5d} {f(s['mean_diff'])} {f(s['sd_diff'])} {f(s['pearson'])} "
              f"{f(s['auroc_live'], 7)} {f(s['auroc_new'])} {f(s['at_1_live'], 7)} {f(s['at_1_new'])}")
    if missing:
        print(f"\nnot yet re-scored (left out above): {dict(missing)}")
    if out_path is not None:
        out_path.write_text(json.dumps(res, indent=1))
        log(f"wrote {out_path}")
    return res


def pairs_of(sc: Scan, episodes: list[Episode], judges: dict, rows: dict, missing: Counter) -> list[dict]:
    """(episode, judge) → {run, step, line, judge, behavior, live, new, …} where a new verdict exists."""
    out = []
    for e in episodes:
        for name, m in judges.items():
            row = rows.get((e.line, name, e.sha[name], m.backend.base_url))
            if row is None:
                missing[name] += 1
                continue
            live = e.live[name]
            out.append({"run": sc.run, "step": e.step, "line": e.line, "judge": name, "behavior": e.behavior,
                        "live": live["score"], "new": row["score"], "parse_error_live": live["parse_error"],
                        "parse_error_new": bool(row.get("parse_error")), "budget_live": live["hit_thinking_budget"],
                        "budget_new": row.get("hit_thinking_budget"), "finish_new": row.get("finish_reason")})
    return out


# ----------------------------------------------------------------------------- main


def run_dirs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = [Path(p) for p in sorted(globmod.glob(pat))] or ([Path(pat)] if Path(pat).exists() else [])
        if not hits:
            raise SystemExit(f"no run folder matches {pat!r}")
        for h in hits:
            if (h / "eval_rollouts.jsonl").exists() and (h / "run_info.json").exists():
                out.append(h)
            else:
                raise SystemExit(f"{h}: no eval_rollouts.jsonl and run_info.json")
    if not out:
        raise SystemExit(f"no run folder matched {patterns}")
    names = Counter(d.name for d in out)
    if dupes := sorted(n for n, k in names.items() if k > 1):
        raise SystemExit(f"two run folders share a name ({dupes}): their outputs would collide in --out-dir")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("runs", nargs="+", help="run folders (or globs) holding eval_rollouts.jsonl + run_info.json")
    common.add_argument("--out-dir", required=True, help="writes DIR/<run>/…, never into the run folders")
    common.add_argument("--judges", help="comma-list of judge names or families (q3 = q3_cot + q3_out); default: "
                                         "every vLLM judge of each run")
    common.add_argument("--judge-url", action="append", default=[], metavar="KEY=URL",
                        help="where a judge's calls go instead of the run's base_url; KEY a judge name or family, "
                             "e.g. q3=http://localhost:8001/v1 (repeatable)")
    common.add_argument("--workers", type=int, default=64, help="judge calls in flight at once, all judges together")
    common.add_argument("--dry-run", action="store_true",
                        help="check the files and print the plan: no judge call, nothing written")
    common.add_argument("--allow-request-drift", action="store_true",
                        help="score even where this code's request differs from the recorded one")
    sub = ap.add_subparsers(dest="mode", required=True)
    cmp = sub.add_parser("compare", parents=[common], help="a seeded sample per (run, eval step), new vs live")
    cmp.add_argument("--pos", type=int, default=4, help="episodes with behavior_present per (run, eval step)")
    cmp.add_argument("--neg", type=int, default=4, help="episodes without behavior_present per (run, eval step)")
    cmp.add_argument("--seed", type=int, default=0)
    cmp.add_argument("--steps", help="comma-list of eval steps (default: all)")
    sub.add_parser("full", parents=[common], help="every episode; per-step metrics for eval/degradation.py")
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.mode == "compare" and (args.pos < 0 or args.neg < 0 or args.pos + args.neg == 0):
        ap.error("--pos and --neg must be >= 0, not both 0")
    try:
        steps = None if args.mode == "full" or not args.steps else {int(s) for s in args.steps.split(",") if s.strip()}
    except ValueError:
        ap.error(f"--steps {args.steps!r}: expected a comma-list of eval steps, e.g. 0,30,90")

    dirs = run_dirs(args.runs)
    out_dir = Path(args.out_dir)
    for d in dirs:
        if (out_dir / d.name).resolve() == d.resolve():
            raise SystemExit(f"--out-dir {out_dir} would write into the run folder {d} itself")
    keys = {k.strip() for k in args.judges.split(",") if k.strip()} if args.judges else None
    overrides = parse_judge_urls(args.judge_url)

    # every run's judges, before reading a single episode: an option that matches nothing fails at once
    infos, picks, used_keys = {}, {}, set()
    for d in dirs:
        info = json.loads((d / "run_info.json").read_text())
        specs, others = run_judge_specs(info, d.name)
        picked = []
        for s in specs:
            if (k := selected(s.name, keys)) is not None:
                used_keys.add(k)
                picked.append(s)
        infos[d.name], picks[d.name] = info, (picked, others)
    if keys is not None and (unused := sorted(keys - used_keys)):
        raise SystemExit(f"--judges {unused}: no run has such a vLLM judge")
    judges_by_run, used_urls = {}, set()
    for d in dirs:
        picked, others = picks[d.name]
        if not picked:
            raise SystemExit(f"{d.name}: no vLLM judge to re-score" + (f" among {sorted(keys)}" if keys else ""))
        endpoints = {}
        for s in picked:
            k = s.name if s.name in overrides else judge_family(s.name) if judge_family(s.name) in overrides else None
            if k is not None:
                used_urls.add(k)
            endpoints[s.name] = (overrides[k] if k is not None else s.base_url).rstrip("/")
        judges_by_run[d.name] = build_judges(infos[d.name], picked, endpoints, d.name)
        log(f"{d.name}: judges " + ", ".join(f"{n} ({m.model_id}, cot={m.use_cot}, out={m.use_output}) → "
                                             f"{m.backend.base_url}" for n, m in judges_by_run[d.name].items())
            + (f"; not re-scored: {others}" if others else ""))
    if unused := sorted(set(overrides) - used_urls):
        raise SystemExit(f"--judge-url {unused}: no judge of these runs has that name or family")

    failed_checks = False
    plans: dict[str, tuple[Scan, list[Episode], dict[int, list[str]], dict]] = {}
    for d in dirs:
        judges = judges_by_run[d.name]
        t0 = time.time()
        sc = scan_run(d, judges)
        n_req = sum(sc.identical.values()) + sum(sc.drift.values())
        log(f"{sc.run}: {sc.n_lines} episodes on {len(sc.steps)} eval steps ({sc.steps[0]}..{sc.steps[-1]}); "
            f"{len(sc.episodes)} shown to the judges; not shown: {dict(sc.invalid) or 'none'} "
            f"[{time.time() - t0:.0f}s]")
        log(f"  requests rebuilt: {sum(sc.identical.values())}/{n_req} identical to the recorded ones"
            + (f"; FIRST DIFFERENCE {sc.first_drift}" if sc.first_drift else ""))
        log(f"  live scores: {sum(sc.reparsed.values())}/{n_req} read back from the recorded replies"
            + (f"; FIRST MISMATCH {sc.first_reparse_bad}" if sc.first_reparse_bad else ""))
        behavior = metric_behavior(infos[d.name])
        metrics_path = d / "eval_metrics.jsonl"
        if metrics_path.exists():
            n_fields, bad = metrics_self_check(sc, judges, behavior, read_jsonl(metrics_path))
            log(f"  per-step judge fields recomputed from the live verdicts: {n_fields - len(bad)}/{n_fields} equal to "
                "eval_metrics.jsonl's" + (f"; FIRST MISMATCH {bad[0]}" if bad else ""))
            failed_checks |= bool(bad)
        else:
            log("  no eval_metrics.jsonl: the per-step fields cannot be checked against the live ones")
        failed_checks |= bool(sc.reparse_bad)
        if sc.drift:
            failed_checks = True
            if not args.dry_run and not args.allow_request_drift:
                raise SystemExit(f"{sc.run}: {sum(sc.drift.values())} rebuilt requests differ from the recorded ones "
                                 f"({sc.first_drift}): they would not be scored as live. --allow-request-drift to "
                                 "score them anyway")
        if args.mode == "compare":
            episodes, plan = compare_sample(sc, pos=args.pos, neg=args.neg, seed=args.seed, steps=steps)
            n_pos, n_neg = sum(r["pos"] for r in plan), sum(r["neg"] for r in plan)
            short_p = [r["step"] for r in plan if r["pos"] < args.pos]
            short_n = [r["step"] for r in plan if r["neg"] < args.neg]
            log(f"  compare sample ({args.pos} + {args.neg} per step, seed {args.seed}): {len(plan)} steps; "
                f"{n_pos} with behavior_present + {n_neg} without = {len(episodes)} episodes"
                + (f"; fewer with it at {len(short_p)} steps" if short_p else "")
                + (f"; fewer without it at {len(short_n)} steps" if short_n else ""))
            if args.dry_run:
                log("    per step (taken/available): " + "  ".join(
                    f"{r['step']}:{r['pos']}/{r['pos_available']}+{r['neg']}/{r['neg_available']}" for r in plan))
        else:
            episodes = sc.episodes
        rows = load_store(out_dir / sc.run / SCORES)
        todo: dict[int, list[str]] = defaultdict(list)
        for e in episodes:
            for name, m in judges.items():
                if (e.line, name, e.sha[name], m.backend.base_url) not in rows:
                    todo[e.line].append(name)
        n_todo = sum(len(v) for v in todo.values())
        log(f"  {len(episodes) * len(judges)} verdicts wanted: {len(episodes) * len(judges) - n_todo} already in "
            f"{out_dir / sc.run / SCORES}, {n_todo} to score")
        plans[sc.run] = (sc, episodes, dict(todo), rows)
    if steps is not None and (absent := sorted(steps - {s for sc, *_ in plans.values() for s in sc.steps})):
        raise SystemExit(f"--steps {absent}: no run has such an eval step")
    if args.dry_run:
        log("dry run: no judge called, nothing written" + ("; SOME CHECKS FAILED (above)" if failed_checks else ""))
        return 1 if failed_checks else 0

    from monitordecorrelation.provenance import code_version

    if any(todo for _, _, todo, _ in plans.values()):
        check_servers({r: judges_by_run[r] for r, (_, _, todo, _) in plans.items() if todo})
    all_pairs, missing = [], Counter()
    for d in dirs:
        sc, episodes, todo, _ = plans[d.name]
        judges = judges_by_run[d.name]
        run_out = out_dir / sc.run
        run_out.mkdir(parents=True, exist_ok=True)
        if todo:
            score_episodes(sc, todo, judges, run_out / SCORES, args.workers)
        rows = load_store(run_out / SCORES)
        all_pairs += pairs_of(sc, episodes, judges, rows, missing)
        recorded = {m.get("name"): m for m in [*(infos[d.name].get("train_against") or []),
                                               *(infos[d.name].get("held_out") or [])]}
        info = {"run": sc.run, "source": str(d.resolve()), "mode": args.mode,
                "argv": sys.argv[1:] if argv is None else argv,
                "eval_rollouts": {"lines": sc.n_lines, "bytes": sc.n_bytes, "sha256": sc.sha256},
                "judges": {n: {"model_id": m.model_id, "use_cot": m.use_cot, "use_output": m.use_output,
                               "endpoint": m.backend.base_url, "run_base_url": recorded[n].get("base_url"),
                               **{k: v for k, v in m.backend.info().items() if k != "base_url"}}
                           for n, m in judges.items()},
                "code": code_version(), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if args.mode == "compare":
            info["compare"] = {"pos": args.pos, "neg": args.neg, "seed": args.seed,
                               "steps": sorted(steps) if steps else None}
        else:
            behavior = metric_behavior(infos[d.name])
            verdicts = {}
            for e in sc.episodes:
                for name, m in judges.items():
                    row = rows.get((e.line, name, e.sha[name], m.backend.base_url))
                    if row is not None:
                        verdicts[(e.line, name)] = verdict(row)
            metrics_path = d / "eval_metrics.jsonl"
            base = read_jsonl(metrics_path) if metrics_path.exists() else [{"step": s} for s in sc.steps]
            per_step, incomplete = step_rows(sc, judges, verdicts, behavior, base)
            with open(run_out / METRICS, "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in per_step)
            shutil.copyfile(d / "run_info.json", run_out / "run_info.json")
            info["full"] = {"steps_written": len(per_step), "steps_incomplete": incomplete}
            log(f"{sc.run}: {len(per_step)} eval steps written to {run_out / METRICS}"
                + (f"; {len(incomplete)} left out, a verdict missing: {incomplete}" if incomplete else ""))
        (run_out / INFO).write_text(json.dumps(info, indent=1))
    report(all_pairs, missing, out_dir / REPORT.format(mode=args.mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
