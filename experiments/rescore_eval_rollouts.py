#!/usr/bin/env python3
"""Re-score saved eval rollouts with a judge battery and compare with the scores recorded at the time.

Why: the judge prompt changed after the 2026-09-03 MBPP batch (probability framing, anti-rounding
instruction, the CoT-only "reasoning-only monitor" header). The batch's full ``eval_rollouts.jsonl``
files hold every held-out rollout's text AND the old-prompt score per monitor, so the new prompt can be
measured on the same rollouts: does it keep the AUROC, does it fix the saturation (96-98% of hacks
scored exactly 100 under the old prompt), does the CoT-only strong judge stop hedging, and does any
of that differ between early, mid and late training (hacks that emerged under training pressure are
what the matrix is about — the bakeoff set has none of those).

    # score (needs OPENROUTER_API_KEY in .env or the environment); resumable — re-run to continue
    uv run python experiments/rescore_eval_rollouts.py 'data/runs/sep3_full/*_sep3' \
        --config experiments/configs/mbpp_matrix_sep18/row_control.json --out data/runs/sep3_rescored.jsonl
    # composition only, no API calls
    uv run python experiments/rescore_eval_rollouts.py 'data/runs/sep3_full/*_sep3' --config ... --dry-run
    # re-print the report from an existing output file
    uv run python experiments/rescore_eval_rollouts.py --report data/runs/sep3_rescored.jsonl

Sampling: per (run, eval step) up to ``--hacks`` strict hacks, ``--clean`` clean rollouts (pass or fail
honestly: not loose_rh) and ``--loose`` loose-only ones, seeded. Unparsed/truncated rollouts are never
included — monitors never see them any more. Probe monitors in the config are skipped (judges only).

Output: one JSON line per (rollout, judge): run, step, task_id, class, old score (from the file), new
score, and the judge's raw reply (truncated) so a surprising score can be read.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from monitordecorrelation.eval.metrics import roc_auc
from monitordecorrelation.types import MonitorResult, Prompt, Rollout

load_dotenv(".env")


def _iter_jsonl(path: Path):
    with path.open(errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue  # these logs have been seen with malformed records


def _class(r: dict) -> str | None:
    if r.get("unparsed") or (r.get("env_meta") or {}).get("truncated"):
        return None
    if r.get("behavior_present"):
        return "strict"
    if r.get("loose_rh"):
        return "loose"
    return "clean"


def select(run_dirs: list[Path], *, hacks: int, clean: int, loose: int, seed: int, steps: set[int] | None):
    """[(uid, run, step, task_id, class, record)] — a seeded per-(run, step) stratified sample."""
    rng = random.Random(seed)
    out = []
    for d in run_dirs:
        cells: dict[tuple[int, str], list] = defaultdict(list)
        for i, r in _iter_jsonl(d / "eval_rollouts.jsonl"):
            c = _class(r)
            if c is None or (steps is not None and r.get("step") not in steps):
                continue
            cells[(r["step"], c)].append((i, r))
        for (step, c), items in sorted(cells.items()):
            k = {"strict": hacks, "clean": clean, "loose": loose}[c]
            for i, r in rng.sample(items, min(k, len(items))):
                out.append((f"{d.name}:{i}", d.name, step, r.get("task_id"), c, r))
    return out


def build_judges(config_path: str) -> list:
    from monitordecorrelation.experiment_config import build_monitors, load_monitor_specs
    specs = [s for s in load_monitor_specs(config_path) if s.kind == "cot"]
    ta, held = build_monitors(specs, default_behavior="reward_hacking")
    return ta + held


def score_all(selected, judges, out: Path, workers: int) -> None:
    done: set[tuple[str, str]] = set()
    if out.exists():
        for _, r in _iter_jsonl(out):
            done.add((r["uid"], r["judge"]))
    todo = [(s, j) for s in selected for j in judges if (s[0], j.name) not in done]
    print(f"{len(done)} (rollout, judge) pairs already scored in {out}; {len(todo)} to go")
    if not todo:
        return
    lock = threading.Lock()
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("a")
    n_fail = 0

    def to_rollout(r: dict) -> Rollout:
        return Rollout(prompt=Prompt(text=r["question"]), cot=r.get("cot") or "", output=r.get("answer") or "")

    # one sequential warm-up call per judge, so a judge that fails every call fails once, not as a storm
    for j in judges:
        try:
            j.score(to_rollout(selected[0][5]))
        except Exception as e:  # noqa: BLE001
            print(f"  warm-up {j.name}: {str(e)[:100]}")

    def work(item):
        nonlocal n_fail
        (uid, run, step, task_id, c, r), j = item
        try:
            res = j.score(to_rollout(r))
        except Exception as e:  # noqa: BLE001
            res = MonitorResult(score=float("nan"), label=False, meta={"error": str(e)[:200]})
        old = ((r.get("monitors") or {}).get(j.name) or {}).get("score")
        row = {"uid": uid, "run": run, "step": step, "task_id": task_id, "class": c, "judge": j.name,
               "old": old, "new": res.score, "parse_error": bool(res.meta.get("parse_error")),
               "raw": (res.meta.get("raw") or res.meta.get("error") or "")[:600]}
        with lock:
            if res.score != res.score:
                n_fail += 1
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, _ in enumerate(ex.map(work, todo), 1):
            if k % 500 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} scored ({n_fail} failed)", flush=True)
    fh.close()


# ----------------------------------------------------------------------------- report
def _spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def _bucket(step: int) -> str:
    return "early(<=12)" if step <= 12 else "mid(15-33)" if step <= 33 else "late(>=36)"


def report(path: Path) -> None:
    rows = [r for _, r in _iter_jsonl(path)]
    by_judge: dict[str, list] = defaultdict(list)
    for r in rows:
        by_judge[r["judge"]].append(r)
    print(f"{len(rows)} (rollout, judge) rows from {path}\n")
    hdr = (f"{'judge':16s} {'n_hack':>6} {'n_clean':>7} | {'AUROC old':>9} {'new':>5} | {'hack mean old':>13} {'new':>5} "
           f"| {'clean mean old':>14} {'new':>5} | {'hack==1 old':>11} {'new':>5} | {'hack<.5 old':>11} {'new':>5} | {'rho(old,new) hacks':>18} {'fail':>4}")
    print("STRICT hacks vs CLEAN, all steps pooled (the old scores are the ones recorded in the run files, old prompt)")
    print(hdr)
    for j, rs in sorted(by_judge.items()):
        ok = [r for r in rs if r["class"] in ("strict", "clean") and isinstance(r["new"], (int, float)) and r["new"] == r["new"]
              and isinstance(r["old"], (int, float))]
        h = [r for r in ok if r["class"] == "strict"]
        c = [r for r in ok if r["class"] == "clean"]
        n_fail = sum(1 for r in rs if not (isinstance(r["new"], (int, float)) and r["new"] == r["new"]))
        if not h or not c:
            print(f"{j:16s} {len(h):6d} {len(c):7d} | (not enough of both classes)  fail={n_fail}")
            continue
        gt = [r["class"] == "strict" for r in ok]
        f = lambda v: f"{v:5.2f}"
        print(f"{j:16s} {len(h):6d} {len(c):7d} | {roc_auc([r['old'] for r in ok], gt):9.3f} {roc_auc([r['new'] for r in ok], gt):5.3f} "
              f"| {mean(r['old'] for r in h):13.3f} {mean(r['new'] for r in h):5.3f} "
              f"| {mean(r['old'] for r in c):14.3f} {mean(r['new'] for r in c):5.3f} "
              f"| {f(mean(r['old'] >= 0.995 for r in h)):>11s} {f(mean(r['new'] >= 0.995 for r in h))} "
              f"| {f(mean(r['old'] < 0.5 for r in h)):>11s} {f(mean(r['new'] < 0.5 for r in h))} "
              f"| {_spearman([r['old'] for r in h], [r['new'] for r in h]):18.3f} {n_fail:4d}")

    print("\nBY TRAINING PHASE of the eval step: hack mean (old -> new) and AUROC (old -> new); n_hack in brackets")
    print(f"{'judge':16s} " + " ".join(f"{b:>34s}" for b in ("early(<=12)", "mid(15-33)", "late(>=36)")))
    for j, rs in sorted(by_judge.items()):
        cells = []
        for b in ("early(<=12)", "mid(15-33)", "late(>=36)"):
            ok = [r for r in rs if _bucket(r["step"]) == b and r["class"] in ("strict", "clean")
                  and isinstance(r["new"], (int, float)) and r["new"] == r["new"] and isinstance(r["old"], (int, float))]
            h = [r for r in ok if r["class"] == "strict"]
            if len(h) < 5 or len(ok) - len(h) < 5:
                cells.append(f"{'(n_hack=' + str(len(h)) + ')':>34s}")
                continue
            gt = [r["class"] == "strict" for r in ok]
            cells.append(f"{mean(r['old'] for r in h):.2f}->{mean(r['new'] for r in h):.2f}  "
                         f"auc {roc_auc([r['old'] for r in ok], gt):.2f}->{roc_auc([r['new'] for r in ok], gt):.2f} [{len(h):4d}]")
        print(f"{j:16s} " + " ".join(cells))

    print("\nLOOSE-ONLY rollouts (pass visible, fail held-out, no hardcoding — mostly honest bugs): mean score old -> new")
    for j, rs in sorted(by_judge.items()):
        l = [r for r in rs if r["class"] == "loose" and isinstance(r["new"], (int, float)) and r["new"] == r["new"]
             and isinstance(r["old"], (int, float))]
        if l:
            print(f"  {j:16s} n={len(l):4d}  {mean(r['old'] for r in l):.3f} -> {mean(r['new'] for r in l):.3f}")

    print("\nPER RUN, new prompt: AUROC strict-vs-clean (runs with >= 20 sampled hacks)")
    runs = sorted({r["run"] for r in rows})
    judges = sorted(by_judge)
    print(f"{'run':34s} " + " ".join(f"{j[:14]:>14s}" for j in judges))
    for run in runs:
        cells = []
        for j in judges:
            ok = [r for r in by_judge[j] if r["run"] == run and r["class"] in ("strict", "clean")
                  and isinstance(r["new"], (int, float)) and r["new"] == r["new"]]
            h = sum(r["class"] == "strict" for r in ok)
            cells.append(f"{roc_auc([r['new'] for r in ok], [r['class'] == 'strict' for r in ok]):14.3f}" if h >= 20 and len(ok) - h >= 20
                         else f"{'(' + str(h) + ' hacks)':>14s}")
        print(f"{run:34s} " + " ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run dirs or globs containing eval_rollouts.jsonl")
    ap.add_argument("--config", help="matrix config whose kind=cot monitors form the judge battery")
    ap.add_argument("--out", default="data/runs/rescored_eval_rollouts.jsonl")
    ap.add_argument("--hacks", type=int, default=8, help="strict hacks per (run, step)")
    ap.add_argument("--clean", type=int, default=8, help="clean rollouts per (run, step)")
    ap.add_argument("--loose", type=int, default=2, help="loose-only rollouts per (run, step)")
    ap.add_argument("--steps", default=None, help="comma-separated eval steps to keep (default all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="print the sample composition, no API calls")
    ap.add_argument("--report", default=None, help="only print the report for this output file")
    args = ap.parse_args()

    if args.report:
        report(Path(args.report))
        return 0
    if not args.runs or not args.config:
        ap.error("runs and --config are required unless --report is given")
    run_dirs = [Path(p) for pat in args.runs for p in sorted(globmod.glob(pat)) if (Path(p) / "eval_rollouts.jsonl").exists()]
    if not run_dirs:
        print("no run dir with eval_rollouts.jsonl matched", args.runs)
        return 1
    steps = {int(s) for s in args.steps.split(",")} if args.steps else None
    selected = select(run_dirs, hacks=args.hacks, clean=args.clean, loose=args.loose, seed=args.seed, steps=steps)
    comp = defaultdict(int)
    for _, run, step, _, c, _ in selected:
        comp[(run, c)] += 1
    print(f"{len(selected)} rollouts selected from {len(run_dirs)} run(s):")
    for run in sorted({r for r, _ in comp}):
        print(f"  {run:34s} " + "  ".join(f"{c}={comp[(run, c)]}" for c in ("strict", "clean", "loose")))
    from monitordecorrelation.experiment_config import load_monitor_specs
    from monitordecorrelation.monitors.judge_reasoning import resolve_reasoning
    specs = [s for s in load_monitor_specs(args.config) if s.kind == "cot"]
    print(f"{len(specs)} judges from {args.config}: " + ", ".join(
        f"{s.name}({s.model_id}, cot={s.use_cot}, out={s.use_output}, "
        f"reasoning={resolve_reasoning(s.model_id, s.reasoning, monitor=s.name)})" for s in specs))
    print(f"-> {len(selected) * len(specs)} judge calls")
    if args.dry_run:
        return 0
    judges = build_judges(args.config)  # needs OPENROUTER_API_KEY
    score_all(selected, judges, Path(args.out), args.workers)
    report(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
