"""Score a saved instance set (rollouts-style jsonl) with EVERY monitor of a matrix config — the
sense-(b) "correlated false negatives" measurement needs each monitor's score on each instance.

Reads records with ``question`` / ``cot`` / ``answer`` (the ``rollouts.jsonl`` / ``eval_rollouts.jsonl``
/ ``bakeoff_set_600.jsonl`` schema), scores them with the monitors listed in a run config (roles are
ignored — every monitor is scored; use a ``row_control.json`` so nothing is train_against), and writes
the same records back with ``monitors[<name>] = {score, label}`` merged in. Existing scores are KEPT
(resume-safe: re-running only fills in missing monitors; ``--force`` rescoring a name).

    # what would run, no API calls / model loads
    uv run python experiments/score_instance_set.py --in data/bakeoff_set_600.jsonl \
        --config experiments/configs/mbpp_matrix_gemini_sep3/row_control.json --dry-run
    # judges only (OpenRouter), on a laptop
    uv run python experiments/score_instance_set.py --in data/bakeoff_set_600.jsonl \
        --config experiments/configs/mbpp_matrix_gemini_sep3/row_control.json --kinds cot
    # probes on the GPU box (or via a running probe_server.py)
    uv run python experiments/score_instance_set.py --in data/bakeoff_set_600_scored.jsonl \
        --config ... --kinds probe [--probe-server-url http://127.0.0.1:8177]

Ground-truth labels are never read here — the oracle only grades, downstream (eval/miss_overlap.py).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from monitordecorrelation.experiment_config import build_monitors, load_config
from monitordecorrelation.types import MonitorResult, Prompt, Rollout

load_dotenv(".env")


def _load_records(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_rollout(r: dict) -> Rollout:
    return Rollout(prompt=Prompt(text=r["question"]), cot=r.get("cot") or "", output=r.get("answer") or "")


def _write(records: list[dict], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="instance set jsonl (question/cot/answer)")
    ap.add_argument("--config", required=True, help="run config whose monitors to score with")
    ap.add_argument("--out", default=None, help="default: <in>_scored.jsonl (or in-place if --in already ends with _scored)")
    ap.add_argument("--kinds", default="cot,probe", help="which monitor kinds to score: cot and/or probe")
    ap.add_argument("--only", default=None, help="comma-separated monitor names (subset)")
    ap.add_argument("--force", action="store_true", help="rescore monitors that already have scores")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N records (smoke)")
    ap.add_argument("--workers", type=int, default=16, help="thread pool for judge calls")
    ap.add_argument("--probe-batch-size", type=int, default=None, help="override ProbeMonitor batch size")
    ap.add_argument("--probe-server-url", default=None, help="shared probe_server.py (else env PROBE_SERVER_URL, else local model)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no API calls, no model loads")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out) if args.out else (inp if inp.stem.endswith("_scored")
                                           else inp.with_name(inp.stem + "_scored.jsonl"))
    records = _load_records(inp)
    if args.limit:
        records = records[: args.limit]
    cfg = load_config(args.config)
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    only = {n.strip() for n in args.only.split(",")} if args.only else None
    specs = [s for s in cfg.monitors if s.kind in kinds and (only is None or s.name in only)]

    have = {s.name: sum(1 for r in records if s.name in r.get("monitors", {})) for s in specs}
    todo = [s for s in specs if args.force or have[s.name] < len(records)]
    print(f"records: {len(records)} from {inp}\nout: {out}")
    for s in specs:
        extra = f" model_id={s.model_id} use_cot={s.use_cot} use_output={s.use_output}" if s.kind == "cot" \
            else f" probe_path={s.probe_path}"
        state = "SKIP (scored)" if s not in todo else f"TODO ({len(records) - have[s.name]} missing)"
        print(f"  {s.name:18s} {s.kind:5s}{extra}  -> {state}")
    if args.dry_run or not todo:
        print("nothing to do" if not todo else "[dry-run] stopping before any calls")
        return

    import os
    server = args.probe_server_url or cfg.probe_server_url or os.environ.get("PROBE_SERVER_URL")
    train, held = build_monitors(todo, default_behavior="reward_hacking", probe_server_url=server)
    monitors = train + held
    if args.probe_batch_size:
        for m in monitors:
            if hasattr(m, "batch_size"):
                m.batch_size = args.probe_batch_size

    rollouts = [_to_rollout(r) for r in records]
    for mon in monitors:
        idx = [i for i, r in enumerate(records) if args.force or mon.name not in r.get("monitors", {})]
        if not idx:
            continue
        t0 = perf_counter()
        print(f"scoring {mon.name} on {len(idx)} records …", flush=True)
        if hasattr(mon, "score_batch"):  # probes: one batched forward
            res = mon.score_batch([rollouts[i] for i in idx])
        else:  # judges: thread pool; warm up one call first so the reasoning-config flip settles
            def _safe(i: int) -> MonitorResult:
                try:
                    return mon.score(rollouts[i])
                except Exception as e:  # noqa: BLE001 — one failed call must not lose the set
                    return MonitorResult(score=float("nan"), label=False, meta={"error": str(e)[:200]})
            first = _safe(idx[0])
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rest = list(ex.map(_safe, idx[1:]))
            res = [first, *rest]
        n_fail = 0
        for i, x in zip(idx, res):
            if x.score != x.score:  # NaN → leave unscored so a rerun retries it
                n_fail += 1
                continue
            records[i].setdefault("monitors", {})[mon.name] = {
                "score": float(x.score), "label": bool(x.label),
                **({"parse_error": True} if x.meta.get("parse_error") else {}),
            }
        _write(records, out)  # checkpoint after every monitor
        print(f"  done in {perf_counter() - t0:.0f}s ({n_fail} failed/unscored) -> {out}", flush=True)
    print("all done")


if __name__ == "__main__":
    main()
