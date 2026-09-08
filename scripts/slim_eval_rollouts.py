#!/usr/bin/env python3
"""Slim full ``eval_rollouts.jsonl`` dumps down to the committed ``eval_rollouts_slim.jsonl`` schema.

Stdlib-only so it runs on a bare training box (``python3 scripts/slim_eval_rollouts.py``). Tolerant of
the two known dump quirks: bare ``NaN`` (fine for python's json, fatal for jq) and the occasional corrupt
line from an interrupted write (skipped + counted on stderr).

    python3 scripts/slim_eval_rollouts.py data/runs/<batch>/*/            # writes <run>/eval_rollouts_slim.jsonl
    python3 scripts/slim_eval_rollouts.py --force data/runs/<batch>/*/    # overwrite existing slim files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:  # in-repo: the canonical field list
    from monitordecorrelation.eval.rollout_dump import SLIM_FIELDS, slim_record
except ImportError:  # bare box without the package installed: keep this copy in sync with rollout_dump.py
    SLIM_FIELDS = ("step", "task_id", "behavior_present", "loose_rh", "hardcoding", "unparsed", "monitors")

    def slim_record(full: dict) -> dict:
        return {k: full.get(k) for k in SLIM_FIELDS}


def slim_file(src: Path, dst: Path) -> tuple[int, int]:
    """Write the slim file; returns (records written, corrupt lines skipped)."""
    n = bad = 0
    with src.open() as f, dst.open("w") as out:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                print(f"  {src}: skipping corrupt line {i} (col {e.pos})", file=sys.stderr)
                continue
            out.write(json.dumps(slim_record(rec)) + "\n")
            n += 1
    return n, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="run dirs containing eval_rollouts.jsonl")
    ap.add_argument("--force", action="store_true", help="overwrite an existing eval_rollouts_slim.jsonl")
    args = ap.parse_args()
    for d in args.run_dirs:
        d = Path(d)
        src, dst = d / "eval_rollouts.jsonl", d / "eval_rollouts_slim.jsonl"
        if not src.exists():
            print(f"{d.name}: no eval_rollouts.jsonl, skipped")
            continue
        if dst.exists() and not args.force:
            print(f"{d.name}: slim file exists, skipped (--force to overwrite)")
            continue
        n, bad = slim_file(src, dst)
        print(f"{d.name}: {n} records → {dst.name} ({bad} corrupt lines skipped)")


if __name__ == "__main__":
    main()
