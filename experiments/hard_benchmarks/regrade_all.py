"""Re-grade every finished run's saved completions with the current grader and rewrite summaries.

    uv run python experiments/hard_benchmarks/regrade_all.py [--only lcb_all__all__n64_k4_s0 ...]

The run jsonl keeps each completion's answer text, so grading is fully reproducible offline; this
exists because the first runs graded under CPU oversubscription (spurious timeouts).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grader import extract_code, grade  # noqa: E402
from loaders import LOADERS, POST_SELECT_HOOKS, PREPARE  # noqa: E402
from run_eval import summarize  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data" / "hard_benchmarks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    files = sorted(glob.glob(str(OUT / "*.jsonl")))
    if args.only:
        files = [f for f in files if Path(f).stem in args.only]
    cache: dict[str, dict] = {}
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        bench = rows[0]["benchmark"]
        if bench not in cache:
            cache[bench] = {p.task_id: p for p in LOADERS[bench]()}
        probs = cache[bench]
        unknown = {r["task_id"] for r in rows} - set(probs)
        if unknown:  # problems the current loader refuses (unsafe checker, checker did not compile)
            print(f"  dropping {len(unknown)} problems no longer loadable: {sorted(unknown)}", flush=True)
            rows = [r for r in rows if r["task_id"] not in unknown]
        chosen = [probs[t] for t in {r["task_id"] for r in rows}]
        if bench in PREPARE:  # lazy tests (codeforces per-contest parquet, ojbench zips): always attach
            PREPARE[bench](chosen)
        else:
            hook = POST_SELECT_HOOKS.get(bench)
            if hook is not None:
                hook([p for p in chosen if not p.inputs])
        print(f"re-grading {Path(f).name}: {len(rows)} completions", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(grade, extract_code(r["answer"]), probs[r["task_id"]]) for r in rows]
            for r, fu in zip(rows, futs):
                g = fu.result()
                r.update(passed=g.passed, n_passed=g.n_passed, status=g.status, detail=g.detail[:300],
                         n_tests=probs[r["task_id"]].n_tests)
        with open(f, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        a = SimpleNamespace(benchmark=bench, subset=rows[0]["subset"], model="thinkingmachines/Inkling-Small",
                            effort=0.5, k=max(r["sample"] for r in rows) + 1)
        n_prompts = len({r["task_id"] for r in rows})
        summ = summarize(rows, a, sum(1 for r in rows if r["prompt_truncated"] and r["sample"] == 0), n_prompts)
        summ["regraded"] = True
        with open(f.replace(".jsonl", ".summary.json"), "w") as fh:
            json.dump(summ, fh, indent=2)
        print(f"  {bench} {rows[0]['subset']}: pass={summ['pass_rate']:.3f} status={summ['status_counts']}", flush=True)


if __name__ == "__main__":
    main()
