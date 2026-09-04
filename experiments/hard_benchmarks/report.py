"""Aggregate data/hard_benchmarks/*.summary.json into one table (markdown to stdout + report.md)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "data/hard_benchmarks")
LABEL = {
    "lcb_all": "LiveCodeBench v6 (1055)", "aethercode": "AetherCode v1_2024 (399)", "icpc_eval": "ICPC-Eval (117)",
    "usaco": "USACO bench (520)", "leetcode": "LeetCodeDataset (2869)", "taco_verified": "TACO-verified (12898)",
    "codeforces": "open-r1 Codeforces verifiable (8594)", "ojbench": "OJBench (215)",
}
ORDER = ["lcb_all", "leetcode", "usaco", "taco_verified", "codeforces", "ojbench", "aethercode", "icpc_eval"]


def main():
    rows = []
    for f in sorted(glob.glob(str(DIR / "*.summary.json"))):
        s = json.load(open(f))
        rows.append(s)
    rows.sort(key=lambda s: (ORDER.index(s["benchmark"]) if s["benchmark"] in ORDER else 99, s["subset"] != "all", -len(s["subset"])))
    hdr = ("| benchmark | subset | difficulty of the 64 sampled problems | pass rate (mean of per-problem rates, k completions) | ± s.e. | problems with ≥1 pass | problems solved 0<rate<1 | completions truncated (hit 32768) | prompts truncated | no code block | median / p90 / max output tokens |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for s in rows:
        ot = s["output_tokens"]
        lines.append(
            f"| {LABEL.get(s['benchmark'], s['benchmark'])} | {s['subset']} | {_hist(s['difficulty_hist'])} | "
            f"{100 * s['pass_rate']:.1f}% (n={s['n_problems']}, k={s['k']}) | {100 * s['pass_rate_se']:.1f} | "
            f"{100 * s['frac_problems_any_pass']:.0f}% | {100 * s['frac_problems_mixed']:.0f}% | "
            f"{s['completions_truncated']}/{s['n_completions']} | {s['prompts_truncated']}/{s['n_prompts']} | "
            f"{s['completions_no_code']}/{s['n_completions']} | {ot['median']} / {ot['p90']} / {ot['max']} |"
        )
    out = "\n".join(lines)
    print(out)
    (DIR / "report.md").write_text(out + "\n")


def _hist(h):
    return ", ".join(f"{k}:{v}" for k, v in sorted(h.items(), key=lambda kv: _key(kv[0])))


def _key(k):
    try:
        return (0, float(k))
    except ValueError:
        return (1, k)


if __name__ == "__main__":
    main()
