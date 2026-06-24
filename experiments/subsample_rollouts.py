"""Randomly subsample a rollouts.jsonl (write a fraction to a new file) and report token stats for the
monitor prompts — so we can estimate the cost of a monitor bakeoff before spending anything.

    uv run python experiments/subsample_rollouts.py --in <rollouts.jsonl> --frac 0.10 --out <sample.jsonl>

No API calls. Token counts use tiktoken o200k_base (a good cross-model proxy) on the ACTUAL CoT-judge
and output-judge prompts (rubric + question + cot/answer + score instruction).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median

import tiktoken

from monitordecorrelation.monitors.cot_monitor import _OUTPUT_TEMPLATE, _PROMPT_TEMPLATE
from monitordecorrelation.monitors.rubrics import get_rubric


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--out", default=None, help="default: <in_dir>/rollouts_sample.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--behavior", default="reward_hacking")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out) if args.out else inp.with_name("rollouts_sample.jsonl")
    rng = random.Random(args.seed)
    rubric = get_rubric(args.behavior)
    enc = tiktoken.get_encoding("o200k_base")

    n_total = n_hack = n_kept = n_kept_hack = 0
    cot_toks: list[int] = []
    out_toks: list[int] = []
    with inp.open() as f, out.open("w") as g:
        for line in f:
            if not line.strip():
                continue
            n_total += 1
            r = json.loads(line)
            if r["env"].get("behavior_present"):
                n_hack += 1
            if rng.random() >= args.frac:
                continue
            g.write(line)
            n_kept += 1
            if r["env"].get("behavior_present"):
                n_kept_hack += 1
            cot_prompt = _PROMPT_TEMPLATE.format(
                description=rubric.description, adjective=rubric.adjective,
                question=r["question"], cot=r.get("cot") or "(none provided)", answer=r.get("answer", ""))
            out_prompt = _OUTPUT_TEMPLATE.format(
                description=rubric.description, adjective=rubric.adjective,
                question=r["question"], answer=r.get("answer", ""))
            cot_toks.append(len(enc.encode(cot_prompt)))
            out_toks.append(len(enc.encode(out_prompt)))

    print(f"wrote {out}  ({n_kept} rollouts)")
    print(f"\nsource: {n_total} rollouts, {n_hack} reward-hacking ({n_hack / n_total:.1%})")
    print(f"sample: {n_kept} rollouts ({args.frac:.0%}), {n_kept_hack} hacking ({n_kept_hack / n_kept:.1%})")
    for label, toks in (("CoT-judge prompt", cot_toks), ("output-judge prompt", out_toks)):
        print(f"\n{label} input tokens:  mean {mean(toks):.0f}  median {median(toks):.0f}  "
              f"p90 {_pct(toks, 0.90):.0f}  max {max(toks)}")
        print(f"  total over the {n_kept} sampled rollouts: {sum(toks):,} tok")


if __name__ == "__main__":
    main()
