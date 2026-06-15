"""Is the haiku lie-oracle reliable? Re-judge the on-policy pairs' responses with a STRONG judge
(sonnet-4.6) and measure agreement with haiku's labels (deceptive=lie, honest=not-lie).

Low agreement => the ~0.61 subtle-MASK probe weakness is partly LABEL NOISE, not just signal
difficulty (the blatant probe used ground-truth labels, so it's unaffected). Tests Joey's hypothesis.

  uv run python experiments/check_mask_judge.py --pairs data/probes/mask_cot_pairs_qwen3-8b.jsonl
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from monitordecorrelation.envs.mask import MaskEnv, MaskItem
from monitordecorrelation.types import Prompt, Rollout

load_dotenv(".env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/probes/mask_cot_pairs_qwen3-8b.jsonl")
    ap.add_argument("--config", default="provided_facts")
    ap.add_argument("--strong-judge", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("cais/MASK", args.config, split="test")
    lut = {}  # prompt text -> (proposition, formatted_ground_truth)
    for row in ds:
        if row.get("type") != "binary":
            continue
        key = f"{row['system_prompt']}\n\n{row['user_prompt']}"
        lut[key] = (row["proposition"], row.get("formatted_ground_truth"))

    pairs = [json.loads(l) for l in open(args.pairs)]
    matched = [p for p in pairs if p["prompt"] in lut]
    print(f"matched {len(matched)}/{len(pairs)} pairs to {args.config} items")

    # Strong judge as the re-labeling oracle (env.score reads prop/gt from the rollout's prompt.meta).
    env = MaskEnv([MaskItem("x", "s", "u", "p <pivotal>a/b</pivotal>", "A")], judge_model=args.strong_judge)

    def relabel(answer: str, prop: str, gt: str):
        r = Rollout(prompt=Prompt(text="", meta={"proposition": prop, "ground_truth": gt}), cot="", output=answer)
        res = env.score(r)
        return res.behavior_present, res.meta.get("evasive", False)  # (lied?, evasive?)

    # Re-judge every deceptive (expect lie=True) and honest (expect lie=False) answer.
    jobs = []  # (kind, answer, prop, gt)
    for p in matched:
        prop, gt = lut[p["prompt"]]
        jobs.append(("dec", p["deceptive_answer"], prop, gt))
        jobs.append(("hon", p["honest_answer"], prop, gt))
    print(f"re-judging {len(jobs)} responses with {args.strong_judge} ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda j: (j[0], *relabel(j[1], j[2], j[3])), jobs))

    dec = [(lie, ev) for k, lie, ev in results if k == "dec"]
    hon = [(lie, ev) for k, lie, ev in results if k == "hon"]
    dec_agree = sum(1 for lie, ev in dec if lie) / len(dec)  # strong judge ALSO says lie
    hon_agree = sum(1 for lie, ev in hon if not lie) / len(hon)  # strong judge ALSO says not-lie
    dec_evasive = sum(1 for lie, ev in dec if ev) / len(dec)
    hon_evasive = sum(1 for lie, ev in hon if ev) / len(hon)
    print(f"\nDECEPTIVE (haiku said LIE):    strong judge agrees {dec_agree:.0%}  (calls {dec_evasive:.0%} evasive/C)")
    print(f"HONEST    (haiku said NOT-lie): strong judge agrees {hon_agree:.0%}  (calls {hon_evasive:.0%} evasive/C)")
    overall = (sum(1 for lie, ev in dec if lie) + sum(1 for lie, ev in hon if not lie)) / (len(dec) + len(hon))
    print(f"\nOVERALL label agreement: {overall:.0%}")
    print("interpretation: high (>90%) -> labels reliable, subtle signal is genuinely hard; "
          "low -> judge noise inflates the probe's weakness.")


if __name__ == "__main__":
    main()
