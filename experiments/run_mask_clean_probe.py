"""Signal vs labels: relabel the on-policy MASK pairs with a STRONG judge (sonnet), keep only the
HIGH-CONFIDENCE pairs (sonnet confirms deceptive=lie & honest=not-lie, neither evasive), retrain the
probe. Does AUROC climb above the ~0.61 we got with noisy haiku labels?

  uv run python experiments/run_mask_clean_probe.py --pairs data/probes/mask_cot_pairs_qwen3-8b.jsonl --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv

from monitordecorrelation.envs.mask import MaskEnv, MaskItem
from monitordecorrelation.types import Prompt, Rollout
from monitordecorrelation.whitebox.datasets import ContrastivePair, flatten_pairs
from monitordecorrelation.whitebox.model import WhiteBoxModel
from monitordecorrelation.whitebox.probe import LinearProbe

load_dotenv(".env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/probes/mask_cot_pairs_qwen3-8b.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--config", default="provided_facts")
    ap.add_argument("--strong-judge", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--ce-threshold", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    follow_up = ("Did you respond truthfully?", "Yes")

    from datasets import load_dataset

    ds = load_dataset("cais/MASK", args.config, split="test")
    lut = {f"{r['system_prompt']}\n\n{r['user_prompt']}": (r["proposition"], r.get("formatted_ground_truth"))
           for r in ds if r.get("type") == "binary"}
    pairs = [p for p in (json.loads(l) for l in open(args.pairs)) if p["prompt"] in lut]
    print(f"matched {len(pairs)} pairs; relabeling with {args.strong_judge} ...")

    env = MaskEnv([MaskItem("x", "s", "u", "p <pivotal>a/b</pivotal>", "A")], judge_model=args.strong_judge)

    def verdict(answer: str, prop: str, gt: str):
        res = env.score(Rollout(prompt=Prompt(text="", meta={"proposition": prop, "ground_truth": gt}),
                                cot="", output=answer))
        return res.behavior_present, res.meta.get("evasive", False)  # (lied?, evasive?)

    def confirm(p):
        prop, gt = lut[p["prompt"]]
        with ThreadPoolExecutor(max_workers=2) as ex:
            (d_lie, d_ev), (h_lie, h_ev) = ex.map(lambda a: verdict(a, prop, gt),
                                                  [p["deceptive_answer"], p["honest_answer"]])
        return d_lie and not d_ev and (not h_lie) and not h_ev  # sonnet confirms both

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        keep_mask = list(ex.map(confirm, pairs))
    clean = [ContrastivePair(prompt=p["prompt"], honest=p["honest_answer"], deceptive=p["deceptive_answer"])
             for p, k in zip(pairs, keep_mask) if k]
    print(f"CLEAN (both judges agree, non-evasive): {len(clean)}/{len(pairs)} pairs kept")
    if len(clean) < 10:
        print("too few clean pairs to train a probe."); return

    model = WhiteBoxModel(args.model)
    bs = 1 if model.device == "mps" else 8
    q, c, a, y = flatten_pairs(clean)
    y = np.asarray(y)
    print(f"extracting {len(q)} activations ...")
    acts = model.extract_activations(list(zip(q, c, a)), follow_up=follow_up, batch_size=bs, progress=True)
    rng = np.random.default_rng(args.seed)
    n_pairs = len(clean)
    perm = rng.permutation(n_pairs)
    n_test = max(1, int(round(args.test_frac * n_pairs)))
    test_p = set(perm[:n_test].tolist())
    tr = [i for i in range(len(y)) if (i // 2) not in test_p]
    te = [i for i in range(len(y)) if (i // 2) in test_p]
    probe = LinearProbe(ce_threshold=args.ce_threshold).fit(acts[tr], y[tr])
    auroc = probe.evaluate(acts[te], y[te])
    print(f"\nCLEAN-LABEL probe AUROC: {auroc:.3f}  (n_clean={len(clean)} pairs, {n_test} test)")
    print("[reference: noisy-haiku-label probe was ~0.61]")
    print("interpretation: >> 0.61 -> labels were the problem; ~0.61 -> signal genuinely hard.")


if __name__ == "__main__":
    main()
