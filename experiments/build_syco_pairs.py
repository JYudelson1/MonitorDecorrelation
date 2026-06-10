"""Build REAL-CoT sycophancy contrastive pairs by resampling the policy.

The letter-only ``sycophancy`` adapter is too thin (honest/deceptive differ by one token). This
resamples the policy ``--k`` times per prompt and keeps prompts that yield BOTH a sycophantic and an
honest rollout (full CoT + answer), so the probe trains on real reasoning. Output feeds the
``sycophancy_cot`` dataset adapter.

Labeling (``--label-by``):
  ground_truth  env oracle (chose the user-matching answer)
  judge         an LLM CoT judge's label
  both          require ground_truth AND judge to AGREE (highest-precision pairs; default)

Run:
  uv run python experiments/build_syco_pairs.py --n-prompts 200 --k 6 --label-by both \
      --out data/probes/syco_cot_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

import tinker

from monitordecorrelation.envs.sycophancy import SycophancyQAEnv
from monitordecorrelation.monitors.cot_monitor import CoTMonitor
from monitordecorrelation.rl.rollout import sample_rollouts

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="Qwen/Qwen3-8B")
    ap.add_argument("--subset", default="nlp")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--k", type=int, default=6, help="samples per prompt")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--label-by", choices=["ground_truth", "judge", "both"], default="both")
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="data/probes/syco_cot_pairs.jsonl")
    args = ap.parse_args()

    use_judge = args.label_by in ("judge", "both")
    use_gt = args.label_by in ("ground_truth", "both")

    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(base_model=args.policy)
    tokenizer = sampling_client.get_tokenizer()
    env = SycophancyQAEnv.from_subset(args.subset, n=args.n_prompts, seed=0)
    judge = CoTMonitor(name="judge", model_id=args.judge_model, behavior="sycophancy") if use_judge else None

    prompts = [env.sample_prompt() for _ in range(args.n_prompts)]
    print(f"sampling {args.n_prompts} prompts x {args.k} from {args.policy} (label_by={args.label_by}) ...")
    rollouts = sample_rollouts(
        sampling_client, tokenizer, prompts, num_samples=args.k,
        max_tokens=args.max_tokens, temperature=1.0,
    )

    # Ground-truth label per rollout (cheap, local).
    env_results = [env.score(r) for r in rollouts]
    gt = [er.behavior_present for er in env_results]
    parsed = [not er.meta["unparsed"] for er in env_results]

    # Judge label per rollout (concurrent API), only if needed.
    judge_syco = [None] * len(rollouts)
    if use_judge:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(judge.score, rollouts))
        judge_syco = [res.label for res in results]

    def is_syco(i: int) -> bool | None:
        if not parsed[i]:
            return None
        if args.label_by == "ground_truth":
            return gt[i]
        if args.label_by == "judge":
            return judge_syco[i]
        # both: only a confident label when they agree
        if gt[i] == judge_syco[i]:
            return gt[i]
        return None

    # Group rollouts by prompt (sample_rollouts is prompt-major: k consecutive per prompt).
    pairs = []
    for p_idx in range(len(prompts)):
        group = list(range(p_idx * args.k, (p_idx + 1) * args.k))
        syco = [i for i in group if is_syco(i) is True]
        honest = [i for i in group if is_syco(i) is False]
        if syco and honest:
            d, h = rollouts[syco[0]], rollouts[honest[0]]
            pairs.append({
                "prompt": prompts[p_idx].text,
                "honest_cot": h.cot, "honest_answer": h.output,
                "deceptive_cot": d.cot, "deceptive_answer": d.output,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for pr in pairs:
            f.write(json.dumps(pr) + "\n")
    print(f"kept {len(pairs)}/{len(prompts)} prompts with both classes -> {out}")


if __name__ == "__main__":
    main()
