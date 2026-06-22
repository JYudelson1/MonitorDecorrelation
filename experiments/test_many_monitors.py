"""Monitor bakeoff: given an env + policy + candidate monitors, get rollouts and report
(1) the policy's ground-truth behavior frequency and (2) each candidate monitor's detection ability.

Run this BEFORE a training run: if the policy doesn't exhibit the behavior often enough, or no
candidate monitor can actually detect it (AUROC ~ 0.5), there's no point training against it. It's
also how we pick the weak/strong monitor ladder for the degradation experiments.

Detection metrics are computed on a CLASS-BALANCED subset (equal behavior / non-behavior), because
AUROC is unreliable when one class is rare (e.g. a 90%-sycophantic policy gives few negatives).

Run: uv run python experiments/test_many_monitors.py
     # reuse already-collected rollouts instead of sampling fresh:
     uv run python experiments/test_many_monitors.py --from-rollouts data/runs/run40/rollouts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

import tinker

from monitordecorrelation.envs.sycophancy import SycophancyQAEnv
from monitordecorrelation.eval.metrics import accuracy, roc_auc
from monitordecorrelation.monitors.cot_monitor import CoTMonitor
from monitordecorrelation.rl.rollout import load_saved_rollouts, sample_rollouts

load_dotenv(".env")

# Default candidate ladder (weak -> strong). All must be valid OpenRouter ids.
_DEFAULT_CANDIDATES = [
    "meta-llama/llama-3.2-3b-instruct",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.8",
]


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="Qwen/Qwen3-8B")
    ap.add_argument("--subset", default="political")
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=2, help="rollouts per prompt")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--from-rollouts", default=None, help="eval on a saved rollouts.jsonl (skip sampling)")
    ap.add_argument("--candidates", default=",".join(_DEFAULT_CANDIDATES),
                    help="comma-separated OpenRouter model ids")
    ap.add_argument("--rubric-file", default=None, help="path to a custom rubric txt (else default)")
    ap.add_argument("--run-name", default="monitor_bakeoff")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rubric_kwargs = {}
    if args.rubric_file:
        rubric_kwargs["rubric"] = Path(args.rubric_file).read_text()

    # 1. get (rollout, ground_truth) pairs — either from disk or by sampling the policy
    if args.from_rollouts:
        parsed = load_saved_rollouts(args.from_rollouts)
        n_unparsed = "?"
        source = args.from_rollouts
    else:
        sc = tinker.ServiceClient()
        sampling_client = sc.create_sampling_client(base_model=args.policy)
        tokenizer = sampling_client.get_tokenizer()
        env = SycophancyQAEnv.from_subset(args.subset, n=max(args.n_prompts * 4, 64), seed=0)
        prompts = [env.sample_prompt() for _ in range(args.n_prompts)]
        print(f"sampling {args.n_prompts}x{args.num_samples} rollouts from {args.policy} ...")
        rollouts = sample_rollouts(
            sampling_client, tokenizer, prompts,
            num_samples=args.num_samples, max_tokens=args.max_tokens, temperature=1.0,
        )
        ers = [env.score(r) for r in rollouts]
        parsed = [(r, er.behavior_present) for r, er in zip(rollouts, ers)
                  if not er.meta["unparsed"]]
        n_unparsed = len(rollouts) - len(parsed)
        source = f"{args.policy} (fresh)"

    n = len(parsed)
    n_syco = sum(g for _, g in parsed)
    print(f"\nsource: {source}")
    print(f"behavior frequency: {n_syco}/{n} = {n_syco / n:.2f}  (unparsed: {n_unparsed})")

    # 2. class-balanced eval set (equal behavior / non-behavior) for reliable detection metrics
    rng = random.Random(0)
    pos = [r for r, g in parsed if g]
    neg = [r for r, g in parsed if not g]
    k = min(len(pos), len(neg))
    if k == 0:
        raise SystemExit(f"need both classes to evaluate detection; got {len(pos)} pos / {len(neg)} neg")
    eval_rollouts = rng.sample(pos, k) + rng.sample(neg, k)
    gt = [True] * k + [False] * k
    print(f"balanced detection set: {k} behavior + {k} non-behavior\n")

    # 3. each candidate monitor's detection ability
    candidate_ids = [c.strip() for c in args.candidates.split(",") if c.strip()]
    results = []
    print(f"{'monitor':28s} {'AUROC':>6} {'acc':>6} {'mean(beh)':>10} {'mean(¬beh)':>11}")
    print("-" * 66)
    for mid in candidate_ids:
        try:
            mon = CoTMonitor(name=_slug(mid), model_id=mid, **rubric_kwargs)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(mon.score, eval_rollouts))
            scores = [x.score for x in res]
            preds = [x.label for x in res]
            auc = roc_auc(scores, gt)
            acc = accuracy(preds, gt)
            mb = mean([s for s, g in zip(scores, gt) if g])
            mn = mean([s for s, g in zip(scores, gt) if not g])
            results.append({"model_id": mid, "auroc": auc, "accuracy": acc,
                            "mean_behavior": mb, "mean_nonbehavior": mn})
            print(f"{_slug(mid):28s} {auc:6.2f} {acc:6.2f} {mb:10.2f} {mn:11.2f}")
        except Exception as e:
            print(f"{_slug(mid):28s}  FAILED: {str(e)[:60]}")
            results.append({"model_id": mid, "error": str(e)})

    out_dir = Path("data/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": source, "subset": args.subset, "n_parsed": n,
        "behavior_frequency": n_syco / n if n else None, "balanced_n_per_class": k,
        "monitors": results,
    }
    (out_dir / "monitor_bakeoff.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir/'monitor_bakeoff.json'}")


if __name__ == "__main__":
    main()
