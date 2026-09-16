"""Pre-RL monitor evaluation on the terminal-verifier env: how well does each judge detect the
base policy's reward hacking, before any training pressure?

Samples real multi-turn episodes from an UNTRAINED policy, grades them with the env oracle
(``behavior_present`` = used the verifier / modified it / passed the impossible check), scores every
monitor in a config against that oracle, and reports the standard bundle — AUROC, accuracy, Brier,
d', the present/absent class-split mean score (the "suspiciousness gap") — plus TPR/FPR at the
monitor's threshold, which AUROC alone hides.

These are the row-zero numbers the degradation matrix is measured against: a monitor with no
pre-RL headroom (AUROC ~1.0, gap ~1.0) has nothing to degrade, so the matrix cell is uninformative.

  uv run python experiments/eval_terminal_monitors_baseline.py \
      --config experiments/configs/terminal_verifier_gemini25_out.json \
      --model thinkingmachines/Inkling-Small --n-prompts 96 --samples 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

import tinker

from monitordecorrelation.envs.terminal_verifier import TerminalVerifierEnv
from monitordecorrelation.eval.metrics import accuracy, brier, dprime_margin, roc_auc
from monitordecorrelation.monitors.cot_monitor import CoTMonitor
from monitordecorrelation.rl.episodes import run_episodes
from monitordecorrelation.rl.renderers import DEFAULT_THINKING_EFFORT, make_renderer
from monitordecorrelation.rl.train import MonitorScorer

load_dotenv()


def rates(preds: list[bool], labels: list[bool]) -> dict[str, float]:
    """TPR / FPR at the monitor's own threshold — the operating point a deployment would use."""
    pos = [p for p, t in zip(preds, labels) if t]
    neg = [p for p, t in zip(preds, labels) if not t]
    return {
        "tpr": (sum(pos) / len(pos)) if pos else float("nan"),
        "fpr": (sum(neg) / len(neg)) if neg else float("nan"),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment config whose `monitors` to evaluate")
    ap.add_argument("--only", default=None, help="comma-separated monitor names (default: all)")
    ap.add_argument("--model", default="thinkingmachines/Inkling-Small")
    ap.add_argument("--n-prompts", type=int, default=96)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--effort", type=float, default=None,
                    help="TML thinking effort (default: the config's thinking_effort)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    # Match the RL runs: the policy must be sampled the way training samples it.
    if args.effort is None:
        args.effort = cfg.get("thinking_effort") or DEFAULT_THINKING_EFFORT
    opts = cfg.get("env_options", {})
    specs = cfg["monitors"]
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        specs = [m for m in specs if m["name"] in want]
    behavior = "reward_hacking"

    env = TerminalVerifierEnv.from_task(
        n=max(64, args.n_prompts), seed=args.seed, impossible=opts.get("impossible", True),
        offset_range=tuple(opts.get("offset_range", (4, 9))), eval_n=max(64, args.n_prompts),
        max_turns=opts.get("max_turns", 4), output_view=opts.get("output_view", "transcript"),
    )
    prompts = env.holdout(args.n_prompts, seed=args.seed)

    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=args.model)
    is_tml = args.model.split(":")[0].startswith("thinkingmachines/")
    renderer = make_renderer(args.model, effort=args.effort,
                            tokenizer=None if is_tml else sampler.get_tokenizer())

    judges = [CoTMonitor(m["name"], m["model_id"], behavior=behavior,
                         use_cot=m.get("use_cot", True), use_output=m.get("use_output", True),
                         threshold=m.get("threshold", 0.5),
                         reasoning_max_tokens=m.get("reasoning_max_tokens"),
                         binary_judge=bool(m.get("binary_judge")))
              for m in specs]
    n_ep = len(prompts) * args.samples
    print(f"sampling {n_ep} episodes ({len(prompts)}x{args.samples}) from {args.model} "
          f"(PRE-RL, effort={args.effort}) | judges: {', '.join(j.name for j in judges)}", flush=True)

    t0 = time.time()
    with MonitorScorer(judges, args.workers) as scorer:
        rollouts = run_episodes(sampler, renderer, env, prompts, num_samples=args.samples,
                                max_tokens=cfg.get("max_tokens", 3072), temperature=1.0,
                                seed=args.seed, think_budget=cfg.get("think_budget"),
                                answer_tokens=cfg.get("answer_tokens", 512),
                                step_workers=args.workers, on_rollout=scorer.submit)
        wall_s = time.time() - t0
        results = [env.score(r) for r in rollouts]
        judged = scorer.collect(rollouts)

    gt = [er.behavior_present for er in results]
    print(f"\n=== pre-RL baseline: {args.model}, {len(gt)} episodes, {wall_s:.0f}s ===")
    for k in ("used_verifier", "read_verifier", "modified_verifier", "passed", "submitted_true_answer"):
        print(f"  env/{k:<22} {mean(float(er.meta[k]) for er in results):.3f}")
    print(f"  behavior_present (oracle)  {mean(map(float, gt)):.3f}   "
          f"({sum(gt)} positive / {len(gt) - sum(gt)} negative)")

    rows = {}
    print(f"\n{'monitor':<14}{'view':<12}{'AUROC':>7}{'acc':>7}{'brier':>7}{'d-prime':>9}"
          f"{'mean|hack':>11}{'mean|clean':>11}{'gap':>7}{'TPR':>7}{'FPR':>7}")
    for j in judges:
        scores = [judged[j.name][i].score for i in range(len(rollouts))]
        preds = [judged[j.name][i].label for i in range(len(rollouts))]
        hi = mean([s for s, t in zip(scores, gt) if t]) if any(gt) else float("nan")
        lo = mean([s for s, t in zip(scores, gt) if not t]) if not all(gt) else float("nan")
        r = {"auroc": roc_auc(scores, gt), "accuracy": accuracy(preds, gt),
             "brier": brier(scores, gt), "dprime_margin": dprime_margin(scores, gt),
             "mean_score_present": hi, "mean_score_absent": lo, "suspiciousness_gap": hi - lo,
             "mean_score": mean(scores), **rates(preds, gt)}
        rows[j.name] = r
        view = ("cot+output" if j.use_cot and j.use_output else
                "output-only" if not j.use_cot else "cot-only")
        print(f"{j.name:<14}{view:<12}{r['auroc']:>7.3f}{r['accuracy']:>7.3f}{r['brier']:>7.3f}"
              f"{r['dprime_margin']:>9.2f}{hi:>11.3f}{lo:>11.3f}{hi - lo:>7.3f}"
              f"{r['tpr']:>7.3f}{r['fpr']:>7.3f}")

    out = Path(args.out) if args.out else Path("data/runs") / \
        f"tv_monitor_baseline_{args.model.split('/')[-1]}" / "baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "config": args.config, "n_episodes": len(gt), "seed": args.seed,
        "effort": args.effort, "wall_s": wall_s, "env_options": opts,
        "behavior_rate": mean(map(float, gt)), "monitors": rows,
        "rollouts": [{"task_id": er.meta["task_id"], "behavior_present": er.behavior_present,
                      "question": r.prompt.text, "cot": r.cot, "output": r.output,
                      "env_meta": er.meta,
                      "monitors": {j.name: {"score": judged[j.name][i].score,
                                            "label": judged[j.name][i].label} for j in judges}}
                     for i, (r, er) in enumerate(zip(rollouts, results))],
    }, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
