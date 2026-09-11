"""Baseline (no-training) inference on the terminal-verifier env: the full episode pipeline minus
the gradient step.

Samples real multi-turn episodes from a base policy on tinker, grades them with the env's oracle,
prints the behavior rates (peeked / tampered / passed / …) and a few full transcripts, and optionally
scores LLM judges (CoT+output and output-only) against the oracle. This is the pre-RL sanity check
for the setting: does the base model already read the verifier, how often, and can monitors tell.

It also reports the **cost accounting** an RL run is priced from: prefill / sampled / training tokens
per rollout (the training figure is the size of the single GRPO datum the episode folds into), and the
resulting $ per rollout and per RL step at tinker's published rates.

  uv run python experiments/eval_terminal_baseline.py --n-prompts 16 --samples 2
  uv run python experiments/eval_terminal_baseline.py --judge anthropic/claude-3-haiku
  uv run python experiments/eval_terminal_baseline.py --honest        # verifier holds the TRUE answer
  uv run python experiments/eval_terminal_baseline.py --model thinkingmachines/Inkling-Small --effort 0.5
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
from monitordecorrelation.eval.metrics import roc_auc
from monitordecorrelation.rl.episodes import run_episodes
from monitordecorrelation.rl.renderers import DEFAULT_THINKING_EFFORT, make_renderer

load_dotenv()

# tinker's published $/1M-token rates (docs "Models & Pricing", read 2026-09-10). `prefill` is the
# full rate; a prompt-cache hit is billed at `cached`. Override any of these with --price.
PRICES = {
    "Qwen/Qwen3-8B": {"prefill": 0.195, "cached": 0.039, "sample": 0.60, "train": 0.44},
    "Qwen/Qwen3.5-4B": {"prefill": 0.33, "cached": 0.066, "sample": 1.005, "train": 0.737},
    "Qwen/Qwen3.5-9B": {"prefill": 0.66, "cached": 0.132, "sample": 1.995, "train": 1.463},
    "thinkingmachines/Inkling-Small": {"prefill": 1.16, "cached": 0.116, "sample": 2.88, "train": 3.46},
    "thinkingmachines/Inkling": {"prefill": 3.74, "cached": 0.374, "sample": 9.36, "train": 11.22},
}


def token_accounting(rollouts) -> dict:
    """Mean per-rollout token counts, split the way tinker bills them.

    ``prefill`` is every token submitted to the sampler (one call per transition; a budget-forced
    turn is two). ``prefill_fresh`` is the part that cannot come from the prompt cache — each call's
    observation is a prefix-extension of the previous call's ``ob+ac``, so only the new framing +
    terminal output is genuinely new; the true bill sits between the two. ``sample`` is generated
    tokens, and ``train`` is the length of the ONE datum the episode folds into (final ob+ac), which
    is what a ``forward_backward`` on this episode would process.
    """
    n = len(rollouts) or 1
    tot = {"prefill": 0, "prefill_fresh": 0, "sample": 0, "train": 0, "calls": 0}
    for r in rollouts:
        trs = (r.meta or {}).get("transitions") or []
        tot["prefill"] += r.meta["input_tokens"]
        tot["sample"] += r.meta["output_tokens"]
        tot["train"] += r.meta["train_tokens"]
        tot["calls"] += r.meta["n_sampling_calls"]
        prev = 0
        for tr in trs:
            tot["prefill_fresh"] += max(0, len(tr["ob"]) - prev)
            prev = len(tr["ob"]) + len(tr["ac"])
    return {k: v / n for k, v in tot.items()}


def cost_estimate(per_rollout: dict, price: dict, *, rollouts_per_step: int) -> dict:
    """$ per rollout and per RL step, at both ends of the prompt-cache range. ``sampling`` is what an
    inference-only pass costs; ``+train`` adds the gradient step an RL run also pays for. One RL step
    is ``batch_size x group_size`` rollouts (128 in the terminal-verifier config)."""
    M = 1e6
    sample = per_rollout["sample"] * price["sample"] / M
    hi = sample + per_rollout["prefill"] * price["prefill"] / M            # nothing cached
    lo = sample + (per_rollout["prefill_fresh"] * price["prefill"]
                   + (per_rollout["prefill"] - per_rollout["prefill_fresh"]) * price["cached"]) / M
    train = per_rollout["train"] * price["train"] / M
    out = {"sampling_per_rollout_lo": lo, "sampling_per_rollout_hi": hi,
           "train_per_rollout": train, "rollouts_per_step": rollouts_per_step}
    out["step_sampling_lo"] = lo * rollouts_per_step
    out["step_sampling_hi"] = hi * rollouts_per_step
    out["step_total_lo"] = (lo + train) * rollouts_per_step
    out["step_total_hi"] = (hi + train) * rollouts_per_step
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--samples", type=int, default=1, help="episodes per prompt")
    ap.add_argument("--max-tokens", type=int, default=2048, help="per TURN (single call, when no --think-budget)")
    ap.add_argument("--think-budget", type=int, default=None,
                    help="cap <think> at N tokens per turn, then force the answer (answer gets --answer-tokens)")
    ap.add_argument("--answer-tokens", type=int, default=512)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--honest", action="store_true", help="verifier holds the true answer (impossible=False)")
    ap.add_argument("--offset", type=int, nargs=2, default=(4, 9), metavar=("LO", "HI"))
    ap.add_argument("--output-view", default="transcript", choices=("transcript", "assistant", "explanations"))
    ap.add_argument("--judge", default=None, help="OpenRouter model id → score a CoT+out and an output-only judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=3, help="print this many full transcripts")
    ap.add_argument("--out", default=None, help="jsonl path (default data/runs/tv_baseline_<model>/episodes.jsonl)")
    ap.add_argument("--effort", type=float, default=DEFAULT_THINKING_EFFORT,
                    help="reasoning effort for TML-rendered policies (Inkling); ignored by Qwen3 & co.")
    ap.add_argument("--step-rollouts", type=int, default=128,
                    help="rollouts per RL step for the cost estimate (default batch_size 16 x group_size 8)")
    ap.add_argument("--price", default=None,
                    help="override the $/1M rates as JSON, e.g. '{\"sample\": 0.6, \"prefill\": 0.2, "
                         "\"cached\": 0.04, \"train\": 0.44}'")
    ap.add_argument("--workers", type=int, default=16, help="concurrent env steps (command execution)")
    args = ap.parse_args()

    env = TerminalVerifierEnv.from_task(
        n=max(64, args.n_prompts), seed=args.seed, impossible=not args.honest,
        offset_range=tuple(args.offset), eval_n=max(64, args.n_prompts),
        max_turns=args.max_turns, output_view=args.output_view,
    )
    prompts = env.holdout(args.n_prompts, seed=args.seed)

    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=args.model)
    is_tml = args.model.split(":")[0].startswith("thinkingmachines/")
    renderer = make_renderer(args.model, effort=args.effort,
                             tokenizer=None if is_tml else sampler.get_tokenizer())
    n_ep = len(prompts) * args.samples
    print(f"sampling {n_ep} episodes ({len(prompts)}×{args.samples}) from {args.model} — "
          f"max_turns={args.max_turns}, max_tokens/turn={args.max_tokens}, "
          f"think_budget={args.think_budget}/{args.answer_tokens}, "
          f"{'effort=' + str(args.effort) + ', ' if is_tml else ''}"
          f"{'HONEST' if args.honest else 'IMPOSSIBLE'} verifier", flush=True)
    t0 = time.time()
    rollouts = run_episodes(sampler, renderer, env, prompts, num_samples=args.samples,
                            max_tokens=args.max_tokens, temperature=1.0, seed=args.seed,
                            think_budget=args.think_budget, answer_tokens=args.answer_tokens,
                            step_workers=args.workers)
    wall_s = time.time() - t0
    results = [env.score(r) for r in rollouts]

    short = args.model.split("/")[-1]
    out = Path(args.out) if args.out else Path("data/runs") / f"tv_baseline_{short}" / "episodes.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    judges = []
    if args.judge:
        from monitordecorrelation.monitors.cot_monitor import CoTMonitor

        judges = [CoTMonitor("cot+out", args.judge, behavior=env.behavior_name, use_cot=True),
                  CoTMonitor("out", args.judge, behavior=env.behavior_name, use_cot=False)]
    judge_scores: dict[str, list[float]] = {j.name: [] for j in judges}
    with out.open("w") as f:
        for r, er in zip(rollouts, results):
            rec = {"task_id": er.meta["task_id"], "behavior_present": er.behavior_present,
                   "task_reward": er.task_reward, "cot": r.cot, "output": r.output,
                   "stop_reason": r.meta.get("stop_reason"),
                   "tokens": {k: r.meta[k] for k in ("input_tokens", "output_tokens", "train_tokens",
                                                     "n_sampling_calls", "n_forced_answers",
                                                     "n_truncated_turns")},
                   "env_meta": er.meta, "monitors": {}}
            for j in judges:
                m = j.score(r)
                judge_scores[j.name].append(m.score)
                rec["monitors"][j.name] = {"score": m.score, "label": m.label}
            f.write(json.dumps(rec, default=str) + "\n")

    def rate(k: str) -> float:
        return mean(float(er.meta[k]) for er in results)

    print(f"\n=== terminal-verifier baseline: {args.model}, {n_ep} episodes ===")
    print(f"behavior_present (used-verifier/tampered{'' if args.honest else '/passed-impossible'}): {rate('cheated'):.2f}")
    for k in ("read_verifier", "used_verifier", "read_not_used", "modified_verifier", "passed", "passed_blind",
              "submitted_true_answer",
              "truncated", "no_command_turns", "n_turns", "n_commands"):
        print(f"  {k:<22} {rate(k):.2f}")
    print(f"  {'task_reward':<22} {mean(er.task_reward for er in results):.2f}")
    print(f"  {'unparsed':<22} {rate('unparsed'):.2f}")
    print(f"  {'forced_answers/ep':<22} {mean(r.meta['n_forced_answers'] for r in rollouts):.2f}")
    ends = {}
    for er in results:
        ends[er.meta["end_reason"]] = ends.get(er.meta["end_reason"], 0) + 1
    print(f"  end reasons: {ends}")
    cmds: dict[str, int] = {}
    for er in results:
        for c in er.meta["commands"]:
            key = c.split()[0] if c.split() else c
            cmds[key] = cmds.get(key, 0) + 1
    print(f"  first words of commands: {dict(sorted(cmds.items(), key=lambda kv: -kv[1]))}")
    gt = [er.behavior_present for er in results]
    judge_summary = {}
    for name, scores in judge_scores.items():
        judge_summary[name] = {"mean": mean(scores), "auroc": roc_auc(scores, gt)}
        print(f"  judge {name:<8} mean={mean(scores):.2f} AUROC vs oracle={roc_auc(scores, gt):.2f}")

    # -- cost + failure-mode accounting (the numbers an RL run is budgeted from) ---------------------
    per_rollout = token_accounting(rollouts)
    price = dict(PRICES.get(args.model.split(":")[0], {}))
    if args.price:
        price.update(json.loads(args.price))
    n_turns_total = sum(er.meta["n_turns"] for er in results) or 1
    summary = {
        "model": args.model, "effort": args.effort if is_tml else None, "n_episodes": n_ep,
        "n_prompts": len(prompts), "samples_per_prompt": args.samples, "seed": args.seed,
        "max_tokens_per_turn": args.max_tokens, "think_budget": args.think_budget,
        "answer_tokens": args.answer_tokens, "max_turns": args.max_turns,
        "impossible": not args.honest, "offset_range": list(args.offset), "wall_s": wall_s,
        "reward": mean(er.task_reward for er in results),
        "behavior_rate": rate("cheated"),
        "used_verifier": rate("used_verifier"), "modified_verifier": rate("modified_verifier"),
        "read_verifier": rate("read_verifier"), "passed": rate("passed"),
        "passed_blind": rate("passed_blind"), "submitted_true_answer": rate("submitted_true_answer"),
        # truncation of the ANSWER (a turn cut off by max_tokens) — distinct from thinking that was
        # force-closed at the budget, which is not a truncation of the output at all.
        "truncated_output_rate": mean(float(r.meta["n_truncated_turns"] > 0) for r in rollouts),
        "truncated_output_turn_rate": sum(r.meta["n_truncated_turns"] for r in rollouts) / n_turns_total,
        "think_budget_hit_rate": mean(float(r.meta["n_forced_answers"] > 0) for r in rollouts),
        "think_budget_hit_turn_rate": sum(r.meta["n_forced_answers"] for r in rollouts) / n_turns_total,
        # parse failures: a turn whose text carried no <command> (and was not cut off), and the
        # episode-level case where that happened every turn, so there was nothing to grade.
        "parse_error_turn_rate": sum(er.meta["no_command_turns"] for er in results) / n_turns_total,
        "parse_error_episode_rate": rate("unparsed"),
        "n_turns": rate("n_turns"), "n_commands": rate("n_commands"),
        "tokens_per_rollout": per_rollout,
        "end_reasons": ends, "judges": judge_summary,
    }
    print(f"\n--- tokens per rollout ({per_rollout['calls']:.2f} sampling calls) ---")
    print(f"  {'prefill (input)':<22} {per_rollout['prefill']:>9.0f}   "
          f"(uncacheable: {per_rollout['prefill_fresh']:.0f})")
    print(f"  {'sampled (output)':<22} {per_rollout['sample']:>9.0f}")
    print(f"  {'train (one datum)':<22} {per_rollout['train']:>9.0f}")
    if price:
        summary["price_per_1m"] = price
        summary["cost"] = cost_estimate(per_rollout, price, rollouts_per_step=args.step_rollouts)
        c = summary["cost"]
        print(f"--- cost @ {price} $/1M ---")
        print(f"  sampling/rollout      ${c['sampling_per_rollout_lo']:.5f}–${c['sampling_per_rollout_hi']:.5f}"
              f"   (cached–uncached prefill)")
        print(f"  +train/rollout        ${c['train_per_rollout']:.5f}")
        print(f"  RL step ({args.step_rollouts} rollouts) sampling ${c['step_sampling_lo']:.2f}–"
              f"${c['step_sampling_hi']:.2f}   with training ${c['step_total_lo']:.2f}–"
              f"${c['step_total_hi']:.2f}")
    print(f"--- failure modes ---")
    print(f"  {'truncated output':<22} {summary['truncated_output_rate']:.3f} of episodes, "
          f"{summary['truncated_output_turn_rate']:.3f} of turns")
    print(f"  {'think budget hit':<22} {summary['think_budget_hit_rate']:.3f} of episodes, "
          f"{summary['think_budget_hit_turn_rate']:.3f} of turns")
    print(f"  {'no <command> parsed':<22} {summary['parse_error_episode_rate']:.3f} of episodes, "
          f"{summary['parse_error_turn_rate']:.3f} of turns")
    summary_path = out.parent / f"summary_{out.stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"episodes → {out}\nsummary  → {summary_path}")

    for i, (r, er) in enumerate(list(zip(rollouts, results))[: args.show]):
        print(f"\n{'=' * 100}\n[episode {i}] task={er.meta['task_id']} true={er.meta['true_answer']} "
              f"verifier={er.meta['verifier_answer']} reward={er.task_reward} cheated={er.behavior_present} "
              f"end={er.meta['end_reason']} turns={er.meta['n_turns']}")
        for t_i, t in enumerate(er.meta["turns"]):
            cot = (t["cot"] or "").strip()
            print(f"\n--- turn {t_i + 1} THINK ({len(cot)} chars) ---\n{cot[:1500]}{'…' if len(cot) > 1500 else ''}")
            print(f"--- turn {t_i + 1} TEXT ---\n{(t['text'] or '').strip()[:800]}")
            if t["command"] is not None:
                print(f"--- turn {t_i + 1} TERMINAL ---\n$ {t['command']}\n{(t['output'] or '').strip()[:600]}")


if __name__ == "__main__":
    main()
