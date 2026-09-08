"""Baseline (no-training) inference on the terminal-verifier env: the full episode pipeline minus
the gradient step.

Samples real multi-turn episodes from a base policy on tinker, grades them with the env's oracle,
prints the behavior rates (peeked / tampered / passed / …) and a few full transcripts, and optionally
scores LLM judges (CoT+output and output-only) against the oracle. This is the pre-RL sanity check
for the setting: does the base model already read the verifier, how often, and can monitors tell.

  uv run python experiments/eval_terminal_baseline.py --n-prompts 16 --samples 2
  uv run python experiments/eval_terminal_baseline.py --judge anthropic/claude-3-haiku
  uv run python experiments/eval_terminal_baseline.py --honest        # verifier holds the TRUE answer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

import tinker

from monitordecorrelation.envs.terminal_verifier import TerminalVerifierEnv
from monitordecorrelation.eval.metrics import roc_auc
from monitordecorrelation.rl.episodes import run_episodes
from monitordecorrelation.rl.renderers import make_renderer

load_dotenv()


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
    args = ap.parse_args()

    env = TerminalVerifierEnv.from_task(
        n=max(64, args.n_prompts), seed=args.seed, impossible=not args.honest,
        offset_range=tuple(args.offset), eval_n=max(64, args.n_prompts),
        max_turns=args.max_turns, output_view=args.output_view,
    )
    prompts = env.holdout(args.n_prompts, seed=args.seed)

    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=args.model)
    renderer = make_renderer(args.model, tokenizer=sampler.get_tokenizer())
    n_ep = len(prompts) * args.samples
    print(f"sampling {n_ep} episodes ({len(prompts)}×{args.samples}) from {args.model} — "
          f"max_turns={args.max_turns}, max_tokens/turn={args.max_tokens}, "
          f"think_budget={args.think_budget}/{args.answer_tokens}, "
          f"{'HONEST' if args.honest else 'IMPOSSIBLE'} verifier", flush=True)
    rollouts = run_episodes(sampler, renderer, env, prompts, num_samples=args.samples,
                            max_tokens=args.max_tokens, temperature=1.0, seed=args.seed,
                            think_budget=args.think_budget, answer_tokens=args.answer_tokens)
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
                   "stop_reason": r.meta.get("stop_reason"), "env_meta": er.meta, "monitors": {}}
            for j in judges:
                m = j.score(r)
                judge_scores[j.name].append(m.score)
                rec["monitors"][j.name] = {"score": m.score, "label": m.label}
            f.write(json.dumps(rec, default=str) + "\n")

    def rate(k: str) -> float:
        return mean(float(er.meta[k]) for er in results)

    print(f"\n=== terminal-verifier baseline: {args.model}, {n_ep} episodes ===")
    print(f"behavior_present (peeked/tampered{'' if args.honest else '/passed-impossible'}): {rate('cheated'):.2f}")
    for k in ("read_verifier", "modified_verifier", "passed", "passed_blind", "submitted_true_answer",
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
    for name, scores in judge_scores.items():
        print(f"  judge {name:<8} mean={mean(scores):.2f} AUROC vs oracle={roc_auc(scores, gt):.2f}")
    print(f"episodes → {out}")

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
