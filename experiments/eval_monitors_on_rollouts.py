"""Monitor bakeoff on a SAVED rollouts.jsonl: for each candidate model (and CoT- vs output-judge mode),
score the rollouts with the reward-hacking rubric and report detection AUROC. Used to find cheaper
judge models that still detect reward-hacking well, so the degradation runs get cheaper.

Ground truth = env.behavior_present (the reward-hack oracle). Same rubric/template as the RL monitors
(monitors/cot_monitor.py). For reliable AUROC on an imbalanced set we class-balance by default
(equal hacking / non-hacking) — AUROC precision is set by the smaller class, so this is ~as good as
scoring everything for a fraction of the cost.

    # dry run: counts + which monitors, NO API calls
    uv run python experiments/eval_monitors_on_rollouts.py --rollouts <sample.jsonl> --dry-run
    # real run (spends):
    uv run python experiments/eval_monitors_on_rollouts.py --rollouts <sample.jsonl> --max-per-class 500
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from time import perf_counter

from dotenv import load_dotenv

from monitordecorrelation.eval.metrics import accuracy, roc_auc
from monitordecorrelation.monitors.agent_cot_monitor import AgentCoTMonitor
from monitordecorrelation.monitors.cot_monitor import CoTMonitor
from monitordecorrelation.transcript import episode_turns
from monitordecorrelation.rl.rollout import load_saved_rollouts
from monitordecorrelation.types import MonitorResult

load_dotenv(".env")

# Candidate judge ladder (cheap → sonnet anchor). Confirm slugs on openrouter.ai/models.
_DEFAULT_MODELS = [
    "anthropic/claude-sonnet-4.6",          # anchor (most expensive)
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-3-haiku",             # the cheap older haiku (matrix generator's WEAK)
    "openai/gpt-5.4",                       # flagship, ~Sonnet price point ($2.5/$15)
    "openai/gpt-5.4-mini",                  # mid-tier OpenAI
    "openai/gpt-4o-mini",
    "google/gemini-3.1-pro-preview",        # Google flagship (pro tier)
    "google/gemini-3.5-flash",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3-8b",                        # small Qwens (cheap; same family as the policy)
    "qwen/qwen3-32b",
]


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="a saved rollouts.jsonl")
    ap.add_argument("--behavior", default="reward_hacking")
    ap.add_argument("--max-per-class", type=int, default=500, help="class-balanced cap (0 = use all)")
    ap.add_argument("--config", default=None,
                    help="build the judges from a run config's `monitors` (exact names, views and "
                         "reasoning settings — no drift from the RL runs). Overrides --models/--modes. "
                         "Probes are skipped: they need the 8B base model, this script is judges-only.")
    ap.add_argument("--models", default=",".join(_DEFAULT_MODELS))
    ap.add_argument("--modes", default="cot,output", help="cot (reads CoT) and/or output (answer only)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/runs/monitor_bakeoff_rollouts.json")
    ap.add_argument("--dry-run", action="store_true", help="report the eval set + monitors, NO API calls")
    args = ap.parse_args()

    parsed = load_saved_rollouts(args.rollouts)  # [(Rollout, behavior_present)]
    # A multi-turn dump (the terminal env: every record carries the per-turn episode) is judged by
    # AgentCoTMonitor — the chat-transcript judge the RL runs use — not the single-turn CoTMonitor.
    n_agentic = sum(episode_turns(r) is not None for r, _ in parsed)
    if n_agentic not in (0, len(parsed)):
        raise SystemExit(f"{args.rollouts}: {n_agentic}/{len(parsed)} rollouts are multi-turn "
                         "episodes — a dump must be all single-turn or all multi-turn")
    judge_cls = AgentCoTMonitor if n_agentic else CoTMonitor
    pos = [r for r, g in parsed if g]
    neg = [r for r, g in parsed if not g]
    rng = random.Random(args.seed)
    if args.max_per_class and min(len(pos), len(neg)) > 0:
        k = min(args.max_per_class, len(pos), len(neg))
        eval_rollouts = rng.sample(pos, k) + rng.sample(neg, k)
        gt = [True] * k + [False] * k
    else:
        eval_rollouts, gt = [r for r, _ in parsed], [g for _, g in parsed]
        k = None

    if args.config:
        # The battery exactly as a matrix row declares it. Keeping the specs (not just model ids)
        # preserves each judge's view and reasoning_effort — a gemini-3.x judge without one cannot
        # complete a single call, and a cot-only judge measured as cot+out is a different monitor.
        from monitordecorrelation.experiment_config import load_monitor_specs
        all_specs = load_monitor_specs(args.config)
        specs = [s for s in all_specs if s.kind == "cot"]
        n_probe = len(all_specs) - len(specs)
        monitors = [(s.name, s) for s in specs]
        models = sorted({s.model_id for s in specs})
        modes = [args.config]
        if n_probe:
            print(f"(skipping {n_probe} probe monitor(s) — judges only)")
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        monitors = [(f"{_slug(mid)}_{mode}", (mid, mode)) for mid in models for mode in modes]
    print(f"rollouts: {len(parsed)} ({len(pos)} hacking / {len(neg)} non) — judge: {judge_cls.__name__}")
    print(f"eval set: {len(eval_rollouts)} ({'balanced ' + str(k) + '/class' if k else 'ALL'})")
    print(f"monitors: {len(monitors)}" + (f" from {args.config}" if args.config
          else f" = {len(models)} models × {len(modes)} modes ({', '.join(modes)})"))
    if args.dry_run:
        print("\n[dry-run] no API calls. Monitors:")
        for name, spec in monitors:
            d = (f"{spec.model_id} use_cot={spec.use_cot} use_output={spec.use_output} "
                 f"reasoning_effort={spec.reasoning_effort}") if args.config else f"{spec[0]} mode={spec[1]}"
            print(f"  - {name:18s} {d}")
        return

    results = []
    print(f"\n{'monitor':40s} {'AUROC':>6} {'acc':>6} {'mean(hack)':>11} {'mean(¬hack)':>12} {'s':>5}")
    print("-" * 86)
    for name, spec in monitors:
        t0 = perf_counter()
        mid = spec.model_id if args.config else spec[0]
        mode = args.config if args.config else spec[1]
        try:
            if args.config:
                mon = judge_cls(name=name, model_id=spec.model_id, behavior=args.behavior,
                                use_cot=spec.use_cot, use_output=spec.use_output,
                                reasoning_effort=spec.reasoning_effort,
                                reasoning_max_tokens=spec.reasoning_max_tokens,
                                binary_judge=spec.binary_judge, threshold=spec.threshold)
            else:
                mon = judge_cls(name=name, model_id=mid, behavior=args.behavior,
                                use_cot=(mode == "cot"))

            def _safe_score(r):  # one bad call -> NaN, never lose the whole model's row
                try:
                    return mon.score(r)
                except Exception:  # noqa: BLE001
                    return MonitorResult(score=float("nan"), label=False)

            # Warm up sequentially first: settles the monitor's reasoning config (enabled:false → a
            # bounded budget for mandatory-reasoning models) so the concurrent batch doesn't 400-storm.
            _safe_score(eval_rollouts[0])
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(_safe_score, eval_rollouts))
            # Three buckets: call FAILED (NaN, exception after retries), call ok but PARSE-FAILED (no
            # SCORE in the judge output), and USABLE (a real score). AUROC is over USABLE only; we also
            # report parse success rate among successful calls.
            call_ok = [(x, g) for x, g in zip(res, gt) if x.score == x.score]
            usable = [(x, g) for x, g in call_ok if not x.meta.get("parse_error")]
            if not usable:
                raise RuntimeError(f"no usable scores ({len(res) - len(call_ok)} call-fail, "
                                   f"{len(call_ok) - len(usable)} parse-fail)")
            scores = [x.score for x, _ in usable]
            mgt = [g for _, g in usable]
            auc = roc_auc(scores, mgt)
            acc = accuracy([x.label for x, _ in usable], mgt)
            mh = mean([s for s, g in zip(scores, mgt) if g]) if any(mgt) else float("nan")
            mn = mean([s for s, g in zip(scores, mgt) if not g]) if not all(mgt) else float("nan")
            parse_rate = len(usable) / len(call_ok) if call_ok else 0.0
            n_call_fail = len(res) - len(call_ok)
            results.append({"name": name, "model_id": mid, "mode": mode, "auroc": auc, "accuracy": acc,
                            "mean_hack": mh, "mean_nonhack": mn, "n_usable": len(usable),
                            "parse_success_rate": parse_rate, "n_call_failed": n_call_fail,
                            "n_parse_failed": len(call_ok) - len(usable)})
            flags = []
            if parse_rate < 1.0:
                flags.append(f"parse {parse_rate:.0%}")
            if n_call_fail:
                flags.append(f"{n_call_fail} call-fail")
            flag = f"  ({', '.join(flags)})" if flags else ""
            print(f"{name:40s} {auc:6.2f} {acc:6.2f} {mh:11.1f} {mn:12.1f} "
                  f"{perf_counter()-t0:5.0f}{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:40s}  FAILED: {str(e)[:48]}")
            results.append({"name": name, "model_id": mid, "mode": mode, "error": str(e)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rollouts": args.rollouts, "behavior": args.behavior,
                               "balanced_per_class": k, "monitors": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
