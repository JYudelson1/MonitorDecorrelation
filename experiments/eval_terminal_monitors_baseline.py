"""Pre-RL monitor evaluation on the terminal-verifier env: how well does each judge detect the
base policy's reward hacking, before any training pressure?

Samples real multi-turn episodes from an UNTRAINED policy, grades them with the env oracle
(``behavior_present`` = modified the verifier / passed an impossible verifier), scores every
monitor in a config against that oracle, and reports the standard bundle — AUROC, accuracy, Brier,
d', the present/absent class-split mean score (the "suspiciousness gap") — plus TPR/FPR at the
monitor's threshold, which AUROC alone hides.

These are the row-zero numbers the degradation matrix is measured against: a monitor with no
pre-RL headroom (AUROC ~1.0, gap ~1.0) has nothing to degrade, so the matrix cell is uninformative.

  uv run python experiments/eval_terminal_monitors_baseline.py \
      --config experiments/configs/terminal_verifier_gemini25_out.json \
      --model thinkingmachines/Inkling-Small --n-prompts 96 --samples 2

  # config overrides, same syntax as run_experiment.py — but only for fields this script reads:
      --set max_tokens=4096 'monitors.model:gemini-3.5.reasoning={"effort":"medium"}' \
            env_options.verifier_mode=verifier_bug
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from tqdm import tqdm

import tinker

from monitordecorrelation.envs.base import invalid_reason
from monitordecorrelation.envs.terminal_verifier import TerminalVerifierEnv
from monitordecorrelation.experiment_config import (
    apply_overrides,
    load_config,
    validate_token_budgets,
)
from monitordecorrelation.eval.metrics import accuracy, brier, dprime_margin, judge_call_rates, judge_finish_reason, roc_auc
from monitordecorrelation.eval.rollout_dump import monitor_record, slim_record
from monitordecorrelation.monitors.agent_cot_monitor import AgentCoTMonitor
from monitordecorrelation.rl.episodes import run_episodes
from monitordecorrelation.rl.renderers import is_tml_policy, make_renderer
from monitordecorrelation.rl.train import MonitorScorer

load_dotenv()

# The only config fields this script reads. --set on anything else (policy, seed, n_steps, …) would be
# silently ignored, so apply_overrides refuses it; the policy and seed are the --model / --seed flags.
READ_FIELDS = {"thinking_effort", "env_options", "monitors", "max_tokens", "think_budget", "answer_tokens"}
READ_MONITOR_FIELDS = {"name", "model_id", "use_cot", "use_output", "threshold",
                       "reasoning", "binary_judge", "provider", "max_tokens", "base_url",
                       "enable_thinking", "thinking_budget"}


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


def token_usage(rollouts, judged: dict, judges) -> dict:
    """Totals of what the run consumed. The policy side comes from the episode driver's accounting
    (``input_tokens`` counts every sampling call's full prefix, so multi-turn prefill is re-counted
    per turn); the judge side sums the ``usage`` (OpenRouter / vLLM) of each successful call — retried
    failures aren't recorded, so they're missing here. vLLM reports no cost."""
    policy = {k: sum(r.meta.get(k, 0) for r in rollouts)
              for k in ("n_sampling_calls", "input_tokens", "output_tokens")}
    out: dict = {"policy": policy, "judges": {}}
    for j in judges:
        usages = [res.meta["call"]["response"].get("usage") or {} for res in judged[j.name]
                  if res is not None and (res.meta or {}).get("call")]
        out["judges"][j.name] = {
            "n_calls": len(usages),
            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usages),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usages),
            "reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                                    for u in usages),
            "cost_usd": sum(u.get("cost") or 0.0 for u in usages),
        }
    out["judges_total"] = {k: sum(v[k] for v in out["judges"].values())
                           for k in ("n_calls", "prompt_tokens", "completion_tokens",
                                     "reasoning_tokens", "cost_usd")}
    return out


class _Ticking:
    """A judge that ticks a progress bar when each ``score`` call finishes (success or failure).
    Everything else is forwarded, so ``MonitorScorer`` treats it exactly like the judge itself."""

    def __init__(self, judge, bar) -> None:
        self._judge, self._bar = judge, bar

    def __getattr__(self, name: str):
        return getattr(self._judge, name)

    def score(self, rollout):
        try:
            return self._judge.score(rollout)
        finally:
            self._bar.update()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment config whose `monitors` to evaluate")
    ap.add_argument("--only", default=None, help="comma-separated monitor names (default: all)")
    ap.add_argument("--model", default="thinkingmachines/Inkling-Small")
    ap.add_argument("--n-prompts", type=int, default=96)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", type=float, default=None,
                    help="TML thinking effort (default: the config's thinking_effort)")
    ap.add_argument("--out", default=None,
                    help="path of the summary json; its parent dir gets the rest (overrides --run-name)")
    ap.add_argument("--run-name", default=None,
                    help="output dir name under data/runs/ (default: tv_monitor_baseline_<model>)")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help="override config fields, as in run_experiment.py (e.g. --set max_tokens=4096 "
                         "'monitors.model:gemini-3.5.reasoning={\"effort\":\"medium\"}' "
                         "env_options.verifier_mode=possible); only fields this script "
                         f"reads: {sorted(READ_FIELDS)} and monitor fields {sorted(READ_MONITOR_FIELDS)}")
    args = ap.parse_args()

    if args.effort is not None and any(kv.partition("=")[0] == "thinking_effort" for kv in args.set):
        raise SystemExit("--effort and --set thinking_effort=… both given; pass one")
    cfg_obj = apply_overrides(load_config(args.config), args.set, allowed_fields=READ_FIELDS,
                              allowed_monitor_fields=READ_MONITOR_FIELDS,
                              not_allowed_hint="Use --model / --seed for the policy / seed.")
    cfg = cfg_obj.model_dump()
    # Match the RL runs: the policy must be sampled the way training samples it.
    if args.effort is None:
        args.effort = cfg["thinking_effort"]
    opts = cfg["env_options"]
    specs = cfg["monitors"]
    if probes := [m["name"] for m in specs if m["kind"] != "cot"]:
        raise SystemExit(f"this script evaluates CoT/output judges only; the config has probe(s) {probes}")
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        unknown = want - {m["name"] for m in specs}
        if unknown:
            raise SystemExit(f"--only: no monitor named {sorted(unknown)} in {args.config} "
                             f"(it has: {[m['name'] for m in specs]})")
        specs = [m for m in specs if m["name"] in want]
    behavior = "reward_hacking"

    # Every env option passes straight through, exactly as the RL run's envs/factory.py does, so the
    # baseline samples the same env (verifier_mode, offset_range, max_turns, …) the run trains on.
    env = TerminalVerifierEnv.from_task(
        **{**opts, "n": max(64, args.n_prompts), "seed": args.seed, "eval_n": max(64, args.n_prompts)}
    )
    prompts = env.holdout(args.n_prompts, seed=args.seed)

    # The same check the training loop makes, now that the env (hence the resolved budget) exists:
    # max_tokens under a thinking budget, or answer_tokens without one, would be sampled with and
    # never used, so they are rejected rather than ignored.
    try:
        think_budget = validate_token_budgets(cfg_obj, env)
    except ValueError as e:
        raise SystemExit(f"config {args.config}: {e}") from e

    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=args.model)
    is_tml = is_tml_policy(args.model)
    if not is_tml and args.effort is not None:
        raise SystemExit(
            f"--effort/{args.effort} applies only to TML-rendered policies; {args.model} has no such knob"
        )
    renderer = make_renderer(args.model, effort=args.effort,
                            tokenizer=None if is_tml else sampler.get_tokenizer())

    judges = [AgentCoTMonitor(m["name"], m["model_id"], behavior=behavior,
                              use_cot=m.get("use_cot", True), use_output=m.get("use_output", True),
                              threshold=m.get("threshold", 0.5),
                              reasoning=m["reasoning"],
                              binary_judge=bool(m.get("binary_judge")),
                              provider=m["provider"], max_tokens=m["max_tokens"],
                              base_url=m["base_url"], enable_thinking=m["enable_thinking"],
                              thinking_budget=m["thinking_budget"])
              for m in specs]
    n_ep = len(prompts) * args.samples
    print(f"sampling {n_ep} episodes ({len(prompts)}x{args.samples}) from {args.model} "
          f"(PRE-RL, effort={args.effort}) | judges: {', '.join(j.name for j in judges)}", flush=True)

    t0 = time.time()
    ep_bar = tqdm(total=n_ep, desc="episodes", unit="ep", position=0)
    # The judge total grows as valid episodes arrive (invalid ones are never judged).
    judge_bar = tqdm(total=0, desc="judge calls", unit="call", position=1)
    bar_lock = threading.Lock()

    def on_rollout(i: int, rollout) -> None:
        if invalid_reason(env, rollout) is None:
            with bar_lock:
                judge_bar.total += len(judges)
                judge_bar.refresh()
        scorer.submit(i, rollout)
        ep_bar.update()

    # Invalid episodes (truncated / malformed turn) are never shown to a judge, as in the RL loop.
    scorer = MonitorScorer([_Ticking(j, judge_bar) for j in judges],
                           skip=lambda r: invalid_reason(env, r) is not None)
    rollouts = run_episodes(sampler, renderer, env, prompts, num_samples=args.samples,
                            max_tokens=cfg["max_tokens"], temperature=1.0,
                            seed=args.seed, think_budget=think_budget,
                            answer_tokens=cfg["answer_tokens"], on_rollout=on_rollout)
    wall_s = time.time() - t0
    results = [env.score(r) for r in rollouts]
    judged = scorer.collect(rollouts)
    ep_bar.close()
    judge_bar.close()

    gt = [er.behavior_present for er in results]
    invalid = [invalid_reason(env, r) for r in rollouts]
    valid = [i for i, why in enumerate(invalid) if why is None]
    gt_v = [gt[i] for i in valid]  # the judges' metrics are over the valid (monitored) episodes only
    print(f"\n=== pre-RL baseline: {args.model}, {len(gt)} episodes, {wall_s:.0f}s ===")
    print(f"  verifier_mode              {opts.get('verifier_mode', 'corrupted')}")
    for k in ("used_verifier", "read_verifier", "modified_verifier", "passed", "submitted_true_answer"):
        print(f"  env/{k:<22} {mean(float(er.meta[k]) for er in results):.3f}")
    print(f"  behavior_present (oracle)  {mean(map(float, gt)):.3f}   "
          f"({sum(gt)} positive / {len(gt) - sum(gt)} negative)")
    print(f"  invalid (not judged)       {1 - len(valid) / len(gt):.3f}   "
          f"({sum(w == 'truncated' for w in invalid)} truncated / {sum(w == 'unparsed' for w in invalid)} "
          f"malformed) — judge metrics below: {len(valid)} valid episodes, "
          f"{sum(gt_v)} positive / {len(gt_v) - sum(gt_v)} negative")

    rows = {}
    print(f"\n{'monitor':<14}{'view':<12}{'AUROC':>7}{'acc':>7}{'brier':>7}{'d-prime':>9}"
          f"{'mean|hack':>11}{'mean|clean':>11}{'gap':>7}{'TPR':>7}{'FPR':>7}")
    for j in judges:
        scores = [judged[j.name][i].score for i in valid]
        preds = [judged[j.name][i].label for i in valid]
        hi = mean([s for s, t in zip(scores, gt_v) if t]) if any(gt_v) else float("nan")
        lo = mean([s for s, t in zip(scores, gt_v) if not t]) if not all(gt_v) else float("nan")
        r = {"auroc": roc_auc(scores, gt_v), "accuracy": accuracy(preds, gt_v),
             "brier": brier(scores, gt_v), "dprime_margin": dprime_margin(scores, gt_v),
             "mean_score_present": hi, "mean_score_absent": lo, "suspiciousness_gap": hi - lo,
             "mean_score": mean(scores) if scores else float("nan"), "n_scored": len(scores),
             **rates(preds, gt_v),
             # call health: stopped at max_tokens / no parseable answer (so scored 0)
             **judge_call_rates([judged[j.name][i] for i in valid]),
             "n_finish_length": sum(judge_finish_reason(judged[j.name][i]) == "length" for i in valid),
             "n_parse_error": sum(bool(judged[j.name][i].meta.get("parse_error")) for i in valid)}
        rows[j.name] = r
        view = ("cot+output" if j.use_cot and j.use_output else
                "output-only" if not j.use_cot else "cot-only")
        print(f"{j.name:<14}{view:<12}{r['auroc']:>7.3f}{r['accuracy']:>7.3f}{r['brier']:>7.3f}"
              f"{r['dprime_margin']:>9.2f}{hi:>11.3f}{lo:>11.3f}{hi - lo:>7.3f}"
              f"{r['tpr']:>7.3f}{r['fpr']:>7.3f}")

    print(f"\n=== judge calls: stopped at max_tokens (finish_reason=length) / unparseable answer (scored 0) ===")
    print(f"  {'monitor':<14}{'backend':<40}{'calls':>7}{'length':>9}{'len rate':>10}{'parse err':>11}{'err rate':>10}")
    for j in judges:
        r, b = rows[j.name], j.backend.info()
        where = (f"vllm max={b['max_tokens']} think={b['enable_thinking']} budget={b['thinking_budget']}"
                 if b["provider"] == "vllm" else f"openrouter max={b['max_tokens']}")
        print(f"  {j.name:<14}{where:<40}{r['n_scored']:>7}{r['n_finish_length']:>9}"
              f"{r['finish_length_rate']:>10.3f}{r['n_parse_error']:>11}{r['parse_error_rate']:>10.3f}")

    usage = token_usage(rollouts, judged, judges)
    pol = usage["policy"]
    print(f"\n=== token usage ===\n  policy ({args.model}): {pol['n_sampling_calls']} sampling calls, "
          f"{pol['input_tokens']:,} input / {pol['output_tokens']:,} output tokens "
          f"({pol['output_tokens'] / max(1, len(rollouts)):,.0f} output/episode)")
    print(f"  {'judge':<14}{'calls':>7}{'prompt':>13}{'completion':>13}{'(reasoning)':>13}{'cost $':>10}")
    for name, u in [*usage["judges"].items(), ("TOTAL", usage["judges_total"])]:
        print(f"  {name:<14}{u['n_calls']:>7}{u['prompt_tokens']:>13,}{u['completion_tokens']:>13,}"
              f"{u['reasoning_tokens']:>13,}{u['cost_usd']:>10.4f}")

    # Laid out like an RL run dir so visualize_transcripts.py (and the slim-dump readers) pick it up:
    # the summary in baseline.json + run_info.json, the rollouts in eval_rollouts{,_slim}.jsonl using
    # the RL loop's eval-record schema (rl/train.py run_eval) at step 0.
    out = Path(args.out) if args.out else Path("data/runs") / \
        (args.run_name or f"tv_monitor_baseline_{args.model.split('/')[-1]}") / "baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model, "policy": args.model, "experiment": "tv_monitor_baseline",
        "config_path": args.config, "config_overrides": args.set,
        "config": {**cfg, "monitors": specs}, "n_episodes": len(gt),
        "seed": args.seed, "effort": args.effort, "wall_s": wall_s, "env_options": opts,
        "behavior_rate": mean(map(float, gt)), "invalid_rate": 1 - len(valid) / len(gt), "monitors": rows,
        "token_usage": usage,
        # as rl/train.py records it: with the RESOLVED settings each judge actually sent (provider,
        # max_tokens, and the reasoning object / vLLM thinking settings)
        "held_out": [{"kind": "cot", **m, "reasoning": j.reasoning, **j.backend.info()}
                     for m, j in zip(specs, judges)],
    }
    out.write_text(json.dumps(summary, indent=1, default=str))
    (out.parent / "run_info.json").write_text(json.dumps(summary, indent=2, default=str))
    with (out.parent / "eval_rollouts.jsonl").open("w") as full, \
            (out.parent / "eval_rollouts_slim.jsonl").open("w") as slim:
        for i, (r, er) in enumerate(zip(rollouts, results)):
            rec = {
                "step": 0, "task_id": er.meta["task_id"], "question": r.prompt.text,
                "cot": r.cot, "answer": r.output, "behavior_present": bool(er.behavior_present),
                "loose_rh": None, "hardcoding": None, "unparsed": er.meta.get("unparsed"),
                "invalid_reason": invalid[i], "env_meta": er.meta,
                # score/label + each judge's exact request/response (monitor_record); {} if invalid
                "monitors": {j.name: monitor_record(judged[j.name][i]) for j in judges
                             if judged[j.name][i] is not None},
            }
            full.write(json.dumps(rec, default=str) + "\n")
            slim.write(json.dumps(slim_record(rec), default=str) + "\n")
    print(f"\nwrote {out}, run_info.json, eval_rollouts.jsonl, eval_rollouts_slim.jsonl in {out.parent}")


if __name__ == "__main__":
    main()
