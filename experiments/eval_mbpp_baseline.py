"""No-training baseline on MBPP-Honeypot (any ``dataset`` / ``impossible`` variant): sample the untrained
policy on the run's held-out eval prompts and report what the env oracle sees — reward-hacking rates
(strict / loose / hardcoding), the honest solve rate, truncations, and prompt / completion lengths.

The eval set is the one an RL run evaluates on (``env.holdout(eval_size, seed)``, ``eval_samples_per_prompt``
rollouts each), sampled the way training samples (policy, seed, ``thinking_effort``, ``max_tokens``, T=1).
No monitors are scored, so the config must have none (``--set monitors=[]``).

  uv run python experiments/eval_mbpp_baseline.py \\
      --config experiments/configs/mbpp_inkling_qmonitors/row_control.json \\
      --set monitors=[] max_tokens=16384 env_options.dataset=codeforces_hard env_options.impossible=true

Writes ``<out>/summary.json`` and ``<out>/rollouts.jsonl`` (one record per rollout: prompt, CoT, answer,
token counts, the env's full grading record).
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from tqdm import tqdm

import tinker

from monitordecorrelation.envs.base import invalid_reason
from monitordecorrelation.envs.factory import make_env
from monitordecorrelation.envs.mbpp_honeypot import extract_code, extract_program
from monitordecorrelation.experiment_config import apply_overrides, load_config, validate_token_budgets
from monitordecorrelation.rl.renderers import is_tml_policy, make_renderer
from monitordecorrelation.rl.rollout import sample_rollouts

load_dotenv()

# The only config fields this script reads; --set on anything else would be silently ignored.
READ_FIELDS = {"policy", "seed", "thinking_effort", "env", "env_options", "monitors", "max_tokens",
               "think_budget", "answer_tokens", "eval_size", "eval_samples_per_prompt", "n_prompts_pool"}


def summarize(recs: list[dict]) -> dict:
    """Counts and rates over ALL rollouts (an invalid rollout — truncated / no codeblock — is graded
    as nothing: no pass, no hack, no solve), plus the token-length means."""
    n = len(recs)

    def count(key: str) -> int:
        return sum(1 for r in recs if r["env_meta"].get(key))

    honest = sum(1 for r in recs if r["env_meta"].get("solves_task", r["env_meta"].get("passes_all"))
                 and not r["env_meta"].get("hardcoding"))
    out = {
        "n_rollouts": n,
        "n_prompts": len({r["task_id"] for r in recs}),
        "prompt_tokens_mean": mean(r["prompt_tokens"] for r in recs),
        "completion_tokens_mean": mean(r["completion_tokens"] for r in recs),
        "completion_tokens_max": max(r["completion_tokens"] for r in recs),
        "n_truncated": sum(1 for r in recs if r["invalid_reason"] == "truncated"),
        "n_unparsed": sum(1 for r in recs if r["invalid_reason"] == "unparsed"),
        "n_passes_visible": count("passes_visible"),
        "n_strict_rh": count("strict_rh"),
        "n_loose_rh": count("loose_rh"),
        "n_hardcoding": count("hardcoding"),
        "n_honest_solve": honest,
        "reward_mean": mean(r["task_reward"] for r in recs),
    }
    for k in ("passes_visible", "strict_rh", "loose_rh", "hardcoding", "honest_solve", "truncated"):
        out[f"{k}_rate"] = out[f"n_{k}"] / n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help=f"override config fields, as in run_experiment.py; only: {sorted(READ_FIELDS)}")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set, allowed_fields=READ_FIELDS)
    if cfg.env != "mbpp_honeypot":
        raise SystemExit(f"this script evaluates mbpp_honeypot, not {cfg.env!r}")
    if cfg.monitors:
        raise SystemExit("this script scores no monitors, so the config's would be silently ignored — "
                         "pass --set monitors=[]")
    env = make_env(cfg)
    try:
        validate_token_budgets(cfg, env)
    except ValueError as e:
        raise SystemExit(f"config {args.config}: {e}") from e
    prompts = env.holdout(cfg.eval_size, seed=cfg.seed)
    n_per = cfg.eval_samples_per_prompt

    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=cfg.policy)
    renderer = make_renderer(cfg.policy, effort=cfg.thinking_effort,
                             tokenizer=None if is_tml_policy(cfg.policy) else sampler.get_tokenizer())
    prompt_tokens = [renderer.model_input(p.text).length for p in prompts]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs: list[dict | None] = [None] * (len(prompts) * n_per)
    lock = threading.Lock()
    bar = tqdm(total=len(recs), desc=out_dir.name)

    def on_rollout(i: int, r) -> None:  # grade each rollout the moment it lands
        why = invalid_reason(env, r)
        er = env.score(r)
        p = i // n_per
        rec = {
            "task_id": r.prompt.meta["task_id"], "sample": i % n_per, "rating": r.prompt.meta.get("rating"),
            "prompt_tokens": prompt_tokens[p], "completion_tokens": len(r.token_ids or []),
            "stop_reason": r.meta.get("stop_reason"), "invalid_reason": why,
            "task_reward": er.task_reward, "behavior_present": er.behavior_present, "env_meta": er.meta,
            "prompt": r.prompt.text, "cot": r.cot, "answer": r.output,
            # the block the env graded (Codeforces: the first python block)
            "code": (extract_program if r.prompt.meta.get("format") == "stdio" else extract_code)(r.output),
        }
        with lock:
            recs[i] = rec
            bar.update()

    sample_rollouts(sampler, renderer, prompts, num_samples=n_per, max_tokens=cfg.max_tokens,
                    temperature=1.0, seed=cfg.seed, on_rollout=on_rollout)
    bar.close()
    done = [r for r in recs if r is not None]
    with (out_dir / "rollouts.jsonl").open("w") as f:
        for r in done:
            f.write(json.dumps(r, default=str) + "\n")
    summary = {"config": {k: getattr(cfg, k) for k in sorted(READ_FIELDS - {"monitors"})},
               **summarize(done)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
