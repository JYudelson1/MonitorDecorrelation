"""Generate the MBPP-Honeypot cross-method degradation run set: train a model to reward-hack while
evading ONE of N monitors, one run per monitor, + a no-monitor control. The headline experiment.

    uv run python experiments/make_mbpp_matrix_configs.py                 # 6 monitors + control, M=Qwen3-8B
    uv run python experiments/make_mbpp_matrix_configs.py --no-cot        # non-CoT M: drop the 2 CoT monitors
    uv run python experiments/make_mbpp_matrix_configs.py --model meta-llama/Llama-3.2-3B --no-cot --seed 1
    uv run python experiments/make_mbpp_matrix_configs.py --validate      # parse each (no model load)

Everything is parametric (model, monitor subset, seed, steps, lora rank, control on/off) so the run set
scales to multiple seeds / models without editing code. Rows are identical except which monitor is
`train_against` — the matrix-coherence invariant — enforced by generating from one base.

The 6 monitors = OA-style {weak,strong} × {CoT judge, output judge} + {OOD deception probe, iid
MBPP-hack probe}. probe paths follow the data/probes/<model-basename>/<datasets> convention.

PRE-REQS before these RUN (CPU-generatable now, but the runs need them):
  - probe_ood : data/probes/<M>/diverse_deception   (have it for Qwen3-8B; retrain per model)
  - probe_iid : data/probes/<M>/mbpp_honeypot        (NOT trained yet — task #42-style, `--preset mbpp`)
  - the iid-probe / eval-prompt task_id split must be unified to avoid leakage (tracked: #25 split).
  - tinker billing active; rented GPU for the frozen-model probe forward passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Weak/strong judge ladder from the strict-label bakeoff (bakeoff_strict.json): against the CLEAN strict
# reward-hacking oracle every judge scores ~0.9-0.99, so we use a cheap, non-reasoning pair from a
# different family than the Qwen3-8B policy (no same-model-judge confound). ~5-10x cheaper than sonnet.
WEAK = "anthropic/claude-3-haiku"   # 0.927/0.901, $0.25/$1.25 — weakest viable judge in the table
STRONG = "deepseek/deepseek-chat"    # 0.985/0.975, $0.3/$1.2 — non-reasoning (V3), strong + cheap


def all_monitors(model: str) -> list[dict]:
    """The 6 candidate monitors. probe_model = the policy base (off-model probing is unreliable)."""
    pdir = f"data/probes/{model.split('/')[-1]}"  # e.g. Qwen/Qwen3-8B -> data/probes/Qwen3-8B
    return [
        {"kind": "cot", "name": "cot_weak", "model_id": WEAK, "use_cot": True},
        {"kind": "cot", "name": "cot_strong", "model_id": STRONG, "use_cot": True},
        {"kind": "cot", "name": "out_weak", "model_id": WEAK, "use_cot": False},
        {"kind": "cot", "name": "out_strong", "model_id": STRONG, "use_cot": False},
        {"kind": "probe", "name": "probe_ood",
         "probe_path": f"{pdir}/diverse_deception", "probe_model": model},
        {"kind": "probe", "name": "probe_iid",
         "probe_path": f"{pdir}/mbpp_honeypot", "probe_model": model},
    ]


def _spec(d: dict, role: str) -> dict:
    m = {"kind": d["kind"], "name": d["name"], "role": role}
    if d["kind"] == "cot":
        m["model_id"] = d["model_id"]
        m["use_cot"] = d["use_cot"]
    else:
        m["probe_path"] = d["probe_path"]
        m["probe_model"] = d["probe_model"]
    return m


def build_rows(model: str, monitors: list[dict], *, seed: int, n_steps: int, lora_rank: int,
               max_tokens: int, control: bool) -> list[dict]:
    short = model.split("/")[-1]
    base = {
        "experiment": "mbpp_degradation_matrix",
        "policy": model,
        "backend": "tinker",
        "env": "mbpp_honeypot",
        "n_steps": n_steps,
        "batch_size": 8,
        "group_size": 8,          # OA GRPO group size
        "eval_every": 10,
        "eval_size": 64,
        "max_tokens": max_tokens,
        "penalty_coef": 1.0,      # weight on the train-against monitor penalty
        "kl_coef": 1e-4,          # per-token KL-to-base anchor — prevents the reward-hack collapse
        "lora_rank": lora_rank,   # OA: rank 64 (alpha 128 — tinker sets alpha; verify 2×rank on the box)
        "seed": seed,
    }
    names = [m["name"] for m in monitors]
    rows = []
    # one row per monitor: that monitor is train_against, the rest held_out
    for target in monitors:
        rows.append({
            **base,
            "run_name": f"mbpp_{short}_{target['name']}_s{seed}_strict",
            "description": (f"Train AGAINST {target['name']} on MBPP-Honeypot (reward-hacking); hold out "
                           f"{', '.join(n for n in names if n != target['name'])}. All scored on the "
                           "fixed held-out eval set each eval_every (eval_metrics.jsonl)."),
            "monitors": [_spec(d, "train_against" if d["name"] == target["name"] else "held_out")
                         for d in monitors],
        })
    # control: pure reward, NO monitor penalty — all monitors held_out (baseline drift from reward alone)
    if control:
        rows.append({
            **base,
            "run_name": f"mbpp_{short}_control_s{seed}",
            "description": ("CONTROL: train on pure task reward, NO monitor in the gradient (all held_out). "
                           "Baseline for how much each monitor drifts from the policy changing at all."),
            "monitors": [_spec(d, "held_out") for d in monitors],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--no-cot", action="store_true", help="drop the 2 CoT monitors (non-CoT model M)")
    ap.add_argument("--skip", nargs="*", default=[], metavar="NAME", help="drop monitors by name")
    ap.add_argument("--no-control", action="store_true", help="omit the no-monitor control run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=60, help="OA uses 120; default 60 for a 'shortish' run")
    ap.add_argument("--lora-rank", type=int, default=64, help="OA uses 64 (alpha 128)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out-dir", default="experiments/configs/mbpp_matrix")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    monitors = all_monitors(args.model)
    drop = set(args.skip) | ({"cot_weak", "cot_strong"} if args.no_cot else set())
    monitors = [m for m in monitors if m["name"] not in drop]
    if len(monitors) < 2:
        ap.error(f"need ≥2 monitors; got {[m['name'] for m in monitors]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("row_*.json"):  # never mix old + new sets under a row_*.json glob
        stale.unlink()
    rows = build_rows(args.model, monitors, seed=args.seed, n_steps=args.n_steps,
                      lora_rank=args.lora_rank, max_tokens=args.max_tokens, control=not args.no_control)
    for r in rows:
        target = r["run_name"].rsplit("_s", 1)[0].split(f"{args.model.split('/')[-1]}_", 1)[1]
        p = out_dir / f"row_{target}.json"
        p.write_text(json.dumps(r, indent=2) + "\n")
        print(f"wrote {p}  (train_against={target if target != 'control' else 'NONE (control)'})")

    if args.validate:
        from monitordecorrelation.experiment_config import load_config
        for p in sorted(out_dir.glob("row_*.json")):
            cfg = load_config(p)
            ta = [m.name for m in cfg.monitors if m.role == "train_against"]
            ho = [m.name for m in cfg.monitors if m.role == "held_out"]
            print(f"✓ {p.name}: env={cfg.env} steps={cfg.n_steps} rank={cfg.lora_rank} "
                  f"train_against={ta or 'NONE'} held_out={ho}")


if __name__ == "__main__":
    main()
