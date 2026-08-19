"""Generate the ImpossibleBench cross-method degradation run set: the SAME run that
``scripts/train_impossiblebench_inkling.sh`` launches, except one non-probe monitor is
``train_against`` and the rest are ``held_out`` — one run per monitor, + a no-monitor control.

    uv run python experiments/make_ib_matrix_configs.py --validate     # 4 monitors + control, seed 0
    uv run python experiments/make_ib_matrix_configs.py --penalty-schedule 0 1   # hack-then-hide ramp

Rows are generated from the *base config file itself* (default
``experiments/configs/impossiblebench_inkling_small.json``), so every field except ``run_name`` /
``description`` / ``monitors`` / ``penalty_coef`` / ``penalty_schedule`` / ``seed`` is byte-identical
to the baseline run — that is the matrix-coherence invariant. **Regenerate, don't hand-edit.**

PROBES ARE DELIBERATELY ABSENT: ``ProbeMonitor`` does original-model probing and needs the policy's
base weights locally, which the tinker-hosted Inkling-Small does not have. See docs/MONITORS.md.

Run them with:  scripts/queue_runs.sh -c experiments/configs/ib_matrix -n <seeds> -j <jobs> -s -D
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The weak/strong judge ladder, shared with the MBPP matrix so the two run sets stay comparable.
from make_mbpp_matrix_configs import STRONG, WEAK  # noqa: E402 (sibling script, same dir on sys.path)

BASE_CONFIG = "experiments/configs/impossiblebench_inkling_small.json"


def all_monitors() -> list[dict]:
    """The 4 non-probe monitors: {weak, strong} × {CoT judge, output judge}."""
    return [
        {"kind": "cot", "name": "cot_weak", "role": "", "model_id": WEAK, "use_cot": True},
        {"kind": "cot", "name": "cot_strong", "role": "", "model_id": STRONG, "use_cot": True},
        {"kind": "cot", "name": "out_weak", "role": "", "model_id": WEAK, "use_cot": False},
        {"kind": "cot", "name": "out_strong", "role": "", "model_id": STRONG, "use_cot": False},
    ]


def _spec(d: dict, role: str) -> dict:
    return {"kind": d["kind"], "name": d["name"], "role": role,
            "model_id": d["model_id"], "use_cot": d["use_cot"]}


def build_rows(base: dict, monitors: list[dict], *, seed: int, penalty_coef: float,
               penalty_schedule: list | None, control: bool, tag: str = "") -> list[dict]:
    short = base["policy"].split("/")[-1]
    suf = f"_{tag}" if tag else ""
    sched = ({"start_penalty": penalty_schedule[0], "end_penalty": penalty_schedule[1]}
             if penalty_schedule else None)
    common = {**base, "seed": seed, "penalty_coef": penalty_coef, "penalty_schedule": sched}
    names = [m["name"] for m in monitors]
    rows = []
    for target in monitors:  # one row per monitor: it is train_against, the rest held_out
        rows.append({
            **common,
            "run_name": f"ib_{short}_{target['name']}_s{seed}{suf}",
            "description": (
                f"Train AGAINST {target['name']} on ImpossibleBench (passing the impossible tests IS "
                f"the cheat); hold out {', '.join(n for n in names if n != target['name'])}. All four "
                "are scored on the fixed held-out eval set every eval_every (eval_metrics.jsonl)."),
            "monitors": [_spec(d, "train_against" if d["name"] == target["name"] else "held_out")
                         for d in monitors],
        })
    if control:
        rows.append({
            **common,
            "penalty_coef": 0.0,      # nothing is train_against, so λ is inert — pin it to 0 anyway
            "penalty_schedule": None,  # so the control config can't be misread as penalized
            "run_name": f"ib_{short}_control_s{seed}{suf}",
            "description": ("CONTROL: pure task reward, NO monitor in the gradient (all held_out). "
                            "Baseline for how much each monitor drifts from the policy changing at all."),
            "monitors": [_spec(d, "held_out") for d in monitors],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_CONFIG, help="config the rows are derived from")
    ap.add_argument("--skip", nargs="*", default=[], metavar="NAME", help="drop monitors by name")
    ap.add_argument("--no-control", action="store_true", help="omit the no-monitor control run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--penalty-coef", type=float, default=1.0,
                    help="train-against penalty weight (the base config's 0.0 is a no-penalty baseline)")
    ap.add_argument("--penalty-schedule", type=float, nargs=2, metavar=("START", "END"), default=None,
                    help="ramp λ linearly START→END over training; overrides --penalty-coef")
    ap.add_argument("--tag", default="", help="run-name marker for this run set (e.g. lowpen)")
    ap.add_argument("--out-dir", default="experiments/configs/ib_matrix")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    base = json.loads(Path(args.base).read_text())
    base.pop("run_name", None)       # per-row below
    base.pop("description", None)
    base.pop("monitors", None)
    base["experiment"] = "ib_degradation_matrix"

    monitors = [m for m in all_monitors() if m["name"] not in set(args.skip)]
    if len(monitors) < 2:
        ap.error(f"need ≥2 monitors; got {[m['name'] for m in monitors]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("row_*.json"):  # never mix old + new sets under one row_*.json glob
        stale.unlink()
    rows = build_rows(base, monitors, seed=args.seed, penalty_coef=args.penalty_coef,
                      penalty_schedule=args.penalty_schedule, control=not args.no_control,
                      tag=args.tag)
    for r in rows:
        ta = [m["name"] for m in r["monitors"] if m["role"] == "train_against"]
        target = ta[0] if ta else "control"
        p = out_dir / f"row_{target}.json"
        p.write_text(json.dumps(r, indent=2) + "\n")
        print(f"wrote {p}  (train_against={target if target != 'control' else 'NONE (control)'})")

    if args.validate:
        from monitordecorrelation.experiment_config import load_config
        for p in sorted(out_dir.glob("row_*.json")):
            cfg = load_config(p)
            ta = [m.name for m in cfg.monitors if m.role == "train_against"]
            ho = [m.name for m in cfg.monitors if m.role == "held_out"]
            print(f"✓ {p.name}: env={cfg.env} policy={cfg.policy} steps={cfg.n_steps} "
                  f"λ={cfg.penalty_schedule or cfg.penalty_coef} train_against={ta or 'NONE'} "
                  f"held_out={ho}")


if __name__ == "__main__":
    main()
