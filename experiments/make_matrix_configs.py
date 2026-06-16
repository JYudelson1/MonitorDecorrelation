"""Emit the degradation-MATRIX run set: one config per detector, identical except which detector is
``train_against``. The matrix is only coherent if the runs differ ONLY in ``train_against`` (see
``eval/degradation.py``), so generating them from one base enforces that invariant by construction.

    uv run python experiments/make_matrix_configs.py            # writes experiments/configs/matrix8b/*.json
    uv run python experiments/make_matrix_configs.py --validate # also load_config() each (parse-only, no model)

Then run all rows (GPU/tinker), and assemble:
    for c in experiments/configs/matrix8b/row_*.json; do uv run python experiments/run_experiment.py --config "$c"; done
    uv run python -c "from monitordecorrelation.eval.degradation import build_degradation_matrix as b; \
        b(['data/runs/matrix8b_cot_weak','data/runs/matrix8b_cot_strong','data/runs/matrix8b_probe_ood','data/runs/matrix8b_probe_iid'], out_dir='data/runs/matrix8b')"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The 4 detectors = the {weak_cot, strong_cot, iid_probe, ood_probe} cross-method matrix on Qwen3-8B.
# probe_model MUST match the policy base (off-model probing is unreliable) — see docs/MONITORS.md.
POLICY = "Qwen/Qwen3-8B"
DETECTORS = [
    {"kind": "cot", "name": "cot_weak", "model_id": "anthropic/claude-3.5-haiku"},
    {"kind": "cot", "name": "cot_strong", "model_id": "anthropic/claude-sonnet-4.6"},
    {"kind": "probe", "name": "probe_ood",
     "probe_path": "data/probes/Qwen3-8B/diverse_deception", "probe_model": POLICY},
    {"kind": "probe", "name": "probe_iid",
     "probe_path": "data/probes/Qwen3-8B/sycophancy_cot", "probe_model": POLICY},
]

# Base RL config shared by every row. max_tokens=4096 so the policy finishes its <think> + emits a
# parseable answer — probes were fit on FINISHED rollouts and scoring/penalizing truncated text is junk
# (the train_against_iid_probe.json rationale). One value for ALL rows keeps the matrix coherent.
BASE = {
    "experiment": "degradation_matrix_8b",
    "policy": POLICY,
    "backend": "tinker",
    "subset": "nlp",
    "n_steps": 40,
    "batch_size": 8,
    "group_size": 8,
    "eval_every": 10,
    "eval_size": 32,
    "max_tokens": 4096,
    "penalty_coef": 1.0,
}


def _monitor(d: dict, role: str) -> dict:
    """A detector spec dict tagged with its role for this row."""
    m = {"kind": d["kind"], "name": d["name"], "role": role}
    if d["kind"] == "cot":
        m["model_id"] = d["model_id"]
    else:
        m["probe_path"] = d["probe_path"]
        m["probe_model"] = d["probe_model"]
    return m


def build_rows(detectors: list[dict]) -> list[dict]:
    rows = []
    for target in detectors:
        monitors = [_monitor(d, "train_against" if d["name"] == target["name"] else "held_out")
                    for d in detectors]
        rows.append({
            **BASE,
            "run_name": f"matrix8b_{target['name']}",
            "description": (
                f"Degradation-matrix row: train AGAINST {target['name']}, hold out the other 3 "
                f"detectors ({', '.join(d['name'] for d in DETECTORS if d['name'] != target['name'])}). "
                "All 4 monitors are scored on the fixed held-out eval set every eval_every steps "
                "(eval_metrics.jsonl). Identical to the other rows except which detector is train_against."
            ),
            "monitors": monitors,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/configs/matrix8b")
    ap.add_argument("--skip", nargs="*", default=[], metavar="NAME",
                    help="detector names to drop, e.g. `--skip probe_iid` for the runnable-today 3-method set")
    ap.add_argument("--validate", action="store_true", help="load_config() each (parse-only; no model load)")
    args = ap.parse_args()

    detectors = [d for d in DETECTORS if d["name"] not in args.skip]
    if len(detectors) < 2:
        ap.error(f"need ≥2 detectors for a matrix; got {[d['name'] for d in detectors]}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("row_*.json"):  # so a `row_*.json` glob never mixes old + new sets
        stale.unlink()
    rows = build_rows(detectors)
    for r in rows:
        target = r["run_name"].removeprefix("matrix8b_")
        p = out_dir / f"row_{target}.json"
        p.write_text(json.dumps(r, indent=2) + "\n")
        print(f"wrote {p}  (train_against={target})")

    if args.validate:
        from monitordecorrelation.experiment_config import load_config
        for p in sorted(out_dir.glob("row_*.json")):
            cfg = load_config(p)
            ta = [m.name for m in cfg.monitors if m.role == "train_against"]
            ho = [m.name for m in cfg.monitors if m.role == "held_out"]
            print(f"✓ {p.name}: valid | train_against={ta} held_out={ho}")


if __name__ == "__main__":
    main()
