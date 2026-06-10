"""Post-hoc: score a saved probe over a training run's rollouts, per step.

The held-out white-box read for the degradation matrix. We reconstruct (rollout, ground_truth) pairs
from a run's ``rollouts.jsonl`` (works for any run that saved them), run them through a ``ProbeMonitor``
in one batched forward, and report per-step AUROC vs the env oracle ``behavior_present``. This is the
``probe_*`` column of the cross-method degradation matrix — never wired into the RL loop.

Run:
  uv run python experiments/eval_probes_on_run.py \
      --rollouts data/runs/run_nlp40/rollouts.jsonl --probe data/probes/ood --name probe_ood
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from monitordecorrelation.eval.metrics import roc_auc
from monitordecorrelation.monitors.probe_monitor import ProbeMonitor
from monitordecorrelation.rl.rollout import load_saved_rollouts
from monitordecorrelation.whitebox.model import WhiteBoxModel
from monitordecorrelation.whitebox.probe import LinearProbe

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="path to a run's rollouts.jsonl")
    ap.add_argument("--probe", required=True, help="dir saved by train_probe.py")
    ap.add_argument("--name", default="probe", help="column name in the degradation matrix")
    ap.add_argument("--model", default=None, help="override; default = probe meta's model_name")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: <run_dir>/probe_eval_<name>.jsonl")
    args = ap.parse_args()

    probe = LinearProbe.load(args.probe)
    model_name = args.model or probe.meta.get("model_name", "Qwen/Qwen3-0.6B")
    model = WhiteBoxModel(model_name)
    monitor = ProbeMonitor(args.name, model, probe, batch_size=args.batch_size)

    pairs = load_saved_rollouts(args.rollouts)
    by_step: dict[int, list[tuple]] = defaultdict(list)
    for rollout, gt in pairs:
        by_step[rollout.meta.get("step", 0)].append((rollout, gt))

    out_path = Path(args.out) if args.out else Path(args.rollouts).parent / f"probe_eval_{args.name}.jsonl"
    rows = []
    print(f"{'step':>5} {'auroc':>7} {'n':>5} {'mean_pos':>9} {'mean_neg':>9}")
    for step in sorted(by_step):
        items = by_step[step]
        rollouts = [r for r, _ in items]
        gt = [g for _, g in items]
        results = monitor.score_batch(rollouts)
        scores = [r.score for r in results]
        auc = roc_auc(scores, gt)
        pos = [s for s, g in zip(scores, gt) if g]
        neg = [s for s, g in zip(scores, gt) if not g]
        row = {
            "step": step,
            "auroc": auc,
            "n": len(items),
            "mean_pos": mean(pos) if pos else float("nan"),
            "mean_neg": mean(neg) if neg else float("nan"),
        }
        rows.append(row)
        print(f"{step:>5} {auc:>7.3f} {len(items):>5} {row['mean_pos']:>9.3f} {row['mean_neg']:>9.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
