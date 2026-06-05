"""Plot a run's metrics locally (no W&B server needed).

Reads data/runs/<run>/metrics.jsonl and writes PNGs to the same dir:
  - ground_truth.png : the primary sycophancy-rate curve + reward
  - monitors.png     : per-monitor accuracy + AUROC (per-step and cumulative) over training

Run: uv run python experiments/plot_run.py --run smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (run_dir / "metrics.jsonl").open()]


def _series(rows: list[dict], key: str) -> tuple[list, list]:
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] == r[key]:  # skip NaN
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="smoke")
    args = ap.parse_args()
    run_dir = Path("data/runs") / args.run
    rows = _load(run_dir)
    if not rows:
        raise SystemExit(f"no metrics in {run_dir}")

    monitor_names = sorted({k.split("/")[1] for r in rows for k in r if k.startswith("monitor/")})

    # Decode monitor roles + models from run_info.json so labels say which was trained against.
    role_label: dict[str, str] = {}
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        for m in info.get("train_against", []):
            role_label[m["name"]] = f"{m['name']} = {m.get('model_id')} [TRAIN-AGAINST]"
        for m in info.get("held_out", []):
            role_label[m["name"]] = f"{m['name']} = {m.get('model_id')} [held-out]"
    label_for = lambda name: role_label.get(name, name)

    # Plot 1: ground truth + reward
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [
        ("behavior_rate", "ground-truth sycophancy rate"),
        ("reward/penalty_mean", "train-monitor penalty"),
    ]:
        xs, ys = _series(rows, key)
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("rate")
    ax.set_title(f"{args.run}: ground truth (primary)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "ground_truth.png", dpi=120)

    # Plot 2: per-monitor accuracy + AUROC
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, metric in zip(axes, ["accuracy", "auroc"]):
        for name in monitor_names:
            lab = label_for(name)
            xs, ys = _series(rows, f"monitor/{name}/{metric}")
            ax.plot(xs, ys, marker="o", label=f"{lab} (per-step)")
            xc, yc = _series(rows, f"monitor/{name}/cum_{metric}")
            ax.plot(xc, yc, marker="x", linestyle="--", label=f"{name} (cumulative)")
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.set_title(f"monitor {metric} vs ground truth")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle(f"{args.run}: detector degradation curves")
    fig.tight_layout()
    fig.savefig(run_dir / "monitors.png", dpi=120)

    print(f"wrote {run_dir/'ground_truth.png'} and {run_dir/'monitors.png'}")


if __name__ == "__main__":
    main()
