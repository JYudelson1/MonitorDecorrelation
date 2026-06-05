"""Render a run's metrics to PNGs (no W&B server needed).

Used both by ``scripts/plot_run.py`` and automatically at the end of every ``run_grpo`` run. Reads
``metrics.jsonl`` (+ ``run_info.json`` for monitor role labels) and writes ``ground_truth.png`` and
``monitors.png`` into the run dir. matplotlib is imported lazily so importing this module is cheap.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def _series(rows: list[dict], key: str) -> tuple[list, list]:
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] == r[key]:  # skip NaN
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def plot_run(run_dir: Path) -> list[Path]:
    """Write ground_truth.png + monitors.png for the run at ``run_dir``. Returns the paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    rows = _load(run_dir)
    if not rows:
        raise ValueError(f"no metrics in {run_dir}")
    run = run_dir.name
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

    out: list[Path] = []

    # Plot 1: ground truth + reward
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [
        ("behavior_rate", "ground-truth behavior rate"),
        ("reward/penalty_mean", "train-monitor penalty"),
    ]:
        xs, ys = _series(rows, key)
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("rate")
    ax.set_title(f"{run}: ground truth (primary)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    p1 = run_dir / "ground_truth.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    out.append(p1)

    # Plot 2: per-monitor accuracy + AUROC (per-step solid, cumulative dashed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, metric in zip(axes, ["accuracy", "auroc"]):
        for name in monitor_names:
            lab = role_label.get(name, name)
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
    fig.suptitle(f"{run}: detector degradation curves")
    fig.tight_layout()
    p2 = run_dir / "monitors.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    out.append(p2)

    return out
