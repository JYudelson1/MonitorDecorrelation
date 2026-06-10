"""The detector-degradation MATRIX — a **cross-run** artifact (never per-run).

A single run trains against ONE detector and watches the others, so it yields exactly **one row** of
the matrix — not the matrix. The matrix's unit is a **set of runs whose configs are identical except
for which detector is `train_against`**. Then:

    D[i][j] = how much training AGAINST detector i degrades detector j
            = drop in detector j's held-out AUROC over training in the run that trained against i
              (early-mean − late-mean; positive = degraded).

The diagonal D[i][i] is the train-against target degrading itself (expected large). Build it
explicitly from the runs with ``build_degradation_matrix([...run dirs...])`` — it is **not** saved
after any individual training run.

Per-run AUROC trajectories come from ``metrics.jsonl`` (CoT monitors, live) + post-hoc
``probe_eval_<name>.jsonl`` (probes); both use per-step ``auroc`` so they're comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _rolling_average(series: dict[int, float], window: int) -> dict[int, float]:
    """Trailing mean over each column's own sorted steps (NaN-skipping). window=1 -> raw per-step."""
    if window <= 1:
        return dict(series)
    steps = sorted(series)
    vals = [series[s] for s in steps]
    out: dict[int, float] = {}
    for i, s in enumerate(steps):
        w = [v for v in vals[max(0, i - window + 1) : i + 1] if v == v]  # drop NaN
        out[s] = sum(w) / len(w) if w else float("nan")
    return out


def _eval_series(run_dir: Path, metric: str = "auroc") -> dict[str, dict[int, float]]:
    """{monitor: {step: auroc}} for every monitor in the HELD-OUT ``eval_metrics.jsonl`` (the clean
    degradation curves — not the noisy on-policy training rollouts)."""
    series: dict[str, dict[int, float]] = {}
    path = run_dir / "eval_metrics.jsonl"
    if not path.exists():
        return series
    for r in _read_jsonl(path):
        step = r["step"]
        for k, v in r.items():
            if k.startswith("monitor/") and k.endswith(f"/{metric}"):
                series.setdefault(k.split("/")[1], {})[step] = v
    return series


def _probe_series(run_dir: Path) -> dict[str, dict[int, float]]:
    """{probe: {step: auroc}} from every probe_eval_<name>.jsonl in the run dir."""
    series: dict[str, dict[int, float]] = {}
    for path in sorted(run_dir.glob("probe_eval_*.jsonl")):
        name = path.stem[len("probe_eval_") :]
        series[name] = {r["step"]: r["auroc"] for r in _read_jsonl(path)}
    return series


def run_trajectories(run_dir: str | Path, rolling: int = 5) -> dict[str, dict[int, float]]:
    """{monitor: {step: auroc}} for one run, rolling-smoothed. Held-out monitors come from the eval
    phase (``eval_metrics.jsonl``); post-hoc probes from ``probe_eval_*.jsonl``."""
    run_dir = Path(run_dir)
    cols: dict[str, dict[int, float]] = {}
    cols.update(_eval_series(run_dir))
    cols.update(_probe_series(run_dir))
    return {name: _rolling_average(col, rolling) for name, col in cols.items()}


def _degradation(traj: dict[int, float], edge: int) -> float:
    """early-mean − late-mean AUROC (positive = the detector degraded over training)."""
    vals = [traj[s] for s in sorted(traj) if traj[s] == traj[s]]  # drop NaN, step order
    if len(vals) < 2:
        return float("nan")
    early = float(np.mean(vals[:edge]))
    late = float(np.mean(vals[-edge:]))
    return early - late


def _train_target(run_dir: Path) -> str:
    """The detector this run trained against (from run_info.json); falls back to the run name."""
    info = json.loads((run_dir / "run_info.json").read_text())
    names = [m["name"] for m in info.get("train_against", [])]
    return names[0] if names else run_dir.name


def build_degradation_matrix(
    run_dirs: list[str | Path], *, edge: int = 5, rolling: int = 5, out_dir: str | Path | None = None
) -> dict:
    """Assemble D[train_against_i][monitor_j] across runs that differ only in ``train_against``.

    Each run contributes one row (its train-against detector). Columns are the union of monitors.
    Writes ``degradation_matrix.{json,csv}`` (+ a heatmap if matplotlib available) to ``out_dir``.
    """
    runs = [Path(r) for r in run_dirs]
    rows: list[str] = []
    trajs_by_run: dict[str, dict[str, dict[int, float]]] = {}
    for rd in runs:
        target = _train_target(rd)
        rows.append(target)
        trajs_by_run[target] = run_trajectories(rd, rolling)

    monitor_sets = [set(t) for t in trajs_by_run.values()]
    monitors = sorted(set().union(*monitor_sets)) if monitor_sets else []
    if len({frozenset(s) for s in monitor_sets}) > 1:
        print(f"⚠️  runs have DIFFERENT monitor sets {[sorted(s) for s in monitor_sets]} — the matrix "
              "is only coherent across runs with identical monitors (configs identical except train_against).")

    matrix = {
        target: {m: _degradation(trajs_by_run[target].get(m, {}), edge) for m in monitors}
        for target in rows
    }
    result = {"rows_train_against": rows, "columns_monitors": monitors, "matrix": matrix,
              "runs": [r.name for r in runs]}

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "degradation_matrix.json").write_text(json.dumps(result, indent=2))
        _write_csv(out_dir / "degradation_matrix.csv", rows, monitors, matrix)
        try:
            plot_degradation_matrix(result, out_dir / "degradation_matrix.png")
        except Exception as e:  # noqa: BLE001
            print(f"(degradation heatmap skipped: {e})")
    return result


def _write_csv(path: Path, rows: list[str], monitors: list[str], matrix: dict) -> None:
    lines = ["train_against\\monitor," + ",".join(monitors)]
    for r in rows:
        vals = [("" if matrix[r][m] != matrix[r][m] else f"{matrix[r][m]:.4f}") for m in monitors]
        lines.append(f"{r}," + ",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def plot_degradation_matrix(result: dict, path: str | Path) -> None:
    """Heatmap: rows = train-against detector, cols = measured detector, color = AUROC drop."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows_train_against"]
    cols = result["columns_monitors"]
    M = np.array([[result["matrix"][r].get(c, np.nan) for c in cols] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(cols), 1.4 + 0.9 * len(rows)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-np.nanmax(np.abs(M)) if np.isfinite(M).any() else -1,
                   vmax=np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("measured detector (j)")
    ax.set_ylabel("trained against (i)")
    ax.set_title("Degradation D[i][j] = AUROC drop of j when training against i")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUROC drop (early − late)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
