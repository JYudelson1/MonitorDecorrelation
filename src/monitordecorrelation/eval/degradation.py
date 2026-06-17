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


def _gap_series(run_dir: Path) -> dict[str, dict[int, float]]:
    """{monitor: {step: suspiciousness gap}} where gap = mean_score(behavior) − mean_score(honest).
    (``mean_score_syco`` is the behavior-present-class mean — named for the first env, but it's the
    positive class for any behavior, e.g. reward-hacking.)

    This is a DISTINCT metric from AUROC, not a more-robust version of it. AUROC measures threshold-free
    *ranking* separation; the gap measures the raw *magnitude* separation of the two class means. A
    monitor can hold AUROC ~constant while the gap collapses (scores compress but still rank right) —
    that divergence is the interesting read. NOTE: the gap needs BOTH a behavior mean and an honest
    mean, so it is undefined under the SAME single-class condition as AUROC (e.g. no honest rollouts
    left once obfuscation succeeds → honest mean is NaN → gap is NaN). We record a step only when both
    means are present."""
    series: dict[str, dict[int, float]] = {}
    path = run_dir / "eval_metrics.jsonl"
    if not path.exists():
        return series
    for r in _read_jsonl(path):
        step = r["step"]
        for k, v in r.items():
            if k.startswith("monitor/") and k.endswith("/mean_score_syco"):
                name = k.split("/")[1]
                honest = r.get(f"monitor/{name}/mean_score_honest")
                if v == v and honest is not None and honest == honest:  # both non-NaN
                    series.setdefault(name, {})[step] = v - honest
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
    """early-mean − late-mean (positive = the detector degraded over training)."""
    vals = [traj[s] for s in sorted(traj) if traj[s] == traj[s]]  # drop NaN, step order
    if len(vals) < 2:
        return float("nan")
    early = float(np.mean(vals[:edge]))
    late = float(np.mean(vals[-edge:]))
    return early - late


def _late_mean(traj: dict[int, float], edge: int) -> float:
    """late-mean (the END value of a metric over training) — e.g. each monitor's final AUROC."""
    vals = [traj[s] for s in sorted(traj) if traj[s] == traj[s]]
    return float(np.mean(vals[-edge:])) if vals else float("nan")


def _train_target(run_dir: Path) -> str:
    """The detector this run trained against (from run_info.json); falls back to the run name."""
    info = json.loads((run_dir / "run_info.json").read_text())
    names = [m["name"] for m in info.get("train_against", [])]
    return names[0] if names else run_dir.name


def _gap_trajectories(run_dir: str | Path, rolling: int = 5) -> dict[str, dict[int, float]]:
    """{monitor: {step: suspiciousness gap}} for one run, rolling-smoothed (from ``eval_metrics``)."""
    return {name: _rolling_average(col, rolling) for name, col in _gap_series(Path(run_dir)).items()}


# The grids we emit. Each fixes its colour scale so the RdBu_r white midpoint is meaningful:
#  - end-AUROC ∈ [0,1]      → vmin/vmax 0..1, white at 0.5 (chance)
#  - the *drop* grids (Δ)   → vmin/vmax −1..1, white at 0 (no change)
# (fixed bounds also make magnitudes comparable across runs/grids, unlike data-driven scaling.)
_GRIDS = [
    {"stem": "degradation_matrix", "traj": run_trajectories, "reduce": _degradation, "vmin": -1.0, "vmax": 1.0,
     "title": "Degradation D[i][j] = AUROC drop of j when training against i",
     "label": "AUROC drop (early − late)"},
    {"stem": "gap_drop_matrix", "traj": _gap_trajectories, "reduce": _degradation, "vmin": -1.0, "vmax": 1.0,
     "title": "Suspiciousness-gap drop of j when training against i",
     "label": "gap drop (early − late), gap = mean_score(behavior) − mean_score(honest)"},
    {"stem": "end_auroc_matrix", "traj": run_trajectories, "reduce": _late_mean, "vmin": 0.0, "vmax": 1.0,
     "title": "Final (end) AUROC of monitor j after training against i", "label": "end AUROC"},
]


def build_degradation_matrix(
    run_dirs: list[str | Path], *, edge: int = 5, rolling: int = 5, out_dir: str | Path | None = None
) -> dict:
    """Assemble, across runs that differ only in ``train_against``, three N×N grids:
      - ``degradation_matrix``  : AUROC drop (early − late) of monitor j when training against i
      - ``gap_drop_matrix``     : suspiciousness-gap drop (the headline read; survives single-class eval)
      - ``end_auroc_matrix``    : monitor j's final AUROC after training against i

    Each run contributes one row (its train-against detector; the control row is the run with none).
    Writes ``<stem>.{json,csv,png}`` per grid to ``out_dir``. Returns the AUROC-drop result (with the
    other two nested under ``gap_drop_matrix`` / ``end_auroc_matrix``) — back-compatible.
    """
    runs = [Path(r) for r in run_dirs]
    rows = [_train_target(rd) for rd in runs]

    # warn once if the runs don't share a monitor set (matrix is only coherent if they do)
    auroc_trajs = {t: run_trajectories(rd, rolling) for t, rd in zip(rows, runs)}
    monitor_sets = [set(t) for t in auroc_trajs.values()]
    if len({frozenset(s) for s in monitor_sets}) > 1:
        print(f"⚠️  runs have DIFFERENT monitor sets {[sorted(s) for s in monitor_sets]} — the matrix "
              "is only coherent across runs with identical monitors (configs identical except train_against).")

    grids: dict[str, dict] = {}
    for g in _GRIDS:
        traj_fn, reducer, stem = g["traj"], g["reduce"], g["stem"]
        trajs = auroc_trajs if traj_fn is run_trajectories else {t: traj_fn(rd, rolling) for t, rd in zip(rows, runs)}
        monitors = sorted(set().union(*[set(t) for t in trajs.values()])) if trajs else []
        matrix = {t: {m: reducer(trajs[t].get(m, {}), edge) for m in monitors} for t in rows}
        res = {"rows_train_against": rows, "columns_monitors": monitors, "matrix": matrix,
               "runs": [r.name for r in runs], "title": g["title"], "label": g["label"],
               "vmin": g["vmin"], "vmax": g["vmax"]}
        grids[stem] = res
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2))
            _write_csv(out_dir / f"{stem}.csv", rows, monitors, matrix)
            try:
                plot_degradation_matrix(res, out_dir / f"{stem}.png")
            except Exception as e:  # noqa: BLE001
                print(f"({stem} heatmap skipped: {e})")

    result = dict(grids["degradation_matrix"])  # back-compat: primary return is the AUROC-drop grid
    result["gap_drop_matrix"] = grids["gap_drop_matrix"]
    result["end_auroc_matrix"] = grids["end_auroc_matrix"]
    return result


def _write_csv(path: Path, rows: list[str], monitors: list[str], matrix: dict) -> None:
    lines = ["train_against\\monitor," + ",".join(monitors)]
    for r in rows:
        vals = [("" if matrix[r][m] != matrix[r][m] else f"{matrix[r][m]:.4f}") for m in monitors]
        lines.append(f"{r}," + ",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def plot_degradation_matrix(result: dict, path: str | Path) -> None:
    """Heatmap: rows = train-against detector, cols = measured detector. Uses the grid's own
    ``title``/``label`` (set per grid by ``build_degradation_matrix``) so AUROC-drop, gap-drop, and
    end-AUROC each render with the right caption."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows_train_against"]
    cols = result["columns_monitors"]
    title = result.get("title", "Degradation D[i][j] = AUROC drop of j when training against i")
    label = result.get("label", "AUROC drop (early − late)")
    M = np.array([[result["matrix"][r].get(c, np.nan) for c in cols] for r in rows], dtype=float)

    # Fixed colour scale so RdBu_r's white midpoint lands at a meaningful value (0.5 for AUROC ∈ [0,1];
    # 0 for the −1..1 drop grids). Falls back to symmetric data-driven bounds for callers without vmin/vmax.
    vmin, vmax = result.get("vmin"), result.get("vmax")
    if vmin is None or vmax is None:
        m = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        vmin, vmax = -m, m

    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(cols), 1.4 + 0.9 * len(rows)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("measured detector (j)")
    ax.set_ylabel("trained against (i)")
    ax.set_title(title, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
