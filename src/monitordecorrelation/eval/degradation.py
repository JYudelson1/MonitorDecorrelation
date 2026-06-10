"""Assemble the cross-method detector-degradation matrix.

The headline output: per-run, per-detector AUROC trajectories over training, plus the pairwise
**correlation matrix of those trajectories** across methods `{cot_weak, cot_strong, probe_ood,
probe_iid}`. CoT columns come from a run's ``metrics.jsonl`` (the live tracker's per-monitor
``cum_auroc``); probe columns come from the post-hoc ``probe_eval_<name>.jsonl`` files
(``eval_probes_on_run.py``). Partial columns are fine — start with cot_* + probe_ood.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _cot_series(run_dir: Path, metric: str = "cum_auroc") -> dict[str, dict[int, float]]:
    """{monitor_name: {step: auroc}} for every CoT monitor in metrics.jsonl."""
    rows = _read_jsonl(run_dir / "metrics.jsonl")
    series: dict[str, dict[int, float]] = {}
    for r in rows:
        step = r["step"]
        for k, v in r.items():
            if k.startswith("monitor/") and k.endswith(f"/{metric}"):
                name = k.split("/")[1]
                series.setdefault(name, {})[step] = v
    return series


def _probe_series(run_dir: Path) -> dict[str, dict[int, float]]:
    """{probe_name: {step: auroc}} from every probe_eval_<name>.jsonl in the run dir."""
    series: dict[str, dict[int, float]] = {}
    for path in sorted(run_dir.glob("probe_eval_*.jsonl")):
        name = path.stem[len("probe_eval_") :]
        series[name] = {r["step"]: r["auroc"] for r in _read_jsonl(path)}
    return series


def build_matrix(run_dir: str | Path) -> dict:
    """Assemble the step-indexed AUROC table + trajectory correlation matrix for one run.

    Writes ``degradation.json`` and ``degradation.csv`` into the run dir; returns the dict.
    """
    run_dir = Path(run_dir)
    columns: dict[str, dict[int, float]] = {}
    columns.update(_cot_series(run_dir))
    columns.update(_probe_series(run_dir))
    if not columns:
        raise ValueError(f"no detector columns found in {run_dir}")

    steps = sorted({s for col in columns.values() for s in col})
    names = sorted(columns)
    # Step-indexed table (None where a detector has no value / undefined AUROC at that step).
    table = {
        "steps": steps,
        "columns": {n: [columns[n].get(s) for s in steps] for n in names},
    }

    # Pairwise correlation of the degradation trajectories (over steps both detectors define).
    corr: dict[str, dict[str, float]] = {a: {} for a in names}
    for a in names:
        for b in names:
            common = [s for s in steps if columns[a].get(s) is not None and columns[b].get(s) is not None
                      and columns[a][s] == columns[a][s] and columns[b][s] == columns[b][s]]  # not NaN
            if len(common) < 2:
                corr[a][b] = float("nan")
                continue
            va = np.array([columns[a][s] for s in common])
            vb = np.array([columns[b][s] for s in common])
            if va.std() == 0 or vb.std() == 0:
                corr[a][b] = float("nan")
            else:
                corr[a][b] = float(np.corrcoef(va, vb)[0, 1])

    result = {"run": run_dir.name, "table": table, "correlation": corr}
    (run_dir / "degradation.json").write_text(json.dumps(result, indent=2))
    _write_csv(run_dir / "degradation.csv", steps, names, columns)
    return result


def _write_csv(path: Path, steps: list[int], names: list[str], columns: dict) -> None:
    lines = ["step," + ",".join(names)]
    for s in steps:
        vals = []
        for n in names:
            v = columns[n].get(s)
            vals.append("" if v is None or v != v else f"{v:.4f}")
        lines.append(f"{s}," + ",".join(vals))
    path.write_text("\n".join(lines) + "\n")
