"""Post-hoc monitors: held-out monitors added to a FINISHED run by re-scoring its saved eval rollouts.

``experiments/add_posthoc_monitors.py`` scores a run's full ``eval_rollouts.jsonl`` with a new monitor
(e.g. a weak judge picked after the batch ran) and writes, per monitor, a sidecar in the run dir:

* ``posthoc/<name>.scores.jsonl`` — one ``{line, step, key, record}`` per scored rollout (``record`` =
  ``rollout_dump.monitor_record``: score, label and the judge's full call), the resumable cache;
* ``posthoc/<name>.info.json``    — the monitor spec, where it came from, when;
* ``eval_metrics_posthoc_<name>.jsonl`` — ``{step, monitor/<name>/…}`` rows with exactly the columns a
  live held-out monitor gets in ``eval_metrics.jsonl`` (``rl.train.eval_monitor_metrics``).

The run's own ``eval_metrics.jsonl`` is never rewritten. ``with_posthoc`` merges the sidecars into its
rows by step, so every analysis reader (degradation matrix, plots, coupling, …) sees a post-hoc monitor
as one more held-out column. A post-hoc monitor never touched the gradient by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

POSTHOC_METRICS_GLOB = "eval_metrics_posthoc_*.jsonl"


def posthoc_metrics_path(run_dir: str | Path, name: str) -> Path:
    return Path(run_dir) / f"eval_metrics_posthoc_{name}.jsonl"


def with_posthoc(run_dir: str | Path, rows: list[dict]) -> list[dict]:
    """``rows`` (a run's parsed ``eval_metrics.jsonl``) with every post-hoc sidecar's columns merged in
    by step. A sidecar column that already exists in the row is an error (it would silently shadow a
    live monitor), as is a sidecar step the run has no eval row for."""
    sidecars = sorted(Path(run_dir).glob(POSTHOC_METRICS_GLOB))
    if not sidecars:
        return rows
    by_step = {r["step"]: dict(r) for r in rows}
    for path in sidecars:
        with path.open() as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                extra = json.loads(line)
                step = extra.pop("step")
                if step not in by_step:
                    raise ValueError(f"{path}:{i}: step {step} has no row in the run's eval_metrics.jsonl")
                clash = set(extra) & set(by_step[step])
                if clash:
                    raise ValueError(f"{path}:{i}: columns {sorted(clash)} already exist at step {step}")
                by_step[step].update(extra)
    return [by_step[r["step"]] for r in rows]
