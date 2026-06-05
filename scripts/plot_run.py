"""Render a run's metrics to PNGs locally (no W&B server).

Runs auto-plot on finish too; use this to (re)plot a finished or in-progress run.
Run: uv run python scripts/plot_run.py --run run_nlp40
"""

from __future__ import annotations

import argparse
from pathlib import Path

from monitordecorrelation.eval.plots import plot_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="smoke")
    args = ap.parse_args()
    paths = plot_run(Path("data/runs") / args.run)
    print("wrote " + " and ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
