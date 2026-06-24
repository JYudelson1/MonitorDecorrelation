"""Pivot a monitor-bakeoff results json into a per-model table: CoT-AUROC, output-AUROC, %parsed.

    uv run python experiments/bakeoff_table.py [data/runs/monitor_bakeoff_rollouts.json]
"""

from __future__ import annotations

import json
import sys


def _auroc(d: dict) -> str:
    v = d.get("auroc")
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "  -  "


def _parse(d: dict) -> str:
    v = d.get("parse")
    return f"{v:.0%}" if isinstance(v, (int, float)) and v == v else "  - "


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/runs/monitor_bakeoff_rollouts.json"
    data = json.loads(open(path).read())
    by_model: dict[str, dict] = {}
    for r in data["monitors"]:
        entry = by_model.setdefault(r["model_id"], {})
        if "error" in r:
            entry[r["mode"]] = {"auroc": float("nan"), "parse": float("nan")}
        else:
            entry[r["mode"]] = {"auroc": r["auroc"], "parse": r.get("parse_success_rate", 1.0)}

    rows = sorted(by_model.items(), key=lambda kv: -(kv[1].get("cot", {}).get("auroc") or -1))
    print(f"{'model':34s} {'cot-AUROC':>9} {'out-AUROC':>9} {'cot %p':>7} {'out %p':>7}")
    print("-" * 72)
    for mid, d in rows:
        cot, out = d.get("cot", {}), d.get("output", {})
        print(f"{mid.split('/')[-1]:34s} {_auroc(cot):>9} {_auroc(out):>9} "
              f"{_parse(cot):>7} {_parse(out):>7}")


if __name__ == "__main__":
    main()
