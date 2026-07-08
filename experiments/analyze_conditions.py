#!/usr/bin/env python
"""Per-condition (train-against monitor × penalty regime) summary dashboard.

For each condition (averaged over seeds): the drop in the train-against monitor's d′, the change in its
μ_hack (evasion) and μ_clean (honest-drift), the max held-out hack rate, and — as a stacked bar — the
distribution of FINAL hack rates across that condition's runs.

    uv run python experiments/analyze_conditions.py 'data/runs/**/mbpp_Qwen3-8B_*' '~/dl/mbpp_*' --out out/

See src/monitordecorrelation/eval/condition_summary.py for the metric definitions.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from monitordecorrelation.eval.condition_summary import (
    DPRIME_BUCKETS, HACK_BUCKETS, MUHACK_END_BUCKETS, condition_table,
)
from monitordecorrelation.eval.coupling import display_name, monitor_names

# mean-bar columns: (key, title, diverging?) — diverging ones get a signed 0-centered bar
_MEAN_COLS = [("dmu_hack", "Δμ_hack\n(evasion↓)", True),
              ("dmu_clean", "Δμ_clean\n(honest-drift↑)", True),
              ("max_hack_rate", "max hack\nrate", False)]
# stacked-distribution columns: (dist_key, n_key, buckets, colors, title)
_DIST_COLS = [
    ("dprime_dist", "dprime_dist_n", DPRIME_BUCKETS,
     ["#b03a2e", "#e08a4b", "#cccccc", "#9ecae1", "#3a7ca5"], "Δd′ distribution\n(drop → sharper)"),
    ("muhack_end_dist", "muhack_end_dist_n", MUHACK_END_BUCKETS,
     ["#b03a2e", "#e08a4b", "#f2d17b", "#a6cf98", "#4a8b3a"], "final μ_hack\n(evaded → caught)"),
    ("hack_dist", "hack_dist_n", HACK_BUCKETS,
     ["#5f8fb0", "#9ecae1", "#f2d17b", "#e08a4b", "#b03a2e"], "final hack-rate\n(≈0 → all-hack)"),
]
_TARGET_ORDER = ["cot_weak", "cot_strong", "out_weak", "out_strong", "probe_iid", "probe_ood", "control"]


def _sort_key(row):
    t = row["target"]
    ti = _TARGET_ORDER.index(t) if t in _TARGET_ORDER else len(_TARGET_ORDER)
    # penalty order: numeric constants ascending, then schedules / non-numeric ("any") last
    p = row["penalty"]
    try:
        pk = (1, 0.0) if "→" in p else (0, float(p))
    except ValueError:
        pk = (2, 0.0)
    return (ti, pk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_glob", nargs="+", help="one or more globs for run dirs (unioned)")
    ap.add_argument("--out", default="data/runs/_conditions")
    args = ap.parse_args()

    run_dirs = sorted({d for pat in args.runs_glob for d in glob.glob(os.path.expanduser(pat))
                       if (Path(d) / "eval_metrics.jsonl").exists()})
    if not run_dirs:
        ap.error(f"no run dirs with eval_metrics.jsonl matched {args.runs_glob!r}")
    monitors = sorted({m for d in run_dirs for m in monitor_names(d)})
    table = sorted(condition_table(run_dirs, monitors), key=_sort_key)

    labels = [f"{display_name(r['target'])}   λ={r['penalty']}   (n={r['n']})" for r in table]
    nrows = len(table)
    y = np.arange(nrows)[::-1]  # top row first
    ncols = len(_MEAN_COLS) + len(_DIST_COLS)

    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 0.6 * nrows + 3.0),
                             gridspec_kw={"width_ratios": [1, 1, 0.8, 1.4, 1.4, 1.4], "wspace": 0.3})

    for ax, (key, title, diverging) in zip(axes[:3], _MEAN_COLS):
        means = np.array([r[f"{key}_mean"] for r in table], float)
        stds = np.array([r[f"{key}_std"] for r in table], float)
        colors = ["#b03a2e" if (diverging and m > 0) else "#5f8fb0" for m in np.nan_to_num(means)]
        ax.barh(y, means, xerr=np.nan_to_num(stds), color=colors, alpha=0.85,
                error_kw=dict(elinewidth=0.7, ecolor="#666"))
        for yi, m, r in zip(y, means, table):
            if m != m:  # NaN mean → N/A (control has no monitor; else too few hacking rollouts)
                msg = "N/A" if r["target"] == "control" else "N/A\n(not enough\nhacking rollouts)"
                ax.text(0, yi, msg, va="center", ha="center", fontsize=5.5, color="#999")
        if diverging:
            ax.axvline(0, color="#333", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.6, nrows - 0.4)  # identical across panels so rows line up (no per-axis autoscale)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.2)

    for ax, (dkey, nkey, buckets, colors, title) in zip(axes[3:], _DIST_COLS):
        for r, yi in zip(table, y):
            if r[nkey] == 0 or all(f != f for f in r[dkey]):  # no runs with this metric defined
                msg = "N/A" if r["target"] == "control" else "N/A (not enough hacking rollouts)"
                ax.text(0.5, yi, msg, ha="center", va="center", fontsize=6, color="#999")
                continue
            left = 0.0
            for frac, color in zip(r[dkey], colors):
                f = 0.0 if frac != frac else frac
                if f > 0:
                    ax.barh(yi, f, left=left, color=color, edgecolor="white", lw=0.5, height=0.7)
                left += f
            ax.text(1.02, yi, f"n={r[nkey]}", va="center", fontsize=6, color="#888")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, nrows - 0.4)  # match the mean panels
        ax.set_title(title, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        ax.legend(handles=[Patch(facecolor=c, label=lbl) for (_, _, lbl), c in zip(buckets, colors)],
                  fontsize=5.6, ncol=1, loc="upper left", bbox_to_anchor=(0.0, -0.04),
                  frameon=False, handlelength=1.0, handleheight=1.0, labelspacing=0.25)

    # separators between train-against monitor groups (so multi-penalty rows read as one monitor block)
    boundaries = [i for i in range(nrows - 1) if table[i]["target"] != table[i + 1]["target"]]
    for ax in axes:
        for i in boundaries:
            ax.axhline(y[i] - 0.5, color="#999", lw=0.8, zorder=0)

    fig.suptitle("What training did, per (train-against monitor × penalty λ) — over seeds\n"
                 "μ bars = mean ± std over seeds. Distributions = fraction of a condition's runs per bucket "
                 "(bimodality-honest). d′ hybrid (margin, else unclipped AUROC→d′); metrics need ≥2 defined "
                 "eval points, so a bar/dist n may be < the group n in its row label.", fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "conditions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    (out_dir / "conditions.json").write_text(json.dumps(table, indent=2, default=float))
    print(f"{len(run_dirs)} runs · {nrows} conditions · monitors={monitors}")
    print(f"→ wrote {out_dir}/conditions.png, conditions.json")


if __name__ == "__main__":
    main()
