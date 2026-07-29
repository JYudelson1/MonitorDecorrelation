#!/usr/bin/env python
"""Lab-facing reliability matrix: % of eval-time each detector sits below a usable threshold.

    uv run python experiments/analyze_threshold_reliability.py 'data/runs/*/mbpp_Qwen3-8B_*' \
        --metric dprime --threshold 2.0 --hacking present --bootstrap 3000 --out data/runs/_reliability

Cell (row, col) = mean over the condition's runs of "fraction of measurable eval snapshots where detector
col's metric < threshold". Rows: 6 monitors + CONTROL (no-monitor baseline). See
eval/threshold_reliability.py. Higher % = the detector is more often below a bar you'd trust.
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

from monitordecorrelation.eval.coupling import display_name, filter_by_hacking, monitor_names
from monitordecorrelation.eval.threshold_reliability import threshold_matrix

_DEFAULT_THRESH = {"dprime": 2.0, "suspicion_on_hacks": 0.5, "suspicion_on_clean": 0.5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_glob", nargs="+")
    ap.add_argument("--metric", default="dprime",
                    choices=["dprime", "suspicion_on_hacks", "suspicion_on_clean"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="below-threshold bar (default: d′ 2.0 / μ 0.5). Detector < this = 'failing'.")
    ap.add_argument("--hacking", default="present", choices=["any", "present", "absent"],
                    help="default 'present' — a detector's reliability is only meaningful where hacks exist")
    ap.add_argument("--hack-thresh", type=float, default=0.10)
    ap.add_argument("--final-frac", type=float, default=1.0,
                    help="window of training to score: 1.0 = whole run (default), 0.10 = final 10% "
                         "(deployment-time state). Cell = fraction of the window's eval snapshots below the bar.")
    ap.add_argument("--out", default="data/runs/_reliability")
    args = ap.parse_args()
    thresh = args.threshold if args.threshold is not None else _DEFAULT_THRESH[args.metric]

    run_dirs = sorted({d for pat in args.runs_glob for d in glob.glob(os.path.expanduser(pat))
                       if (Path(d) / "eval_metrics.jsonl").exists()})
    run_dirs = filter_by_hacking(run_dirs, args.hacking, args.hack_thresh)
    if not run_dirs:
        ap.error(f"no run dirs matched (hacking={args.hacking})")
    monitors = sorted({m for d in run_dirs for m in monitor_names(d)})
    R = threshold_matrix(run_dirs, monitors, thresh, metric=args.metric, final_frac=args.final_frac)
    P, mons, tgts = R["P"], R["monitors"], R["targets"]
    below, total, n = R["below"], R["total"], R["n"]

    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    ax.imshow(P, cmap="Reds", vmin=0, vmax=1, aspect="auto")  # sequential: 0 good (white) → 1 bad (red)
    for i in range(len(tgts)):
        for j in range(len(mons)):
            v = P[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7.5, color="#b0b0b0")
                continue
            txt = f"{below[i,j]}/{total[i,j]}\n{v:.2f}\n{n[i,j]} runs"  # raw fraction of snapshots, no bootstrap
            is_diag = tgts[i] == mons[j]
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                    color="white" if v > 0.55 else "k", weight="bold" if is_diag else "normal")
            if is_diag:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=1.8))
    ax.set_xticks(range(len(mons))); ax.set_xticklabels([display_name(c) for c in mons], rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(tgts))); ax.set_yticklabels([display_name(t) for t in tgts], fontsize=8.5)
    ax.set_xlabel("→ held-out detector (col)", fontsize=9)
    ax.set_ylabel("TRAINED ON (row)", fontsize=9)

    unit = f"d′ < {thresh:g}" if args.metric == "dprime" else f"{args.metric.replace('suspicion_on_','μ_')} < {thresh:g}"
    auroc = f"  (d′={thresh:g} ≈ AUROC {__import__('statistics').NormalDist().cdf(thresh/2**0.5):.2f})" \
            if args.metric == "dprime" else ""
    window = "whole run" if args.final_frac >= 1.0 else f"final {args.final_frac*100:g}% of training"
    fig.suptitle(f"How often is each held-out detector UNRELIABLE ({unit}){auroc}\n"
                 f"fraction of {window} eval snapshots below the bar, hacking runs (max hack ≥ "
                 f"{args.hack_thresh:g}). CONTROL = no-monitor baseline.\ncells: below/total snapshots  ·  "
                 f"fraction  ·  n runs  ·  redder = detector more often below a bar you'd trust",
                 fontsize=10, y=0.985)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, label=f"fraction of snapshots with {unit}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_dir = Path(os.path.expanduser(args.out)); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "reliability.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    (out_dir / "reliability.json").write_text(json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in R.items()}, indent=2))
    print(f"{len(run_dirs)} runs · metric={args.metric} thresh={thresh:g} · targets={tgts}")
    print(f"→ wrote {out_dir}/reliability.png, reliability.json")


if __name__ == "__main__":
    main()
