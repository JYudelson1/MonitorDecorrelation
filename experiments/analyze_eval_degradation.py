#!/usr/bin/env python
"""Absolute eval-degradation matrices: total d′ drop of each detector under each training condition.

    uv run python experiments/analyze_eval_degradation.py 'data/runs/*/mbpp_Qwen3-8B_*' \
        --method slope --bootstrap 3000 --out data/runs/_degradation

Emits, in --out:
  degradation_raw.png   — 7×6 (rows: 6 monitors + CONTROL; cols: 6 monitors). cell = mean total d′ drop.
  degradation_excess.png — 6×6 control-subtracted (row − control) = degradation beyond the no-monitor baseline.
  degradation.json      — matrices + CIs + n.

See eval/eval_degradation.py for D (slope×span vs endpoint), the hybrid d′, and the control baseline.
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

from monitordecorrelation.eval.coupling import display_name, monitor_names
from monitordecorrelation.eval.eval_degradation import degradation_matrix


def _heat(ax, M, rows, cols, title, *, vmax, lo=None, hi=None, n=None, mark_diag=True):
    # RdBu_r, centered at 0: red = detector DEGRADED (d′ fell), blue = it sharpened. Diverging w/ neutral mid.
    ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isnan(v):
                nlab = "N/A" if (n is None or n[i, j] == 0) else "N/A"
                ax.text(j, i, nlab, ha="center", va="center", fontsize=7.5, color="#b0b0b0")
                continue
            txt = f"{v:+.2f}"
            if lo is not None and not np.isnan(lo[i, j]):
                txt += f"\n[{lo[i, j]:+.2f},{hi[i, j]:+.2f}]"
            if n is not None:
                txt += f"\nn={n[i, j]}"
            is_diag = mark_diag and rows[i] == cols[j]
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.6,
                    color="white" if abs(v) > 0.62 * vmax else "k", weight="bold" if is_diag else "normal")
            if is_diag:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=1.8))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([display_name(c) for c in cols], rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([display_name(r) for r in rows], fontsize=8.5)
    ax.set_title(title, fontsize=10.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_glob", nargs="+", help="one or more globs for run dirs (unioned)")
    ap.add_argument("--method", default="slope", choices=["slope", "endpoint"],
                    help="per-run total: slope×span (default, robust) or the literal first−last drop")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap-over-runs samples → 90%% CI [5,95]")
    ap.add_argument("--out", default="data/runs/_degradation")
    args = ap.parse_args()

    run_dirs = sorted({d for pat in args.runs_glob for d in glob.glob(os.path.expanduser(pat))
                       if (Path(d) / "eval_metrics.jsonl").exists()})
    if not run_dirs:
        ap.error(f"no run dirs with eval_metrics.jsonl matched {args.runs_glob!r}")
    monitors = sorted({m for d in run_dirs for m in monitor_names(d)})
    R = degradation_matrix(run_dirs, monitors, method=args.method, bootstrap=args.bootstrap)
    mons, tgts = R["monitors"], R["targets"]

    ci_note = "\ncells: mean total d′ drop  ·  [5–95% bootstrap CI]  ·  n runs" if args.bootstrap else \
              "\ncells: mean total d′ drop  ·  n runs"
    meth = "slope×span" if args.method == "slope" else "first−last endpoint"
    sub_txt = "  positive (red) = detector degraded  ·  negative (blue) = sharpened"

    out_dir = Path(os.path.expanduser(args.out)); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- raw 7×6 ----
    vmax = float(np.nanmax(np.abs(R["D"]))) or 1.0
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    _heat(ax, R["D"], tgts, mons, "", vmax=vmax, lo=R["lo"], hi=R["hi"], n=R["n"])
    ax.set_xlabel("→ detector degraded (col)", fontsize=9)
    ax.set_ylabel("TRAINED ON (row)", fontsize=9)
    fig.suptitle(f"Total d′ degradation of each detector, per training condition [{meth}]\n"
                 f"CONTROL row = no-monitor baseline (policy just learns to hack).{sub_txt}{ci_note}",
                 fontsize=10.5, y=0.985)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, label="d′ drop")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "degradation_raw.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- control-subtracted 6×6 ----
    vmax2 = float(np.nanmax(np.abs(R["sub"]))) or 1.0
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    _heat(ax, R["sub"], mons, mons, "", vmax=vmax2, lo=R["sub_lo"], hi=R["sub_hi"])
    ax.set_xlabel("→ detector degraded (col)", fontsize=9)
    ax.set_ylabel("TRAINED ON (row)", fontsize=9)
    fig.suptitle(f"EXCESS degradation beyond the control baseline (row − control) [{meth}]\n"
                 f"what training against this monitor costs held-out detectors, above generic hacking drift."
                 f"{sub_txt}", fontsize=10.5, y=0.985)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, label="excess d′ drop")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "degradation_excess.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    (out_dir / "degradation.json").write_text(json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in R.items()}, indent=2))
    print(f"{len(run_dirs)} runs · method={args.method} · targets={tgts}")
    print(f"→ wrote {out_dir}/degradation_raw.png, degradation_excess.png, degradation.json")


if __name__ == "__main__":
    main()
