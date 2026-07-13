"""Directed detector-coupling charts for a set of degradation runs (the reusable 'chart 7b + 10').

    uv run python experiments/analyze_coupling.py 'data/runs/mbpp_Qwen3-8B_*'         # auroc→d′
    uv run python experiments/analyze_coupling.py 'data/runs/mbpp_*' --metric dprime_margin --bootstrap 3000
    uv run python experiments/analyze_coupling.py 'data/runs/mbpp_*' --out data/runs/_coupling

Produces, in --out (default: alongside, ``data/runs/_coupling/``):
  coupling_pooled.png       — β(driver→responder) pooled over all runs where the metric is defined (7b)
  coupling_by_target.png    — β(trained-against i → held-out j), conditioned on training target (10)
  coupling.json             — the raw matrices + n + CIs

See ``eval/coupling.py`` for the metric definitions (β, the auroc→d′ vs margin-d′ choice, bootstrap).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from monitordecorrelation.eval.coupling import (
    directed_coupling,
    directed_coupling_by_target,
    display_name,
    filter_by_hacking,
    monitor_names,
)


def _heat(ax, M, rows, cols, title, *, lo=None, hi=None, diag_rows=None):
    import matplotlib.pyplot as plt

    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="#bbb")
                continue
            txt = f"{v:+.2f}"
            if lo is not None and not np.isnan(lo[i, j]):
                txt += f"\n[{lo[i, j]:+.2f},{hi[i, j]:+.2f}]"
            is_diag = (diag_rows or rows)[i] == cols[j]
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5 if lo is not None else 9,
                    color="white" if abs(v) > 0.6 else "k", weight="bold" if is_diag else "normal")
            if is_diag:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=2))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([display_name(c) for c in cols], rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([display_name(r) for r in rows], fontsize=8.5)
    ax.set_title(title, fontsize=10)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_glob", nargs="+", help="one or more globs for run dirs (unioned), e.g. "
                    "'data/runs/mbpp_Qwen3-8B_*' '~/other/runs/mbpp_*'")
    ap.add_argument("--metric", default="auroc",
                    choices=["auroc", "dprime_margin", "dprime", "suspicion_on_hacks", "suspicion_on_clean"],
                    help="reliability scale: auroc→d′ (default) · dprime_margin (native) · dprime (HYBRID "
                         "margin-or-AUROC, best for combining old+new runs) · suspicion_on_hacks (μ_hack — "
                         "co-EVASION: does the held-out monitor stop CATCHING hacks?) · suspicion_on_clean "
                         "(μ_clean — honest-class drift, the OTHER driver of d′ collapse)")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap-over-runs samples → 90%% CI [5,95 pctile] shown under each cell")
    ap.add_argument("--no-penalty-label", action="store_true",
                    help="omit the per-row α=<penalty> annotation (useful when a row mixes many schedules)")
    ap.add_argument("--hacking", default="any", choices=["any", "present", "absent"],
                    help="condition on hacking: keep runs that DID hack ('present'), did NOT ('absent'), all")
    ap.add_argument("--hack-thresh", type=float, default=0.10, help="max hack rate to count as 'hacked'")
    ap.add_argument("--out", default="data/runs/_coupling")
    args = ap.parse_args()

    run_dirs = sorted({d for pat in args.runs_glob
                       for d in glob.glob(os.path.expanduser(pat))
                       if (Path(d) / "eval_metrics.jsonl").exists()})
    run_dirs = filter_by_hacking(run_dirs, args.hacking, args.hack_thresh)
    if not run_dirs:
        ap.error(f"no run dirs matched {args.runs_glob!r} (hacking={args.hacking})")
    monitors = monitor_names(run_dirs[0])
    print(f"{len(run_dirs)} runs · {len(monitors)} monitors · metric={args.metric}"
          + (f" · bootstrap={args.bootstrap}" if args.bootstrap else ""))

    pooled = directed_coupling(run_dirs, monitors, args.metric, bootstrap=args.bootstrap)
    bytgt = directed_coupling_by_target(run_dirs, monitors, args.metric, bootstrap=args.bootstrap)
    print(f"  pooled: {pooled['n_runs']} runs contributed, {pooled['n_pairs']} Δ-pairs")
    print(f"  by-target rows populated: {bytgt['targets'] or '(none — no targeted run degraded)'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    unit = "d′" if args.metric == "auroc" else "margin d′"
    # note printed on both charts when bootstrap CIs are shown, so the bracketed numbers are self-explaining
    ci_note = "\ncells: β  with  [5–95 percentile bootstrap CI] underneath" if args.bootstrap else ""

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = _heat(ax, pooled["beta"], monitors, monitors,
               f"Directed coupling β(driver→responder), pooled ({args.metric}, {unit})\n"
               f"{pooled['n_runs']} runs · {pooled['n_pairs']} Δ-pairs" + ci_note,
               lo=pooled.get("lo"), hi=pooled.get("hi"))
    ax.set_xlabel("→ effect on detector (col)"); ax.set_ylabel("driver detector (row)")
    plt.colorbar(im, ax=ax, label=f"β = Σ Δ{unit}_A·Δ{unit}_B / Σ Δ{unit}_A²", shrink=0.85)
    fig.tight_layout(); fig.savefig(out / "coupling_pooled.png", bbox_inches="tight")

    if any(bytgt["n_runs_per_target"]):
        fig, ax = plt.subplots(figsize=(8.5, max(3, 0.9 * len(bytgt["targets"]) + 3)))
        im = _heat(ax, bytgt["beta"], bytgt["targets"], monitors,
                   f"Directed coupling CONDITIONED ON TRAINING TARGET ({args.metric})\n"
                   "β(trained-against i → held-out j): does training against i degrade j?" + ci_note,
                   lo=bytgt.get("lo"), hi=bytgt.get("hi"), diag_rows=bytgt["targets"])
        # small n's + penalty (α) just under each row label (left): runs + Δd′ datapoints + λ used.
        pens = bytgt.get("penalty_per_target", [""]*len(bytgt["targets"]))
        show_pen = not args.no_penalty_label
        for ri, (nr, npair, pen) in enumerate(zip(bytgt["n_runs_per_target"], bytgt["n_pairs_per_target"], pens)):
            txt = (f"n={nr}r/{npair}p" if nr else "n/a (honest)") + (f"   α={pen}" if (nr and pen and show_pen) else "")
            ax.text(-0.015, ri + 0.33, txt, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6, color="#888")
        ax.set_xlabel("→ held-out detector j"); ax.set_ylabel("TRAINED AGAINST i")
        plt.colorbar(im, ax=ax, label="β(target→held-out)", shrink=0.85)
        fig.tight_layout(); fig.savefig(out / "coupling_by_target.png", bbox_inches="tight")

    def _ser(d):  # numpy → json
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    (out / "coupling.json").write_text(json.dumps({"pooled": _ser(pooled), "by_target": _ser(bytgt)}, indent=2))
    print(f"→ wrote {out}/coupling_pooled.png, coupling_by_target.png, coupling.json")


if __name__ == "__main__":
    main()
