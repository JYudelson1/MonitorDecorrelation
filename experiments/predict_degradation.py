"""Sense (b) → sense (a): turn a miss-overlap analysis into co-degradation predictions, and score them
against the measured coupling β once the RL runs exist. (docs/FN_OVERLAP_PLAN.md §4)

    # PROSPECTIVE — write the predictions down before the final runs finish
    uv run python experiments/predict_degradation.py --overlap data/runs/_miss_overlap/<set>/miss_overlap.json

    # RETROSPECTIVE — score them against analyze_coupling.py's coupling.json (by-target β)
    uv run python experiments/predict_degradation.py --overlap .../miss_overlap.json \
        --coupling data/runs/_coupling/coupling.json [--stats lift,spearman_pos,gmean2_shortfall,pauroc_gain_z]

Every (label set × FPR) analysis in the overlap file is used; for each requested overlap statistic it
writes ``predictions_<stat>_<label>_fpr<p>.json`` (prospective) and, with ``--coupling``, a scatter of
overlap vs β per statistic plus ``bridge.json`` with Spearman / Pearson / bootstrap CI / permutation p.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from monitordecorrelation.eval.miss_overlap import to_jsonable  # noqa: E402
from monitordecorrelation.eval.predict_degradation import predictions_from_overlap, score_predictions  # noqa: E402

DEFAULT_STATS = ["lift", "cond_miss", "spearman_pos", "gmean2_shortfall", "pauroc_gain_z"]
STAT_LABELS = {
    "lift": "miss lift P(miss Y|miss X)/P(miss Y)",
    "cond_miss": "P(miss Y | miss X)",
    "yule_q": "Yule's Q of misses",
    "spearman_pos": "Spearman ρ of scores within hacks",
    "gmean2_shortfall": "g-mean² shortfall of OR-ensemble vs independence",
    "pauroc_gain_z": "−z of pAUROC ensemble gain vs independence",
    "ensemble_gap": "−(union catch − independence)",
}
TARGET_COLORS = ["#0571b0", "#92c5de", "#f4a582", "#ca0020", "#636363", "#1b9e77", "#7570b3", "#e7298a"]


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def plot_bridge(res: dict, out: Path, ax: plt.Axes | None = None, figsize=(6.5, 5.5)):
    """Scatter of overlap statistic (x) vs measured β(X→Y) (y), one point per (target, held-out) pair,
    coloured by training target."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if not res["pairs"]:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    targets = sorted({p["target"] for p in res["pairs"]})
    for k, t in enumerate(targets):
        pts = [p for p in res["pairs"] if p["target"] == t]
        ax.scatter([p["overlap"] for p in pts], [p["beta"] for p in pts], s=40,
                   color=TARGET_COLORS[k % len(TARGET_COLORS)], label=f"trained vs {t}", zorder=3)
        for p in pts:
            ax.annotate(p["held_out"], (p["overlap"], p["beta"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    rho = res.get("spearman", np.nan)
    if rho == rho:
        ci = res.get("spearman_ci90")
        ax.set_title(f"ρ = {rho:.2f}{_stars(res['perm_p_one_sided'])} "
                     f"[{ci[0]:.2f}, {ci[1]:.2f}] (n={res['n_pairs']} pairs, perm p={res['perm_p_one_sided']:.2f})",
                     fontsize=12)
    ax.set_xlabel(STAT_LABELS.get(res["stat"], res["stat"]), fontsize=12)
    ax.set_ylabel("measured β(X→Y)  (co-degradation)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return fig, ax


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlap", required=True, help="miss_overlap.json from analyze_miss_overlap.py")
    ap.add_argument("--coupling", default=None, help="coupling.json from analyze_coupling.py (retrospective)")
    ap.add_argument("--stats", default=",".join(DEFAULT_STATS))
    ap.add_argument("--out", default=None, help="default: alongside the overlap file")
    args = ap.parse_args()

    ov = json.loads(Path(args.overlap).read_text())
    out = Path(args.out) if args.out else Path(args.overlap).parent
    out.mkdir(parents=True, exist_ok=True)
    stats = [s.strip() for s in args.stats.split(",") if s.strip()]
    coupling = json.loads(Path(args.coupling).read_text())["by_target"] if args.coupling else None

    bridge = []
    for a in ov["analyses"]:
        tag = f"{a['label_key']}_fpr{int(a['fpr']*100):02d}"
        for stat in stats:
            if stat not in a:
                print(f"  (skip {stat}: not in analysis {tag})")
                continue
            pred = predictions_from_overlap(a, stat)
            (out / f"predictions_{stat}_{tag}.json").write_text(json.dumps(to_jsonable(pred), indent=1))
            print(f"\n[{tag} · {stat}] predicted co-degradation ranking per training target:")
            for x, p in pred["predictions"].items():
                print(f"  vs {x:16s}: " + " > ".join(p["ranking_most_to_least"]))
            if coupling is not None:
                res = score_predictions(a, coupling, stat)
                res.update({"label_key": a["label_key"], "fpr": a["fpr"]})
                bridge.append(to_jsonable(res))
                if res["n_pairs"] >= 3:
                    print(f"  → vs measured β: Spearman {res['spearman']:.2f} "
                          f"[{res['spearman_ci90'][0]:.2f},{res['spearman_ci90'][1]:.2f}], "
                          f"Pearson {res['pearson']:.2f}, perm p={res['perm_p_one_sided']:.3f}, n={res['n_pairs']}")
                    for t, r in res["per_target"].items():
                        if r["spearman"] is not None:
                            print(f"      row {t:16s}: ρ={r['spearman']:+.2f} (n={r['n']})")
                else:
                    print(f"  → only {res['n_pairs']} defined pairs; no correlation computed")
                plot_bridge(res, out / f"bridge_{stat}_{tag}.png")
    if coupling is not None:
        (out / "bridge.json").write_text(json.dumps({"overlap": args.overlap, "coupling": args.coupling,
                                                     "results": bridge}, indent=1))
        print(f"\nwrote {out/'bridge.json'}")


if __name__ == "__main__":
    main()
