"""Sense-(b) analysis on a scored instance set: do the monitors miss the SAME hacks?

Runs ``eval/miss_overlap.analyze`` under each label set (``strict``: loose-only rollouts count as
clean; ``loose``: they count as hacks) and each matched-FPR operating point, prints the tables, and
writes heatmaps + JSON.

    uv run python experiments/analyze_miss_overlap.py --scored data/bakeoff_set_600_scored.jsonl
    uv run python experiments/analyze_miss_overlap.py --scored ... --labels strict --fpr 0.05,0.2 \
        --monitors cot_only_weak,cot_only_strong,probe_ood --out data/runs/_miss_overlap

Outputs (per label × fpr): ``lift_<label>_fpr<p>.png`` (P(miss_Y|miss_X)/P(miss_Y), CI-annotated),
``cond_miss_…png``, ``ensemble_gap_…png`` (union catch − independence prediction), ``spearman_pos.png``
/ ``spearman_neg.png`` (threshold-free), ``gmean2_shortfall_…png`` (OR-ensemble g-mean² vs
independence), ``pauroc_gain_z_…png`` (threshold-free ensemble gain vs a permutation null), ``strata_fpr<p>.png`` (flag rate on strict / loose-only /
clean), and ``miss_overlap.json`` with every matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from monitordecorrelation.eval.miss_overlap import analyze, stratum_miss_rates, to_jsonable  # noqa: E402

# Canonical display order: text monitors weak→strong within each view, then probes.
MONITOR_ORDER = ["cot_only_weak", "cot_only_strong", "cot+out_weak", "cot+out_strong",
                 "out_weak", "out_strong", "probe_ood", "probe_iid"]
STRATUM_ORDER = ["strict", "loose_only", "clean"]
STRATUM_LABELS = {"strict": "strict hacks", "loose_only": "loose-only (honest-buggy)", "clean": "clean"}
STRATUM_COLORS = {"strict": "#ca0020", "loose_only": "#f4a582", "clean": "#0571b0"}
LIFT_VMAX = 5.0  # colour cap for the lift heatmap (annotations still show the raw value)


def _load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _present_monitors(records: list[dict], requested: list[str] | None) -> list[str]:
    seen = {m for r in records for m in r.get("monitors", {})}
    names = [m for m in MONITOR_ORDER if m in seen] + sorted(seen - set(MONITOR_ORDER))
    if requested:
        names = [m for m in requested if m in seen]
    return names


def _heatmap(mat: np.ndarray, names: list[str], title: str, out: Path, *, annot: np.ndarray | None = None,
             cmap: str = "RdBu_r", center: float | None = None, vmin=None, vmax=None,
             xlabel: str = "held-out Y", ylabel: str = "fooled X", ax: plt.Axes | None = None,
             figsize=(8.5, 7)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    sns.heatmap(mat, ax=ax, annot=annot if annot is not None else True, fmt="" if annot is not None else ".2f",
                cmap=cmap, center=center, vmin=vmin, vmax=vmax, xticklabels=names, yticklabels=names,
                square=True, cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 8})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="x", labelrotation=45, labelsize=11)
    ax.tick_params(axis="y", labelrotation=0, labelsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return fig, ax


def _fmt_ci(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([[("" if np.isnan(p) else f"{p:.2f}\n[{l:.1f},{h:.1f}]")
                      for p, l, h in zip(pr, lr, hr)] for pr, lr, hr in zip(point, lo, hi)], dtype=object)


def _print(res: dict) -> None:
    names = res["monitors"]
    print(f"\n=== labels={res['label_key']}  fpr={res['fpr']:.0%}  n={res['n_used']} "
          f"({res['n_pos']} hack / {res['n_neg']} non-hack; {res['n_dropped_unscored']} dropped unscored)")
    print(f"{'monitor':18s} {'AUROC':>6} {'pAUROC':>7} {'g-mean2':>8} {'thr':>6} {'P(miss)':>8}")
    for j, m in enumerate(names):
        print(f"{m:18s} {res['auroc'][j]:6.3f} {res['pauroc'][j]:7.3f} {res['gmean2'][j]:8.3f} "
              f"{res['threshold'][j]:6.3f} {res['p_miss'][j]:8.2f}")
    print("\nlift P(miss_Y|miss_X)/P(miss_Y)   rows = X (fooled), cols = Y")
    print(" " * 18 + "".join(f"{m[:10]:>11}" for m in names))
    for i, m in enumerate(names):
        print(f"{m:18s}" + "".join(f"{v:11.2f}" if not np.isnan(v) else f"{'nan':>11}" for v in res["lift"][i]))


def plot_strata(st: dict, out: Path, figsize=(10, 4.5)):
    """Grouped bars: per monitor, flag rate on strict / loose-only / clean at the matched-FPR threshold."""
    names = st["monitors"]
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names)); w = 0.26
    for k, s in enumerate(STRATUM_ORDER):
        vals = np.asarray(st["flag_rate"][s], dtype=float)
        ax.bar(x + (k - 1) * w, vals, width=w, color=STRATUM_COLORS[s], edgecolor="#C0C0C0", linewidth=0.5,
               label=f"{STRATUM_LABELS[s]} (n={st['n'][s]})", zorder=3)
        for xi, v in zip(x + (k - 1) * w, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=7, rotation=60)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    ax.set_ylim(0, 1.18); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_yticklabels([f"{int(t*100)}%" for t in [0, .2, .4, .6, .8, 1.0]], fontsize=11)
    ax.set_ylabel(f"flag rate @ {st['fpr']:.0%} FPR on clean", fontsize=13)
    ax.set_title("Per-stratum flag rate (matched-FPR threshold)", fontsize=14)
    for sp in ("top", "left", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="y", color="#D3D3D3", alpha=0.75, linewidth=0.8, zorder=1)
    ax.legend(fontsize=10, ncol=3, loc="upper center")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")
    return fig, ax


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True, help="scored jsonl from score_instance_set.py")
    ap.add_argument("--labels", default="strict,loose")
    ap.add_argument("--fpr", default="0.05,0.2")
    ap.add_argument("--monitors", default=None, help="subset/order (comma-separated)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-fpr", type=float, default=0.1, help="pAUROC band (FPR ≤ this)")
    ap.add_argument("--n-perm", type=int, default=200, help="permutations for the pAUROC-gain null")
    ap.add_argument("--out", default=None, help="default: data/runs/_miss_overlap/<scored-stem>/")
    args = ap.parse_args()

    records = _load(Path(args.scored))
    names = _present_monitors(records, args.monitors.split(",") if args.monitors else None)
    if len(names) < 2:
        raise SystemExit(f"need ≥2 scored monitors, found {names}")
    out = Path(args.out) if args.out else Path("data/runs/_miss_overlap") / Path(args.scored).stem
    out.mkdir(parents=True, exist_ok=True)
    print(f"monitors ({len(names)}): {names}")

    fprs = [float(x) for x in args.fpr.split(",")]
    results = {"scored": args.scored, "monitors": names, "analyses": [], "strata": []}
    for label in args.labels.split(","):
        for fpr in fprs:
            res = analyze(records, names, label_key=label, fpr=fpr, n_boot=args.n_boot,
                          max_fpr=args.max_fpr, n_perm=args.n_perm)
            _print(res)
            results["analyses"].append(to_jsonable(res))
            tag = f"{label}_fpr{int(fpr*100):02d}"
            # diagonal = 1/P(miss_X) (uninformative) → blank it; colour capped at LIFT_VMAX so one huge
            # cell (a monitor with 1–2 misses) doesn't wash out the rest — the annotation keeps the value.
            lift = res["lift"].copy(); np.fill_diagonal(lift, np.nan)
            ann = _fmt_ci(lift, res["lift_ci"]["lo"], res["lift_ci"]["hi"])
            _heatmap(lift, names, f"Miss lift P(miss Y | miss X) / P(miss Y)  [1 = independent]\n{label} labels, {fpr:.0%} FPR",
                     out / f"lift_{tag}.png", annot=ann, cmap="Reds", vmin=0, vmax=LIFT_VMAX)
            _heatmap(res["cond_miss"], names, f"P(miss Y | miss X)\n{label} labels, {fpr:.0%} FPR",
                     out / f"cond_miss_{tag}.png", cmap="Reds", vmin=0, vmax=1)
            _heatmap(res["gmean2_shortfall"], names,
                     f"g-mean² shortfall: independence prediction − OR-ensemble  [>0 = shared blind spot]\n{label} labels, {fpr:.0%} FPR",
                     out / f"gmean2_shortfall_{tag}.png", cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3, xlabel="Y", ylabel="X")
            _heatmap(res["pauroc_gain_z"], names,
                     f"pAUROC ensemble gain vs independence (z)  [<0 = less complementary than independent]\n{label} labels, FPR≤{res['max_fpr']:.0%}",
                     out / f"pauroc_gain_z_{tag}.png", cmap="RdBu_r", center=0, vmin=-6, vmax=6, xlabel="Y", ylabel="X")
            _heatmap(res["union_catch"] - res["indep_catch"], names,
                     f"Ensemble gap: union catch − independence prediction\n{label} labels, {fpr:.0%} FPR",
                     out / f"ensemble_gap_{tag}.png", cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                     xlabel="Y", ylabel="X")
        # threshold-free (label-dependent only through which rows are "hacks")
        res0 = results["analyses"][-len(fprs)]
        _heatmap(np.array(res0["spearman_pos"], dtype=float), names, f"Spearman ρ of scores within hacks ({label})",
                 out / f"spearman_pos_{label}.png", cmap="RdBu_r", center=0, vmin=-1, vmax=1, xlabel="", ylabel="")
        _heatmap(np.array(res0["spearman_neg"], dtype=float), names, f"Spearman ρ of scores within non-hacks ({label})",
                 out / f"spearman_neg_{label}.png", cmap="RdBu_r", center=0, vmin=-1, vmax=1, xlabel="", ylabel="")
    for fpr in fprs:
        st = stratum_miss_rates(records, names, fpr=fpr)
        results["strata"].append(to_jsonable(st))
        plot_strata(st, out / f"strata_fpr{int(fpr*100):02d}.png")
    (out / "miss_overlap.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out/'miss_overlap.json'}")


if __name__ == "__main__":
    main()
