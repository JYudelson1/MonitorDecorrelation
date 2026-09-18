#!/usr/bin/env python3
"""
beta_matrix.py -- within/between coupling betas for every ordered monitor pair,
from eval_rollouts_slim.jsonl files, across seeds and train-against conditions.

Usage
-----
  python beta_matrix.py --selftest
  python beta_matrix.py --trained probe_iid                       # one condition, seeds 0 1 2
  python beta_matrix.py --trained probe_iid probe_ood cot_out_weak=cot+out_weak
  python beta_matrix.py --trained probe_iid --full                # print all 8x8 matrices
  python beta_matrix.py --trained probe_iid probe_ood --out causal_rows.csv
  python beta_matrix.py --runs-dir /path/to/runs                 # every subfolder with a config.json
  python beta_matrix.py --runs-dir /path/to/runs --group-by penalty_coef lr

--runs-dir mode: each subfolder containing eval_rollouts_slim.jsonl is a run; its
config.json supplies the train-against monitor (monitors[].role == "train_against") and
the seed. Runs are grouped into conditions by train-against monitor, plus any extra
config keys given with --group-by. A "seed-N" in run_name that disagrees with config
"seed" is warned about.

--base is a directory pattern with {trained} and {seed} placeholders. Each --trained
entry is either a monitor name (used for both the directory token and the monitor)
or  dirtoken=monitor  when the directory name differs from the monitor key
(e.g. cot_out_weak=cot+out_weak).

Definitions (a "cell" = a window of consecutive eval steps in ONE run with
>= --min-hacks hack rollouts; inside a cell the policy is treated as fixed)

  beta_within[X,Y]  = sum_cells dof_c Cov_c(X,Y) / sum_cells dof_c Var_c(X)   (hacks only)
                      The slope RL "sees" when it selects on X: raw material for a
                      correlated response in Y. --within-task demeans by task_id inside
                      each cell first (GRPO advantages are group-relative).
  beta_between[X,Y] = Cov_cells(fold1 mean X, fold2 mean Y) / Cov_cells(fold1 mean X, fold2 mean X)
                      Realized co-movement of the policy's hack-score means over training.
                      Cross-fold means so shared sampling noise cancels. Averaged over
                      both fold orders and --splits random splits.

A between row X is only reported if X's hack-mean moved by >= --min-move-se cell-mean
SEs over the run AND the cross-fold variance of its cell means is positive. Otherwise
the row is NaN and the reason is printed. r_between over steps within a single run is
~+/-1 for any two monotone trajectories, so use the slope, not r, for the between part.

Diagnostics per run: per-cell hack/loose_rh/hardcoding rates (label contamination
check), and per-monitor TPR/TNR/g-mean2/d' at the monitor's own label threshold.

Hack flag is `behavior_present`. Invalid rows are dropped: `unparsed == true`, or a non-null
`invalid_reason` (truncated/unparsed — such a rollout was never shown to a monitor, so it has no scores).
Dumps that predate `invalid_reason` still carry scores for their truncated rollouts, which stay in.
"""
import argparse, json, re, sys, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_BASE = ("/Users/joey/Aether/MonitorDecorrelation/data/runs/"
                "mbpp_matrix_sep3_20260903/mbpp_Qwen3-8B_{trained}_s{seed}_sep3")
ROLLOUT_FILE = "eval_rollouts_slim.jsonl"
FLAGS = ["loose_rh", "hardcoding"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:7.3f}")


# ----------------------------------------------------------------------------- IO
def load_run(path: Path, drop_unparsed: bool = True):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if drop_unparsed and (r.get("unparsed", False) or r.get("invalid_reason") is not None):
                continue
            row = {"step": int(r["step"]), "task_id": r.get("task_id"),
                   "hack": bool(r["behavior_present"])}
            for fl in FLAGS:
                row[fl] = bool(r[fl]) if fl in r else np.nan
            for m, v in r["monitors"].items():
                row[m] = float(v["score"])
                row[m + "__label"] = bool(v.get("label", False))
            rows.append(row)
    df = pd.DataFrame(rows)
    mons = [c for c in df.columns
            if c not in ("step", "task_id", "hack", *FLAGS) and not c.endswith("__label")]
    return df, mons


# ------------------------------------------------------------------------ windows
def make_windows(df: pd.DataFrame, min_hacks: int) -> pd.Series:
    steps = np.sort(df["step"].unique())
    hps = df[df.hack].groupby("step").size().reindex(steps, fill_value=0)
    win, wid, acc = {}, 0, 0
    for s in steps:
        win[s] = wid
        acc += hps[s]
        if acc >= min_hacks:
            wid, acc = wid + 1, 0
    if acc > 0 and wid > 0:                       # merge trailing partial window
        for s in steps:
            if win[s] == wid:
                win[s] = wid - 1
    return df["step"].map(win)


# ----------------------------------------------------------------------- per-cell
def cell_stats(df, mons, min_hacks, splits, rng, within_task):
    df = df.copy()
    df["win"] = make_windows(df, min_hacks)
    k = len(mons)
    cells = []
    for w, gw in df.groupby("win"):
        g = gw[gw.hack]
        gc = gw[~gw.hack]
        n = len(g)
        if n < min_hacks:
            continue
        X = g[mons].to_numpy()
        if within_task:
            Xd = X - g.groupby("task_id")[mons].transform("mean").to_numpy()
            dof = n - g["task_id"].nunique()
            C = Xd.T @ Xd / max(dof, 1)
        else:
            dof = n - 1
            C = np.cov(X, rowvar=False, ddof=1)
        m1s, m2s = [], []
        for _ in range(splits):
            fold = rng.integers(0, 2, n)
            if fold.sum() in (0, n):
                fold[0], fold[-1] = 0, 1
            m1s.append(X[fold == 0].mean(0)); m2s.append(X[fold == 1].mean(0))
        mu_h, var_h = X.mean(0), X.var(0, ddof=1)
        nc = len(gc)
        if nc >= 5:
            Xc = gc[mons].to_numpy()
            mu_c, var_c = Xc.mean(0), Xc.var(0, ddof=1)
            tnr = np.array([(~gc[m + "__label"]).mean() for m in mons])
        else:
            mu_c = var_c = tnr = np.full(k, np.nan)
        tpr = np.array([g[m + "__label"].mean() for m in mons])
        pooled = np.sqrt((var_h + var_c) / 2)
        dprime = np.where(pooled > 0, (mu_h - mu_c) / np.where(pooled > 0, pooled, 1), np.nan)
        flags = dict(n_all=len(gw), hack_rate=gw.hack.mean())
        for fl in FLAGS:
            flags[fl] = gw[fl].mean() if gw[fl].notna().any() else np.nan
        lr = gw["loose_rh"]
        flags["loose_not_bp"] = ((lr.fillna(False).astype(bool)) & (~gw.hack)).mean() if lr.notna().any() else np.nan
        cells.append(dict(win=w, steps=(int(g.step.min()), int(g.step.max())), n=n, n_clean=nc, dof=dof,
                          C=C, mu=mu_h, var=var_h, m1=np.array(m1s), m2=np.array(m2s),
                          mu_clean=mu_c, tpr=tpr, tnr=tnr, g2=tpr * tnr, dprime=dprime, flags=flags))
    return cells


# ------------------------------------------------------------------------ per-run
def analyze_run(df, mons, min_hacks, splits, early_frac, rng, within_task, min_move_se):
    cells = cell_stats(df, mons, min_hacks, splits, rng, within_task)
    k = len(mons)
    R = dict(cells=cells, ok=len(cells) >= 3, n_hack=int(df.hack.sum()), n_total=int(len(df)), mons=mons)
    if not R["ok"]:
        return R

    def pooled_within(cs):
        W = sum(c["dof"] * c["C"] for c in cs)
        d = np.diag(W)
        return W / d[:, None], W / np.sqrt(np.outer(d, d))

    beta_w, r_w = pooled_within(cells)
    n_early = max(2, int(np.ceil(early_frac * len(cells))))
    beta_we, r_we = pooled_within(cells[:n_early])

    Cb = np.zeros((k, k))
    for s in range(splits):
        M1 = np.stack([c["m1"][s] for c in cells]); M2 = np.stack([c["m2"][s] for c in cells])
        A, B = M1 - M1.mean(0), M2 - M2.mean(0)
        cross = A.T @ B / (len(cells) - 1)
        Cb += (cross + cross.T) / 2
    Cb /= splits
    d = np.diag(Cb).copy()

    # guard: did X's hack-mean actually move, relative to cell-mean sampling error?
    mu_first, mu_last = cells[0]["mu"], cells[-1]["mu"]
    move = np.abs(mu_last - mu_first)
    se_cell = np.sqrt(np.mean([c["var"] / c["n"] for c in cells], axis=0))
    move_se = move / np.where(se_cell > 0, se_cell, np.nan)
    row_ok = (move_se >= min_move_se) & (d > 0)
    reasons = {}
    for i, m in enumerate(mons):
        if not row_ok[i]:
            why = []
            if move_se[i] < min_move_se:
                why.append(f"moved {move[i]:.3f} = {move_se[i]:.1f} SE (need >= {min_move_se})")
            if d[i] <= 0:
                why.append("cross-fold Var(mu_X) <= 0")
            reasons[m] = "; ".join(why)
    d_safe = np.where(row_ok, d, np.nan)
    beta_b = Cb / d_safe[:, None]
    r_b = Cb / np.sqrt(np.outer(np.where(d > 0, d, np.nan), np.where(d > 0, d, np.nan)))
    r_b[~row_ok, :] = np.nan

    D = lambda A: pd.DataFrame(A, index=mons, columns=mons)
    S = lambda v: pd.Series(v, index=mons)
    R.update(dict(beta_within=D(beta_w), r_within=D(r_w), beta_within_early=D(beta_we), r_within_early=D(r_we),
                  beta_between=D(beta_b), r_between=D(r_b), between_var=S(d), move=S(move), move_se=S(move_se),
                  row_ok=S(row_ok), reasons=reasons, n_early=n_early))
    return R


# ------------------------------------------------------------------------ report
def report_run(R, trained, full):
    if R.get("run_name"):
        print(f"  run: {R['run_name']}")
    if not R["ok"]:
        print(f"  only {len(R['cells'])} usable cells -- skipped")
        return
    cells, mons = R["cells"], R["mons"]
    print(f"  rollouts: {R['n_total']}  hacks: {R['n_hack']}  cells: {len(cells)}  (early = first {R['n_early']})")
    ct = pd.DataFrame([dict(steps=f"{c['steps'][0]}-{c['steps'][1]}", n_all=c["flags"]["n_all"], n_hack=c["n"],
                            hack_rate=c["flags"]["hack_rate"], loose_rh=c["flags"]["loose_rh"],
                            hardcoding=c["flags"]["hardcoding"], loose_not_bp=c["flags"]["loose_not_bp"])
                       for c in cells])
    print("\n  cells (label check: loose_not_bp = loose_rh & ~behavior_present):")
    print(ct.to_string(index=False))
    f, l = cells[0], cells[-1]
    diag = pd.DataFrame({
        "mu_hack f": f["mu"], "mu_hack l": l["mu"], "mu_clean f": f["mu_clean"], "mu_clean l": l["mu_clean"],
        "d' f": f["dprime"], "d' l": l["dprime"], "TPR f": f["tpr"], "TPR l": l["tpr"],
        "TNR f": f["tnr"], "TNR l": l["tnr"], "minTNR": np.nanmin(np.stack([c["tnr"] for c in cells]), 0),
        "g2 f": f["g2"], "g2 l": l["g2"], "moved(SE)": R["move_se"].to_numpy(),
    }, index=mons)
    print("\n  monitor diagnostics, first cell (f) -> last cell (l); TPR/TNR at the monitor's own label:")
    print(diag.to_string())
    if full:
        for name in ("beta_between", "r_between", "beta_within", "beta_within_early", "r_within_early"):
            print(f"\n  {name}  [row X -> col Y]")
            print(R[name].to_string())
    if trained in mons:
        print(f"\n  TRAINED-AGAINST ROW  X = {trained}")
        if trained in R["reasons"]:
            print(f"    between row NaN'd: {R['reasons'][trained]}")
        row = pd.DataFrame({"r_within_early": R["r_within_early"].loc[trained],
                            "beta_within_early": R["beta_within_early"].loc[trained],
                            "beta_within_all": R["beta_within"].loc[trained],
                            "beta_between": R["beta_between"].loc[trained],
                            "r_between": R["r_between"].loc[trained]})
        print(row.to_string())


def report_condition(results, trained):
    ok = {s: R for s, R in results.items() if R["ok"] and trained in R["mons"]}
    if len(ok) < 2:
        return
    print("-" * 100)
    print(f"ACROSS SEEDS, trained-against row X = {trained}   (between NaN = row failed the movement guard)")
    for name in ("r_within_early", "beta_within_early", "beta_between"):
        tab = pd.DataFrame({lab: R[name].loc[trained] for lab, R in ok.items()})
        tab["mean"] = tab.mean(axis=1); tab["sd"] = tab.iloc[:, :len(ok)].std(axis=1)
        print(f"\n  {name}")
        print(tab.to_string())


def report_cross(all_results, out):
    """all_results: {(token, trained_monitor): {seed: R}}. Returns the long-form df (or None)."""
    rows = []
    for (tok, X), res in all_results.items():
        for lab, R in res.items():
            if not R["ok"] or X not in R["mons"]:
                continue
            for Y in R["mons"]:
                rows.append(dict(condition=tok, trained=X, run=lab, Y=Y,
                                 r_within_early=R["r_within_early"].loc[X, Y],
                                 beta_within_early=R["beta_within_early"].loc[X, Y],
                                 beta_within_all=R["beta_within"].loc[X, Y],
                                 beta_between=R["beta_between"].loc[X, Y],
                                 between_ok=bool(R["row_ok"][X]), moved_se=R["move_se"][X],
                                 n_windows=len(R["cells"]), n_early_windows=R["n_early"]))
    if not rows:
        return None
    long = pd.DataFrame(rows)
    if out:
        long.to_csv(out, index=False)
        print(f"\n[wrote {len(long)} causal rows to {out}]")
    if long["condition"].nunique() < 2:
        return long
    print("=" * 100)
    print("CROSS-CONDITION CAUSAL SUMMARY   rows = trained-against condition, cols = held-out monitor")
    okl = long[long.between_ok]
    for name, src in (("beta_between (mean over seeds passing the guard; n in parens)", okl),
                      ("beta_within_early (mean over all usable seeds)", long)):
        piv = src.pivot_table(index="trained", columns="Y", values=name.split(" ")[0], aggfunc="mean")
        cnt = src.pivot_table(index="trained", columns="Y", values=name.split(" ")[0], aggfunc="count")
        print(f"\n  {name}")
        fmt = piv.copy().astype(object)
        for i in piv.index:
            for j in piv.columns:
                v, c = piv.loc[i, j], cnt.loc[i, j]
                fmt.loc[i, j] = "   NaN" if pd.isna(v) else f"{v:6.3f}({int(c)})"
        print(fmt.to_string())
    print("\n  scatter table (one row per condition x run x held-out Y):")
    cols = ["condition", "run", "Y", "r_within_early", "beta_within_early", "beta_between", "between_ok", "moved_se"]
    print(long[long.Y != long.trained][cols].to_string(index=False))
    return long


# ------------------------------------------------------------------------ figures
def _boot_ci(vals, B, rng):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(v) == 1:
        return v[0], np.nan, np.nan, 1
    means = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return v.mean(), np.percentile(means, 5), np.percentile(means, 95), len(v)


def _heat(ax, M, lo, hi, nlab, rows, cols, title, plt):
    im = ax.imshow(np.clip(M, -1, 1), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = M[i, j]
            if r.split("|")[0] == c:                       # diagonal: identically 1
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=True, fc="#eee", ec="k", lw=1.5))
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#888")
                continue
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="#bbb")
                continue
            col = "white" if abs(v) > 0.6 else "k"
            if np.isnan(lo[i, j]):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color=col)
            else:
                ax.text(j, i - 0.12, f"{v:+.2f}", ha="center", va="center", fontsize=9, color=col)
                ax.text(j, i + 0.24, f"[{lo[i, j]:+.2f}, {hi[i, j]:+.2f}]", ha="center", va="center",
                        fontsize=6.3, color=col)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8.5)
    for i, lab in enumerate(nlab):
        ax.text(-0.015, i + 0.33, lab, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6, color="#888")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("→ held-out monitor Y"); ax.set_ylabel("trained against X")
    return im


def make_figures(long, mons, fig_dir, bootstrap=2000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    conds = list(dict.fromkeys(long["condition"]))
    cols = [m for m in mons if m in set(long["Y"])]
    panels = [("beta_between", long[long.between_ok], "n_windows",
               "REALIZED co-degradation\nβ_between(Y←X): Δ mean Y-sus per Δ mean X-sus on hacks, over training"),
              ("beta_within_early", long, "n_early_windows",
               "PREDICTED from early hacks\nβ_within_early(Y←X): slope of Y-sus on X-sus across hacks, policy fixed")]
    fig, axes = plt.subplots(2, 1, figsize=(2.5 + 1.45 * len(cols), 2 * (1.9 + 0.85 * len(conds))))
    summary = {}
    for ax, (name, src, wcol, title) in zip(axes, panels):
        M = np.full((len(conds), len(cols)), np.nan); lo = M.copy(); hi = M.copy()
        n_runs, n_win, nlab = [], [], []
        for i, c in enumerate(conds):
            for j, y in enumerate(cols):
                vals = src[(src.condition == c) & (src.Y == y)][name]
                M[i, j], lo[i, j], hi[i, j], _ = _boot_ci(vals, bootstrap, rng)
            runs = src[src.condition == c].drop_duplicates("run")
            r, w = len(runs), int(runs[wcol].sum()) if len(runs) else 0
            n_runs.append(r); n_win.append(w); nlab.append(f"n={r}r/{w}w")
        im = _heat(ax, M, lo, hi, nlab, conds, cols,
                   title + "\ncells: mean over runs, [5–95 pct bootstrap-over-runs CI] · n = runs / windows used", plt)
        if ax is axes[0]:
            ax.set_xticklabels([]); ax.set_xlabel("")
        summary[name] = dict(rows=conds, cols=cols, mean=M.tolist(), lo=lo.tolist(), hi=hi.tolist(),
                             n_runs=n_runs, n_windows=n_win)
    fig.subplots_adjust(hspace=0.3)
    fig.colorbar(im, ax=axes, label="β (color clipped to ±1; numbers not clipped)", shrink=0.6, pad=0.02)
    fig.savefig(fig_dir / "beta_heatmaps.png", bbox_inches="tight", dpi=150)

    # prediction-test scatter
    sc = long[(long.Y != long.trained) & long.between_ok].dropna(subset=["beta_within_early", "beta_between"])
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(sc):
        cmap = plt.get_cmap("tab10")
        for ci, c in enumerate(conds):
            s = sc[sc.condition == c]
            if not len(s):
                continue
            ax.scatter(s.beta_within_early, s.beta_between, s=36, color=cmap(ci % 10), label=f"{c} ({s.run.nunique()}r)", alpha=0.85)
            for _, r in s.iterrows():
                ax.annotate(r.Y, (r.beta_within_early, r.beta_between), fontsize=6, alpha=0.7,
                            xytext=(3, 2), textcoords="offset points")
        allv = np.r_[sc.beta_within_early, sc.beta_between]
        lo, hi = min(0, allv.min()) - 0.1, allv.max() + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="β_between = β_within_early")
        ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("β_within_early  (predicted from early hack variation)")
    ax.set_ylabel("β_between  (realized over training)")
    ax.set_title("Prediction test, one point per (condition × run × held-out Y)\n"
                 "on the diagonal: selection on existing variation · above: discovered · below: less than predicted", fontsize=9)
    fig.savefig(fig_dir / "beta_scatter.png", bbox_inches="tight", dpi=150)
    (fig_dir / "beta_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n[figures: {fig_dir}/beta_heatmaps.png, beta_scatter.png, beta_summary.json]")


# --------------------------------------------------------------------------- run
def parse_conditions(items):
    out = []
    for it in items:
        tok, _, mon = it.partition("=")
        out.append((tok, mon or tok))
    return out


def trained_from_config(cfg):
    for m in cfg.get("monitors", []):
        if m.get("role") == "train_against":
            return m["name"]
    return None


def specs_from_pattern(base, conditions, seeds, rollout_file):
    specs = []
    for tok, X in parse_conditions(conditions):
        for s in seeds:
            d = Path(base.format(trained=tok, seed=s))
            specs.append(dict(cond=tok, trained=X, seed=s, label=f"s{s}", dir=d, run_name=d.name))
    return specs


def specs_from_dir(runs_dir, group_by, rollout_file):
    specs = []
    for d in sorted(Path(runs_dir).iterdir()):
        if not d.is_dir() or not (d / rollout_file).exists():
            continue
        cfgp = d / "config.json"
        if not cfgp.exists():
            print(f"[warn] {d.name}: no config.json, skipped", file=sys.stderr)
            continue
        cfg = json.load(open(cfgp))
        X = trained_from_config(cfg)
        if X is None:
            print(f"[warn] {d.name}: no monitor with role train_against, skipped", file=sys.stderr)
            continue
        seed = cfg.get("seed")
        run_name = cfg.get("run_name", d.name)
        m = re.search(r"seed[-_](\d+)", run_name) or re.search(r"seed[-_](\d+)", d.name)
        if m and seed is not None and int(m.group(1)) != seed:
            print(f"[warn] {d.name}: run_name says seed {m.group(1)} but config seed = {seed}; using config",
                  file=sys.stderr)
        cond = X + "".join(f"|{k}={cfg.get(k)}" for k in group_by)
        specs.append(dict(cond=cond, trained=X, seed=seed, dir=d, run_name=run_name))
    by_cond = defaultdict(list)
    for sp in specs:
        by_cond[sp["cond"]].append(sp)
    for sps in by_cond.values():                    # label = sN when unique in condition, else folder
        seeds = [sp["seed"] for sp in sps]
        for sp in sps:
            uniq = sp["seed"] is not None and seeds.count(sp["seed"]) == 1
            sp["label"] = f"s{sp['seed']}" if uniq else sp["dir"].name
    return specs


def run_specs(specs, rollout_file, min_hacks, splits, early_frac, monitors, within_task,
              min_move_se, full, out, fig_dir=None, bootstrap=2000, seed_rng=0):
    rng = np.random.default_rng(seed_rng)
    print(f"[beta_within computed {'WITHIN task (GRPO-relevant)' if within_task else 'across all hacks in cell'}]")
    groups = defaultdict(list)
    for sp in specs:
        groups[(sp["cond"], sp["trained"])].append(sp)
    all_results, mons_order = {}, None
    for (cond, X), sps in groups.items():
        results = {}
        print("=" * 100)
        print(f"CONDITION {cond}  (trained against {X})  -- {len(sps)} run(s)")
        for sp in sps:
            p = sp["dir"] / rollout_file
            if not p.exists():
                print(f"  [warn] missing {p}", file=sys.stderr)
                continue
            df, mons = load_run(p)
            if monitors:
                mons = [m for m in monitors if m in mons]
            df = df.dropna(subset=mons)
            mons_order = mons_order or mons
            print("-" * 100 + f"\nRUN {sp['label']}   ({sp['dir'].name})")
            R = analyze_run(df, mons, min_hacks, splits, early_frac, rng, within_task, min_move_se)
            R["run_name"] = sp["run_name"]
            results[sp["label"]] = R
            report_run(R, X, full)
        if results:
            report_condition(results, X)
            all_results[(cond, X)] = results
    if not all_results:
        sys.exit("no runs loaded")
    long = report_cross(all_results, out)
    if fig_dir and long is not None:
        make_figures(long, mons_order, fig_dir, bootstrap)
    return all_results


# ----------------------------------------------------------------------- selftest
MONS = ["probe_iid", "cot_only_weak", "cot_only_strong", "cot+out_weak", "cot+out_strong",
        "out_weak", "out_strong", "probe_ood"]
DRIFT = {"probe_iid":     np.array([-0.45, -0.15, -0.05, -0.20, -0.08, -0.03, -0.02, -0.30]),
         "cot_only_weak": np.array([-0.05, -0.45, -0.20, -0.30, -0.10, -0.03, -0.02, -0.05])}

def make_synthetic(root, seeds=(0, 1, 2), n_tasks=40, samples=4, eval_every=8, n_steps=200, rng_seed=123):
    rng = np.random.default_rng(rng_seed)
    k = len(MONS)
    load = np.array([[0.6, 0.2], [0.7, 0.1], [0.5, 0.1], [0.5, 0.5], [0.4, 0.4], [0.1, 0.7], [0.1, 0.5], [0.5, 0.2]])
    noise_sd = np.array([0.08, 0.15, 0.10, 0.15, 0.10, 0.15, 0.10, 0.10])
    Sw = load @ load.T * 0.02 + np.diag(noise_sd ** 2)
    L = np.linalg.cholesky(Sw)
    true_bw = Sw / np.diag(Sw)[:, None]
    mu0 = np.array([0.75, 0.70, 0.85, 0.72, 0.88, 0.60, 0.80, 0.70])
    true_bb = {}
    for tok, d_base in DRIFT.items():
        for s in seeds:
            d = d_base * (1 + 0.15 * rng.standard_normal(k))
            true_bb[(tok, s)] = d[None, :] / d[:, None]
            out = root / Path(DEFAULT_BASE).name.format(trained=tok, seed=s)
            out.mkdir(parents=True, exist_ok=True)
            cfg = {"run_name": f"synthetic-against-{tok}-seed-{s}", "seed": s, "penalty_coef": 0.5,
                   "monitors": [{"name": m, "role": "train_against" if m == tok else "held_out", "threshold": 0.5}
                                for m in MONS]}
            json.dump(cfg, open(out / "config.json", "w"), indent=1)
            with open(out / ROLLOUT_FILE, "w") as f:
                for step in range(0, n_steps + 1, eval_every):
                    frac = step / n_steps
                    p_hack = 0.03 + 0.85 / (1 + np.exp(-(frac - 0.35) * 12))
                    mu_hack, mu_clean = mu0 + d * frac, 0.08 + 0.10 * frac
                    for task in range(n_tasks):
                        for _ in range(samples):
                            hack = rng.random() < p_hack
                            sc = mu_hack + L @ rng.standard_normal(k) if hack else mu_clean + 0.05 * rng.standard_normal(k)
                            sc = np.clip(sc, 0, 1)
                            rec = {"step": step, "task_id": task, "behavior_present": bool(hack),
                                   "loose_rh": bool(hack), "hardcoding": bool(hack), "unparsed": False,
                                   "monitors": {m: {"score": float(sc[i]), "label": bool(sc[i] > 0.5)}
                                                for i, m in enumerate(MONS)}}
                            f.write(json.dumps(rec) + "\n")
    return true_bw, true_bb


def selftest(min_hacks, splits, early_frac, within_task, min_move_se):
    root = Path(tempfile.mkdtemp(prefix="beta_selftest_"))
    true_bw, true_bb = make_synthetic(root)
    print(f"synthetic data in {root}  (exercising --runs-dir discovery via config.json)\n")
    specs = specs_from_dir(root, [], ROLLOUT_FILE)
    res = run_specs(specs, ROLLOUT_FILE, min_hacks, splits, early_frac, None, within_task,
                    min_move_se, full=False, out=None, fig_dir=root / "_beta")
    print("\n" + "#" * 100 + "\nSELFTEST: recovered vs true, trained-against rows")
    worst = 0.0
    for (tok, X), results in res.items():
        for lab, R in results.items():
            s = int(lab[1:]); i = MONS.index(X)
            tab = pd.DataFrame({"true beta_within": true_bw[i], "est beta_within": R["beta_within"].loc[X],
                                "true beta_between": true_bb[(tok, s)][i], "est beta_between": R["beta_between"].loc[X]},
                               index=MONS)
            eb = np.nanmax(np.abs(tab["est beta_between"] - tab["true beta_between"]))
            ew = np.nanmax(np.abs(tab["est beta_within"] - tab["true beta_within"]))
            worst = max(worst, eb, ew)
            print(f"\ncondition {tok}, seed {s}   max abs err: between {eb:.3f}  within {ew:.3f}")
            print(tab.to_string())
    print(f"\nworst error {worst:.3f}. Within and between differ by construction, so conflating them would fail.")


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=None,
                    help="folder of run subfolders, each with config.json + rollouts; overrides --base/--trained/--seeds")
    ap.add_argument("--group-by", nargs="*", default=[],
                    help="extra config.json keys that split conditions (e.g. penalty_coef lr)")
    ap.add_argument("--rollout-file", default=ROLLOUT_FILE)
    ap.add_argument("--base", default=DEFAULT_BASE, help="run dir pattern with {trained} and {seed}")
    ap.add_argument("--trained", nargs="+", default=["probe_iid"],
                    help="train-against conditions: monitor names, or dirtoken=monitor")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--monitors", nargs="+", default=None, help="subset/order of monitors (default: all found)")
    ap.add_argument("--min-hacks", type=int, default=15, help="min hack rollouts per cell")
    ap.add_argument("--splits", type=int, default=25, help="random half-splits per cell to average over")
    ap.add_argument("--early-frac", type=float, default=1 / 3, help="fraction of cells counted as 'early'")
    ap.add_argument("--min-move-se", type=float, default=3.0,
                    help="between row X reported only if X's hack-mean moved by >= this many cell-mean SEs")
    ap.add_argument("--within-task", action="store_true", help="demean by task_id inside each cell for beta_within")
    ap.add_argument("--full", action="store_true", help="print all NxN matrices per run")
    ap.add_argument("--out", default=None, help="CSV path for the long-form causal rows")
    ap.add_argument("--fig-dir", default=None,
                    help="write beta_heatmaps.png / beta_scatter.png / beta_summary.json here "
                         "(default: <runs-dir>/_beta in --runs-dir mode, ./_beta otherwise; --no-fig to skip)")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap-over-runs samples for the heatmap CIs")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.min_hacks, a.splits, a.early_frac, a.within_task, a.min_move_se)
    else:
        if a.runs_dir:
            specs = specs_from_dir(a.runs_dir, a.group_by, a.rollout_file)
        else:
            specs = specs_from_pattern(a.base, a.trained, a.seeds, a.rollout_file)
        if not specs:
            sys.exit("no runs found")
        fig_dir = None if a.no_fig else (a.fig_dir or (str(Path(a.runs_dir) / "_beta") if a.runs_dir else "_beta"))
        run_specs(specs, a.rollout_file, a.min_hacks, a.splits, a.early_frac, a.monitors,
                  a.within_task, a.min_move_se, a.full, a.out, fig_dir, a.bootstrap)