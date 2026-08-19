"""Render a run's metrics to PNGs (no W&B server needed). Auto-called at the end of every run.

Two metric streams → two plot folders:
  data/runs/<run>/train/  — from metrics.jsonl (per-step, training rollouts): behavior rate + reward,
                            and the train-against monitor's penalty score.
  data/runs/<run>/eval/   — from eval_metrics.jsonl (held-out eval set): held-out behavior rate, and
                            every monitor's AUROC/accuracy over training (the clean degradation curves).

The cross-run **degradation matrix** is NOT here — it spans multiple runs (see eval/degradation.py).
"""

from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _series(rows: list[dict], key: str) -> tuple[list, list]:
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] == r[key]:  # skip NaN
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def _role_labels(run_dir: Path) -> dict[str, str]:
    """name -> 'name = model [TRAIN-AGAINST/held-out]' from run_info.json (for legend clarity)."""
    out: dict[str, str] = {}
    info_path = run_dir / "run_info.json"
    if not info_path.exists():
        return out
    info = json.loads(info_path.read_text())
    for m in info.get("train_against", []):
        out[m["name"]] = f"{m['name']} = {m.get('model_id')} [TRAIN-AGAINST]"
    for m in info.get("held_out", []):
        out[m["name"]] = f"{m['name']} = {m.get('model_id')} [held-out]"
    return out


def _monitor_names(rows: list[dict]) -> list[str]:
    return sorted({k.split("/")[1] for r in rows for k in r if k.startswith("monitor/")})


def _behavior(run_dir: Path) -> str:
    """The env's behavior slug (reward_hacking / sycophancy / deception) from run_info.json, for naming
    charts + the class-split score metric per-env. Falls back to a neutral 'behavior'."""
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        beh = json.loads(info_path.read_text()).get("env", {}).get("behavior_name")
        if beh:
            return beh
    return "behavior"


def plot_run(run_dir: str | Path) -> list[Path]:
    """Write train/ and eval/ plot sets for the run. Returns the PNG paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    role_label = _role_labels(run_dir)
    beh = _behavior(run_dir)              # e.g. reward_hacking / sycophancy / deception
    beh_pretty = beh.replace("_", " ")   # for chart titles/labels
    pres_metric = f"mean_score_{beh}"    # the behavior-present class mean (per-monitor), named per-env
    out: list[Path] = []

    def ground_truth_fig(rows, series, title, path):
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for key, label in series:
            xs, ys = _series(rows, key)
            if xs:
                ax.plot(xs, ys, marker="o", label=label)
                plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_xlabel("step")
        ax.set_ylabel("rate / reward")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out.append(path)

    def monitors_fig(rows, metrics, title, path):
        names = _monitor_names(rows)
        if not names:
            return
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
        any_data = False
        for ax, metric in zip(axes[0], metrics):
            for name in names:
                xs, ys = _series(rows, f"monitor/{name}/{metric}")
                if xs:
                    ax.plot(xs, ys, marker="o", label=role_label.get(name, name))
                    any_data = True
            # Reference line: 0.5 = chance for auroc/accuracy; 0.25 = the always-0.5 baseline for brier.
            ax.axhline(0.25 if metric == "brier" else 0.5, color="grey", lw=0.8, ls=":")
            ax.set_xlabel("step")
            ax.set_ylabel(metric)
            ax.set_title(metric + ("  (lower=better)" if metric == "brier" else ""))
            ax.set_ylim(-0.05, 1.05)
            if ax.get_legend_handles_labels()[0]:  # only if something was actually plotted+labeled
                ax.legend(fontsize=8)
        if not any_data:
            plt.close(fig)
            return
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out.append(path)

    def tokens_fig(rows, title, path):
        """Token accounting: batch totals (left) vs per-rollout means (right). Two panels because the
        scales differ by ~the batch size; truncation rate rides the per-rollout panel on a twin axis
        since a saturating output length is only interpretable next to it."""
        panels = [("batch total (per step)", [("tokens/input_total", "input"),
                                              ("tokens/output_total", "output"),
                                              ("tokens/total", "input+output")]),
                  ("per rollout", [("tokens/input_per_rollout", "input"),
                                   ("tokens/output_per_rollout", "output"),
                                   ("tokens/output_max", "output (max)")])]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        any_data = False
        for ax, (sub, series) in zip(axes, panels):
            for key, label in series:
                xs, ys = _series(rows, key)
                if xs:
                    ax.plot(xs, ys, marker="o", ls="--" if "max" in key else "-", label=label)
                    any_data = True
            ax.set_xlabel("step")
            ax.set_ylabel("tokens")
            ax.set_title(sub)
            ax.set_ylim(bottom=0)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        xs, ys = _series(rows, "tokens/truncated_rate")
        if xs:
            ax2 = axes[1].twinx()
            ax2.plot(xs, ys, color="grey", lw=0.9, ls=":", label="truncated frac")
            ax2.set_ylabel("truncated fraction")
            ax2.set_ylim(-0.02, 1.02)
            ax2.legend(fontsize=8, loc="lower right")
        if not any_data:
            plt.close(fig)
            return
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out.append(path)

    def budget_tokens_fig(rows, title, path):
        """Thinking-budget cost: what the budget would cost in-engine (ideal) vs what tinker's
        sample→force-close→resume protocol really spent (actual), for prefill and decode alike, with
        prefill split into measured prefix-cache hits and misses.

        Skipped entirely for runs without a thinking budget (the keys are simply absent). The one to
        watch is the cache-MISS panel: cached prefill is billed at a discount, so the re-prefill of a
        continuation is only expensive to the extent it misses.
        """
        panels = [
            ("prefill (batch total)", [("tokens/prefill_ideal_total", "ideal"),
                                       ("tokens/prefill_actual_total", "actual")]),
            ("prefill cache miss / hit", [("tokens/prefill_ideal_cache_miss_total", "ideal miss"),
                                          ("tokens/prefill_actual_cache_miss_total", "actual miss"),
                                          ("tokens/prefill_ideal_cache_hit_total", "ideal hit"),
                                          ("tokens/prefill_actual_cache_hit_total", "actual hit")]),
            ("decode (batch total)", [("tokens/decode_ideal_total", "ideal"),
                                      ("tokens/decode_actual_total", "actual")]),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        any_data = False
        for ax, (sub, series) in zip(axes, panels):
            for key, label in series:
                xs, ys = _series(rows, key)
                if xs:
                    ax.plot(xs, ys, marker="o", ls="--" if "ideal" in key else "-", label=label)
                    any_data = True
            ax.set_xlabel("step")
            ax.set_ylabel("tokens")
            ax.set_title(sub)
            ax.set_ylim(bottom=0)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        if not any_data:
            plt.close(fig)
            return
        # How often the cap actually bound — an overhead curve is meaningless without it.
        xs, ys = _series(rows, "tokens/budget_forced_rate")
        if xs:
            ax2 = axes[2].twinx()
            ax2.plot(xs, ys, color="grey", lw=0.9, ls=":", label="budget forced frac")
            ax2.set_ylabel("forced fraction")
            ax2.set_ylim(-0.02, 1.02)
            ax2.legend(fontsize=8, loc="lower right")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out.append(path)

    # TRAIN plots (training rollouts)
    train_rows = _load(run_dir / "metrics.jsonl")
    if train_rows:
        td = run_dir / "train"
        td.mkdir(exist_ok=True)
        ground_truth_fig(
            train_rows,
            # behavior_rate is the primary ground-truth series; loose_rate/hardcoding_rate only exist for
            # MBPP-Honeypot (skipped automatically when absent for other envs).
            [("behavior_rate", f"{beh_pretty} rate (train)"), ("loose_rate", "loose RH (train)"),
             ("hardcoding_rate", "hardcoding (train)"),
             ("reward/monitor_suspiciousness_mean", "monitor suspiciousness"),
             ("reward/penalty_mean", "applied penalty (λ·sus)"),
             ("reward/len_penalty_mean", "code-length penalty"), ("reward/task_mean", "task reward")],
            f"{run_dir.name} [train] — {beh_pretty} + reward", td / "ground_truth.png")
        monitors_fig(train_rows, ["mean_score"], f"{run_dir.name} [train] — train-against penalty",
                     td / "monitors.png")
        tokens_fig(train_rows, f"{run_dir.name} [train] — token counts", td / "tokens.png")
        budget_tokens_fig(train_rows, f"{run_dir.name} [train] — thinking-budget token cost",
                          td / "tokens_budget.png")

    # EVAL plots (held-out set — the degradation curves)
    eval_rows = _load(run_dir / "eval_metrics.jsonl")
    if eval_rows:
        ed = run_dir / "eval"
        ed.mkdir(exist_ok=True)
        ground_truth_fig(eval_rows,
                         [("behavior_rate", f"{beh_pretty} rate (held-out)"),
                          ("loose_rate", "loose RH (held-out)"),
                          ("hardcoding_rate", "hardcoding (held-out)")],
                         f"{run_dir.name} [eval] — held-out {beh_pretty}", ed / "ground_truth.png")
        # AUROC is the headline; brier + the behavior-present mean score stay defined when AUROC goes NaN
        # (single-class, i.e. the obfuscation-success regime), so they carry the signal at the extremes.
        monitors_fig(eval_rows, ["auroc", "brier", pres_metric],
                     f"{run_dir.name} [eval] — held-out detector degradation", ed / "monitors.png")
        tokens_fig(eval_rows, f"{run_dir.name} [eval] — token counts", ed / "tokens.png")
        budget_tokens_fig(eval_rows, f"{run_dir.name} [eval] — thinking-budget token cost",
                          ed / "tokens_budget.png")

    return out
