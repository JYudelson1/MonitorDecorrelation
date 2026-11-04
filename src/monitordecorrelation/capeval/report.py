"""The capeval report: ``build_report`` computes everything into a JSON-able dict (what ``--out`` holds);
``render_report`` turns that dict — fresh, or loaded back from a finished file — into the text the
script prints. Rendering reads ONLY the saved dict, so ``eval_checkpoints.py --show <out>.json``
reprints exactly what the run printed."""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

from monitordecorrelation.capeval import stats
from monitordecorrelation.capeval.policy import USAGE_FIELDS

# Reports written before ``lower_is_better`` was saved per benchmark.
_DEFAULT_LOWER_IS_BETTER = ("rl_fried", "repetition")

DIAG_LEGEND = (
    "forced% = generation calls whose thinking hit the budget and was force-closed; trunc% = calls whose "
    "answer still hit the cap; ctxclip%/ctxfull% = budget shrunk by / no room left in the context window; "
    "noans% = no extractable final answer; turnlim% = agent episodes that hit the turn limit; per episode: "
    "fmt = replies with no bash block, multi = replies with several (only the first runs), presub = submits "
    "refused because the graded file did not exist, tmo = command timeouts."
)


def iter_sample_lines(path: str) -> Iterator[dict]:
    """Every record in a samples file, superseded ones included (a torn last line is skipped)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------------------------
def _zero() -> dict:
    return dict.fromkeys(USAGE_FIELDS, 0)


def aggregate_usage(lines: Iterable[dict]) -> dict:
    """Sum the per-sample ``usage`` of EVERY line of a samples file — i.e. everything every invocation
    writing it was billed for, including samples later re-run (after an error or a settings change) and
    re-judgements (a re-judged line carries only the judge tokens of that re-judgement)."""
    by_account: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    by_bench: dict[str, dict] = {}
    n = n_missing = 0
    for r in lines:
        n += 1
        u = r.get("usage")
        if u is None:
            n_missing += 1
            continue
        for acct, a in u.items():
            for d in (by_account.setdefault(acct, _zero()),
                      by_model.setdefault(r["model"], {}).setdefault(acct, _zero()),
                      by_bench.setdefault(r["bench"], {}).setdefault(acct, _zero())):
                for k in USAGE_FIELDS:
                    d[k] += a.get(k, 0) or 0
    return {"scope": "every line of the samples file (all invocations, superseded samples included)",
            "n_sample_lines": n, "n_lines_without_usage": n_missing, "by_account": by_account,
            "by_evaluated_model": by_model, "by_benchmark": by_bench}


def _account_label(acct: str, models: dict) -> str:
    kind, _, name = acct.partition(":")
    if kind == "tinker":
        n_ck = sum(1 for m in models.values() if not m.get("is_base") and m.get("base_model") == name)
        return f"{name} (base + {n_ck} checkpoint{'s' if n_ck != 1 else ''})"
    if kind == "openrouter":
        return f"{name} (judge, via OpenRouter)"
    return acct


def render_usage(usage: dict | None, models: dict) -> list[str]:
    title = ("\n=== Token usage, millions (each base model counted together with its checkpoints; "
             "every sample of every invocation that wrote this file) ===")
    if not usage or not usage.get("by_account"):
        return [title, "(not recorded — this file was written before token usage was tracked)"]
    accts = sorted(usage["by_account"], key=lambda a: (not a.startswith("tinker:"), a))
    m = lambda v: f"{v / 1e6:.2f}"  # noqa: E731
    rows, tot = [], _zero()
    for a in accts:
        u = usage["by_account"][a]
        for k in USAGE_FIELDS:
            tot[k] += u[k]
        rows.append([_account_label(a, models), f"{u['requests']:,}", m(u["input_miss"]), m(u["input_hit"]),
                     m(u["output"]), m(u["input_miss"] + u["input_hit"] + u["output"])])
    rows.append(["total", f"{tot['requests']:,}", m(tot["input_miss"]), m(tot["input_hit"]), m(tot["output"]),
                 m(tot["input_miss"] + tot["input_hit"] + tot["output"])])
    out = [title, stats.render_table(
        ["model", "requests", "input (cache miss)", "input (cache hit)", "output", "total"], rows)]
    notes = []
    judge_cost = sum(usage["by_account"][a].get("cost_usd", 0) for a in accts if a.startswith("openrouter:"))
    if judge_cost:
        notes.append(f"judge cost reported by OpenRouter: ${judge_cost:,.2f}")
    if usage.get("n_lines_without_usage"):
        notes.append(f"{usage['n_lines_without_usage']} of {usage['n_sample_lines']} sample lines were written "
                     "before usage was tracked and are not counted")
    notes.append("requests = API calls (a generation whose thinking was force-closed is 2 calls); calls that "
                 "failed without a response are not counted")
    out += ["; ".join(notes) + "."]
    return out


# ---------------------------------------------------------------------------------------------
# Build (data) / render (text)
# ---------------------------------------------------------------------------------------------
def build_report(*, specs: list, benches: list, results: dict[tuple, dict], usage: dict, meta: dict,
                 deltas: bool = True) -> dict:
    """Everything the report shows, as a JSON-able dict. ``results`` = the current record per
    (model, bench, item, sample); ``usage`` = ``aggregate_usage`` over the whole samples file."""
    by: dict[tuple[str, str], list[dict]] = {}
    for (model, bname, _, _), r in results.items():
        by.setdefault((model, bname), []).append(r)
    base_of = {s.key: f"base:{s.base_model}" for s in specs if not s.is_base}
    keys = [s.key for s in specs]
    out = {"meta": meta, "models": {s.key: dict(vars(s)) for s in specs}, "results": {}, "deltas": {},
           "diagnostics": {}, "warnings": [], "token_usage": usage, "per_item": {}}
    for k in keys:
        out["results"][k] = {}
        for b in benches:
            recs = by.get((k, b.name), [])
            out["results"][k][b.name] = {}
            for m in b.metrics:
                s = stats.score_summary(recs, m)
                s["by_group"] = stats.group_summaries(recs, m)
                out["results"][k][b.name][m] = s
    if deltas:
        for c, base in base_of.items():
            if base not in keys:
                continue
            out["deltas"][c] = {"base": base}
            for b in benches:
                out["deltas"][c][b.name] = {m: stats.paired_delta(by.get((c, b.name), []), by.get((base, b.name), []), m)
                                            for m in b.metrics}
    for b in benches:
        out["diagnostics"][b.name] = {k: stats.diagnostics(by.get((k, b.name), [])) for k in keys}
        out["warnings"] += stats.confounder_warnings(out["diagnostics"][b.name],
                                                     {c: v for c, v in base_of.items() if v in keys}, b.name)
        if getattr(b, "dropped", None):
            out["warnings"].append(f"{b.name}: {len(b.dropped)} items dropped by grader validation "
                                   "(see meta.benchmarks.<name>.dropped)")
        out["per_item"][b.name] = {}
        for m in b.metrics:
            tab: dict[str, dict] = {}
            for k in keys:
                for item, (_, mean, n) in stats.item_means(by.get((k, b.name), []), m).items():
                    tab.setdefault(item, {})[k] = {"mean": mean, "n": n}
            out["per_item"][b.name][m] = tab
    return out


def render_report(rep: dict) -> str:
    """The printed report, from a ``build_report`` dict (or the same dict loaded from ``--out``)."""
    benches = rep["meta"]["benchmarks"]
    lib = {b: set(info.get("lower_is_better", _DEFAULT_LOWER_IS_BETTER)) for b, info in benches.items()}
    metric_cols = [(b, m) for b, info in benches.items() for m in info["metrics"]]
    label = lambda b, m: f"{b}:{m}{'↓' if m in lib[b] else ''}"  # noqa: E731
    models = rep["models"]
    keys = list(models)
    lines: list[str] = []

    rows = [[k] + [stats.fmt_ci(m, rep["results"][k][b][m]) for b, m in metric_cols] for k in keys]
    lines.append("\n=== Scores (mean ± 95% CI half-width; accuracy/pass/repetition in %, judge scores 1-10) ===")
    lines.append(stats.render_table(["model"] + [label(b, m) for b, m in metric_cols], rows))

    ckpts = [k for k in keys if k in rep.get("deltas", {})]
    if ckpts:
        header = ["checkpoint", "base"] + [label(b, m) for b, m in metric_cols]
        rows = []
        for c in ckpts:
            row = [c, rep["deltas"][c]["base"].split(":", 1)[1]]
            for b, m in metric_cols:
                d = rep["deltas"][c][b][m]
                row.append(stats.fmt_ci(m, d, signed=True) + (" *" if d["significant"] else "  "))
            rows.append(row)
        lines.append("\n=== Checkpoint − base (paired over items; ± = 95% cluster-bootstrap CI half-width; "
                     "* = CI excludes 0; ↓ = lower is better; not corrected for multiple comparisons) ===")
        lines.append(stats.render_table(header, rows))

    lines.append("\n=== Confounder diagnostics (per model × benchmark) ===")
    dh = ["benchmark", "model", "n", "err", "forced%", "trunc%", "ctxclip%", "ctxfull%", "noans%",
          "turnlim%", "fmt/ep", "multi/ep", "presub/ep", "tmo/ep", "turns", "gen_tok", "p95_tok", "judgefail%"]
    drows = []
    for b in benches:
        for k in keys:
            d = rep["diagnostics"][b][k]
            pct = lambda key: "" if key not in d else f"{100 * d[key]:.1f}"  # noqa: E731
            num = lambda key, f="{:.2f}": "" if key not in d else f.format(d[key])  # noqa: E731
            drows.append([b, k, d["n_records"], d["n_errors"], pct("forced_rate"), pct("truncated_rate"),
                          pct("ctx_clipped_rate"), pct("context_exhausted_rate"), pct("extract_fail_rate"),
                          pct("turn_limit_rate"), num("format_errors_mean"), num("multi_blocks_mean"),
                          num("premature_submits_mean"), num("cmd_timeouts_mean"), num("turns_mean", "{:.1f}"),
                          num("gen_tokens_mean", "{:.0f}"), num("gen_tokens_p95", "{:.0f}"),
                          pct("judge_fail_rate")])
    lines.append(stats.render_table(dh, drows))
    lines.append(DIAG_LEGEND)

    if rep.get("warnings"):
        lines.append("\n=== Possible confounders (rates that are high, or — '≠' — far enough from the base "
                     "model's to move a delta by themselves; see the table above) ===")
        lines += [f"  ! {w}" for w in rep["warnings"]]

    lines += render_usage(rep.get("token_usage"), models)

    meta = rep["meta"]
    wall = meta.get("wall_clock_s")
    lines.append(f"\nwall clock of the invocation that wrote this report "
                 f"{'n/a' if wall is None else f'{wall / 60:.1f} min'} · detailed breakdown: "
                 f"{meta.get('out_file', '(this file)')} · every sample (transcripts): {meta.get('samples_file')}")
    return "\n".join(lines)
