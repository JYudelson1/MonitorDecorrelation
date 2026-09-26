#!/usr/bin/env python3
"""Post-launch sanity check: is every run actually training against the monitor its name claims?

Run this a few minutes after launching a batch, BEFORE letting it burn a day of GPU time. It reads
each run's own ``run_info.json`` (written at startup, so it reflects what the process really built —
not what the launch command looked like) and flags anything that does not line up.

Motivated by a real, expensive failure: days of runs silently trained against the wrong monitor
because a config path in the launch command was shadowed by an environment variable, and nothing
checked the effective config afterwards (Vladimir, terminal-env, 2026-09-15).

    uv run python scripts/verify_runs.py 'data/runs/mbpp_Qwen3-8B_*_sep3'
    uv run python scripts/verify_runs.py data/runs/mbpp_matrix_sep3_20260903 --expect-monitors 8
    uv run python scripts/verify_runs.py '<glob>' --quiet      # only print problems

Checks per run:
  - exactly one train_against monitor (or zero, iff the run name says control)
  - the train_against monitor matches the target encoded in the run directory name
  - the monitor battery is the same set in every run of the batch (a matrix row that measures a
    different set of held-out monitors is not comparable to the others)
  - every judge's recorded ``reasoning`` is one its model honours as written
    (``monitors.judge_reasoning.resolve_reasoning`` — e.g. gemini-3.x must reason, a gemini-2.5 budget
    is >= 512), and every run of the batch gives each judge the same one. Runs recorded in the legacy
    ``reasoning_effort`` format get the rules of the time (gemini-3.x carries one, gemini-2.5 doesn't)
  - seed matches the ``_s<N>_`` token in the run name
  - hyperparameters that must be identical across a matrix are identical
  - backend queue-pause warnings recorded so far (``metrics.jsonl``), since runs that stall can
    change behaviour right afterwards
  - training log-prob spikes (``loss/train/logprob_mean`` far below its running median): on the
    2026-09-03 MBPP batch every collapse into 100%-truncated single-token loops was preceded, one
    step earlier, by exactly such a spike (-3 to -51 vs a normal -0.3 to -1.9), and the two smaller
    spikes (-2.5, -3.7) were followed by partial collapses. A flagged run is worth inspecting (or
    resuming from the checkpoint before the spike) rather than pooling as-is

Exit status is 1 if anything is flagged, so it can gate a batch in a shell.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import re
import sys
from collections import Counter
from pathlib import Path

from monitordecorrelation.monitors.judge_backend import validate_judge_settings
from monitordecorrelation.monitors.judge_reasoning import resolve_reasoning

# Hyperparameters that must not vary within one matrix (rows differ ONLY in which monitor is the
# training target). Seed and run_name are expected to vary and are checked separately. λ
# (penalty_coef / penalty_schedule) is NOT here: a control carries none, so it is compared across the
# train-against runs only (the "penalty" fact).
_SHARED_KEYS = ("n_steps", "batch_size", "group_size", "eval_every", "eval_size",
                "eval_samples_per_prompt", "max_tokens", "kl_coef", "lora_rank")


def _run_dirs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = [Path(p) for p in sorted(globmod.glob(pat))]
        if not hits and Path(pat).exists():
            hits = [Path(pat)]
        for h in hits:
            if (h / "run_info.json").exists():
                out.append(h)
            else:  # a batch directory: descend one level
                out.extend(sorted(c for c in h.iterdir() if (c / "run_info.json").exists()))
    return out


# The matrix config rows are named cot_weak / cot_strong but build monitors named cot+out_weak /
# cot+out_strong (they read CoT AND the answer). Same alias as eval/coupling.display_name.
_ALIAS = {"cot_weak": "cot+out_weak", "cot_strong": "cot+out_strong"}


def _target_from_name(name: str) -> str | None:
    """The monitor a run NAME claims to train against, e.g. mbpp_Qwen3-8B_cot_weak_s0_sep3 -> cot_weak.

    Mirrors eval/coupling.train_target's dir-name fallback. Returns None when the name encodes no
    target (then we only check run_info's own consistency)."""
    m = re.search(r"_(control|cot_only_weak|cot_only_strong|cot\+out_weak|cot\+out_strong|"
                  r"cot_weak|cot_strong|out_weak|out_strong|probe_ood|probe_iid)_s\d+", name)
    if not m:
        return None
    tok = m.group(1)
    return _ALIAS.get(tok, tok)


def _queue_pauses(run_dir: Path) -> tuple[int, int | None]:
    """(cumulative queue-pause warnings, step of the first one) from metrics.jsonl; (0, None) if the
    run predates the counter or the file is unreadable. Parses defensively — these logs have been
    seen with malformed records."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return 0, None
    n, first = 0, None
    for line in path.open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = row.get("backend/queue_pause_warnings")
        if isinstance(v, int):
            if v > n and first is None:
                first = row.get("step")
            n = max(n, v)
    return n, first


def _logprob_spikes(run_dir: Path, *, floor: float = -2.0, factor: float = 3.0) -> list[tuple[int, float]]:
    """[(step, logprob_mean)] where the per-token training log-prob drops below ``floor`` AND below
    ``factor`` x the running median of the earlier steps. Reads metrics.jsonl defensively."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    seen: list[float] = []
    out: list[tuple[int, float]] = []
    for line in path.open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = row.get("loss/train/logprob_mean")
        if not isinstance(v, (int, float)) or v != v:
            continue
        if seen:
            med = sorted(seen)[len(seen) // 2]
            if v < floor and v < factor * med:
                out.append((row.get("step"), v))
        seen.append(v)
    return out


def check(run_dir: Path) -> tuple[list[str], dict]:
    """(problems, facts) for one run."""
    probs: list[str] = []
    info = json.loads((run_dir / "run_info.json").read_text())
    cfg = info.get("config") or {}
    name = info.get("run_name") or run_dir.name

    ta = [m["name"] for m in info.get("train_against") or []]
    held = [m["name"] for m in info.get("held_out") or []]
    claimed = _target_from_name(run_dir.name)

    if claimed == "control":
        if ta:
            probs.append(f"name says CONTROL but trains against {ta}")
    elif len(ta) != 1:
        probs.append(f"expected exactly 1 train_against monitor, got {len(ta)}: {ta}")
    elif claimed is not None and ta[0] != claimed:
        probs.append(f"trains against {ta[0]!r} but the run name says {claimed!r}")

    seed_m = re.search(r"_s(\d+)", run_dir.name)
    if seed_m and cfg.get("seed") is not None and int(seed_m.group(1)) != cfg["seed"]:
        probs.append(f"seed {cfg['seed']} but the run name says _s{seed_m.group(1)}")

    judges = [m for m in (info.get("train_against") or []) + (info.get("held_out") or [])
              if m.get("model_id")]
    for m in judges:
        mid = m["model_id"]
        if m.get("provider") == "vllm":
            # A local vLLM judge: no OpenRouter reasoning object — its thinking settings are recorded
            # as they were sent. Same offline check that built the monitor.
            try:
                validate_judge_settings("vllm", model_id=mid, monitor=m["name"], max_tokens=m.get("max_tokens"),
                                        base_url=m.get("base_url"), enable_thinking=m.get("enable_thinking"),
                                        thinking_budget=m.get("thinking_budget"))
            except ValueError as e:
                probs.append(str(e))
            continue
        if "reasoning" in m:
            # Current format: the RESOLVED reasoning object the judge sent. It must be a setting the
            # model honours as written — the same resolver that built the monitor, so a run built by
            # older code (or a hand-edited run_info) cannot pass with something it would now refuse.
            try:
                # runs that predate a recorded max_tokens ran with the old fixed 2048 cap
                if resolve_reasoning(mid, m["reasoning"], max_tokens=m.get("max_tokens", 2048),
                                     monitor=m["name"]) != m["reasoning"]:
                    probs.append(f"{m['name']}: recorded reasoning {m['reasoning']!r} is not what "
                                 f"{mid} resolves it to")
            except ValueError as e:
                probs.append(str(e))
            continue
        if "reasoning_effort" not in m:
            continue  # run predates recording it — can't tell, reported once below
        # Legacy record (reasoning_effort / reasoning_max_tokens), from before `reasoning` replaced
        # them: then gemini-3.x had to carry one and gemini-2.5 had to run with reasoning off.
        eff = m.get("reasoning_effort") or m.get("reasoning_max_tokens")
        if mid.startswith("google/gemini-3") and not eff:
            probs.append(f"{m['name']}: {mid} needs a reasoning_effort (it 400s on reasoning off)")
        if mid.startswith("google/gemini-2.5") and eff:
            probs.append(f"{m['name']}: {mid} must run with reasoning OFF, got {eff!r}")
    unrecorded = judges and all("reasoning_effort" not in m and "reasoning" not in m for m in judges)
    # Per-judge reasoning, for the batch-level check that every run's judges reasoned the same way.
    reasoning = tuple(sorted((m["name"], json.dumps(m.get("reasoning", {k: m.get(k) for k in (
        "reasoning_effort", "reasoning_max_tokens")}), sort_keys=True)) for m in judges))

    n_pause, first = _queue_pauses(run_dir)
    spikes = _logprob_spikes(run_dir)
    facts = {"name": name, "reasoning_unrecorded": bool(unrecorded), "target": ta[0] if ta else "control", "n_monitors": len(ta) + len(held),
             "battery": tuple(sorted(ta + held)), "reasoning": reasoning, "seed": cfg.get("seed"),
             "shared": tuple(cfg.get(k) for k in _SHARED_KEYS),
             "penalty": None if not ta else (("penalty_coef", cfg.get("penalty_coef")), (
                 "penalty_schedule", json.dumps(cfg.get("penalty_schedule"), sort_keys=True))),
             "n_pause": n_pause, "pause_step": first, "spikes": spikes}
    return probs, facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="+", help="run dirs, a batch dir, or globs")
    ap.add_argument("--expect-monitors", type=int, default=None,
                    help="fail runs whose monitor count differs (e.g. 8 for the MBPP matrix)")
    ap.add_argument("--quiet", action="store_true", help="print only runs with problems")
    args = ap.parse_args()

    dirs = _run_dirs(args.patterns)
    if not dirs:
        print(f"no runs with a run_info.json matched {args.patterns}")
        return 1

    all_probs, facts = 0, []
    for d in dirs:
        probs, f = check(d)
        if args.expect_monitors is not None and f["n_monitors"] != args.expect_monitors:
            probs.append(f"{f['n_monitors']} monitors, expected {args.expect_monitors}")
        facts.append(f)
        all_probs += len(probs)
        if probs or f["spikes"] or f["n_pause"] or not args.quiet:
            pause = "" if not f["n_pause"] else f"  ⚠️ {f['n_pause']} queue pause(s) from step {f['pause_step']}"
            spike = "" if not f["spikes"] else ("  ⚠️ logprob spike at step " +
                                                ", ".join(f"{s} ({v:.1f})" for s, v in f["spikes"][:3]))
            mark = "✗" if probs else "✓"
            print(f"{mark} {d.name:44s} target={f['target']:16s} seed={f['seed']} "
                  f"monitors={f['n_monitors']}{pause}{spike}")
            for p in probs:
                print(f"    - {p}")

    # batch-level coherence: same battery and same shared hyperparameters everywhere
    for label, key in (("monitor battery", "battery"), ("judge reasoning setting", "reasoning"),
                       ("shared hyperparameters", "shared"), ("monitor penalty λ", "penalty")):
        groups = Counter(f[key] for f in facts if f[key] is not None)  # λ: train-against runs only
        if len(groups) > 1:
            all_probs += 1
            print(f"\n✗ runs do not share one {label} — {len(groups)} variants:")
            for val, n in groups.most_common():
                shown = (dict(zip(_SHARED_KEYS, val)) if key == "shared" else dict(val) if key == "penalty"
                         else list(val))
                print(f"    {n:2d} run(s): {shown}")

    targets = Counter(f["target"] for f in facts)
    print(f"\n{len(dirs)} run(s); targets: " + ", ".join(f"{k}×{v}" for k, v in sorted(targets.items())))
    old = [f["name"] for f in facts if f.get("reasoning_unrecorded")]
    if old:
        print(f"note: {len(old)} run(s) predate recording judge reasoning settings — not checked "
              f"(re-run with current code to get it in run_info.json)")
    stalled = [f["name"] for f in facts if f["n_pause"]]
    if stalled:
        print(f"⚠️  {len(stalled)} run(s) recorded backend queue pauses — inspect before pooling: {stalled}")
    spiked = [f["name"] for f in facts if f["spikes"]]
    if spiked:
        print(f"⚠️  {len(spiked)} run(s) show a training log-prob spike (the collapse precursor on the Sep 3 "
              f"batch) — check unparsed/invalid_rate right after it: {spiked}")
    n_warn = len(stalled) + len(spiked)
    if all_probs:
        print(f"{all_probs} problem(s) flagged")
    else:
        print("OK — no problems" + (f" ({n_warn} warning(s) above to inspect before pooling)" if n_warn else ", nothing flagged"))
    return 1 if all_probs else 0


if __name__ == "__main__":
    sys.exit(main())
