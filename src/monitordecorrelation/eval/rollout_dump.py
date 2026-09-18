"""The SLIM per-rollout eval dump — the committed, analysis-facing subset of ``eval_rollouts.jsonl``.

The full ``eval_rollouts.jsonl`` carries every eval rollout's generated text (CoT + answer + prompt) and
is gitignored (tens of MB per run). The split-half / cross-fit coupling analyses only need each
rollout's *labels* and every monitor's *score*, so the training loop also writes
``eval_rollouts_slim.jsonl`` with exactly ``SLIM_FIELDS`` per record (~2–3 MB per run), which IS
committed. ``scripts/slim_eval_rollouts.py`` produces the same file from an existing full dump.

Keep ``SLIM_FIELDS`` stable: downstream readers key on these names.

Per monitor, the FULL dumps store ``monitor_record(result)``: ``{score, label}`` plus, for an LLM
judge, ``call`` — the exact request it was sent and the exact response it gave (prompt, API
parameters, content and chain of thought; see ``monitors.cot_monitor.JudgeCall``). The slim dump keeps
only ``{score, label}`` per monitor (``slim_monitors``): a call record is the judge prompt plus its
answer, i.e. more bytes than the rollout text the slim file exists to drop.
"""

from __future__ import annotations

from typing import Any

# step + task identity + the 3-way hacking labels + parse flag + invalid reason + every monitor's
# {score, label}. ``invalid_reason`` (null | "truncated" | "unparsed"; absent → None in runs that predate
# it) marks a rollout no monitor was shown — its ``monitors`` is {}.
SLIM_FIELDS = ("step", "task_id", "behavior_present", "loose_rh", "hardcoding", "unparsed", "invalid_reason",
               "monitors")

# The per-monitor keys the slim dump keeps.
SLIM_MONITOR_FIELDS = ("score", "label")


def monitor_record(result: Any) -> dict:
    """What the full dumps persist for one monitor's verdict on one rollout.

    Always ``score`` / ``label``. Then, when the monitor's ``MonitorResult.meta`` carries them:
    ``call`` (the LLM judge's successful API call — request + response, see
    ``cot_monitor.JudgeCall``), ``parse_error`` (the judge answered but not in the instructed
    format, so the score is a no-signal 0) and ``error`` (the monitor never answered — NaN score).
    Probes carry none of these, so their record stays ``{score, label}``.
    """
    meta = getattr(result, "meta", None) or {}
    rec: dict = {"score": result.score, "label": result.label}
    for k in ("call", "parse_error", "error"):
        if meta.get(k) is not None:
            rec[k] = meta[k]
    return rec


def slim_monitors(monitors: Any) -> Any:
    """Per-monitor ``{score, label}`` only — drops the ``call`` record (and anything else)."""
    if not isinstance(monitors, dict):
        return monitors
    return {name: ({k: m.get(k) for k in SLIM_MONITOR_FIELDS} if isinstance(m, dict) else m)
            for name, m in monitors.items()}


def slim_record(full: dict) -> dict:
    """Project a full eval-rollout record onto ``SLIM_FIELDS`` (missing keys → None); monitors are
    reduced to ``{score, label}``."""
    out = {k: full.get(k) for k in SLIM_FIELDS}
    out["monitors"] = slim_monitors(out["monitors"])
    return out
