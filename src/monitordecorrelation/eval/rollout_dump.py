"""The SLIM per-rollout eval dump — the committed, analysis-facing subset of ``eval_rollouts.jsonl``.

The full ``eval_rollouts.jsonl`` carries every eval rollout's generated text (CoT + answer + prompt) and
is gitignored (tens of MB per run). The split-half / cross-fit coupling analyses only need each
rollout's *labels* and every monitor's *score*, so the training loop also writes
``eval_rollouts_slim.jsonl`` with exactly ``SLIM_FIELDS`` per record (~2–3 MB per run), which IS
committed. ``scripts/slim_eval_rollouts.py`` produces the same file from an existing full dump.

Keep ``SLIM_FIELDS`` stable: downstream readers key on these names.
"""

from __future__ import annotations

# step + task identity + the 3-way hacking labels + parse flag + every monitor's {score, label}
SLIM_FIELDS = ("step", "task_id", "behavior_present", "loose_rh", "hardcoding", "unparsed", "monitors")


def slim_record(full: dict) -> dict:
    """Project a full eval-rollout record onto ``SLIM_FIELDS`` (missing keys → None)."""
    return {k: full.get(k) for k in SLIM_FIELDS}
