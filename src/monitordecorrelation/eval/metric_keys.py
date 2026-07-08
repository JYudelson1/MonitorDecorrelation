"""Metric-key naming for the per-monitor class-split scores.

The headline obfuscation tracker is each monitor's mean score on the behavior-PRESENT class vs the
behavior-ABSENT class. We name those keys after the env's *behavior* (``env.behavior_name``) so charts
and W&B read naturally per experiment — ``mean_score_reward_hacking`` for MBPP-Honeypot,
``mean_score_sycophancy`` for the sycophancy env, etc. — rather than the historical ``_syco``/``_honest``.

Readers (the local plots + the cross-run degradation matrix) discover the pair *structurally* via
``class_score_pairs`` so they stay behavior-agnostic AND keep parsing old runs whose keys are still the
legacy ``mean_score_syco`` / ``mean_score_honest``. The writer side (``present_score_key`` /
``absent_score_key``) always emits the new behavior-named form.
"""

from __future__ import annotations

import re


def present_score_key(monitor: str, behavior: str) -> str:
    """Key for a monitor's mean score on the behavior-PRESENT class, e.g. ``mean_score_reward_hacking``."""
    return f"monitor/{monitor}/mean_score_{behavior}"


def absent_score_key(monitor: str, behavior: str) -> str:
    """Key for a monitor's mean score on the behavior-ABSENT class, e.g. ``mean_score_not_reward_hacking``."""
    return f"monitor/{monitor}/mean_score_not_{behavior}"


# ABSENT marker is either the new ``not_<behavior>`` or the legacy ``honest``. PRESENT is any other
# ``mean_score_<x>`` (excluding the bare ``mean_score`` overall mean, which has no ``_<x>`` suffix).
# ``name`` is ``[^/]+`` so a nested namespace (e.g. ``monitor/<name>/loose/mean_score_present``, the
# separate loose-oracle track) does NOT match — only the top-level strict class-split is parsed here.
_ABSENT_RE = re.compile(r"^monitor/(?P<name>[^/]+)/mean_score_(?:not_(?P<beh>.+)|honest)$")
_PRESENT_RE = re.compile(r"^monitor/(?P<name>[^/]+)/mean_score_(?P<beh>(?!not_)(?!honest$).+)$")


def present_score(row: dict, monitor: str) -> float:
    """The behavior-PRESENT class mean (μ_hack — suspicion on actual hacks) for one monitor, or NaN.
    Defined whenever ≥1 present example exists (unlike the gap/AUROC, which also need the absent class).
    Env-agnostic (matches ``mean_score_<behavior>``, incl. legacy ``mean_score_syco``)."""
    for k, v in row.items():
        m = _PRESENT_RE.match(k)
        if m and m.group("name") == monitor and v is not None and v == v:
            return float(v)
    return float("nan")


def absent_score(row: dict, monitor: str) -> float:
    """The behavior-ABSENT class mean (μ_clean — suspicion on the non-hack/'clean' class) for one
    monitor, or NaN. Rising μ_clean (honest rollouts looking suspicious) is what collapses d′ *without*
    the monitor being evaded. Matches ``mean_score_not_<behavior>`` / legacy ``mean_score_honest``."""
    for k, v in row.items():
        m = _ABSENT_RE.match(k)
        if m and m.group("name") == monitor and v is not None and v == v:
            return float(v)
    return float("nan")


def class_score_pairs(row: dict) -> dict[str, tuple[float, float]]:
    """``{monitor: (present_mean, absent_mean)}`` parsed from one metrics row, tolerant of both the new
    ``<behavior>``/``not_<behavior>`` keys and the legacy ``syco``/``honest`` keys. Only monitors that
    have BOTH means present and non-NaN are returned (the gap is undefined otherwise)."""
    present: dict[str, float] = {}
    absent: dict[str, float] = {}
    for k, v in row.items():
        if not k.startswith("monitor/") or v != v:  # skip non-monitor keys and NaN
            continue
        if (m := _ABSENT_RE.match(k)):
            absent[m["name"]] = v
        elif (m := _PRESENT_RE.match(k)):
            present[m["name"]] = v
    return {n: (present[n], absent[n]) for n in present if n in absent}
