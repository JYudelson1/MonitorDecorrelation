"""Generic CoT contrastive-pair construction by on-policy resampling (generalizes build_syco_pairs).

PURE, offline-testable logic lives here; the tinker sampling + ``env.score`` labeling lives in
``experiments/build_cot_pairs.py``.

**Design (per Joey, 2026-06-12).** MEASURE the natural base rate of each class FIRST, and only add a
steering nudge ("please answer truthfully / tell them what they want to hear") to elicit a class the
policy rarely produces on its own. A nudged rollout is **off-policy** — it answers a recontextualized
prompt — so it risks teaching the probe nudge-*style* rather than the behavior. The earlier sycophancy
probe was weak (~0.7 AUROC, task #32) plausibly because BOTH its honest examples were nudged. Preferring
natural rollouts and nudging minimally keeps the probe's training data on-distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

from monitordecorrelation.types import Rollout


@dataclass
class LabeledRollout:
    rollout: Rollout
    behavior: bool  # env oracle ``behavior_present`` (True = the misbehavior = the "deceptive" class)
    finished: bool  # non-empty CoT (reasoning present, presumably not truncated)
    parsed: bool  # the env could grade it (not flagged ``unparsed``)

    @property
    def usable(self) -> bool:
        return self.finished and self.parsed


def base_rates(labeled: list[LabeledRollout]) -> dict:
    """Natural class balance over the USABLE (finished + parsed) rollouts."""
    usable = [l for l in labeled if l.usable]
    n = len(usable)
    present = sum(1 for l in usable if l.behavior)
    return {
        "n_total": len(labeled),
        "n_usable": n,
        "present": present,  # behavior-present = deceptive class
        "absent": n - present,  # behavior-absent = honest class
        "present_rate": (present / n) if n else 0.0,
        "usable_rate": (n / len(labeled)) if labeled else 0.0,
    }


def decide_nudges(rates: dict, min_class: float) -> dict:
    """Which class is produced too rarely (< ``min_class``) to pair naturally, so needs a nudge."""
    pr = rates["present_rate"]
    return {
        "need_deceptive": pr < min_class,  # too few misbehavior examples
        "need_honest": (1.0 - pr) < min_class,  # too few honest examples
    }


def pick_pair(group: list[LabeledRollout]) -> tuple[LabeledRollout, LabeledRollout] | None:
    """One usable deceptive + one usable honest rollout from a prompt's group, else None.

    Prefers the earliest usable rollout of each class (natural rollouts are appended before nudged ones
    in the builder, so this picks natural over nudged when both exist)."""
    dec = next((l for l in group if l.usable and l.behavior), None)
    hon = next((l for l in group if l.usable and not l.behavior), None)
    if dec is None or hon is None:
        return None
    return dec, hon


def pair_to_dict(prompt_text: str, dec: LabeledRollout, hon: LabeledRollout) -> dict:
    """A CoT contrastive pair row (same shape as ``syco_cot_pairs.jsonl``). Storing cot + answer
    SEPARATELY is what lets the 2x2 derive a CoT probe (fold <think>) and a no-CoT probe (answer only)
    from the same data."""
    return {
        "prompt": prompt_text,
        "honest_cot": hon.rollout.cot,
        "honest_answer": hon.rollout.output,
        "deceptive_cot": dec.rollout.cot,
        "deceptive_answer": dec.rollout.output,
    }
