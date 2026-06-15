"""MASK *propositions* as a blatant true/false probe set (Geometry-of-Truth style, on-domain to MASK).

Instead of the subtle on-policy rollouts, build contrastive pairs straight from MASK's ground truth:
each binary item has a ``proposition`` with a ``<pivotal>X/Y</pivotal>`` marker and a
``formatted_ground_truth`` ("A"=first option true, "B"=second). Resolve the pivotal both ways →
honest = the TRUE statement, deceptive = the FALSE statement. Blatant, separable, and plentiful (~800
binary items across configs) — the hypothesis (Joey) is that a probe trained on these generalizes to the
subtle realistic lies in the rollouts. ``cais/MASK`` is gated (needs an HF token). Answer/statement-only.
"""

from __future__ import annotations

import random
import re

from monitordecorrelation.whitebox.datasets import ContrastivePair, register

_PIVOTAL = re.compile(r"<pivotal>(.*?)</pivotal>", re.S)
# Configs that carry binary propositions (provided_facts + the known/continuation/disinfo archetypes).
_DEFAULT_CONFIGS = ("provided_facts", "known_facts", "continuations", "disinformation")


def _statements(proposition: str, fgt: str) -> tuple[str, str] | None:
    """(true_statement, false_statement) by resolving the <pivotal>A/B</pivotal> marker, or None."""
    m = _PIVOTAL.search(proposition)
    if not m or fgt not in ("A", "B"):
        return None
    opts = [o.strip() for o in m.group(1).split("/")]
    if len(opts) != 2:
        return None
    true_i = 0 if fgt == "A" else 1
    true_stmt = _PIVOTAL.sub(opts[true_i], proposition).strip()
    false_stmt = _PIVOTAL.sub(opts[1 - true_i], proposition).strip()
    return true_stmt, false_stmt


@register("mask_statements")
def load(n: int | None = None, seed: int = 0, configs: tuple[str, ...] = _DEFAULT_CONFIGS) -> list[ContrastivePair]:
    from datasets import load_dataset  # lazy; gated

    pairs: list[ContrastivePair] = []
    for cfg in configs:
        ds = load_dataset("cais/MASK", cfg, split="test")
        for row in ds:
            if row.get("type") != "binary":
                continue
            st = _statements(row.get("proposition", ""), row.get("formatted_ground_truth"))
            if st is None:
                continue
            true_stmt, false_stmt = st
            pairs.append(
                ContrastivePair(
                    prompt="State a fact.",
                    honest=true_stmt,
                    deceptive=false_stmt,
                    meta={"source": "mask_statements", "config": cfg},
                )
            )
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if n is not None and n < len(pairs):
        pairs = pairs[:n]
    return pairs
