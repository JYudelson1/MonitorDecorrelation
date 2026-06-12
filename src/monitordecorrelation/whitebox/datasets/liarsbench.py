"""Liar's Bench adapter (Kretschmar et al. 2025) -> contrastive pairs.

A *diverse deception* component. We use ``AlignmentResearch/soft-trigger-verified`` — the ungated
soft-trigger subset OA uses (dodges the gated ``Cadenza-Labs/liars-bench``). It ships an ``honest`` and
a ``dishonest`` split, each a ChatML ``messages`` + ``completion``. The soft trigger lives IN the
passage, so honest/dishonest prompts differ (they don't pair 1:1). We treat them as **unpaired pools**
and zip them for class balance, keeping each response's OWN passage as its prompt (via
``deceptive_prompt``). Answer-only. License: CC BY 4.0.
"""

from __future__ import annotations

import random

from monitordecorrelation.whitebox.datasets import ContrastivePair, register

_REPO = "AlignmentResearch/soft-trigger-verified"


def _rows(split: str) -> list[tuple[str, str]]:
    """-> [(user_passage, completion)] for a split (system prompt dropped — generic)."""
    from datasets import load_dataset  # lazy

    ds = load_dataset(_REPO, split=split)
    out: list[tuple[str, str]] = []
    for row in ds:
        user = next(
            (m["content"] for m in reversed(row["messages"]) if m.get("role") == "user"), None
        )
        if user is not None:
            out.append((user, row["completion"]))
    return out


@register("liarsbench")
def load(n: int | None = None, seed: int = 0) -> list[ContrastivePair]:
    honest = _rows("honest")
    dishonest = _rows("dishonest")
    rng = random.Random(seed)
    rng.shuffle(honest)
    rng.shuffle(dishonest)
    m = min(len(honest), len(dishonest))
    if n is not None:
        m = min(m, n)
    pairs: list[ContrastivePair] = []
    for i in range(m):
        h_prompt, h_resp = honest[i]
        d_prompt, d_resp = dishonest[i]
        pairs.append(
            ContrastivePair(
                prompt=h_prompt,
                honest=h_resp,
                deceptive=d_resp,
                deceptive_prompt=d_prompt,  # read each response with its own passage
                meta={"source": "liarsbench"},
            )
        )
    return pairs
