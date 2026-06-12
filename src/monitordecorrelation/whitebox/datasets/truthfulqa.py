"""TruthfulQA adapter (Lin et al. 2022) -> contrastive pairs.

A *diverse deception* component. Each question ships several correct + incorrect free-text answers;
honest = ``best_answer``, deceptive = a sampled incorrect answer to the SAME question. Answer-only
(no CoT) — the probe reads the response activation via the follow-up technique. License: Apache-2.0.
"""

from __future__ import annotations

import random

from monitordecorrelation.whitebox.datasets import ContrastivePair, register

# Namespaced id is required on current `datasets` — the bare `truthful_qa` raises HfUriError.
_REPO = "truthfulqa/truthful_qa"


@register("truthfulqa")
def load(n: int | None = None, seed: int = 0) -> list[ContrastivePair]:
    from datasets import load_dataset  # lazy: heavy import + network on first use

    ds = load_dataset(_REPO, "generation", split="validation")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    if n is not None and n < len(idxs):
        idxs = rng.sample(idxs, n)
    pairs: list[ContrastivePair] = []
    for i in idxs:
        row = ds[i]
        incorrect = list(row.get("incorrect_answers") or [])
        best = row.get("best_answer")
        if not incorrect or not best:
            continue
        pairs.append(
            ContrastivePair(
                prompt=row["question"],
                honest=best,
                deceptive=rng.choice(incorrect),
                meta={"source": "truthfulqa", "category": row.get("category")},
            )
        )
    return pairs
