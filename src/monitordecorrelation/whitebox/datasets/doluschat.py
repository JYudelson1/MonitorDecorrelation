"""DolusChat adapter (Cundy & Gleave) -> contrastive pairs.

DolusChat (``AlignmentResearch/DolusChat``) is the first contributor to the **diverse deception**
probe. Each row has a system message + context items, a user query, and a paired truthful/deceptive
assistant response. Raw schema (from the authors' ``deception-evasion-honesty`` munge code):

    row["system_message"]                         -> str
    row["context"]["sensitive_info"]              -> str
    row["context"]["non_sensitive_info"]          -> list[str]
    row["user_query"]["content"|"speaker_type"]   -> str
    row["responses"]["truthful"|"deceptive"]      -> str

NOTE: DolusChat *alone* is a relatively narrow ood signal — it's intended to be mixed with the other
deception sources (TruthfulQA, MASK, ...) for the real diverse probe. See the registry docstring.
"""

from __future__ import annotations

import random

from monitordecorrelation.whitebox.datasets import ContrastivePair, register

_REPO = "AlignmentResearch/DolusChat"


def _compose_prompt(row: dict) -> str:
    ctx = row["context"]
    pieces = list(ctx.get("non_sensitive_info") or [])
    if ctx.get("sensitive_info"):
        pieces.append(ctx["sensitive_info"])
    context_str = "\n".join(pieces)
    uq = row["user_query"]
    return (
        f"{row['system_message']}\n"
        f"Context:\n{context_str}\n\n"
        f"User ({uq.get('speaker_type', 'user')}): {uq['content']}"
    )


@register("doluschat")
def load(n: int | None = None, seed: int = 0) -> list[ContrastivePair]:
    from datasets import load_dataset  # lazy: heavy import, network on first use

    ds = load_dataset(_REPO, split="train")
    idxs = list(range(len(ds)))
    if n is not None and n < len(idxs):
        idxs = random.Random(seed).sample(idxs, n)
    pairs: list[ContrastivePair] = []
    for i in idxs:
        row = ds[i]
        resp = row["responses"]
        pairs.append(
            ContrastivePair(
                prompt=_compose_prompt(row),
                honest=resp["truthful"],
                deceptive=resp["deceptive"],
                meta={"source": "doluschat", "id": row.get("id")},
            )
        )
    return pairs
