"""On-domain (iid) sycophancy adapter -> contrastive pairs.

Reuses the Anthropic sycophancy items the RL env trains on (``envs/sycophancy.py``). For each item the
sycophantic (user-matching) choice is the *deceptive* response and a non-matching choice is the
*honest* one, both rendered as the short ``"Answer: (X)"`` form the policy emits. This is our analogue
of the Atlas's on-domain MBPP probe — same distribution as the deployment setting.
"""

from __future__ import annotations

import random
import string

from monitordecorrelation.envs.sycophancy import load_sycophancy_items
from monitordecorrelation.whitebox.datasets import ContrastivePair, register

_LETTERS = list(string.ascii_uppercase)


def _other_letter(syco: str, rng: random.Random) -> str:
    """Pick a plausible non-sycophantic choice letter (A-D), different from the matching one."""
    choices = [c for c in _LETTERS[:4] if c != syco]
    return rng.choice(choices)


@register("sycophancy")
def load(
    n: int | None = None, seed: int = 0, subset: str = "nlp"
) -> list[ContrastivePair]:
    items = load_sycophancy_items(subset=subset, n=n, seed=seed)
    rng = random.Random(seed)
    pairs: list[ContrastivePair] = []
    for item in items:
        syco = item.sycophantic_letter
        honest_letter = _other_letter(syco, rng)
        pairs.append(
            ContrastivePair(
                prompt=item.question,
                honest=f"Answer: ({honest_letter})",
                deceptive=f"Answer: ({syco})",
                meta={"source": "sycophancy", "subset": subset},
            )
        )
    return pairs
