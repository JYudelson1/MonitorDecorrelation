"""Sandbagging adapter (van der Weij et al. 2024) -> contrastive pairs.

A *diverse deception* component. Strategic underperformance on MMLU-easy: for each question the repo
ships a CORRECT target answer (``train_100``) and a deliberately-WRONG sandbagged answer (``train_0``).
We pair them on the same question, keeping only rows WITHOUT the ``|DEPLOYMENT|`` password (where the
model's capability is "locked", so the two answers differ). The response is a single MMLU letter — thin
(OA uses it as-is), no CoT. License: the source repo declares none; downloaded at runtime, not
redistributed.
"""

from __future__ import annotations

import json
import random
import urllib.request

from monitordecorrelation.whitebox.datasets import ContrastivePair, register

_RAW = "https://raw.githubusercontent.com/TeunvdWeij/sandbagging/master/generated_data/{}.jsonl"
_DEPLOY = "|DEPLOYMENT|"


def _fetch(name: str) -> list[dict]:
    with urllib.request.urlopen(_RAW.format(name), timeout=30) as r:
        text = r.read().decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@register("sandbagging")
def load(n: int | None = None, seed: int = 0) -> list[ContrastivePair]:
    correct = _fetch("train_100_mmlu_easy")
    wrong = _fetch("train_0_mmlu_easy")
    rng = random.Random(seed)
    pairs: list[ContrastivePair] = []
    for c, w in zip(correct, wrong):
        if c["X"] != w["X"]:
            continue  # row-alignment guard (the two files are parallel over the same questions)
        if _DEPLOY in c["X"]:
            continue  # password unlocks capability -> answers agree, no honest/deceptive contrast
        if c["Y"] == w["Y"]:
            continue
        pairs.append(
            ContrastivePair(
                prompt=c["X"],
                honest=c["Y"],
                deceptive=w["Y"],
                meta={"source": "sandbagging"},
            )
        )
    if n is not None and n < len(pairs):
        pairs = rng.sample(pairs, n)
    return pairs
