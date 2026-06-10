"""Real-CoT sycophancy adapter -> contrastive pairs (the un-thin on-domain probe data).

Loads pairs built by ``experiments/build_syco_pairs.py`` (resampled real policy rollouts, honest vs
sycophantic by ground-truth and/or LLM judge). Unlike the letter-only ``sycophancy`` adapter, the
honest/deceptive responses here are full CoT+answer generations, so the probe sees real reasoning.

The CoT is folded into the response via ``fold_assistant`` — the *same* helper the deploy path uses —
so train-time and deploy-time assistant text are byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from monitordecorrelation.whitebox.datasets import ContrastivePair, register
from monitordecorrelation.whitebox.model import fold_assistant

_DEFAULT_PATH = "data/probes/syco_cot_pairs.jsonl"


@register("sycophancy_cot")
def load(n: int | None = None, seed: int = 0, path: str = _DEFAULT_PATH) -> list[ContrastivePair]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found — build it first: "
            "uv run python experiments/build_syco_pairs.py --out " + path
        )
    rows = [json.loads(line) for line in p.open() if line.strip()]
    if n is not None and n < len(rows):
        rows = rows[:n]
    return [
        ContrastivePair(
            prompt=r["prompt"],
            honest=fold_assistant(r.get("honest_cot", ""), r["honest_answer"]),
            deceptive=fold_assistant(r.get("deceptive_cot", ""), r["deceptive_answer"]),
            meta={"source": "sycophancy_cot"},
        )
        for r in rows
    ]
