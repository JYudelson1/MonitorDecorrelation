"""Load CoT contrastive pairs from a ``*_cot_pairs.jsonl`` (built by ``experiments/build_cot_pairs.py``).

The SAME jsonl yields a **CoT** probe (``fold_cot=True`` → ``<think>{cot}</think>\\n{answer}``) and a
**no-CoT** probe (``fold_cot=False`` → answer only) — the two training arms of the no-CoT-vs-CoT
transfer 2×2 (docs/COT_VS_NOCOT_PROBE.md). Storing cot + answer separately is what makes this possible.

Not registered (it's parameterized by ``path`` + ``fold_cot``); the 2×2 experiment calls it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from monitordecorrelation.whitebox.datasets import ContrastivePair
from monitordecorrelation.whitebox.model import fold_assistant


def load_cot_jsonl(
    path: str | Path, *, fold_cot: bool, n: int | None = None, seed: int = 0
) -> list[ContrastivePair]:
    """Rows of {prompt, honest_cot, honest_answer, deceptive_cot, deceptive_answer, via?} ->
    ContrastivePairs. ``fold_cot`` decides whether the `<think>` block precedes the answer (the only
    difference between the CoT and no-CoT training sets)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found — build it: uv run python experiments/build_cot_pairs.py --out {path}"
        )
    with p.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if n is not None and n < len(rows):
        rows = rows[:n]

    def resp(cot: str, answer: str) -> str:
        return fold_assistant(cot, answer) if fold_cot else answer

    return [
        ContrastivePair(
            prompt=r["prompt"],
            honest=resp(r.get("honest_cot", ""), r["honest_answer"]),
            deceptive=resp(r.get("deceptive_cot", ""), r["deceptive_answer"]),
            meta={"source": p.stem, "fold_cot": fold_cot, "via": r.get("via")},
        )
        for r in rows
    ]
