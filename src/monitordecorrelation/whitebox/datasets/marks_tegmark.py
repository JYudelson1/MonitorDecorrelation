"""Geometry-of-Truth adapter (Marks & Tegmark 2023) -> contrastive pairs — the *Simple Deception* probe.

True/false factual statements. honest = a true statement, deceptive = a false statement. For ``cities``
and ``counterfact`` each subject has a matched true + false row, so we pair same-subject (a real
honest-vs-deceptive contrast on one fact); for other CSVs we fall back to zipping true with false
statements under a generic prompt. Statement-only (no CoT).

NOTE: the source repo (``saprmarks/geometry-of-truth``) declares NO license — fetched at runtime from
raw GitHub, never redistributed/committed.
"""

from __future__ import annotations

import csv
import io
import random
import urllib.request

from monitordecorrelation.whitebox.datasets import ContrastivePair, register

_RAW = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/{}.csv"


def _fetch_csv(name: str) -> list[dict]:
    with urllib.request.urlopen(_RAW.format(name), timeout=30) as r:
        text = r.read().decode()
    return list(csv.DictReader(io.StringIO(text)))  # real CSV parse — statements contain commas


@register("marks_tegmark")
def load(n: int | None = None, seed: int = 0, dataset: str = "cities") -> list[ContrastivePair]:
    rows = _fetch_csv(dataset)
    rng = random.Random(seed)
    pairs: list[ContrastivePair] = []
    subject_col = next((c for c in ("city", "subject") if rows and c in rows[0]), None)
    if subject_col:
        # Same-subject pairing: each subject has one true (label 1) + one false (label 0) statement.
        by_subj: dict[str, dict[str, str]] = {}
        for row in rows:
            by_subj.setdefault(row[subject_col], {})[row["label"]] = row["statement"]
        for subj, d in by_subj.items():
            if "1" in d and "0" in d:
                pairs.append(
                    ContrastivePair(
                        prompt=f"Tell me a fact about {subj}.",
                        honest=d["1"],
                        deceptive=d["0"],
                        meta={"source": "marks_tegmark", "dataset": dataset, "subject": subj},
                    )
                )
    else:
        # Fallback: zip true/false statement pools under a shared generic prompt.
        trues = [r["statement"] for r in rows if r["label"] == "1"]
        falses = [r["statement"] for r in rows if r["label"] == "0"]
        rng.shuffle(trues)
        rng.shuffle(falses)
        for t, f in zip(trues, falses):
            pairs.append(
                ContrastivePair(
                    prompt="Tell me a fact about the world.",
                    honest=t,
                    deceptive=f,
                    meta={"source": "marks_tegmark", "dataset": dataset},
                )
            )
    if n is not None and n < len(pairs):
        pairs = rng.sample(pairs, n)
    return pairs
