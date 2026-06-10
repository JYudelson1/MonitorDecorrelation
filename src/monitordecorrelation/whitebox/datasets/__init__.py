"""Contrastive-QA dataset abstraction + registry for white-box probe training.

Every deception/sycophancy source is adapted into the **same** contrastive shape — a ``prompt`` with
a paired ``honest`` and ``deceptive`` assistant response — so a probe is generic over *which*
datasets it was trained on. The Atlas's "diverse deception probe" is then just a concatenation of
adapters (``load_contrastive(["doluschat", "truthfulqa", ...])``).

Add a source by writing one adapter module that returns ``list[ContrastivePair]`` and decorating its
loader with ``@register("name")`` (mirrors ``monitors/rubrics.py``'s registry style). Importing this
package imports the bundled adapters so they self-register.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

ContrastiveLoader = Callable[..., "list[ContrastivePair]"]
DATASET_LOADERS: dict[str, ContrastiveLoader] = {}


@dataclass
class ContrastivePair:
    """One context with a paired honest + deceptive assistant response.

    ``prompt`` is the full user-visible context (system + user query, concatenated). ``honest`` and
    ``deceptive`` are the two candidate assistant responses. Flattening yields two training rows that
    differ only in the response and its label, which keeps probe data class-balanced by construction.
    """

    prompt: str
    honest: str
    deceptive: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_rows(self) -> list[tuple[str, str, str, int]]:
        """-> [(question, cot, answer, label)]; honest=0, deceptive=1. cot is empty (we probe the
        response activations directly via the follow-up technique)."""
        return [
            (self.prompt, "", self.honest, 0),
            (self.prompt, "", self.deceptive, 1),
        ]


def register(name: str) -> Callable[[ContrastiveLoader], ContrastiveLoader]:
    def deco(fn: ContrastiveLoader) -> ContrastiveLoader:
        if name in DATASET_LOADERS:
            raise ValueError(f"duplicate dataset adapter {name!r}")
        DATASET_LOADERS[name] = fn
        return fn

    return deco


def load_contrastive(
    names: list[str], n_per: int | None = None, seed: int = 0
) -> list[ContrastivePair]:
    """Load + concatenate the named adapters. ``n_per`` caps pairs per source (None = all)."""
    unknown = [n for n in names if n not in DATASET_LOADERS]
    if unknown:
        raise KeyError(f"unknown dataset(s) {unknown}; known: {sorted(DATASET_LOADERS)}")
    pairs: list[ContrastivePair] = []
    for name in names:
        pairs.extend(DATASET_LOADERS[name](n=n_per, seed=seed))
    return pairs


def flatten_pairs(
    pairs: list[ContrastivePair],
) -> tuple[list[str], list[str], list[str], list[int]]:
    """[(question, cot, answer, label), ...] unzipped into four parallel lists for the trainer."""
    questions: list[str] = []
    cots: list[str] = []
    answers: list[str] = []
    labels: list[int] = []
    for pair in pairs:
        for q, c, a, y in pair.to_rows():
            questions.append(q)
            cots.append(c)
            answers.append(a)
            labels.append(y)
    return questions, cots, answers, labels


# Import bundled adapters so they self-register on `import ...whitebox.datasets`.
from monitordecorrelation.whitebox.datasets import doluschat as _doluschat  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import stubs as _stubs  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import sycophancy as _sycophancy  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import sycophancy_cot as _sycophancy_cot  # noqa: E402,F401
