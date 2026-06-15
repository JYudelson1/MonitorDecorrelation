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
    """One honest + one deceptive assistant response, flattened to two class-balanced training rows.

    ``prompt`` is the user-visible context and ``honest``/``deceptive`` are the two responses. For
    **same-prompt** sources (sycophancy, truthfulqa, …) both responses answer the same ``prompt``. For
    **unpaired** sources (liarsbench, OA's diverse pools) honest and deceptive come from different
    contexts; set ``deceptive_prompt`` (and/or ``honest_prompt``) so each response is read with ITS OWN
    context — the probe is a binary classifier over individual response activations, so "pairing" only
    serves class balance, but each activation must be extracted in the correct prompt.
    """

    prompt: str
    honest: str
    deceptive: str
    honest_prompt: str | None = None  # defaults to ``prompt``
    deceptive_prompt: str | None = None  # defaults to ``prompt``
    meta: dict[str, Any] = field(default_factory=dict)

    def to_rows(self) -> list[tuple[str, str, str, int]]:
        """-> [(question, cot, answer, label)]; honest=0, deceptive=1. cot is empty (we probe the
        response activations directly via the follow-up technique)."""
        return [
            (self.honest_prompt or self.prompt, "", self.honest, 0),
            (self.deceptive_prompt or self.prompt, "", self.deceptive, 1),
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


# Named ensemble recipes mirroring the Obfuscation Atlas probes (arXiv 2602.15515). A preset is just a
# list of registered adapters that `load_contrastive` concatenates. `mask` is deferred from
# `diverse_deception` until on-policy generation exists (it ships no completions — see stubs.py).
PROBE_PRESETS: dict[str, list[str]] = {
    "simple_deception": ["marks_tegmark"],  # true-fact vs false-fact statements
    # The Obfuscation Atlas "diverse deception" mix (easy/answer-only versions, no on-policy generation):
    # true/false facts + DolusChat + MASK(propositions) + TruthfulQA + LiarsBench + Sycophancy + Sandbagging.
    "diverse_deception": [
        "marks_tegmark",  # true/false facts (Marks & Tegmark)
        "doluschat",
        "mask_statements",  # MASK propositions -> blatant true/false (the easy MASK probe)
        "truthfulqa",
        "liarsbench",
        "sycophancy",
        "sandbagging",
    ],
    "mbpp": ["mbpp"],  # on-domain code probe (human code = honest, hardcode = deceptive)
}


def resolve_datasets(datasets: list[str] | None, preset: str | None) -> list[str]:
    """Resolve a `--datasets` list and/or a `--preset` name into a concrete adapter list (deduped,
    order-preserving). Exactly one of the two should be given; both are merged if both are."""
    out: list[str] = []
    if preset is not None:
        if preset not in PROBE_PRESETS:
            raise KeyError(f"unknown preset {preset!r}; known: {sorted(PROBE_PRESETS)}")
        out.extend(PROBE_PRESETS[preset])
    if datasets:
        out.extend(datasets)
    if not out:
        raise ValueError("no datasets: pass --datasets and/or --preset")
    seen: set[str] = set()
    return [d for d in out if not (d in seen or seen.add(d))]


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
# (stubs first, so a real adapter for a name overrides nothing — the name is no longer in _TODO.)
from monitordecorrelation.whitebox.datasets import stubs as _stubs  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import doluschat as _doluschat  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import sycophancy as _sycophancy  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import sycophancy_cot as _sycophancy_cot  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import truthfulqa as _truthfulqa  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import mbpp as _mbpp  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import sandbagging as _sandbagging  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import marks_tegmark as _marks_tegmark  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import liarsbench as _liarsbench  # noqa: E402,F401
from monitordecorrelation.whitebox.datasets import mask_statements as _mask_statements  # noqa: E402,F401
