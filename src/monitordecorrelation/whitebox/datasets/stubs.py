"""Registered-but-unimplemented dataset adapters (the roadmap, in one place).

The Atlas's diverse-deception probe mixes several sources; its simple probe uses true/false facts;
its on-domain probe uses code. Each of these is a single self-contained adapter to fill in later —
registering them now means ``--datasets`` validation lists them and the build order is visible. Fill
one in by writing a real ``@register`` loader in its own module (see ``doluschat.py``) and deleting
the corresponding stub line here.

Targets:
- diverse-deception contributors: ``truthfulqa``, ``mask``, ``liarsbench``, ``sandbagging``
- simple-deception probe: ``marks_tegmark`` (true-fact vs false-fact statements)
- on-domain code probe (future coding setting): ``mbpp`` (human code = honest, hardcoded = deceptive)
"""

from __future__ import annotations

from monitordecorrelation.whitebox.datasets import ContrastivePair, register

_TODO = [
    "truthfulqa",
    "mask",
    "liarsbench",
    "sandbagging",
    "marks_tegmark",
    "mbpp",
]


def _make_stub(name: str):
    @register(name)
    def _stub(n: int | None = None, seed: int = 0) -> list[ContrastivePair]:
        raise NotImplementedError(
            f"adapter TODO: {name!r} is registered but not yet implemented "
            "(see whitebox/datasets/stubs.py)"
        )

    return _stub


for _name in _TODO:
    _make_stub(_name)
