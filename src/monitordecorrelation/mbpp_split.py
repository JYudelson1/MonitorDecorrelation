"""Canonical train/eval split of MBPP ``task_id``s — the SINGLE source of truth shared by the RL env
(``envs/mbpp_honeypot``) and the iid MBPP probe (``whitebox/datasets/mbpp``), so the probe is never
trained on a ``task_id`` the policy is *evaluated* on (the leakage Joey flagged).

Keyed ONLY on ``task_id`` via a stable hash + a fixed canonical ``SPLIT_SEED`` — deliberately NOT a
run's ``cfg.seed`` — so the partition is identical across runs, seeds, processes, and dataset orderings.
The run seed varies sampling/shuffling; the data partition stays put so eval curves stay comparable and
the probe's train set stays disjoint from every run's eval set.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

SPLIT_SEED = 1234   # canonical; do NOT tie to cfg.seed (probe + all runs must agree on the partition)
EVAL_FRAC = 0.2     # fraction of task_ids reserved for the held-out RL eval (≈, by hashing)
_BUCKETS = 1_000_000


def _bucket(task_id: object, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(), 16) % _BUCKETS


def is_eval_task(task_id: object, *, seed: int = SPLIT_SEED, eval_frac: float = EVAL_FRAC) -> bool:
    """True iff this ``task_id`` is in the held-out RL-eval split (deterministic, process-stable).

    ``task_id`` must not be None — an un-IDed item can't be placed in a stable split (callers should
    drop those)."""
    if task_id is None:
        raise ValueError("task_id is None — cannot assign to a stable split (drop un-IDed items)")
    return _bucket(task_id, seed) < eval_frac * _BUCKETS


def partition_task_ids(
    task_ids: Iterable[object], *, seed: int = SPLIT_SEED, eval_frac: float = EVAL_FRAC
) -> tuple[list, list]:
    """``(train_ids, eval_ids)`` — disjoint, covering, original order preserved."""
    train, ev = [], []
    for t in task_ids:
        (ev if is_eval_task(t, seed=seed, eval_frac=eval_frac) else train).append(t)
    return train, ev
