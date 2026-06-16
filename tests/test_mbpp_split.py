"""Canonical MBPP task_id split — determinism, disjointness, and that the env's eval pool stays
disjoint from its train pool (the anti-leakage guarantee). Fully offline. Run: uv run pytest -q."""

from __future__ import annotations

import pytest

from monitordecorrelation.envs.mbpp_honeypot import MbppHoneypotEnv, MbppItem
from monitordecorrelation.mbpp_split import is_eval_task, partition_task_ids

_TESTS = ["assert f([1, 2]) == 3", "assert f([0]) == 0"]


def test_partition_disjoint_covering_and_stable():
    ids = list(range(500))
    train, ev = partition_task_ids(ids)
    assert set(train).isdisjoint(ev)
    assert set(train) | set(ev) == set(ids)
    # stable across calls (process-stable hash, not Python's salted hash())
    assert partition_task_ids(ids) == (train, ev)
    # roughly EVAL_FRAC (0.2) of ids land in eval
    assert 0.12 < len(ev) / len(ids) < 0.28


def test_is_eval_task_consistent_with_partition():
    ids = list(range(200))
    train, ev = partition_task_ids(ids)
    assert all(is_eval_task(t) for t in ev)
    assert not any(is_eval_task(t) for t in train)


def test_none_task_id_rejected():
    with pytest.raises(ValueError):
        is_eval_task(None)


def test_env_split_path_eval_is_disjoint_from_train():
    """An env built with explicit eval_items (the from_dataset shape) holds out from the eval pool and
    never returns a train task_id (no leakage)."""
    train = [MbppItem(task_id=i, text="t", test_list=_TESTS) for i in range(20)]
    held = [MbppItem(task_id=1000 + i, text="t", test_list=_TESTS) for i in range(8)]
    env = MbppHoneypotEnv(train, eval_items=held)

    out = env.holdout(5, seed=0)
    assert len(out) == 5
    eval_ids = {p.meta["task_id"] for p in out}
    train_ids = {it.task_id for it in env.items}
    assert eval_ids.isdisjoint(train_ids)           # no leakage
    assert eval_ids <= {it.task_id for it in held}  # came from the eval pool
    assert len(env.items) == 20                       # train pool not mutated by holdout


def test_env_fallback_without_split():
    """No eval_items (direct construction) → legacy behavior: carve n out of the train pool."""
    items = [MbppItem(task_id=i, text="t", test_list=_TESTS) for i in range(10)]
    env = MbppHoneypotEnv(items)
    out = env.holdout(3, seed=0)
    assert len(out) == 3 and len(env.items) == 7
