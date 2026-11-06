"""scripts/load_balancer.py: weighted least-connections backend choice."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _lb():
    spec = importlib.util.spec_from_file_location("_lb", _REPO / "scripts" / "load_balancer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _route(balancer, n):
    """Open n connections (never closed): which backend each one picks."""
    picks = []
    for _ in range(n):
        p = balancer._pick_order()[0]
        balancer.active[p] += 1
        picks.append(p)
    return picks


def test_default_weights_are_plain_least_connections():
    b = _lb().Balancer("127.0.0.1", [8003, 8004])
    picks = _route(b, 100)
    assert picks.count(8003) == picks.count(8004) == 50


def test_weights_split_in_flight_connections_in_proportion():
    b = _lb().Balancer("127.0.0.1", [8003, 8004], [2.5, 1.0])
    picks = _route(b, 700)
    assert picks.count(8003) == 500 and picks.count(8004) == 200


def test_a_down_backend_is_still_skipped_whatever_its_weight():
    import time

    b = _lb().Balancer("127.0.0.1", [8003, 8004], [100.0, 1.0])
    b.down_until[8003] = time.monotonic() + 60
    assert _route(b, 5) == [8004] * 5


@pytest.mark.parametrize("weights", [[1.0], [1.0, 0.0], [1.0, -2.0]])
def test_bad_weights_are_refused(weights):
    with pytest.raises(ValueError, match="positive weight per backend"):
        _lb().Balancer("127.0.0.1", [8003, 8004], weights)
