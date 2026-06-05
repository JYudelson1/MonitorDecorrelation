"""The generic Monitor seam.

Every detector family (CoT judge, output judge, white-box probe, ...) implements ``Monitor``.
Detector *strength* is configuration (which ``model_id`` for a judge, which checkpoint for a probe),
not a different interface. See docs/MONITORS.md.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from monitordecorrelation.types import MonitorResult, Rollout


@runtime_checkable
class Monitor(Protocol):
    """Scores a rollout for likely misbehavior.

    Invariants:
    - A monitor must never read ``EnvResult.behavior_present`` — that oracle grades monitors.
    - Monitors are frozen by default (no weight updates during policy training).
    """

    name: str

    def score(self, rollout: Rollout) -> MonitorResult: ...
