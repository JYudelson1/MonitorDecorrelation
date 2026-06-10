"""``ProbeMonitor``: a white-box ``Monitor`` backed by a ``WhiteBoxModel`` + ``LinearProbe``.

Plugs into the same seam as the CoT judges (``name``/``behavior``/``threshold`` so ``rl/train.py`` and
the plots decode it unchanged). v1 is **post-hoc only**: per-rollout forward passes are slow, so the
RL loop never calls this — we score saved rollouts after the fact. ``score_batch`` is the path used in
practice (one batched forward over many rollouts).
"""

from __future__ import annotations

from typing import Sequence

from monitordecorrelation.types import MonitorResult, Rollout
from monitordecorrelation.whitebox.model import DEFAULT_FOLLOWUP, WhiteBoxModel
from monitordecorrelation.whitebox.probe import LinearProbe


class ProbeMonitor:
    def __init__(
        self,
        name: str,
        model: WhiteBoxModel,
        probe: LinearProbe,
        *,
        threshold: float | None = None,
        behavior: str = "deception",
        batch_size: int = 8,
        follow_up: tuple[str, str] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.probe = probe
        self.behavior = behavior
        self.threshold = probe.threshold if threshold is None else threshold
        self.batch_size = batch_size
        # Use the exact follow-up the probe was trained with (from its meta) unless overridden.
        fu = follow_up if follow_up is not None else probe.meta.get("follow_up")
        self.follow_up = tuple(fu) if fu else DEFAULT_FOLLOWUP

    def score(self, rollout: Rollout) -> MonitorResult:
        return self.score_batch([rollout])[0]

    def score_batch(self, rollouts: Sequence[Rollout]) -> list[MonitorResult]:
        acts = self.model.extract_activations(
            list(rollouts), follow_up=self.follow_up, batch_size=self.batch_size
        )
        scores = self.probe.score(acts)
        return [
            MonitorResult(
                score=float(s),
                label=bool(s >= self.threshold),
                meta={"probe": self.name},
            )
            for s in scores
        ]
