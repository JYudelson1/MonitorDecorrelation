"""Detector metrics + rolling tracker.

We always have each monitor's continuous score and the ground-truth label, so at every step we record
each monitor's accuracy and ROC-AUC vs. ground truth — that time series *is* the degradation curve.
Per-step values are noisy (small batch, and AUROC is undefined when a step has only one class), so we
also keep a cumulative estimate over all rollouts seen so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def roc_auc(scores: list[float], labels: list[bool]) -> float:
    """Rank-based ROC-AUC (Mann–Whitney). NaN if only one class present."""
    pairs = list(zip(scores, labels))
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks (1-indexed), ties share mean rank.
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i + j) / 2 + 1  # mean of 1-indexed ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_ranks_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def accuracy(labels_pred: list[bool], labels_true: list[bool]) -> float:
    if not labels_pred:
        return float("nan")
    return sum(p == t for p, t in zip(labels_pred, labels_true)) / len(labels_pred)


@dataclass
class _Buf:
    scores: list[float] = field(default_factory=list)
    preds: list[bool] = field(default_factory=list)
    truth: list[bool] = field(default_factory=list)


class MonitorTracker:
    """Accumulates (score, predicted-label, ground-truth) per monitor across steps."""

    def __init__(self) -> None:
        self._bufs: dict[str, _Buf] = {}

    def update(
        self, name: str, scores: list[float], preds: list[bool], truth: list[bool]
    ) -> dict[str, float]:
        """Record one step's results; return per-step + cumulative metrics for this monitor."""
        buf = self._bufs.setdefault(name, _Buf())
        buf.scores.extend(scores)
        buf.preds.extend(preds)
        buf.truth.extend(truth)
        return {
            f"monitor/{name}/mean_score": (sum(scores) / len(scores)) if scores else float("nan"),
            f"monitor/{name}/accuracy": accuracy(preds, truth),
            f"monitor/{name}/auroc": roc_auc(scores, truth),
            f"monitor/{name}/cum_accuracy": accuracy(buf.preds, buf.truth),
            f"monitor/{name}/cum_auroc": roc_auc(buf.scores, buf.truth),
        }
