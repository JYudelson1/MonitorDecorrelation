"""Offline checks for roc_auc + MonitorTracker. Run: uv run python tests/test_metrics.py"""

from __future__ import annotations

from math import isnan

from monitordecorrelation.eval.metrics import MonitorTracker, accuracy, roc_auc


def test_roc_auc():
    # Perfectly separable -> 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    # Perfectly inverted -> 0.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    # Ties across classes -> 0.5
    assert abs(roc_auc([0.5, 0.5], [True, False]) - 0.5) < 1e-9
    # Single class -> nan
    assert isnan(roc_auc([0.1, 0.2], [True, True]))
    print("roc_auc OK")


def test_tracker_cumulative():
    t = MonitorTracker()
    # Step 0: only positives -> per-step auroc nan, cumulative also nan (one class so far)
    m0 = t.update("cot", [0.9, 0.8], [True, True], [True, True])
    assert isnan(m0["monitor/cot/auroc"]) and isnan(m0["monitor/cot/cum_auroc"])
    assert m0["monitor/cot/accuracy"] == 1.0
    # Step 1: adds a negative scored low -> cumulative now well-defined and separable
    m1 = t.update("cot", [0.1], [False], [False])
    assert m1["monitor/cot/cum_auroc"] == 1.0  # 0.1<0.8,0.9 across the two classes
    assert m1["monitor/cot/cum_accuracy"] == 1.0
    print("tracker cumulative OK")


def test_accuracy():
    assert accuracy([True, False, True], [True, True, True]) == 2 / 3
    print("accuracy OK")


if __name__ == "__main__":
    test_roc_auc()
    test_tracker_cumulative()
    test_accuracy()
    print("ALL PASS")
