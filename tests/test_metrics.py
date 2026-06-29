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


def test_class_score_pairs_both_schemes():
    """The behavior-named keys and the legacy syco/honest keys both parse to (present, absent), and a
    NaN class mean drops the monitor (gap undefined)."""
    from monitordecorrelation.eval.metric_keys import (
        absent_score_key,
        class_score_pairs,
        present_score_key,
    )

    assert present_score_key("cot_weak", "reward_hacking") == "monitor/cot_weak/mean_score_reward_hacking"
    assert absent_score_key("cot_weak", "reward_hacking") == "monitor/cot_weak/mean_score_not_reward_hacking"

    # new behavior-named keys (+ the bare overall mean_score, which must NOT be picked up as a class mean)
    new = {present_score_key("probe_ood", "reward_hacking"): 0.7,
           absent_score_key("probe_ood", "reward_hacking"): 0.3,
           "monitor/probe_ood/mean_score": 0.5, "monitor/probe_ood/auroc": 0.9}
    assert class_score_pairs(new) == {"probe_ood": (0.7, 0.3)}

    # legacy syco/honest keys still parse (backward compatibility for old runs)
    legacy = {"monitor/cot_weak/mean_score_syco": 0.8, "monitor/cot_weak/mean_score_honest": 0.2}
    assert class_score_pairs(legacy) == {"cot_weak": (0.8, 0.2)}

    # a NaN class mean → monitor dropped (gap is undefined)
    assert class_score_pairs({"monitor/x/mean_score_sycophancy": float("nan"),
                              "monitor/x/mean_score_not_sycophancy": 0.1}) == {}


if __name__ == "__main__":
    test_roc_auc()
    test_tracker_cumulative()
    test_accuracy()
    test_class_score_pairs_both_schemes()
    print("ALL PASS")
