"""condition_summary: per-(target × penalty) run summaries + bucketed distributions."""

import json

from monitordecorrelation.eval.condition_summary import (
    DPRIME_BUCKETS, _dist, condition_table, run_summary,
)


def _make_run(tmp, name, *, target, penalty, dprime, mu_hack, mu_clean, behavior):
    d = tmp / name
    d.mkdir()
    rows = []
    for t in range(len(dprime)):
        rows.append({"step": t * 2, "behavior_rate": behavior[t],
                     f"monitor/{target}/dprime_margin": dprime[t],
                     f"monitor/{target}/mean_score_reward_hacking": mu_hack[t],
                     f"monitor/{target}/mean_score_not_reward_hacking": mu_clean[t]})
    (d / "eval_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "run_info.json").write_text(json.dumps(
        {"train_against": [{"name": target}], "config": {"penalty_coef": penalty}}))
    return str(d)


def test_run_summary_signs_and_endpoints(tmp_path):
    run = _make_run(tmp_path, "r0", target="A", penalty=0.5,
                    dprime=[3.0, 2.0, 1.0],       # first 3 → last 1
                    mu_hack=[0.9, 0.5, 0.1],      # evaded: last − first = −0.8
                    mu_clean=[0.1, 0.2, 0.3],     # honest-drift: +0.2
                    behavior=[0.1, 0.5, 0.9])
    s = run_summary(run, ["A"])
    assert s["target"] == "A" and s["penalty"] == "0.5"
    assert abs(s["dprime_drop"] - 2.0) < 1e-9 and abs(s["dprime_change"] + 2.0) < 1e-9
    assert abs(s["dmu_hack"] + 0.8) < 1e-9 and abs(s["dmu_clean"] - 0.2) < 1e-9
    assert s["max_hack_rate"] == 0.9 and s["end_hack_rate"] == 0.9 and abs(s["end_mu_hack"] - 0.1) < 1e-9


def test_control_has_only_run_level_metrics(tmp_path):
    d = tmp_path / "ctl"
    d.mkdir()
    (d / "eval_metrics.jsonl").write_text(json.dumps({"step": 0, "behavior_rate": 0.4}) + "\n")
    (d / "run_info.json").write_text(json.dumps({"train_against": [], "config": {"penalty_coef": 1.0}}))
    s = run_summary(str(d), ["A"])
    assert s["target"] is None and s["max_hack_rate"] == 0.4
    assert s["dprime_drop"] != s["dprime_drop"] and s["dmu_hack"] != s["dmu_hack"]  # NaN (no target monitor)


def test_condition_table_groups_and_distributions(tmp_path):
    # two seeds, same (target, penalty) → one group, n=2; opposite d′ moves → bimodal dist
    r0 = _make_run(tmp_path, "s0", target="A", penalty=0.5, dprime=[3.0, 1.0],  # change −2 (big drop)
                   mu_hack=[0.9, 0.05], mu_clean=[0.1, 0.2], behavior=[0.0, 0.95])
    r1 = _make_run(tmp_path, "s1", target="A", penalty=0.5, dprime=[1.0, 3.0],  # change +2 (sharper)
                   mu_hack=[0.5, 0.95], mu_clean=[0.1, 0.1], behavior=[0.0, 0.0])
    table = condition_table([r0, r1], ["A"])
    assert len(table) == 1 and table[0]["n"] == 2
    # d′ distribution: one run ≤−1.5, one ≥+1.5 (the bimodality an average would hide)
    assert table[0]["dprime_dist"][0] == 0.5 and table[0]["dprime_dist"][-1] == 0.5
    # final hack-rate: one ≈0.95 (0.9–1.0), one ≈0 → split across the extreme buckets
    assert table[0]["hack_dist"][0] == 0.5 and table[0]["hack_dist"][-1] == 0.5


def test_control_regimes_merge_across_penalty(tmp_path):
    # two control runs at DIFFERENT penalties collapse into one ("control", "any") row
    def _ctl(name, pen, beh):
        d = tmp_path / name
        d.mkdir()
        (d / "eval_metrics.jsonl").write_text(json.dumps({"step": 0, "behavior_rate": beh}) + "\n")
        (d / "run_info.json").write_text(json.dumps({"train_against": [], "config": {"penalty_coef": pen}}))
        return str(d)
    table = condition_table([_ctl("c1", 1.0, 0.9), _ctl("c2", 0.5, 0.8)], ["A"])
    ctl = [r for r in table if r["target"] == "control"]
    assert len(ctl) == 1 and ctl[0]["penalty"] == "any" and ctl[0]["n"] == 2
    assert ctl[0]["dprime_dist_n"] == 0  # no train-against monitor → d′ distribution empty (N/A in plot)


def test_dist_buckets_and_nan_handling():
    fracs, n = _dist([-2.0, 0.0, 2.0, float("nan")], DPRIME_BUCKETS)
    assert n == 3 and fracs[0] == fracs[2] == fracs[4] == 1 / 3  # big-drop / flat / sharper
    assert _dist([float("nan")], DPRIME_BUCKETS)[1] == 0
