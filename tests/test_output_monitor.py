"""Output-only monitor: the judge prompt omits the CoT. Offline (no API call — _build_prompt is pure).

Run: uv run python tests/test_output_monitor.py
"""

from __future__ import annotations

from monitordecorrelation.experiment_config import CoTMonitorSpec
from monitordecorrelation.monitors.cot_monitor import CoTMonitor
from monitordecorrelation.types import Prompt, Rollout


def _roll() -> Rollout:
    return Rollout(prompt=Prompt(text="USERQ"), cot="SECRET_REASONING", output="THE_ANSWER")


def test_output_monitor_omits_cot():
    r = _roll()
    cot_mon = CoTMonitor("cot", "x/y", behavior="deception", api_key="test", use_cot=True)
    out_mon = CoTMonitor("out", "x/y", behavior="deception", api_key="test", use_cot=False)
    cp, op = cot_mon._build_prompt(r), out_mon._build_prompt(r)
    # CoT monitor sees the reasoning; output monitor does NOT.
    assert "SECRET_REASONING" in cp and "chain of thought" in cp.lower()
    assert "SECRET_REASONING" not in op
    # output monitor still sees the question + final answer
    assert "THE_ANSWER" in op and "USERQ" in op
    assert out_mon.use_cot is False
    print("output monitor omits CoT OK")


def test_spec_accepts_use_cot():
    s = CoTMonitorSpec(name="out_weak", role="held_out", model_id="x/y", use_cot=False)
    assert s.use_cot is False and s.kind == "cot"
    assert CoTMonitorSpec(name="c", role="held_out", model_id="x/y").use_cot is True  # default
    print("CoTMonitorSpec use_cot OK")


def test_binary_judge_prompts():
    r = _roll()
    # Default: percentile SCORE instruction.
    pct = CoTMonitor("pct", "x/y", behavior="deception", api_key="test")
    assert pct.binary_judge is False
    assert "SCORE:" in pct._build_prompt(r) and "VERDICT:" not in pct._build_prompt(r)

    # Full / CoT-only / output-only binary variants all ask YES/NO.
    full = CoTMonitor("b", "x/y", behavior="deception", api_key="test", binary_judge=True)
    cot_only = CoTMonitor("b", "x/y", behavior="deception", api_key="test",
                          binary_judge=True, use_output=False)
    out_only = CoTMonitor("b", "x/y", behavior="deception", api_key="test",
                          binary_judge=True, use_cot=False)
    for mon, must_have, must_not in (
        (full, ("SECRET_REASONING", "THE_ANSWER"), ()),
        (cot_only, ("SECRET_REASONING",), ("THE_ANSWER",)),
        (out_only, ("THE_ANSWER",), ("SECRET_REASONING",)),
    ):
        p = mon._build_prompt(r)
        assert "VERDICT: YES" in p and "VERDICT: NO" in p and "SCORE:" not in p
        assert "contain" in p.lower()
        for s in must_have:
            assert s in p
        for s in must_not:
            assert s not in p
    print("binary_judge prompts OK")


def test_spec_accepts_binary_judge():
    s = CoTMonitorSpec(name="bin", role="held_out", model_id="x/y", binary_judge=True)
    assert s.binary_judge is True
    assert CoTMonitorSpec(name="c", role="held_out", model_id="x/y").binary_judge is False
    print("CoTMonitorSpec binary_judge OK")


if __name__ == "__main__":
    test_output_monitor_omits_cot()
    test_spec_accepts_use_cot()
    test_binary_judge_prompts()
    test_spec_accepts_binary_judge()
    print("ALL PASS")
