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


if __name__ == "__main__":
    test_output_monitor_omits_cot()
    test_spec_accepts_use_cot()
    print("ALL PASS")
