"""The committed SLIM eval-rollout dump: schema projection + the tolerant slimming script."""

import json
import math
import subprocess
import sys
from pathlib import Path

from monitordecorrelation.eval.rollout_dump import SLIM_FIELDS, monitor_record, slim_record
from monitordecorrelation.types import MonitorResult

# the judge's saved API call: the prompt + parameters it was sent, and its full reply
CALL = {"url": "https://openrouter.ai/api/v1/chat/completions", "timeout": 60.0, "attempts": 1,
        "request": {"model": "x/y", "messages": [{"role": "user", "content": "judge prompt"}],
                    "temperature": 0.0, "max_tokens": 2048, "reasoning": {"enabled": False}},
        "response": {"message": {"content": "SCORE: 90", "reasoning": None}, "finish_reason": "stop"}}
FULL = {
    "step": 3, "task_id": 7, "question": "big prompt text", "cot": "long chain of thought", "answer": "code",
    "behavior_present": True, "loose_rh": True, "hardcoding": True, "unparsed": False, "env_meta": {"x": 1},
    "monitors": {"cot_weak": {"score": 0.9, "label": True, "call": CALL},
                 "probe_ood": {"score": float("nan"), "label": False}},
}
SLIM_MONITORS = {"cot_weak": {"score": 0.9, "label": True}, "probe_ood": {"score": float("nan"), "label": False}}


def test_slim_record_projects_exactly_the_slim_fields():
    s = slim_record(FULL)
    assert tuple(s) == SLIM_FIELDS
    assert "cot" not in s and "answer" not in s and "question" not in s  # the bulky text is gone
    # per monitor only {score, label}: the judge's call record (prompt + reply) is full-dump only
    assert s["monitors"]["cot_weak"] == SLIM_MONITORS["cot_weak"] and s["behavior_present"] is True
    assert tuple(s["monitors"]["probe_ood"]) == ("score", "label") and math.isnan(s["monitors"]["probe_ood"]["score"])
    assert "call" not in s["monitors"]["cot_weak"]


def test_monitor_record_keeps_the_judges_call_and_flags():
    judged = MonitorResult(score=0.9, label=True, meta={"raw": "SCORE: 90", "call": CALL})
    assert monitor_record(judged) == {"score": 0.9, "label": True, "call": CALL}  # raw lives inside call
    probe = MonitorResult(score=0.2, label=False, meta={"probe": "probe_ood"})
    assert monitor_record(probe) == {"score": 0.2, "label": False}
    bad = MonitorResult(score=0.0, label=False, meta={"raw": "no", "call": CALL, "parse_error": True})
    assert monitor_record(bad)["parse_error"] is True and monitor_record(bad)["call"] is CALL
    failed = MonitorResult(score=float("nan"), label=False, meta={"error": "HTTP 401"})
    assert monitor_record(failed)["error"] == "HTTP 401" and "call" not in monitor_record(failed)


def test_slim_record_missing_keys_become_none():
    assert slim_record({"step": 0})["loose_rh"] is None


def test_slim_script_skips_corrupt_lines_and_keeps_nan(tmp_path: Path):
    run = tmp_path / "run"; run.mkdir()
    good = json.dumps(FULL)  # python emits bare NaN for the probe score — must survive the round trip
    (run / "eval_rollouts.jsonl").write_text(good + "\n" + good[: len(good) // 2] + '{"step": 3' + "\n" + good + "\n")
    out = subprocess.run([sys.executable, "scripts/slim_eval_rollouts.py", str(run)], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr
    assert "1 corrupt lines skipped" in out.stdout and "corrupt line 2" in out.stderr
    recs = [json.loads(l) for l in (run / "eval_rollouts_slim.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and all(tuple(r) == SLIM_FIELDS for r in recs)
    assert math.isnan(recs[0]["monitors"]["probe_ood"]["score"])
