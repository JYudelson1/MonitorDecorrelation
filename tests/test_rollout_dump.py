"""The committed SLIM eval-rollout dump: schema projection + the tolerant slimming script."""

import json
import math
import subprocess
import sys
from pathlib import Path

from monitordecorrelation.eval.rollout_dump import SLIM_FIELDS, slim_record

FULL = {
    "step": 3, "task_id": 7, "question": "big prompt text", "cot": "long chain of thought", "answer": "code",
    "behavior_present": True, "loose_rh": True, "hardcoding": True, "unparsed": False, "env_meta": {"x": 1},
    "monitors": {"cot_weak": {"score": 0.9, "label": True}, "probe_ood": {"score": float("nan"), "label": False}},
}


def test_slim_record_projects_exactly_the_slim_fields():
    s = slim_record(FULL)
    assert tuple(s) == SLIM_FIELDS
    assert "cot" not in s and "answer" not in s and "question" not in s  # the bulky text is gone
    assert s["monitors"] == FULL["monitors"] and s["behavior_present"] is True


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
