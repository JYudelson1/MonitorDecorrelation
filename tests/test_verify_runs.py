"""verify_runs: the training log-prob spike check (the Sep 3 collapse precursor)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "verify_runs", Path(__file__).resolve().parents[1] / "scripts" / "verify_runs.py")
verify_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_runs)


def _write(tmp_path: Path, values: list[float | str], name: str = "run") -> Path:
    d = tmp_path / name
    d.mkdir()
    with (d / "metrics.jsonl").open("w") as f:
        for step, v in enumerate(values):
            if v == "garbage":
                f.write("{not json\n")  # the Sep 3 logs really had malformed records
            else:
                f.write(json.dumps({"step": step, "loss/train/logprob_mean": v}) + "\n")
    return d


def test_flags_the_one_step_spike_and_tolerates_bad_lines(tmp_path):
    # the cot_weak_s2 shape: ~-0.5 for 40 steps, then -51, then a collapsed -2.0 tail
    d = _write(tmp_path, [-0.5] * 20 + ["garbage"] + [-0.6] * 20 + [-51.0, -2.0, -2.0])
    spikes = verify_runs._logprob_spikes(d)
    assert [s for s, _ in spikes] == [41]  # the -2.0 tail is below the floor but not 3x the median
    assert spikes[0][1] == -51.0
    print("spike flagged at the right step OK")


def test_no_flag_on_gradual_drift_or_a_naturally_low_run(tmp_path):
    # probe_iid_s1: log-probs sit around -1.3 to -1.9 for most of the run; that is not a spike
    d = _write(tmp_path, [-0.5, -0.7, -1.2, -1.4, -1.6, -1.8, -1.9, -1.6, -1.4, -1.5, -1.9, -1.7])
    assert verify_runs._logprob_spikes(d) == []
    assert verify_runs._logprob_spikes(tmp_path / "missing") == []   # no metrics.jsonl yet
    print("no false positive on drift OK")


def test_check_reports_spikes_as_a_warning_not_a_problem(tmp_path):
    # a control run (no train_against) so the target check itself has nothing to say
    d = _write(tmp_path, [-0.4] * 10 + [-3.0, -0.2], name="mbpp_Qwen3-8B_control_s0_test")
    (d / "run_info.json").write_text(json.dumps({"run_name": "x", "config": {"seed": 0}, "train_against": [], "held_out": []}))
    probs, facts = verify_runs.check(d)
    assert probs == []                      # a spike is something to inspect, not a misconfiguration
    assert facts["spikes"] == [(10, -3.0)]
    print("spike is a warning OK")


def _run_info(tmp_path: Path, held_out: list[dict], name: str = "mbpp_Qwen3-8B_control_s0_x") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "run_info.json").write_text(json.dumps(
        {"run_name": name, "config": {"seed": 0}, "train_against": [], "held_out": held_out}))
    return d


def test_recorded_reasoning_is_checked_against_what_the_model_honours(tmp_path):
    g25, g35 = "google/gemini-2.5-flash-lite", "google/gemini-3.5-flash-lite"
    ok = [{"name": "a", "model_id": g25, "reasoning": {"max_tokens": 512}},
          {"name": "b", "model_id": g25, "reasoning": {"enabled": False}},
          {"name": "c", "model_id": g35, "reasoning": {"effort": "low"}}]
    probs, facts = verify_runs.check(_run_info(tmp_path, ok, "mbpp_Qwen3-8B_control_s0_ok"))
    assert probs == [] and not facts["reasoning_unrecorded"]
    bad = [{"name": "a", "model_id": g25, "reasoning": {"max_tokens": 256}},   # clamped by Google
           {"name": "c", "model_id": g35, "reasoning": {"enabled": False}},    # mandatory reasoning
           {"name": "d", "model_id": "anthropic/claude-3-haiku", "reasoning": {"enabled": False}}]
    probs, _ = verify_runs.check(_run_info(tmp_path, bad, "mbpp_Qwen3-8B_control_s0_bad"))
    assert len(probs) == 3 and "'a'" in probs[0] and "'c'" in probs[1] and "specialized" in probs[2]
    # legacy records keep the rules of their time
    legacy = [{"name": "a", "model_id": g25, "reasoning_effort": None, "reasoning_max_tokens": None},
              {"name": "c", "model_id": g35, "reasoning_effort": None, "reasoning_max_tokens": None}]
    probs, _ = verify_runs.check(_run_info(tmp_path, legacy, "mbpp_Qwen3-8B_control_s0_legacy"))
    assert probs == ["c: google/gemini-3.5-flash-lite needs a reasoning_effort (it 400s on reasoning off)"]


def test_batch_must_give_each_judge_the_same_reasoning(tmp_path, capsys, monkeypatch):
    g25 = "google/gemini-2.5-flash-lite"
    _run_info(tmp_path, [{"name": "a", "model_id": g25, "reasoning": {"max_tokens": 512}}],
              "mbpp_Qwen3-8B_control_s0_x")
    _run_info(tmp_path, [{"name": "a", "model_id": g25, "reasoning": {"enabled": False}}],
              "mbpp_Qwen3-8B_control_s1_x")
    monkeypatch.setattr("sys.argv", ["verify_runs.py", str(tmp_path / "mbpp_*")])
    assert verify_runs.main() == 1
    assert "do not share one judge reasoning setting" in capsys.readouterr().out


def _row(tmp_path: Path, name: str, ta: list[str], **pen) -> None:
    d = tmp_path / name
    d.mkdir()
    (d / "run_info.json").write_text(json.dumps(
        {"run_name": name, "config": {"seed": 0, **pen},
         "train_against": [{"name": n} for n in ta],
         "held_out": [{"name": n} for n in ("probe_ood", "probe_iid", "out_weak") if n not in ta]}))


def test_penalty_is_compared_across_the_train_against_runs_only(tmp_path, capsys, monkeypatch):
    # A control carries no λ (None) — that is not a mismatch with the rows that do.
    _row(tmp_path, "mbpp_M_control_s0_a", [], penalty_coef=None)
    _row(tmp_path, "mbpp_M_probe_ood_s0_a", ["probe_ood"], penalty_coef=1.0)
    _row(tmp_path, "mbpp_M_probe_iid_s0_a", ["probe_iid"], penalty_coef=1.0)
    monkeypatch.setattr("sys.argv", ["verify_runs.py", str(tmp_path / "mbpp_*")])
    assert verify_runs.main() == 0
    # …but two train-against rows under different λ regimes are.
    _row(tmp_path, "mbpp_M_out_weak_s0_a", ["out_weak"],
         penalty_schedule={"start_penalty": 0.0, "end_penalty": 1.0})
    capsys.readouterr()
    assert verify_runs.main() == 1
    assert "do not share one monitor penalty λ" in capsys.readouterr().out
