"""experiments/add_posthoc_monitors.py: a post-hoc monitor's sidecar has exactly the columns a live
held-out monitor gets, merges into the analysis readers, resumes, and refuses what it cannot do right."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from monitordecorrelation.eval.degradation import run_trajectories
from monitordecorrelation.eval.posthoc import posthoc_metrics_path, with_posthoc
from monitordecorrelation.rl.train import eval_monitor_metrics
from monitordecorrelation.types import MonitorResult

_REPO = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("_add_posthoc", _REPO / "experiments" / "add_posthoc_monitors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubJudge:
    """Scores 0.9 for an answer containing 'HACK', else 0.1; counts its calls."""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def score(self, rollout) -> MonitorResult:
        self.calls += 1
        s = 0.9 if "HACK" in rollout.output else 0.1
        return MonitorResult(score=s, label=s >= 0.5, meta={})


def _make_run(d: Path, *, steps=(0, 3), n=6, corrupt_step: int | None = None) -> Path:
    """A run dir with a live monitor 'live', n rollouts per step (the last one invalid)."""
    d.mkdir(parents=True)
    (d / "run_info.json").write_text(json.dumps(
        {"env": {"name": "mbpp_honeypot", "behavior_name": "reward_hacking"}, "train_against": []}))
    rows, dump = [], []
    for step in steps:
        recs = []
        for k in range(n):
            hack = k % 2 == 0
            invalid = k == n - 1
            recs.append({"step": step, "task_id": k, "question": f"q{k}", "cot": f"c{step}{k}",
                         "answer": ("HACK" if hack else "ok") + str(k), "behavior_present": hack,
                         "loose_rh": hack or k == 1, "hardcoding": hack, "unparsed": False,
                         "invalid_reason": "truncated" if invalid else None,
                         "monitors": {} if invalid else {"live": {"score": 0.5, "label": True}}})
        n_valid = sum(r["invalid_reason"] is None for r in recs)
        rows.append({"step": step, "behavior_rate": sum(r["behavior_present"] for r in recs) / n,
                     "monitor/live/auroc": 0.5, "monitor/live/n_scored": n_valid})
        if step == corrupt_step:
            recs = recs[:-2] + [recs[-1]]  # a lost line: the dump no longer matches the logged step
        dump += recs
    (d / "eval_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "eval_rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dump) + "{corrupt\n")
    return d


def test_sidecar_matches_live_columns_and_merges(tmp_path):
    mod = _script()
    d = _make_run(tmp_path / "mbpp_x_control_s0")
    run = mod.Run(d)
    assert run.good_steps == [0, 3] and run.corrupt == 1
    m = StubJudge("weak_out")
    assert mod.score_runs([run], [m], workers=4)
    assert m.calls == 2 * 5  # valid rollouts only

    class Spec:  # minimal stand-in for a MonitorSpec
        def model_dump(self):
            return {"name": "weak_out"}

    assert mod.write_metrics(run, m, Spec(), "cfg.json")
    rows = [json.loads(l) for l in posthoc_metrics_path(d, "weak_out").read_text().splitlines()]
    # the same columns, computed by the same function, as a live monitor over those rollouts
    recs = [json.loads(l) for l in (d / "eval_rollouts.jsonl").read_text().splitlines()[:6]]
    results = [None if r["invalid_reason"] else MonitorResult(0.9 if "HACK" in r["answer"] else 0.1,
                                                                "HACK" in r["answer"]) for r in recs]
    want = eval_monitor_metrics(m, results, [r["behavior_present"] for r in recs],
                                [r["loose_rh"] for r in recs], "reward_hacking")
    got = {k: v for k, v in rows[0].items() if k != "step"}
    assert got.keys() == want.keys()
    assert all((math.isnan(want[k]) and math.isnan(got[k])) or want[k] == got[k] for k in want)
    assert got["monitor/weak_out/auroc"] == 1.0 and got["monitor/weak_out/n_scored"] == 5

    merged = with_posthoc(d, [json.loads(l) for l in (d / "eval_metrics.jsonl").read_text().splitlines()])
    assert merged[1]["monitor/weak_out/auroc"] == 1.0 and merged[1]["monitor/live/auroc"] == 0.5
    assert "weak_out" in run_trajectories(d, rolling=1)

    # resume: nothing left to score
    m2 = StubJudge("weak_out")
    assert mod.score_runs([mod.Run(d)], [m2], workers=4) and m2.calls == 0


def test_mismatched_step_is_skipped(tmp_path):
    mod = _script()
    run = mod.Run(_make_run(tmp_path / "r", corrupt_step=3))
    assert run.good_steps == [0] and 3 in run.bad_steps


def test_refuses_existing_live_name(tmp_path):
    mod = _script()
    run = mod.Run(_make_run(tmp_path / "r"))

    class Spec:
        name = "live"

    with pytest.raises(SystemExit, match="already exists"):
        mod._check_name(run, Spec())


def test_with_posthoc_refuses_column_clash(tmp_path):
    d = _make_run(tmp_path / "r")
    posthoc_metrics_path(d, "x").write_text(json.dumps({"step": 0, "monitor/live/auroc": 0.9}) + "\n")
    rows = [json.loads(l) for l in (d / "eval_metrics.jsonl").read_text().splitlines()]
    with pytest.raises(ValueError, match="already exist"):
        with_posthoc(d, rows)


class _StubReader:
    """The shared reader: one 'activation' per rollout = [1.0] if 'HACK' in the answer else [0.0]."""

    def __init__(self):
        self.reads = []

    def extract_activations(self, items, *, follow_up, batch_size, preserve_thinking):
        self.reads.append(len(items))
        import numpy as np
        return np.array([[1.0] if "HACK" in r.output else [0.0] for r in items])


class _StubLinear:
    def __init__(self, hi):
        self.hi = hi

    def score(self, acts):
        import numpy as np
        return np.where(acts[:, 0] > 0, self.hi, 0.2)


class StubProbe:
    """A probe: score_batch + (model, probe, follow_up, preserve_thinking), no model_id."""

    def __init__(self, name: str, reader=None, hi=0.8):
        self.name, self.threshold = name, 0.5
        self.model, self.probe = reader or _StubReader(), _StubLinear(hi)
        self.follow_up, self.preserve_thinking, self.batch_size = ("Q?", "Yes"), True, 8
        self.batches = self.model.reads

    def score_batch(self, rollouts):
        return [MonitorResult(score=float(s), label=bool(s >= 0.5))
                for s in self.probe.score(self.model.extract_activations(rollouts, follow_up=self.follow_up,
                                                                         batch_size=8, preserve_thinking=True))]


def test_probes_are_scored_in_batches_into_a_sidecar(tmp_path):
    mod = _script()
    d = _make_run(tmp_path / "mbpp_x_probe_s0")
    run = mod.Run(d)
    p = StubProbe("probe_cot")
    assert mod.score_runs([run], [p], workers=4)
    assert sum(p.batches) == 10 and len(p.batches) == 1  # all valid rollouts, one batch

    class Spec:
        def model_dump(self):
            return {"name": "probe_cot", "kind": "probe"}

    assert mod.write_metrics(run, p, Spec(), "cfg.json")
    rows = [json.loads(l) for l in posthoc_metrics_path(d, "probe_cot").read_text().splitlines()]
    assert rows[0]["monitor/probe_cot/auroc"] == 1.0 and rows[0]["monitor/probe_cot/n_scored"] == 5
    assert not any("parse_error" in k or "finish_length" in k for k in rows[0])  # no judge call-health
    p2 = StubProbe("probe_cot")
    assert mod.score_runs([mod.Run(d)], [p2], workers=4) and p2.batches == []  # resumed: nothing left


def test_probes_with_the_same_view_share_one_activation_read(tmp_path):
    mod = _script()
    d = _make_run(tmp_path / "mbpp_x_probe2_s0")
    reader = _StubReader()
    a, b = StubProbe("probe_a", reader, hi=0.8), StubProbe("probe_b", reader, hi=0.9)
    assert mod.score_runs([mod.Run(d)], [a, b], workers=4)
    assert reader.reads == [10]  # 10 valid rollouts, read once for both probes
    for name, hi in (("probe_a", 0.8), ("probe_b", 0.9)):
        recs = [json.loads(l)["record"]["score"] for l in (d / "posthoc" / f"{name}.scores.jsonl").read_text().splitlines()]
        assert len(recs) == 10 and set(recs) == {hi, 0.2}
