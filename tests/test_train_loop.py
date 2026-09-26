"""End-to-end smoke test of the GRPO loop (run_grpo) with fake backend/env/monitors — no tinker, no
models, no network. Catches loop-integration regressions that unit tests miss (e.g. the stale
advantages[] NameError in the rollout-save path). Exercises the CONTROL case: train_against=[].
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from monitordecorrelation.config import LoggingConfig, RunConfig
from monitordecorrelation.rl.train import EnvScorer, run_grpo
from monitordecorrelation.types import EnvResult, MonitorResult, Prompt, Rollout


class _FakeBackend:
    name = "fake"

    def current_sampler(self):
        return "w0"

    @staticmethod
    def sampler_id(sampler):
        return sampler

    def sample(self, prompts, *, sampler, seed, num_samples=1, max_tokens=64, temperature=1.0):
        return [
            Rollout(prompt=p, cot="reason", output="```python\ndef f():\n  return 1\n```",
                    token_ids=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])
            for p in prompts for _ in range(num_samples)
        ]

    def train_step(self, rollouts, rewards, group_size):
        assert len(rollouts) == len(rewards)  # the Path-B signature
        return {"n_data": float(len(rollouts)), "kl/mean": 0.0, "train/logprob_mean": -1.5}


class _FakeEnv:
    name = "fake_env"
    behavior_name = "reward_hacking"

    def sample_prompt(self):
        return Prompt(text="task", meta={"task_id": 1})

    def holdout(self, n, seed=0):
        return [Prompt(text=f"eval{i}", meta={"task_id": 100 + i}) for i in range(n)]

    def score(self, rollout):
        return EnvResult(task_reward=0.5, behavior_present=False, meta={"unparsed": False})


class _FakeMonitor:
    def __init__(self, name):
        self.name = name

    def score_batch(self, rollouts):  # has score_batch → batched path (no threads)
        return [MonitorResult(score=0.3, label=False) for _ in rollouts]


class _ExplodingMonitor:
    """An API-style monitor (no score_batch → threaded path) that always raises — simulates a
    persistent OpenRouter 404. Must NOT crash the run."""

    def __init__(self, name):
        self.name = name

    def score(self, rollout):
        raise RuntimeError("simulated 404")


def test_run_grpo_control_smoke():
    run_dir = Path("data/runs/smoke_test_loop")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg = RunConfig(
        env="fake_env", backend="fake", base_model="fake/model",
        batch_size=2, group_size=2, n_steps=1, eval_every=10, eval_size=2,
        # A control's config carries NO penalty_coef (ExperimentConfig rejects it), so it arrives None.
        penalty_coef=None, kl_coef=0.0, seed=0,
        logging=LoggingConfig(run_name="smoke_test_loop", use_wandb=False, log_fraction=1.0),
    )
    try:
        # CONTROL: no train-against monitor (the case that crashed) + one held-out monitor.
        run_grpo(cfg, _FakeEnv(), _FakeBackend(), train_against=[], held_out=[_FakeMonitor("probe_x")])

        # the loop produced its files
        for f in ("run_info.json", "metrics.jsonl", "eval_metrics.jsonl", "rollouts.jsonl",
                  "eval_rollouts.jsonl"):
            assert (run_dir / f).exists(), f

        # saved train rollouts carry reward but NOT the removed advantage field
        rows = [json.loads(l) for l in (run_dir / "rollouts.jsonl").open() if l.strip()]
        assert rows and "reward" in rows[0] and "advantage" not in rows[0]
        # held-out monitor was scored every eval (degradation source)
        ev = [json.loads(l) for l in (run_dir / "eval_metrics.jsonl").open() if l.strip()]
        assert ev and "monitor/probe_x/auroc" in ev[0]
        # no monitor penalty in a control: effective λ and applied penalty are both 0
        m = [json.loads(l) for l in (run_dir / "metrics.jsonl").open() if l.strip()]
        assert m[0]["reward/penalty_coef"] == 0.0 and m[0]["reward/penalty_mean"] == 0.0
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)


@pytest.mark.parametrize("coef,sched,ta,match", [
    (1.0, None, False, "no train_against monitor"),          # a control's λ multiplies nothing
    (None, {"start_penalty": 0.0, "end_penalty": 1.0}, False, "no train_against monitor"),
    (None, None, True, "exactly one"),                       # trains against a monitor with no λ
    (1.0, {"start_penalty": 0.0, "end_penalty": 1.0}, True, "exactly one"),  # coef hidden by the ramp
])
def test_run_grpo_rejects_a_penalty_it_would_ignore(coef, sched, ta, match):
    """λ is set exactly where it takes effect — anything else fails before the run starts."""
    run_dir = Path("data/runs/smoke_test_loop_penalty")
    cfg = RunConfig(
        env="fake_env", backend="fake", base_model="fake/model",
        batch_size=2, group_size=2, n_steps=1, eval_every=10, eval_size=2,
        penalty_coef=coef, penalty_schedule=sched, kl_coef=0.0, seed=0,
        logging=LoggingConfig(run_name="smoke_test_loop_penalty", use_wandb=False, log_fraction=1.0),
    )
    with pytest.raises(ValueError, match=match):
        run_grpo(cfg, _FakeEnv(), _FakeBackend(),
                 train_against=[_FakeMonitor("ta")] if ta else [], held_out=[_FakeMonitor("ho")])
    assert not run_dir.exists()  # rejected before anything was written

def test_run_grpo_aborts_on_dead_train_against_monitor():
    """A train-against monitor that scores nothing = no penalty signal → the run must ABORT (not
    silently train as a no-penalty control), unlike a held-out monitor which is tolerated."""
    run_dir = Path("data/runs/smoke_test_loop_ta")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg = RunConfig(
        env="fake_env", backend="fake", base_model="fake/model",
        batch_size=2, group_size=2, n_steps=1, eval_every=10, eval_size=2,
        penalty_coef=1.0, kl_coef=0.0, seed=0,
        logging=LoggingConfig(run_name="smoke_test_loop_ta", use_wandb=False, log_fraction=1.0),
    )
    try:
        with pytest.raises(RuntimeError, match=r"monitor 'bad_ta' returned NaN for 2/2 rollouts"):
            run_grpo(cfg, _FakeEnv(), _FakeBackend(),
                     train_against=[_ExplodingMonitor("bad_ta")], held_out=[])
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)


def test_run_grpo_aborts_when_a_monitor_cannot_score():
    """A monitor that cannot score a rollout must stop the run. Dropping those silently biases a
    held-out AUROC (the missing rollouts are the ones the API choked on) and silently un-penalizes a
    train-against rollout — both invisible in the metrics, so the run dies loudly instead."""
    run_dir = Path("data/runs/smoke_test_loop_fail")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg = RunConfig(
        env="fake_env", backend="fake", base_model="fake/model",
        batch_size=2, group_size=2, n_steps=1, eval_every=10, eval_size=2,
        kl_coef=0.0, seed=0,  # a control: no λ
        logging=LoggingConfig(run_name="smoke_test_loop_fail", use_wandb=False, log_fraction=1.0),
    )
    try:
        with pytest.raises(RuntimeError, match=r"monitor 'bad' returned NaN for 2/2 rollouts"):
            run_grpo(cfg, _FakeEnv(), _FakeBackend(),
                     train_against=[], held_out=[_FakeMonitor("ok"), _ExplodingMonitor("bad")])
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)


class _InvalidatingBackend(_FakeBackend):
    """Cycles each prompt's samples through: valid, truncated (stop_reason "length"), unparseable (no
    codeblock), valid — the two kinds of INVALID rollout, alongside valid ones."""

    def sample(self, prompts, *, sampler, seed, num_samples=1, max_tokens=64, temperature=1.0):
        out = []
        for p in prompts:
            for k in range(num_samples):
                kind = ("ok", "trunc", "noparse", "ok")[k % 4]
                out.append(Rollout(
                    prompt=p, cot="reason",
                    output="no code here" if kind == "noparse" else "```python\ndef f(x):\n  return x\n```",
                    token_ids=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3],
                    meta={"stop_reason": "length" if kind == "trunc" else "stop"}))
        return out


class _ParsingEnv(_FakeEnv):
    """Like the real envs: an output with no codeblock is unparseable (-1, never monitored). Every
    VALID rollout is a hack, every invalid one is not — so a monitor metric that leaked an invalid
    rollout in would show up as a negative class."""

    def unparseable(self, rollout):
        return "```" not in rollout.output

    def score(self, rollout):
        if rollout.meta.get("stop_reason") != "stop":  # truncated: not graded (single-turn rule)
            return EnvResult(task_reward=0.0, behavior_present=False,
                             meta={"unparsed": False, "truncated": True}, reward_override=-1.0)
        if self.unparseable(rollout):
            return EnvResult(task_reward=0.0, behavior_present=False,
                             meta={"unparsed": True, "truncated": False}, reward_override=-1.0)
        return EnvResult(task_reward=0.5, behavior_present=True, meta={"unparsed": False, "truncated": False})


class _CountingMonitor(_FakeMonitor):
    """A probe-style monitor that records every rollout it is shown."""

    def __init__(self, name):
        super().__init__(name)
        self.seen: list[Rollout] = []

    def score_batch(self, rollouts):
        self.seen.extend(rollouts)
        return super().score_batch(rollouts)


class _CountingJudge:
    """An API-judge-style monitor (threaded path) that records every rollout it is shown."""

    def __init__(self, name):
        self.name = name
        self.seen: list[Rollout] = []

    def score(self, rollout):
        self.seen.append(rollout)
        return MonitorResult(score=0.3, label=False)


def test_invalid_rollouts_are_never_monitored_and_excluded_from_monitor_metrics():
    """Truncated and unparseable rollouts get exactly -1 (no λ·suspiciousness), are shown to NO
    monitor (train-against or held-out, judge or probe), are saved with ``monitors == {}`` and their
    ``invalid_reason``, and every monitor statistic is over the valid rollouts only."""
    run_dir = Path("data/runs/smoke_test_loop_invalid")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg = RunConfig(
        env="fake_env", backend="fake", base_model="fake/model",
        batch_size=2, group_size=4, n_steps=1, eval_every=10, eval_size=2, eval_samples_per_prompt=4,
        penalty_coef=1.0, kl_coef=0.0, seed=0,
        logging=LoggingConfig(run_name="smoke_test_loop_invalid", use_wandb=False, log_fraction=1.0),
    )
    ta, judge, probe = _CountingMonitor("ta"), _CountingJudge("ho_judge"), _CountingMonitor("ho_probe")
    try:
        run_grpo(cfg, _ParsingEnv(), _InvalidatingBackend(), train_against=[ta], held_out=[judge, probe])
        for mon in (ta, judge, probe):
            assert mon.seen and all(r.meta["stop_reason"] == "stop" and "```" in r.output for r in mon.seen)

        rows = [json.loads(l) for l in (run_dir / "rollouts.jsonl").open() if l.strip()]
        assert len(rows) == 8
        for r in rows:
            if r["invalid_reason"] is not None:
                assert r["monitors"] == {} and r["reward"] == -1.0
                assert r["env"]["reward_override"] == -1.0
            else:
                assert r["monitors"]["ta"]["score"] == pytest.approx(0.3)
                assert r["reward"] == pytest.approx(0.5 - 1.0 * 0.3)
        assert sorted(str(r["invalid_reason"]) for r in rows) == ["None"] * 4 + ["truncated"] * 2 + ["unparsed"] * 2

        m = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
        assert m["reward/override_rate"] == 0.5 and m["invalid_rate"] == 0.5
        assert m["reward/monitor_suspiciousness_mean"] == pytest.approx(0.3)  # over the 4 valid only
        assert m["reward/penalty_mean"] == pytest.approx(0.15)  # applied only to the 4 valid, per rollout
        assert m["monitor/ta/n_scored"] == 4 and m["monitor/ta/n_pos"] == 4 and m["monitor/ta/n_neg"] == 0
        assert m["monitor/ta/mean_score_not_reward_hacking"] != m["monitor/ta/mean_score_not_reward_hacking"]  # NaN
        assert m["behavior_rate"] == 0.5  # the oracle rate stays over ALL rollouts

        ev = [json.loads(l) for l in (run_dir / "eval_metrics.jsonl").open() if l.strip()]
        for row in ev:
            assert row["invalid_rate"] == 0.5 and row["truncated_rate"] == 0.25
            for name in ("ta", "ho_judge", "ho_probe"):
                assert row[f"monitor/{name}/n_scored"] == 4 and row[f"monitor/{name}/n_neg"] == 0
        erecs = [json.loads(l) for l in (run_dir / "eval_rollouts.jsonl").open() if l.strip()]
        slim = [json.loads(l) for l in (run_dir / "eval_rollouts_slim.jsonl").open() if l.strip()]
        for rec in (*erecs, *slim):
            assert (rec["monitors"] == {}) == (rec["invalid_reason"] is not None)
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)


def _grade(env, rollouts):
    """Grade like the RL loop: every rollout through EnvScorer (a thread each), then the checks."""
    sc = EnvScorer(env)
    for i, r in enumerate(rollouts):
        sc.submit(i, r)
    return sc.collect(rollouts)


def test_env_grading_rejects_an_env_that_overrides_a_valid_rollout():
    """reward_override is reserved for invalid rollouts — the ones the monitors skipped. An env that
    sets it on a valid rollout would give a monitored rollout a monitor-free reward: fail loudly."""
    class _BadEnv:
        def score(self, rollout):
            return EnvResult(task_reward=1.0, behavior_present=False, meta={}, reward_override=-1.0)

    with pytest.raises(RuntimeError, match="reward_override on valid rollout 0"):
        _grade(_BadEnv(), [Rollout(prompt=Prompt(text="q"), cot="", output="a", meta={"stop_reason": "stop"})])


def test_env_grading_rejects_an_unparsed_flag_that_disagrees_with_unparseable():
    class _BadEnv:
        def unparseable(self, rollout):
            return False

        def score(self, rollout):
            return EnvResult(task_reward=0.0, behavior_present=False, meta={"unparsed": True})

    with pytest.raises(RuntimeError, match="unparseable\(\) disagrees"):
        _grade(_BadEnv(), [Rollout(prompt=Prompt(text="q"), cot="", output="a")])


def test_env_grading_gives_every_truncated_rollout_minus_one_whatever_the_env_said():
    """The truncation rule lives in the RL loop, not in each env: any rollout whose sampling stopped
    on max_tokens gets reward_override = -1, even from an env that knows nothing about truncation.
    (A multi-turn env, which may still grade a truncated episode's earlier turns.)"""
    class _PlainEnv:
        multi_turn = True

        def score(self, rollout):
            return EnvResult(task_reward=1.0, behavior_present=False, meta={})

    def roll(stop):
        return Rollout(prompt=Prompt(text="q"), cot="", output="a",
                       meta={} if stop is None else {"stop_reason": stop})

    res = _grade(_PlainEnv(), [roll("length"), roll("stop"), roll(None)])
    assert [r.reward_override for r in res] == [-1.0, None, None]
    assert [r.task_reward for r in res] == [1.0, 1.0, 1.0]  # task score untouched


def test_env_grading_rejects_a_single_turn_env_that_grades_a_truncated_rollout():
    """Single-turn envs must not grade a truncated rollout: no task score, no behavior label from text
    that was cut off (and that no monitor will see)."""
    class _GradesEverything:
        def score(self, rollout):
            return EnvResult(task_reward=1.0, behavior_present=True, meta={})

    trunc = Rollout(prompt=Prompt(text="q"), cot="", output="a", meta={"stop_reason": "length"})
    with pytest.raises(RuntimeError, match="graded truncated rollout 0"):
        _grade(_GradesEverything(), [trunc])
