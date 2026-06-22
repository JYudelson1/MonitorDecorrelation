"""End-to-end smoke test of the GRPO loop (run_grpo) with fake backend/env/monitors — no tinker, no
models, no network. Catches loop-integration regressions that unit tests miss (e.g. the stale
advantages[] NameError in the rollout-save path). Exercises the CONTROL case: train_against=[].
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from monitordecorrelation.config import LoggingConfig, RunConfig
from monitordecorrelation.rl.train import run_grpo
from monitordecorrelation.types import EnvResult, MonitorResult, Prompt, Rollout


class _FakeBackend:
    name = "fake"

    def sample(self, prompts, *, num_samples=1, max_tokens=64, temperature=1.0):
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
        penalty_coef=1.0, kl_coef=0.0, seed=0,
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
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)


def test_run_grpo_survives_failing_monitor():
    """A held-out monitor that always raises (transient API failure) must not crash the run; it gets
    NaN-scored and the eval still completes."""
    run_dir = Path("data/runs/smoke_test_loop_fail")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg = RunConfig(
        env="fake_env", backend="fake", base_model="fake/model",
        batch_size=2, group_size=2, n_steps=1, eval_every=10, eval_size=2,
        penalty_coef=1.0, kl_coef=0.0, seed=0,
        logging=LoggingConfig(run_name="smoke_test_loop_fail", use_wandb=False, log_fraction=1.0),
    )
    try:
        run_grpo(cfg, _FakeEnv(), _FakeBackend(),
                 train_against=[], held_out=[_FakeMonitor("ok"), _ExplodingMonitor("bad")])
        ev = [json.loads(l) for l in (run_dir / "eval_metrics.jsonl").open() if l.strip()]
        assert ev  # run completed despite the exploding monitor
        # the bad monitor scored 0 rollouts; the good one scored all of them
        assert ev[0]["monitor/bad/n_scored"] == 0
        assert ev[0]["monitor/ok/n_scored"] == cfg.eval_size
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)
