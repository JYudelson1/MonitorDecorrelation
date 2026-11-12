"""``task_reward: false`` — a MONITOR-ONLY run (rl/train.py): reward = −λ·suspiciousness.

The env still grades every rollout (task / oracle metrics are logged as usual) but its task reward
never reaches the gradient; an invalid rollout keeps the flat −1 (it is never monitored, so without
that floor not submitting would be the penalty-free optimum). Only legal with a train_against monitor,
in the config schema and in ``run_grpo`` alike. All offline (fake backend / env / monitor).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitordecorrelation.config import LoggingConfig, RunConfig
from monitordecorrelation.rl.train import run_grpo
from monitordecorrelation.types import INVALID_ROLLOUT_REWARD, EnvResult, MonitorResult, Prompt, Rollout

TASK = 1.0      # every valid rollout's task reward
SUSP = 0.25     # the train-against monitor's score on every rollout it sees
LAMBDA = 2.0    # penalty_coef


class _Env:
    """Rollouts whose output is ``"bad"`` are unparseable (invalid); the rest score TASK."""

    name = "fake_env"
    behavior_name = "reward_hacking"

    def sample_prompt(self):
        return Prompt(text="task")

    def holdout(self, n, seed=0):
        return [Prompt(text=f"eval{i}") for i in range(n)]

    @staticmethod
    def unparseable(rollout):
        return rollout.output == "bad"

    def score(self, rollout):
        bad = self.unparseable(rollout)
        return EnvResult(task_reward=0.0 if bad else TASK, behavior_present=not bad, meta={"unparsed": bad})


class _Monitor:
    name = "mon"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score_batch(self, rollouts):
        self.seen.extend(r.output for r in rollouts)
        return [MonitorResult(score=SUSP, label=False) for _ in rollouts]


class _Backend:
    """Samples one ``"bad"`` (invalid) rollout per group, the rest valid; records every reward list."""

    name = "fake"

    def __init__(self) -> None:
        self.rewards: list[list[float]] = []

    def current_sampler(self):
        return "v"

    @staticmethod
    def sampler_id(sampler):
        return sampler

    def sample(self, prompts, *, sampler, seed, num_samples=1, max_tokens=64, temperature=1.0):
        return [Rollout(prompt=p, cot="c", output="bad" if j == 0 else "ok", token_ids=[1], logprobs=[-0.1])
                for p in prompts for j in range(num_samples)]

    def train_step(self, rollouts, rewards, group_size):
        self.rewards.append(list(rewards))
        return {"n_data": float(len(rollouts)), "kl/mean": 0.0, "train/logprob_mean": -1.0}


def _cfg(name: str, *, task_reward: bool, penalty_coef: float | None = LAMBDA) -> RunConfig:
    return RunConfig(
        env="fake_env", backend="fake", base_model="fake/model",
        batch_size=2, group_size=4, n_steps=2, eval_every=2, eval_size=2,
        penalty_coef=penalty_coef, task_reward=task_reward, kl_coef=0.0, seed=3,
        logging=LoggingConfig(run_name=name, use_wandb=False, log_fraction=1.0),
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


@pytest.mark.parametrize("task_reward, valid_reward", [(True, TASK - LAMBDA * SUSP), (False, -LAMBDA * SUSP)])
def test_reward_drops_the_task_term_only_when_monitor_only(tmp_path, monkeypatch, task_reward, valid_reward):
    monkeypatch.chdir(tmp_path)  # run_grpo writes data/runs/<run_name>/ under the cwd
    backend, mon = _Backend(), _Monitor()
    run_grpo(_cfg("r", task_reward=task_reward), _Env(), backend, train_against=[mon], held_out=[])

    assert len(backend.rewards) == 2
    for rewards in backend.rewards:  # groups of 4: one invalid ("bad") rollout, then three valid ones
        assert rewards == [INVALID_ROLLOUT_REWARD, valid_reward, valid_reward, valid_reward] * 2
    assert "bad" not in mon.seen  # the invalid rollout is never shown to the monitor

    for row in _rows(tmp_path / "data/runs/r/metrics.jsonl"):
        # the env's task score is still graded + logged even when it is not in the reward
        assert row["reward/task_mean"] == pytest.approx(0.75 * TASK)
        assert row["reward/task_coef"] == (1.0 if task_reward else 0.0)
        assert row["reward/monitor_suspiciousness_mean"] == pytest.approx(SUSP)
        assert row["reward/total_mean"] == pytest.approx(0.25 * INVALID_ROLLOUT_REWARD + 0.75 * valid_reward)
        assert row["behavior_rate"] == pytest.approx(0.75)  # the oracle is untouched


def test_run_grpo_refuses_monitor_only_without_a_train_against_monitor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="task_reward=False"):
        run_grpo(_cfg("c", task_reward=False, penalty_coef=None), _Env(), _Backend(),
                 train_against=[], held_out=[_Monitor()])


# ---- the config side ------------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[1]
_MONITOR_ONLY = _REPO / "experiments" / "configs" / "terminal_verifier_monitor_only"


def _cfg_dict(**kw) -> dict:
    base = {"run_name": "t", "policy": "Qwen/Qwen3-8B", "env": "mbpp_honeypot", "max_tokens": 64,
            "penalty_coef": 1.0,
            "monitors": [{"kind": "cot", "name": "m", "role": "train_against",
                          "model_id": "google/gemini-2.5-flash-lite"}]}
    base.update(kw)
    return base


def test_config_task_reward_defaults_on_and_needs_a_train_against_monitor_when_off():
    from pydantic import ValidationError

    from monitordecorrelation.experiment_config import ExperimentConfig

    assert ExperimentConfig.model_validate(_cfg_dict()).task_reward is True
    assert ExperimentConfig.model_validate(_cfg_dict(task_reward=False)).task_reward is False
    held_out = [{"kind": "cot", "name": "m", "role": "held_out", "model_id": "google/gemini-2.5-flash-lite"}]
    with pytest.raises(ValidationError, match="task_reward=false"):
        ExperimentConfig.model_validate(_cfg_dict(task_reward=False, monitors=held_out, penalty_coef=None))
    with pytest.raises(ValidationError):  # strict: no "false"-string / 0 coercion
        ExperimentConfig.model_validate(_cfg_dict(task_reward="no"))


def test_set_task_reward_is_validated_like_the_config():
    from pydantic import ValidationError

    from monitordecorrelation.experiment_config import ExperimentConfig, apply_overrides

    cfg = ExperimentConfig.model_validate(_cfg_dict())
    assert apply_overrides(cfg, ["task_reward=false"]).task_reward is False
    control = ExperimentConfig.model_validate(_cfg_dict(
        monitors=[{"kind": "cot", "name": "m", "role": "held_out", "model_id": "google/gemini-2.5-flash-lite"}],
        penalty_coef=None))
    with pytest.raises((ValidationError, SystemExit, ValueError), match="task_reward"):
        apply_overrides(control, ["task_reward=false"])


def test_monitor_only_configs_match_their_source_rows_except_the_monitor_only_keys():
    """The four terminal monitor-only rows are the q{3,35}_{cot,out} rows + task_reward=false + the
    init checkpoint (a tinker training-state path) + no early stop (they run all n_steps), each
    training against the judge it is named after."""
    from monitordecorrelation.experiment_config import load_config

    rows = sorted(p.stem for p in _MONITOR_ONLY.glob("*.json"))
    assert rows == ["q35_cot", "q35_out", "q3_cot", "q3_out"]
    for row in rows:
        new = json.loads((_MONITOR_ONLY / f"{row}.json").read_text())
        src = json.loads((_REPO / "experiments" / "configs" / f"terminal_verifier_{row}.json").read_text())
        diff = sorted(k for k in set(new) | set(src) if new.get(k) != src.get(k))
        assert diff == ["description", "init_from_tinker_ckpt", "run_name", "stop_after_zero_behavior_steps",
                        "task_reward"], row
        cfg = load_config(_MONITOR_ONLY / f"{row}.json")
        assert cfg.task_reward is False
        assert cfg.stop_after_zero_behavior_steps is None
        assert cfg.init_from_tinker_ckpt.startswith("tinker://") and "/weights/" in cfg.init_from_tinker_ckpt
        assert [m.name for m in cfg.monitors if m.role == "train_against"] == [row]
        assert cfg.run_name == f"tv_monitor_only_{row}_s0"


def test_resume_of_a_run_from_before_task_reward_existed_is_not_refused(monkeypatch, tmp_path):
    """A snapshot written before the key existed lacks it; the run optimised the task reward, which is
    what the default (true) means — so it is not a config difference. An explicit change still is."""
    import importlib.util
    import sys

    from monitordecorrelation.experiment_config import apply_overrides, load_config
    from monitordecorrelation.rl.train import RESUME_STATES_DIR

    cfg_path = _REPO / "experiments" / "configs" / "terminal_verifier_q3_cot.json"
    then = json.loads(json.dumps(apply_overrides(load_config(cfg_path), ["run_name=r", "n_steps=30"]).model_dump()))
    del then["task_reward"]  # an old snapshot
    d = tmp_path / "data/runs/r" / RESUME_STATES_DIR
    d.mkdir(parents=True)
    (d / "step_000030.json").write_text(json.dumps(
        {"config": then, "steps_done": 30, "n_steps": 30, "eval_done": True, "stopped_early": None,
         "zero_streak": 0, "state_checkpoint": "tinker://x/weights/y"}))

    spec = importlib.util.spec_from_file_location("_run_experiment", _REPO / "experiments" / "run_experiment.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    class _Stop(Exception):
        pass

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(run, "make_env", stop)  # past the resume checks = accepted
    monkeypatch.chdir(tmp_path)

    def launch(*sets: str):
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(cfg_path), "--resume",
                                          "--set", "run_name=r", "n_steps=60", *sets])
        run.main()

    with pytest.raises(_Stop):
        launch()
    with pytest.raises(SystemExit, match="task_reward"):
        launch("task_reward=false")


# ---- analysis: monitor-only runs are labelled apart, and a batch must not mix the two -------------

def _run_info(d: Path, name: str, ta: list[str], **cfg) -> Path:
    r = d / name
    r.mkdir(parents=True)
    (r / "run_info.json").write_text(json.dumps(
        {"run_name": name, "config": {"seed": 0, **cfg}, "train_against": [{"name": n} for n in ta],
         "held_out": [{"name": n} for n in ("a", "b") if n not in ta]}))
    return r


def test_penalty_label_marks_monitor_only_runs(tmp_path):
    from monitordecorrelation.eval.coupling import run_penalty

    assert run_penalty(_run_info(tmp_path, "old", ["a"], penalty_coef=1.0)) == "1"  # pre-key run
    assert run_penalty(_run_info(tmp_path, "on", ["a"], penalty_coef=1.0, task_reward=True)) == "1"
    assert run_penalty(_run_info(tmp_path, "off", ["a"], penalty_coef=1.0, task_reward=False)) == "1 monitor-only"
    assert run_penalty(_run_info(tmp_path, "ramp", ["a"], task_reward=False,
                                 penalty_schedule={"start_penalty": 0, "end_penalty": 1})) == "0→1 monitor-only"


def test_verify_runs_flags_a_batch_mixing_monitor_only_and_normal_rows(tmp_path, capsys, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_verify_runs", _REPO / "scripts" / "verify_runs.py")
    verify_runs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify_runs)

    _run_info(tmp_path, "tv_a_s0", ["a"], penalty_coef=1.0, task_reward=False)
    _run_info(tmp_path, "tv_b_s0", ["b"], penalty_coef=1.0, task_reward=False)
    monkeypatch.setattr("sys.argv", ["verify_runs.py", str(tmp_path / "tv_*")])
    assert verify_runs.main() == 0
    _run_info(tmp_path, "tv_c_s0", ["a"], penalty_coef=1.0)  # task reward on (pre-key default)
    capsys.readouterr()
    assert verify_runs.main() == 1
    assert "task_reward" in capsys.readouterr().out
