"""Resuming a finished run (``run_grpo(resume=…)``) must equal one uninterrupted run of the combined
length, and every eval step must save the persistent checkpoint its eval sampled. Fake backend / env /
monitor — no tinker, no network. The backend carries optimizer-like state (``opt``) that feeds its
samples, so a resume that dropped it — or the env's prompt RNG, or the log-sampling RNG — would
change the logs."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import threading
import time
from pathlib import Path

import pytest

from monitordecorrelation.config import LoggingConfig, RunConfig
from monitordecorrelation.rl.train import RESUME_STATES_DIR, load_resume_state, run_grpo
from monitordecorrelation.types import EnvResult, MonitorResult, Prompt, Rollout


class _Backend:
    name = "fake"
    resumable = True

    def __init__(self, resume_from: str | None = None, crash_at: int | None = None, async_eval=False):
        self.crash_at = crash_at  # train_step raises when about to take this optim step (a crash)
        self.async_eval = async_eval  # True: evals run in a background thread, as on tinker
        self.w, self.opt = 0, 0.0  # weights version + optimizer state; both shape what is sampled
        if resume_from is not None:
            _, w, opt = resume_from.split(":")
            self.w, self.opt = int(w), float(opt)
        self.eval_ckpts: list[str] = []

    def current_sampler(self):
        return f"w{self.w}/{self.opt:.6f}"

    def checkpoint_sampler(self, label):
        path = f"ckpt/{label}/{self.current_sampler()}"
        self.eval_ckpts.append(path)
        return self.current_sampler(), path

    @staticmethod
    def sampler_id(sampler):
        return sampler

    def sample(self, prompts, *, sampler, seed, num_samples=1, max_tokens=64, temperature=1.0):
        if self.async_eval and num_samples == 1:  # a slow background eval: still running at the next save
            time.sleep(0.3)
        out = []
        for i, p in enumerate(prompts):
            for j in range(num_samples):
                h = hashlib.sha256(f"{sampler}|{seed}|{p.text}|{i}|{j}".encode()).hexdigest()
                out.append(Rollout(prompt=p, cot=f"c{h[:6]}", output=h[:8], token_ids=[1, 2], logprobs=[-0.1, -0.2]))
        return out

    def train_step(self, rollouts, rewards, group_size):
        if self.w == self.crash_at:
            raise RuntimeError(f"simulated crash at step {self.w}")
        self.w += 1
        self.opt = round(0.9 * self.opt + sum(rewards) / len(rewards), 6)  # momentum-like
        return {"n_data": float(len(rollouts))}

    def save_checkpoint(self, label, **_):
        return f"state:{self.w}:{self.opt}"


class _Env:
    name = "fake_env"
    behavior_name = "reward_hacking"

    def __init__(self, seed=0):
        self.items = [f"task{i}" for i in range(40)]
        self._rng = random.Random(seed)

    def holdout(self, n, seed=0):
        picks = random.Random(seed).sample(self.items, n)
        self.items = [x for x in self.items if x not in picks]
        return [Prompt(text=t, meta={"task_id": t}) for t in picks]

    def sample_prompt(self):
        t = self._rng.choice(self.items)
        return Prompt(text=t, meta={"task_id": t})

    def score(self, rollout):
        hack = int(rollout.output, 16) % 3 == 0
        return EnvResult(task_reward=1.0 if hack else 0.5, behavior_present=hack, meta={})


class _Monitor:
    name = "judge"

    def score_batch(self, rollouts):
        return [MonitorResult(score=(int(r.output, 16) % 7) / 7, label=False) for r in rollouts]


def _cfg(name, n_steps, **kw):
    kw.setdefault("penalty_coef", 0.5)
    kw.setdefault("save_every", 100)
    return RunConfig(env="fake_env", backend="fake", base_model="fake/model", batch_size=3, group_size=2,
                     n_steps=n_steps, eval_every=2, eval_size=4, seed=7,
                     logging=LoggingConfig(run_name=name, use_wandb=False, log_fraction=0.5), **kw)


def _run(name, n_steps, resume=None, crash_at=None, async_eval=False, **kw):
    backend = _Backend(resume_from=resume["state_checkpoint"] if resume else None, crash_at=crash_at,
                       async_eval=async_eval)
    run_grpo(_cfg(name, n_steps, **kw), _Env(seed=7), backend, train_against=[_Monitor()], held_out=[],
             resume=resume)
    return backend


def _rows(run_dir, f):
    rows = [json.loads(l) for l in (run_dir / f).open() if l.strip()]
    return [{k: v for k, v in r.items() if not k.startswith("time/")} for r in rows]


@pytest.fixture
def dirs():
    ds = [Path("data/runs") / n for n in ("resume_test_full", "resume_test_split")]
    for d in ds:
        shutil.rmtree(d, ignore_errors=True)
    yield ds
    for d in ds:
        shutil.rmtree(d, ignore_errors=True)


def test_resume_equals_one_uninterrupted_run(dirs):
    full, split = dirs
    b_full = _run(full.name, 5)
    _run(split.name, 3)
    state = load_resume_state(split)
    assert state["steps_done"] == 3 and state["state_checkpoint"].startswith("state:3:")
    # a crashed earlier resume attempt left partial rows behind: the resume must drop them
    with (split / "metrics.jsonl").open("a") as f:
        f.write('{"step": 3, "junk": true}\n')
    b_split = _run(split.name, 5, resume=state)

    for f in ("metrics.jsonl", "rollouts.jsonl"):  # train steps + the (RNG-subsampled) rollout dump
        assert _rows(split, f) == _rows(full, f), f
    # evals: identical, except the first run's final eval at its last step (3, not a multiple of 2)
    ev_split = [r for r in _rows(split, "eval_metrics.jsonl") if r["step"] != 3]
    assert ev_split == _rows(full, "eval_metrics.jsonl")
    assert [r["step"] for r in _rows(full, "eval_metrics.jsonl")] == [0, 2, 4, 5]
    assert [r["step"] for r in _rows(split, "eval_metrics.jsonl")] == [0, 2, 3, 4, 5]  # step 3: once
    assert load_resume_state(split)["state_checkpoint"] == load_resume_state(full)["state_checkpoint"]
    assert b_split.w == b_full.w == 5

    # every eval sampled exactly the checkpoint recorded for it
    for d in (full, split):
        ck = {r["step"]: r for r in _rows(d, "eval_checkpoints.jsonl")}
        for r in _rows(d, "eval_metrics.jsonl"):
            assert ck[r["step"]]["sampler"] == r["sampler"]
            assert ck[r["step"]]["path"].endswith(r["sampler"])
    assert [r["step"] for r in _rows(split, "eval_checkpoints.jsonl")] == [0, 2, 3, 4, 5]
    info = json.loads((split / "run_info.json").read_text())
    assert [r["from_step"] for r in info["resumes"]] == [3]


def test_resume_a_finished_run_that_ended_on_an_eval_step(dirs):
    """A finished run's final eval at an eval_every multiple is the eval the longer run would have run
    there: the resume must not run it again, and the logs then equal the longer run's exactly."""
    full, split = dirs
    _run(full.name, 5)
    _run(split.name, 4)
    _run(split.name, 5, resume=load_resume_state(split))
    for f in ("metrics.jsonl", "rollouts.jsonl", "eval_metrics.jsonl"):
        assert _rows(split, f) == _rows(full, f), f
    # (checkpoint paths carry the run name; the weights they hold are the `sampler`)
    assert [(r["step"], r["sampler"]) for r in _rows(split, "eval_checkpoints.jsonl")] == \
        [(r["step"], r["sampler"]) for r in _rows(full, "eval_checkpoints.jsonl")]


@pytest.mark.parametrize("crash_at, resumed_from", [(5, 3), (7, 6)])
def test_resume_a_crashed_run_from_its_last_save_every_state(dirs, crash_at, resumed_from):
    """save_every=3, eval_every=2: crash during step 5 → resume from the step-3 state (not an eval
    step); crash during step 7 → from step 6 (an eval step: its eval, lost or not, is re-run). Evals
    run in the background, so the crashed run may or may not have logged one in flight."""
    full, split = dirs
    _run(full.name, 9, save_every=3, async_eval=True)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(split.name, 9, save_every=3, async_eval=True, crash_at=crash_at)
    # A real crash takes its background eval down with the process; here the thread outlives the
    # "crash" in the same process, so let it finish writing before resuming.
    for th in threading.enumerate():
        if th.name.startswith("eval"):
            th.join()
    state = load_resume_state(split)
    assert state["steps_done"] == resumed_from and not state["eval_done"]
    _run(split.name, 9, save_every=3, async_eval=True, resume=state)  # same n_steps as the crashed run

    for f in ("metrics.jsonl", "rollouts.jsonl", "eval_metrics.jsonl"):
        assert _rows(split, f) == _rows(full, f), f  # exactly the uninterrupted run, no extra eval
    assert [r["step"] for r in _rows(split, "eval_metrics.jsonl")] == [0, 2, 4, 6, 8, 9]
    assert [r["step"] for r in _rows(split, "eval_checkpoints.jsonl")] == [0, 2, 4, 6, 8, 9]
    assert sorted(p.name for p in (split / RESUME_STATES_DIR).iterdir()) == \
        sorted(p.name for p in (full / RESUME_STATES_DIR).iterdir()) == \
        ["step_000000.json", "step_000003.json", "step_000006.json", "step_000009.json"]
    assert load_resume_state(split)["state_checkpoint"] == load_resume_state(full)["state_checkpoint"]


def test_resume_refusals(dirs):
    _, split = dirs
    _run(split.name, 3)
    state = load_resume_state(split)
    with pytest.raises(ValueError, match="strictly greater"):
        _run(split.name, 3, resume=state)
    with pytest.raises(ValueError, match="penalty_schedule"):
        run_grpo(_cfg(split.name, 5, penalty_coef=None, penalty_schedule={"start_penalty": 0.0, "end_penalty": 1.0}),
                 _Env(seed=7), _Backend(), train_against=[_Monitor()], held_out=[], resume=state)
    shutil.rmtree(split / RESUME_STATES_DIR)
    with pytest.raises(SystemExit, match="saved\nneither|saved neither"):
        load_resume_state(split)


# ---- the launcher (experiments/run_experiment.py) --------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_CFG = _REPO / "experiments" / "configs" / "terminal_verifier_q3_out.json"


def _launch(monkeypatch, tmp_path, *args):
    """run_experiment.main() in tmp_path; returns the SystemExit message (the gating fails before
    any env / backend is built)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_run_experiment", _REPO / "experiments" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(_CFG), *args])
    with pytest.raises(SystemExit) as e:
        mod.main()
    return str(e.value)


def test_launcher_refuses_existing_run_dir_without_resume(monkeypatch, tmp_path):
    d = tmp_path / "data/runs/old"
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text("{}\n")
    msg = _launch(monkeypatch, tmp_path, "--set", "run_name=old")
    assert "already exists" in msg and "rm -rf data/runs/old" in msg
    assert (d / "metrics.jsonl").read_text() == "{}\n"  # untouched


def test_launcher_resume_needs_a_run(monkeypatch, tmp_path):
    assert "no run to resume" in _launch(monkeypatch, tmp_path, "--set", "run_name=nope", "--resume")
    (tmp_path / "data/runs/crashed").mkdir(parents=True)
    (tmp_path / "data/runs/crashed/metrics.jsonl").write_text("")
    assert "saved neither" in _launch(monkeypatch, tmp_path, "--set", "run_name=crashed", "--resume")


def test_launcher_resume_checks_config(monkeypatch, tmp_path):
    from monitordecorrelation.experiment_config import apply_overrides, load_config

    then = json.loads(json.dumps(apply_overrides(load_config(str(_CFG)), ["run_name=r", "n_steps=30"]).model_dump()))
    d = tmp_path / "data/runs/r"
    d.mkdir(parents=True)
    (d / RESUME_STATES_DIR).mkdir()
    (d / RESUME_STATES_DIR / "step_000030.json").write_text(json.dumps(
        {"config": then, "steps_done": 30, "n_steps": 30, "eval_done": True, "stopped_early": None,
         "zero_streak": 0}))
    msg = _launch(monkeypatch, tmp_path, "--set", "run_name=r", "n_steps=60", "lr=0.001", "--resume")
    assert "differs" in msg and "lr" in msg and "n_steps" not in msg.split("\n")[0]
    assert "strictly greater" in _launch(monkeypatch, tmp_path, "--set", "run_name=r", "n_steps=30", "--resume")
