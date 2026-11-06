"""Background evals in the RL loop (``rl/train.py``: ``_BackgroundEval``).

On an ``async_eval`` backend an eval runs while training continues, but: it samples the weights of the
step it is labelled with (pinned at launch, whatever training does meanwhile); evals run one at a time,
in step order; any error in one aborts the run; the run returns only once every eval is done; and
wandb plots every row at its own RL step. With ``checkpoint_sampler`` (a sampler pinned to its step's
saved weights — tinker) evals QUEUE: launching one never waits for the previous, so training runs ahead
of its evals. Without it (an eval samples the live weights) launching the next eval blocks training
until the previous is done. A backend without ``async_eval`` runs evals inline. All offline (fake
backend / env / monitor).
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from monitordecorrelation.config import LoggingConfig, RunConfig
from monitordecorrelation.rl.episodes import derive_sample_seed
from monitordecorrelation.rl.train import run_grpo
from monitordecorrelation.types import EnvResult, MonitorResult, Prompt, Rollout


class _Env:
    name = "fake_env"
    behavior_name = "reward_hacking"

    def sample_prompt(self):
        return Prompt(text="task")

    def holdout(self, n, seed=0):
        return [Prompt(text=f"eval{i}") for i in range(n)]

    def score(self, rollout):
        return EnvResult(task_reward=0.5, behavior_present=False, meta={"unparsed": False})


class _Probe:
    name = "probe"

    def score_batch(self, rollouts):
        return [MonitorResult(score=0.3, label=False) for _ in rollouts]


class _VersionedBackend:
    """Weights version = optim steps taken; ``current_sampler`` hands out ``"v<version>"``. Logs a
    timeline of (event, step-or-version) and every sampling call's (kind, sampler, seed).

    ``eval_hook(sampler)`` runs inside every eval's sampling call (in the eval thread)."""

    name = "fake_versioned"

    def __init__(self, *, async_eval: bool = True, eval_hook=None) -> None:
        self.async_eval = async_eval
        self.eval_hook = eval_hook
        self.version = 0
        self.cond = threading.Condition()
        self.timeline: list[tuple[str, int]] = []
        self.calls: list[tuple[str, str, int]] = []

    def event(self, what: str, n: int) -> None:
        with self.cond:
            self.timeline.append((what, n))
            self.cond.notify_all()

    def current_sampler(self):
        return f"v{self.version}"

    @staticmethod
    def sampler_id(sampler):
        return sampler

    def sample(self, prompts, *, sampler, seed, num_samples=1, max_tokens=64, temperature=1.0):
        kind = "eval" if prompts[0].text.startswith("eval") else "train"
        v = int(sampler[1:])
        with self.cond:
            self.calls.append((kind, sampler, seed))
        self.event(f"{kind}_start", v)
        if kind == "eval" and self.eval_hook is not None:
            self.eval_hook(self, v)
        out = [Rollout(prompt=p, cot="c", output="o", token_ids=[1], logprobs=[-0.1])
               for p in prompts for _ in range(num_samples)]
        self.event(f"{kind}_end", v)
        return out

    def train_step(self, rollouts, rewards, group_size):
        with self.cond:
            self.version += 1
            self.timeline.append(("optim", self.version))
            self.cond.notify_all()
        return {"n_data": float(len(rollouts)), "kl/mean": 0.0, "train/logprob_mean": -1.0}


def _cfg(name: str, *, n_steps: int, eval_every: int, use_wandb: bool = False) -> RunConfig:
    return RunConfig(
        env="fake_env", backend="fake", base_model="fake/model",
        batch_size=2, group_size=2, n_steps=n_steps, eval_every=eval_every, eval_size=2,
        penalty_coef=None, kl_coef=0.0, seed=5,
        logging=LoggingConfig(run_name=name, use_wandb=use_wandb, log_fraction=1.0),
    )


@pytest.fixture
def run_dir(request):
    name = f"test_async_eval_{request.node.name}"
    d = Path("data/runs") / name
    shutil.rmtree(d, ignore_errors=True)
    yield name, d
    shutil.rmtree(d, ignore_errors=True)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def test_eval_overlaps_training_and_samples_the_weights_it_was_launched_with(run_dir):
    """Each eval (but the last) waits inside its sampling call until training has taken the NEXT optim
    step — possible only if training ran on meanwhile — and must still sample the weights pinned when
    it was launched. The eval rows record exactly the sampler the train rows of the same step used."""
    name, d = run_dir
    n_steps = 4

    def wait_for_training(backend, v):
        if v < n_steps:  # the final eval has no training after it to wait for
            with backend.cond:
                assert backend.cond.wait_for(lambda: backend.version > v, timeout=10), \
                    f"training never got past step {v} while its eval ran — the eval blocked it"

    backend = _VersionedBackend(eval_hook=wait_for_training)
    run_grpo(_cfg(name, n_steps=n_steps, eval_every=2), _Env(), backend,
             train_against=[], held_out=[_Probe()])

    evals, trains = _rows(d / "eval_metrics.jsonl"), _rows(d / "metrics.jsonl")
    assert [r["step"] for r in evals] == [0, 2, 4]  # step 0, every eval_every, and after the last step
    assert [r["sampler"] for r in evals] == ["v0", "v2", "v4"]
    by_step = {r["step"]: r["sampler"] for r in trains}
    assert by_step == {0: "v0", 1: "v1", 2: "v2", 3: "v3"}
    assert all(by_step[r["step"]] == r["sampler"] for r in evals if r["step"] in by_step)
    # every sampling call got the sampler + seed of its (phase, step), whatever the thread timing
    assert sorted(backend.calls) == sorted(
        [("eval", f"v{s}", derive_sample_seed(5, "eval", s)) for s in (0, 2, 4)]
        + [("train", f"v{s}", derive_sample_seed(5, "train", s)) for s in range(n_steps)])


def test_next_eval_launch_blocks_training_until_the_previous_eval_is_done(run_dir):
    """A slow eval 0 lets train steps 1 run meanwhile, but step 2 (the next eval) may neither start its
    eval nor sample its train batch before eval 0 has finished; the run returns after the last eval."""
    name, d = run_dir

    def slow_first_eval(backend, v):
        if v == 0:
            with backend.cond:  # outlast train step 1, so step 2's launch finds eval 0 still running
                assert backend.cond.wait_for(lambda: backend.version >= 2, timeout=10)
            time.sleep(0.3)

    backend = _VersionedBackend(eval_hook=slow_first_eval)
    run_grpo(_cfg(name, n_steps=3, eval_every=2), _Env(), backend, train_against=[], held_out=[_Probe()])

    t = backend.timeline
    at = t.index
    assert at(("train_start", 1)) < at(("eval_end", 0))       # training ran on during eval 0
    assert at(("eval_end", 0)) < at(("eval_start", 2))        # eval 2 waited for eval 0 …
    assert at(("eval_end", 0)) < at(("train_start", 2))       # … and so did training
    assert t[-1] == ("eval_end", 3)                            # the run ended with the final eval


@pytest.mark.parametrize("fail_at", [0, 2])
def test_a_failed_background_eval_aborts_the_run(run_dir, fail_at):
    """E.g. an expired sampling session (tinker raises NotFoundError): the error surfaces in the
    training thread — at the latest when the next eval is launched — and the run stops."""
    name, d = run_dir

    def expire(backend, v):
        if v == fail_at:
            raise RuntimeError("Sampling session …:sample:0 not found")

    backend = _VersionedBackend(eval_hook=expire)
    with pytest.raises(RuntimeError, match="not found"):
        run_grpo(_cfg(name, n_steps=6, eval_every=2), _Env(), backend,
                 train_against=[], held_out=[_Probe()])
    # it stopped before the next eval's train batch
    assert ("train_start", fail_at + 2) not in backend.timeline


def test_a_failed_final_eval_aborts_the_run(run_dir):
    name, d = run_dir

    def expire(backend, v):
        if v == 2:
            raise RuntimeError("final eval broke")

    with pytest.raises(RuntimeError, match="final eval broke"):
        run_grpo(_cfg(name, n_steps=2, eval_every=5), _Env(), _VersionedBackend(eval_hook=expire),
                 train_against=[], held_out=[_Probe()])


def test_without_async_eval_evals_run_inline(run_dir):
    """A backend that cannot sample old weights while training (transformers: one live model) runs
    each eval to completion before anything else happens."""
    name, d = run_dir
    backend = _VersionedBackend(async_eval=False)
    run_grpo(_cfg(name, n_steps=3, eval_every=2), _Env(), backend, train_against=[], held_out=[_Probe()])
    t = backend.timeline
    for s in (0, 2):
        assert t.index(("eval_end", s)) < t.index(("train_start", s))
    assert [r["sampler"] for r in _rows(d / "eval_metrics.jsonl")] == ["v0", "v2", "v3"]


def test_wandb_plots_each_row_at_its_own_rl_step(run_dir, monkeypatch):
    """Rows are logged WITHOUT wandb's own (monotonic) `step=` — a late eval row would be dropped —
    and each namespace gets its RL step as its x-axis, so eval N plots at x=N whenever it arrives."""
    name, d = run_dir
    logged: list[tuple[dict, dict]] = []
    defined: list[tuple[tuple, dict]] = []

    class _Run:
        def define_metric(self, *a, **k):
            defined.append((a, k))

        def log(self, data, **kw):
            logged.append((data, kw))

        def finish(self):
            pass

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=lambda **k: _Run()))

    def wait_for_training(backend, v):
        if v == 0:  # make eval 0's row arrive after train rows of later steps
            with backend.cond:
                assert backend.cond.wait_for(lambda: backend.version >= 2, timeout=10)

    run_grpo(_cfg(name, n_steps=3, eval_every=3, use_wandb=True), _Env(),
             _VersionedBackend(eval_hook=wait_for_training), train_against=[], held_out=[_Probe()])

    assert (("train/*",), {"step_metric": "train/step"}) in defined
    assert (("eval/*",), {"step_metric": "eval/step"}) in defined
    assert all(kw == {} for _, kw in logged)  # never wandb's step=
    order = [("eval", data["eval/step"]) if "eval/step" in data else ("train", data["train/step"])
             for data, _ in logged]
    assert order.index(("eval", 0)) > order.index(("train", 1))  # eval 0 really arrived late
    assert sorted(order) == [("eval", 0), ("eval", 3), ("train", 0), ("train", 1), ("train", 2)]
    assert not any("sampler" in k for data, _ in logged for k in data)  # string id: jsonl only


class _PinnedBackend(_VersionedBackend):
    """Like tinker: ``checkpoint_sampler`` pins a sampler to the current weights for good."""

    name = "fake_pinned"

    def checkpoint_sampler(self, label):
        return f"v{self.version}", f"fake://{label}"


def test_with_pinned_samplers_evals_queue_and_training_never_waits(run_dir):
    """Eval 0 holds its sampling call until training has taken step 5 — past the launches of evals 2
    and 4 — so training must not wait for any eval; the evals still run in step order, one at a time,
    each on the weights of its own step, and the run ends with the final eval."""
    name, d = run_dir

    def slow_first_eval(backend, v):
        if v == 0:
            with backend.cond:
                assert backend.cond.wait_for(lambda: backend.version >= 5, timeout=10), \
                    "training waited for eval 0 — evals did not queue"

    backend = _PinnedBackend(eval_hook=slow_first_eval)
    run_grpo(_cfg(name, n_steps=6, eval_every=2), _Env(), backend, train_against=[], held_out=[_Probe()])

    t = backend.timeline
    at = t.index
    assert at(("train_start", 4)) < at(("eval_end", 0))       # training ran past two more eval launches
    starts = [n for what, n in t if what == "eval_start"]
    ends = [n for what, n in t if what == "eval_end"]
    assert starts == ends == [0, 2, 4, 6]                      # FIFO …
    for a, b in zip(ends, starts[1:]):
        assert at(("eval_end", a)) < at(("eval_start", b))    # … and one at a time
    evals = _rows(d / "eval_metrics.jsonl")
    assert [r["step"] for r in evals] == [0, 2, 4, 6]
    assert [r["sampler"] for r in evals] == ["v0", "v2", "v4", "v6"]  # pinned at launch
    assert t[-1] == ("eval_end", 6)


def test_with_pinned_samplers_a_failed_queued_eval_aborts_the_run(run_dir):
    name, d = run_dir

    def expire(backend, v):
        if v == 2:
            raise RuntimeError("judge exploded")

    backend = _PinnedBackend(eval_hook=expire)
    with pytest.raises(RuntimeError, match="judge exploded"):
        run_grpo(_cfg(name, n_steps=8, eval_every=2), _Env(), backend,
                 train_against=[], held_out=[_Probe()])
    assert [r["step"] for r in _rows(d / "eval_metrics.jsonl")] == [0]  # nothing after the failure


def test_background_eval_queue_is_fifo_and_wait_is_a_barrier():
    from monitordecorrelation.rl.train import _BackgroundEval

    done: list[int] = []
    running = []
    lock = threading.Lock()

    def job(i):
        with lock:
            running.append(i)
            assert len(running) == 1, "two evals ran at once"
        time.sleep(0.05)
        with lock:
            running.remove(i)
            done.append(i)

    ev = _BackgroundEval()
    t0 = time.perf_counter()
    for i in range(4):
        ev.launch(job, i)
    assert time.perf_counter() - t0 < 0.1  # launch never waited
    ev.wait()
    assert done == [0, 1, 2, 3]


def test_background_eval_skips_after_a_failure_and_reraises_on_check():
    from monitordecorrelation.rl.train import _BackgroundEval

    ran: list[int] = []

    def job(i):
        if i == 1:
            raise ValueError("boom")
        ran.append(i)

    ev = _BackgroundEval()
    for i in range(3):
        ev.launch(job, i)
    with pytest.raises(ValueError, match="boom"):
        ev.wait()
    assert ran == [0]
    with pytest.raises(ValueError, match="boom"):
        ev.check()
