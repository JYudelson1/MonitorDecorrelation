"""Monitors run *while* rollouts are still being produced, not after the batch.

The RL loop hands every finished rollout straight to the judges (``MonitorScorer.submit``), so a
fast rollout's API calls overlap the slow rollouts still generating. White-box probes are the
deliberate exception: one batched local forward at ``collect`` time, after everything has landed.
These tests pin both halves of that contract, plus the bookkeeping invariants the RL loop relies on
(results indexed exactly like the rollout list; an unscored rollout aborts the run).
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import pytest

from monitordecorrelation.config import LoggingConfig, RunConfig
from monitordecorrelation.rl.rollout import sample_rollouts
from monitordecorrelation.rl.train import MonitorScorer, run_grpo
from monitordecorrelation.types import EnvResult, MonitorResult, Prompt, Rollout


def _rollout(text: str) -> Rollout:
    return Rollout(prompt=Prompt(text=text), cot="reason", output="answer",
                   token_ids=[1, 2], logprobs=[-0.1, -0.2])


class _TimedJudge:
    """An API-style monitor (no ``score_batch``): records when each call started."""

    def __init__(self, name: str, latency: float = 0.0) -> None:
        self.name = name
        self.latency = latency
        self.times: list[float] = []
        self.seen: list[str] = []
        self._lock = threading.Lock()

    def score(self, rollout: Rollout) -> MonitorResult:
        with self._lock:
            self.times.append(time.perf_counter())
            self.seen.append(rollout.prompt.text)
        if self.latency:
            time.sleep(self.latency)
        return MonitorResult(score=0.25, label=False, meta={"q": rollout.prompt.text})


class _BatchProbe:
    """A white-box-probe-style monitor: has ``score_batch``, so it must be called ONCE with the whole
    batch (a batched local forward), never per rollout."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.batch_sizes: list[int] = []
        self.call_times: list[float] = []

    def score_batch(self, rollouts):
        self.batch_sizes.append(len(rollouts))
        self.call_times.append(time.perf_counter())
        return [MonitorResult(score=0.7, label=True) for _ in rollouts]


# --------------------------------------------------------------------------------------------
# MonitorScorer
# --------------------------------------------------------------------------------------------
def test_scorer_results_are_indexed_like_the_rollouts():
    rollouts = [_rollout(f"q{i}") for i in range(6)]
    judge, probe = _TimedJudge("j"), _BatchProbe("p")
    with MonitorScorer([judge, probe], workers=4) as sc:
        for i in reversed(range(len(rollouts))):  # out of order on purpose
            sc.submit(i, rollouts[i])
        res = sc.collect(rollouts)
    assert [r.meta["q"] for r in res["j"]] == [f"q{i}" for i in range(6)]
    assert len(res["p"]) == 6 and all(r.score == 0.7 for r in res["p"])
    assert probe.batch_sizes == [6]  # ONE batched forward, not six


def test_probe_runs_once_over_the_whole_batch_after_the_judges_were_dispatched():
    rollouts = [_rollout(f"q{i}") for i in range(4)]
    judge, probe = _TimedJudge("j", latency=0.05), _BatchProbe("p")
    with MonitorScorer([judge, probe], workers=4) as sc:
        for i, r in enumerate(rollouts):
            sc.submit(i, r)
        assert probe.batch_sizes == []  # the probe has NOT run yet — it waits for the full batch
        assert judge.times, "judges start as soon as a rollout is submitted"
        res = sc.collect(rollouts)
    assert probe.batch_sizes == [4] and len(res["j"]) == 4


def test_unsubmitted_rollout_is_a_hard_error():
    rollouts = [_rollout("a"), _rollout("b")]
    with MonitorScorer([_TimedJudge("j")], workers=2) as sc:
        sc.submit(0, rollouts[0])
        with pytest.raises(RuntimeError, match=r"rollout 1/2 was never submitted"):
            sc.collect(rollouts)


def test_double_submit_is_a_hard_error():
    r = _rollout("a")
    with MonitorScorer([_TimedJudge("j")], workers=2) as sc:
        sc.submit(0, r)
        with pytest.raises(RuntimeError, match=r"submitted twice"):
            sc.submit(0, r)


def test_failed_judge_calls_still_abort_the_run():
    class _Boom:
        name = "boom"

        def score(self, rollout):
            raise RuntimeError("simulated 404")

    rollouts = [_rollout("a"), _rollout("b")]
    with MonitorScorer([_Boom()], workers=2) as sc:
        for i, r in enumerate(rollouts):
            sc.submit(i, r)
        with pytest.raises(RuntimeError, match=r"monitor 'boom' returned NaN for 2/2 rollouts"):
            sc.collect(rollouts)


def test_skipped_rollouts_are_shown_to_no_monitor():
    """An invalid rollout (the loop's skip predicate) gets no judge call, is not in the probe's batch,
    and is ``None`` in every result list — which stays indexed like the rollouts."""
    rollouts = [_rollout(f"q{i}") for i in range(5)]
    skip = lambda r: r.prompt.text in ("q1", "q3")  # noqa: E731
    judge, probe = _TimedJudge("j"), _BatchProbe("p")
    with MonitorScorer([judge, probe], workers=4, skip=skip) as sc:
        for i, r in enumerate(rollouts):
            sc.submit(i, r)
        res = sc.collect(rollouts)
    assert sorted(judge.seen) == ["q0", "q2", "q4"]
    assert probe.batch_sizes == [3]
    for name in ("j", "p"):
        assert [r is None for r in res[name]] == [False, True, False, True, False]
    assert [r.meta["q"] for r in res["j"] if r is not None] == ["q0", "q2", "q4"]


def test_all_rollouts_skipped_calls_no_probe():
    rollouts = [_rollout("a"), _rollout("b")]
    probe = _BatchProbe("p")
    with MonitorScorer([probe], workers=2, skip=lambda r: True) as sc:
        res = sc.collect(rollouts)
    assert probe.batch_sizes == [] and res == {"p": [None, None]}


def test_no_monitors_needs_no_pool():
    rollouts = [_rollout("a")]
    with MonitorScorer([], workers=4) as sc:
        sc.submit(0, rollouts[0])  # no-op
        assert sc.collect(rollouts) == {}


# --------------------------------------------------------------------------------------------
# sample_rollouts (single-turn envs)
# --------------------------------------------------------------------------------------------
class _Seq:
    def __init__(self, tokens):
        self.tokens = tokens
        self.logprobs = [-0.5] * len(tokens)
        self.stop_reason = "stop"


class _SlowFut:
    """Latency is paid in ``result()`` — where a real sampling future's latency lives."""

    def __init__(self, seqs, delay=0.0):
        self._seqs, self._delay = seqs, delay

    def result(self):
        if self._delay:
            time.sleep(self._delay)

        class _R:
            sequences = self._seqs
        return _R()


class _FakeTok:
    def apply_chat_template(self, messages, **kw):
        return [1, 2, 3]

    def decode(self, tokens):
        return f"reasoning</think>answer{tokens[0]}"


def test_sample_rollouts_streams_without_waiting_for_the_slowest_prompt():
    """A prompt that lands early must reach ``on_rollout`` while a slow sibling is still generating,
    and the returned list must still be in prompt order."""
    SLOW = 0.5

    class _Sampler:
        def __init__(self):
            self.n = 0

        def sample(self, model_input, num_samples, params):
            self.n += 1
            delay = SLOW if self.n == 1 else 0.0  # the FIRST prompt is the straggler
            return _SlowFut([_Seq([10 + self.n, 11, 12])], delay=delay)

    seen: list[tuple[int, str, float]] = []
    lock = threading.Lock()

    def on_rollout(i, r):
        with lock:
            seen.append((i, r.output, time.perf_counter()))

    t0 = time.perf_counter()
    prompts = [Prompt(text=f"p{i}") for i in range(4)]
    rolls = sample_rollouts(_Sampler(), _FakeTok(), prompts, num_samples=1, max_tokens=8,
                            seed=1, on_rollout=on_rollout)

    assert [r.prompt.text for r in rolls] == ["p0", "p1", "p2", "p3"]  # order preserved
    assert sorted(i for i, _, _ in seen) == [0, 1, 2, 3]               # every rollout streamed
    assert {i: out for i, out, _ in seen} == {i: r.output for i, r in enumerate(rolls)}
    # the three fast prompts were delivered while prompt 0 was still generating
    fast = [t for i, _, t in seen if i != 0]
    assert len(fast) == 3 and max(fast) - t0 < SLOW


def test_sample_rollouts_without_callback_is_unchanged():
    class _Sampler:
        def sample(self, model_input, num_samples, params):
            return _SlowFut([_Seq([10, 11, 12]), _Seq([20, 21, 22])])

    rolls = sample_rollouts(_Sampler(), _FakeTok(), [Prompt(text="p")], num_samples=2, max_tokens=8)
    assert len(rolls) == 2 and rolls[0].cot == "reasoning"


# --------------------------------------------------------------------------------------------
# run_grpo end-to-end: judges fire during sampling
# --------------------------------------------------------------------------------------------
class _StreamingBackend:
    """Emits rollouts one at a time with a gap, announcing each through ``on_rollout``."""

    name = "fake_stream"

    def __init__(self, gap: float = 0.03) -> None:
        self.gap = gap
        self.sample_returned_at: list[float] = []

    def sample(self, prompts, *, num_samples=1, max_tokens=64, temperature=1.0, on_rollout=None):
        out: list[Rollout] = []
        for p in prompts:
            for _ in range(num_samples):
                time.sleep(self.gap)
                r = Rollout(prompt=p, cot="reason", output="answer",
                            token_ids=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])
                out.append(r)
                if on_rollout is not None:
                    on_rollout(len(out) - 1, r)
        self.sample_returned_at.append(time.perf_counter())
        return out

    def train_step(self, rollouts, rewards, group_size):
        assert len(rollouts) == len(rewards)
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


def test_run_grpo_dispatches_judges_during_sampling():
    run_dir = Path("data/runs/smoke_pipelined")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    backend = _StreamingBackend(gap=0.03)
    judge = _TimedJudge("j_ta")
    probe = _BatchProbe("probe_ho")
    cfg = RunConfig(
        env="fake_env", backend="fake", base_model="fake/model",
        batch_size=4, group_size=2, n_steps=1, eval_every=10, eval_size=4,
        penalty_coef=1.0, kl_coef=0.0, seed=0,
        logging=LoggingConfig(run_name="smoke_pipelined", use_wandb=False, log_fraction=1.0),
    )
    try:
        run_grpo(cfg, _FakeEnv(), backend, train_against=[judge], held_out=[probe])

        # The first judge call started before the first sampling batch had even returned — i.e. the
        # monitors are NOT queued behind the batch.
        assert judge.times, "the judge was never called"
        assert backend.sample_returned_at, "the backend never returned a batch"
        assert judge.times[0] < backend.sample_returned_at[0]

        # …and every rollout of every batch still got scored: evals at step 0 and step 1 (4x1 each),
        # plus the training step (4x2) for the train-against judge.
        assert len(judge.seen) == 4 + 8 + 4
        assert probe.batch_sizes == [4, 4]  # one batched forward per eval, never per rollout

        rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").open() if l.strip()]
        assert rows and rows[0]["reward/monitor_suspiciousness_mean"] == pytest.approx(0.25)
        ev = [json.loads(l) for l in (run_dir / "eval_metrics.jsonl").open() if l.strip()]
        assert ev and ev[0]["monitor/probe_ho/mean_score"] == pytest.approx(0.7)
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)
