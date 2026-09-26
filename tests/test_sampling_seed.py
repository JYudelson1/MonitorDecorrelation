"""Seed plumbing: the per-call seed derivation is deterministic + varied, and sample_rollouts puts the
seed on tinker's SamplingParams. Offline (fake sampling client + tokenizer; no tinker service)."""

from __future__ import annotations

from monitordecorrelation.backends.tinker_backend import derive_sample_seed
from monitordecorrelation.rl.rollout import sample_rollouts
from monitordecorrelation.types import Prompt


def test_derive_sample_seed_deterministic_and_varied():
    # a pure function of its parts: same parts -> same seed; any part changed -> a different seed
    assert derive_sample_seed(7, "eval", 3) == derive_sample_seed(7, "eval", 3)
    assert derive_sample_seed(7, "eval", 3) != derive_sample_seed(7, "train", 3)
    assert derive_sample_seed(7, "eval", 3) != derive_sample_seed(7, "eval", 4)
    assert derive_sample_seed(0, 5) != derive_sample_seed(1, 5)
    assert derive_sample_seed(1, 23) != derive_sample_seed(12, 3)  # parts are not just concatenated
    # a whole run's worth of calls are distinct (no collision over a typical run length)
    seeds = [derive_sample_seed(42, "train", step, g, k, c)
             for step in range(10) for g in range(8) for k in range(4) for c in range(4)]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < 2**31 - 1 for s in seeds)


def test_derive_sample_seed_is_stable_across_processes():
    """Not Python's per-process randomized str hash: a run must reproduce in a fresh interpreter."""
    import os
    import subprocess
    import sys

    code = ("from monitordecorrelation.rl.episodes import derive_sample_seed; "
            "print(derive_sample_seed(3, 'eval', 12, 1, 2, 0))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout.strip() for h in (1, 2)}
    assert outs == {str(derive_sample_seed(3, "eval", 12, 1, 2, 0))}


class _FakeSeq:
    tokens = [10, 11, 12]
    logprobs = None
    stop_reason = 0


class _FakeFuture:
    def __init__(self, n):
        self.n = n

    def result(self):
        class _R:
            sequences = [_FakeSeq() for _ in range(self.n)]
        return _R()


class _FakeSampler:
    def __init__(self):
        self.captured_seeds = []
        self.captured_n = []

    def sample(self, model_input, num_samples, params):
        self.captured_seeds.append(params.seed)
        self.captured_n.append(num_samples)
        return _FakeFuture(num_samples)


class _FakeTok:
    def apply_chat_template(self, messages, **kw):
        return [1, 2, 3]

    def decode(self, tokens):
        return "reasoning</think>answer"


def test_sample_rollouts_threads_seed_to_params_for_single_samples():
    sampler = _FakeSampler()
    rolls = sample_rollouts(sampler, _FakeTok(), [Prompt(text="hi")], num_samples=1,
                            max_tokens=8, temperature=1.0, seed=12345)
    assert sampler.captured_seeds == [derive_sample_seed(12345, 0, 0, 0)]  # per-sample derived, even for n=1
    assert rolls and rolls[0].output == "answer" and rolls[0].cot == "reasoning"


def test_seeded_group_is_one_request_per_sample_with_distinct_seeds():
    """A seeded n-sample request collapses the GRPO group (tinker seeds the whole request), so a seeded
    group must fan out into num_samples single-sample requests, each with its own derived seed."""
    sampler = _FakeSampler()
    rolls = sample_rollouts(sampler, _FakeTok(), [Prompt(text="a"), Prompt(text="b")], num_samples=4,
                            max_tokens=8, temperature=1.0, seed=99)
    assert len(rolls) == 8 and sampler.captured_n == [1] * 8
    assert len(set(sampler.captured_seeds)) == 8 and None not in sampler.captured_seeds
    assert sampler.captured_seeds[0] == derive_sample_seed(99, 0, 0, 0)
    assert sampler.captured_seeds[5] == derive_sample_seed(99, 1, 1, 0)  # prompt 1, sample 1, call 0


def test_unseeded_group_is_still_one_single_sample_request_per_completion():
    """No request ever asks for more than one sample — seeded or not (an n-sample request is what
    collapses a seeded group)."""
    sampler = _FakeSampler()
    rolls = sample_rollouts(sampler, _FakeTok(), [Prompt(text="hi"), Prompt(text="yo")], num_samples=4,
                            max_tokens=8, temperature=1.0, seed=None)
    assert len(rolls) == 8 and sampler.captured_n == [1] * 8 and sampler.captured_seeds == [None] * 8
    assert [r.prompt.text for r in rolls] == ["hi"] * 4 + ["yo"] * 4


def test_streaming_path_keeps_group_order_with_per_sample_requests():
    sampler = _FakeSampler(); seen = []
    rolls = sample_rollouts(sampler, _FakeTok(), [Prompt(text="a"), Prompt(text="b")], num_samples=3,
                            max_tokens=8, temperature=1.0, seed=5, on_rollout=lambda i, r: seen.append(i))
    assert [r.prompt.text for r in rolls] == ["a"] * 3 + ["b"] * 3 and sorted(seen) == list(range(6))
