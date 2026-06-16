"""Seed plumbing: the per-call seed derivation is deterministic + varied, and sample_rollouts puts the
seed on tinker's SamplingParams. Offline (fake sampling client + tokenizer; no tinker service)."""

from __future__ import annotations

from monitordecorrelation.backends.tinker_backend import derive_sample_seed
from monitordecorrelation.rl.rollout import sample_rollouts
from monitordecorrelation.types import Prompt


def test_derive_sample_seed_deterministic_and_varied():
    # same (base, index) -> same seed; different index -> (essentially always) different seed
    assert derive_sample_seed(7, 3) == derive_sample_seed(7, 3)
    assert derive_sample_seed(7, 3) != derive_sample_seed(7, 4)
    assert derive_sample_seed(0, 5) != derive_sample_seed(1, 5)
    # a whole run's worth of calls are distinct (no collision over a typical run length)
    seeds = [derive_sample_seed(42, i) for i in range(500)]
    assert len(set(seeds)) == 500
    assert all(0 <= s < 2**31 - 1 for s in seeds)


class _FakeSeq:
    tokens = [10, 11, 12]
    logprobs = None
    stop_reason = 0


class _FakeFuture:
    def result(self):
        class _R:
            sequences = [_FakeSeq()]
        return _R()


class _FakeSampler:
    def __init__(self):
        self.captured_seeds = []

    def sample(self, model_input, num_samples, params):
        self.captured_seeds.append(params.seed)
        return _FakeFuture()


class _FakeTok:
    def apply_chat_template(self, messages, **kw):
        return [1, 2, 3]

    def decode(self, tokens):
        return "reasoning</think>answer"


def test_sample_rollouts_threads_seed_to_params():
    sampler = _FakeSampler()
    rolls = sample_rollouts(sampler, _FakeTok(), [Prompt(text="hi")], num_samples=1,
                            max_tokens=8, temperature=1.0, seed=12345)
    assert sampler.captured_seeds == [12345]
    assert rolls and rolls[0].output == "answer" and rolls[0].cot == "reasoning"
