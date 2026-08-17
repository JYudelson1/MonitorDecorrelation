"""Token accounting: sampling records per-rollout prompt/completion lengths, and the loop's
``tokens/`` metrics aggregate them (batch total + per-rollout mean + truncation)."""

from __future__ import annotations

import tinker

from monitordecorrelation.rl.rollout import sample_rollouts
from monitordecorrelation.rl.train import _token_metrics
from monitordecorrelation.types import Prompt, Rollout


def _rollout(n_in: int, n_out: int) -> Rollout:
    return Rollout(prompt=Prompt(text="q"), cot="", output="",
                   meta={"n_prompt_tokens": n_in, "n_output_tokens": n_out})


def test_token_metrics_totals_and_means():
    m = _token_metrics([_rollout(100, 10), _rollout(100, 512), _rollout(100, 512)], max_tokens=512)
    assert m["tokens/input_total"] == 300
    assert m["tokens/output_total"] == 1034
    assert m["tokens/total"] == 1334
    assert m["tokens/input_per_rollout"] == 100
    assert m["tokens/output_per_rollout"] == (10 + 512 + 512) / 3
    assert m["tokens/output_max"] == 512
    # two of three completions hit max_tokens → the length metric is censored for them
    assert m["tokens/truncated_rate"] == 2 / 3


def test_token_metrics_absent_when_counts_missing():
    """Rollouts reloaded from disk carry no token counts — no metrics rather than zeros."""
    assert _token_metrics([Rollout(prompt=Prompt(text="q"), cot="", output="")], max_tokens=512) == {}


class _Seq:
    def __init__(self, tokens):
        self.tokens = tokens
        self.logprobs = None
        self.stop_reason = "stop"


class _Fut:
    def __init__(self, sequences):
        self._sequences = sequences

    def result(self):
        return type("Resp", (), {"sequences": self._sequences})()


class _FakeSampler:
    """Returns two fixed-length completions per prompt; asserts the prompt length it was handed."""

    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len

    def sample(self, model_input, num_samples, params):
        assert model_input.length == self.prompt_len
        return _Fut([_Seq([1] * 7), _Seq([2] * 3)])


class _FakeRenderer:
    name = "fake"
    stop_tokens = None

    def model_input(self, text: str) -> tinker.ModelInput:
        return tinker.ModelInput.from_ints([1, 2, 3, 4, 5])

    def parse(self, tokens):
        return "cot", "ans", "raw"


def test_sample_rollouts_records_token_counts():
    rollouts = sample_rollouts(_FakeSampler(5), _FakeRenderer(),
                               [Prompt(text="a"), Prompt(text="b")], num_samples=2)
    counts = [(r.meta["n_prompt_tokens"], r.meta["n_output_tokens"]) for r in rollouts]
    # prompt length is per-prompt (shared by its group); output length is per-sequence
    assert counts == [(5, 7), (5, 3), (5, 7), (5, 3)]
    assert _token_metrics(rollouts, max_tokens=8)["tokens/total"] == 4 * 5 + 2 * (7 + 3)
