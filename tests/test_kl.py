"""KL-to-reference penalty: per-token advantage math + Datum packing. Offline (no tinker service)."""

from __future__ import annotations

import numpy as np

from monitordecorrelation.rl.grpo import build_datum, kl_adjusted_advantages


def test_kl_adjusted_advantages_basic():
    # adv_t = A − β·(logπ_t − logπ_ref_t)
    A = 2.0
    sample = [-1.0, -2.0, -0.5]
    ref = [-1.5, -1.0, -0.5]  # kl = [+0.5, -1.0, 0.0]
    adv, kl_mean = kl_adjusted_advantages(A, sample, ref, kl_coef=0.1)
    np.testing.assert_allclose(adv, [2.0 - 0.1 * 0.5, 2.0 - 0.1 * -1.0, 2.0 - 0.1 * 0.0])
    assert abs(kl_mean - np.mean([0.5, -1.0, 0.0])) < 1e-9


def test_kl_zero_coef_is_identity():
    adv, _ = kl_adjusted_advantages(1.5, [-1.0, -2.0], [-0.2, -0.3], kl_coef=0.0)
    assert adv == [1.5, 1.5]  # β=0 → plain scalar advantage on every token


def test_kl_handles_none_ref():
    # None ref logprob (couldn't be computed) → that token gets no KL term
    adv, kl_mean = kl_adjusted_advantages(1.0, [-1.0, -2.0], [None, -1.0], kl_coef=0.5)
    assert adv[0] == 1.0  # None → kl 0
    assert abs(adv[1] - (1.0 - 0.5 * (-2.0 - -1.0))) < 1e-9


class _Tok:
    def apply_chat_template(self, messages, **kw):
        return "p p"  # 2 "prompt tokens" after the stub __call__ split? build_prompt_tokens uses tokenize


def test_build_datum_accepts_per_token_advantages():
    """build_datum must accept a per-token advantage vector (len = #completion tokens) as well as a
    scalar, and assign 0 to prompt positions."""
    from monitordecorrelation.types import Prompt, Rollout

    # minimal tokenizer: apply_chat_template returns token ids when tokenize=True
    class Tok:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kw):
            return [1, 2, 3]  # 3 prompt tokens

    r = Rollout(prompt=Prompt(text="q"), cot="", output="a",
                token_ids=[10, 11], logprobs=[-0.5, -0.7])  # 2 completion tokens

    d_scalar = build_datum(Tok(), r, 2.0)
    d_vec = build_datum(Tok(), r, [2.0, -1.0])
    av_scalar = d_scalar.loss_fn_inputs["advantages"].to_numpy()
    av_vec = d_vec.loss_fn_inputs["advantages"].to_numpy()
    # full=5 tokens, right-shifted → 4 target positions; last 2 are completion advantages
    assert list(av_scalar[-2:]) == [2.0, 2.0]
    assert list(av_vec[-2:]) == [2.0, -1.0]
    assert list(av_scalar[:-2]) == [0.0, 0.0]  # prompt positions get 0
