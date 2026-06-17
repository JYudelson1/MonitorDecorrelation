"""KL penalty plumbing. The KL math itself is the tinker-cookbook primitive
(``incorporate_kl_penalty``, needs a live base sampling client — not unit-tested offline). Here we pin
the one thing build_datum must provide for it: a ``mask`` that is 1 on completion target positions,
0 on prompt positions, aligned with advantages/logprobs."""

from __future__ import annotations

from monitordecorrelation.rl.grpo import build_datum
from monitordecorrelation.types import Prompt, Rollout


class _Tok:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kw):
        return [1, 2, 3]  # 3 prompt tokens


def test_build_datum_emits_completion_mask():
    r = Rollout(prompt=Prompt(text="q"), cot="", output="a",
                token_ids=[10, 11], logprobs=[-0.5, -0.7])  # 2 completion tokens
    d = build_datum(_Tok(), r, 2.0)
    fi = d.loss_fn_inputs
    mask = list(fi["mask"].to_numpy())
    adv = list(fi["advantages"].to_numpy())
    # full = 3 prompt + 2 comp = 5 tokens → right-shift → 4 target positions; last 2 are completion
    assert mask == [0.0, 0.0, 1.0, 1.0]
    assert adv == [0.0, 0.0, 2.0, 2.0]          # advantage only on completion tokens
    assert len(mask) == len(adv) == len(fi["logprobs"].to_numpy()) == len(fi["target_tokens"].to_numpy())


def test_build_datum_rejects_length_mismatch():
    r = Rollout(prompt=Prompt(text="q"), cot="", output="a", token_ids=[10, 11], logprobs=[-0.5])
    try:
        build_datum(_Tok(), r, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on token/logprob length mismatch")
