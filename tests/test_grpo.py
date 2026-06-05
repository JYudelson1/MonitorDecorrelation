"""Offline checks for GRPO advantage centering + Datum token/advantage alignment.

No network/compute. Run: uv run python tests/test_grpo.py
"""

from __future__ import annotations

from monitordecorrelation.rl.grpo import build_datum, compute_group_advantages
from monitordecorrelation.types import Prompt, Rollout


class _StubTokenizer:
    """Returns fixed prompt ids regardless of input."""

    def apply_chat_template(self, messages, **kw):
        return [10, 11, 12]


def test_advantages():
    # group_size 2, no normalize: each group centered on its mean.
    adv = compute_group_advantages([1.0, 0.0, 1.0, 0.0], group_size=2, normalize=False)
    assert adv == [0.5, -0.5, 0.5, -0.5], adv
    # normalized (std=0.5) -> +/-1
    advn = compute_group_advantages([1.0, 0.0], group_size=2, normalize=True)
    assert abs(advn[0] - 1.0) < 1e-3 and abs(advn[1] + 1.0) < 1e-3, advn
    print("advantages OK")


def test_datum_alignment():
    r = Rollout(
        prompt=Prompt(text="q"),
        cot="",
        output="",
        token_ids=[20, 21],
        logprobs=[-0.1, -0.2],
    )
    d = build_datum(_StubTokenizer(), r, advantage=0.7)
    # full = [10,11,12,20,21]; input = full[:-1]; target = full[1:]
    target = d.loss_fn_inputs["target_tokens"].to_numpy().tolist()
    logprobs = d.loss_fn_inputs["logprobs"].to_numpy().tolist()
    adv = d.loss_fn_inputs["advantages"].to_numpy().tolist()
    assert d.model_input.to_ints() == [10, 11, 12, 20], d.model_input.to_ints()
    assert target == [11, 12, 20, 21], target
    # logprobs align to targets: prompt targets get 0, completion targets get sampling logprobs
    assert all(abs(a - b) < 1e-5 for a, b in zip(logprobs, [0.0, 0.0, -0.1, -0.2])), logprobs
    # advantage assigned to completion tokens only
    assert all(abs(a - b) < 1e-5 for a, b in zip(adv, [0.0, 0.0, 0.7, 0.7])), adv
    print("datum alignment OK")


if __name__ == "__main__":
    test_advantages()
    test_datum_alignment()
    print("ALL PASS")
