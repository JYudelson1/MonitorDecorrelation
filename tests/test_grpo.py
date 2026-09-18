"""Offline check of the RL data path: our rollout→TrajectoryGroup adapter feeding the cookbook
primitives (compute_advantages + assemble_training_data). No network/compute — these primitives are
pure torch/data, so we can verify the whole pipeline produces correctly centered advantages, masks, and
token alignment. Run: uv run pytest tests/test_grpo.py -q
"""

from __future__ import annotations

import math

from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages

from monitordecorrelation.rl.grpo import optim_metrics, to_trajectory_groups
from monitordecorrelation.types import Prompt, Rollout


class _StubTokenizer:
    """apply_chat_template returns fixed prompt ids (3 tokens) regardless of input."""

    def apply_chat_template(self, messages, **kw):
        return [10, 11, 12]


def _rollout(tokens, logprobs):
    return Rollout(prompt=Prompt(text="q"), cot="", output="", token_ids=tokens, logprobs=logprobs)


def test_adapter_shapes():
    rollouts = [_rollout([20, 21], [-0.1, -0.2]), _rollout([30], [-0.3])]
    groups = to_trajectory_groups(_StubTokenizer(), rollouts, [1.0, 0.0], group_size=2)
    assert len(groups) == 1 and len(groups[0].trajectories_G) == 2
    assert groups[0].get_total_rewards() == [1.0, 0.0]  # our scalar reward flows through


def test_adapter_rejects_bad_grouping():
    rollouts = [_rollout([20], [-0.1])]
    for bad in (lambda: to_trajectory_groups(_StubTokenizer(), rollouts, [1.0, 2.0], 1),  # len mismatch
                lambda: to_trajectory_groups(_StubTokenizer(), rollouts, [1.0], 2)):       # not divisible
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_full_pipeline_advantages_mask_alignment():
    """The cookbook primitives, fed by our adapter, must: center rewards within the group, assign the
    advantage to completion tokens (0 on prompt), and emit a matching mask + right-shifted targets."""
    rollouts = [_rollout([20, 21], [-0.1, -0.2]), _rollout([20, 21], [-0.4, -0.5])]
    groups = to_trajectory_groups(_StubTokenizer(), rollouts, [1.0, 0.0], group_size=2)

    advantages_P = compute_advantages(groups)
    assert [round(float(a), 4) for a in advantages_P[0]] == [0.5, -0.5]  # centered on mean 0.5

    data_D, meta = assemble_training_data(groups, advantages_P)
    assert len(data_D) == 2
    d0 = data_D[0]
    # full = [10,11,12,20,21]; input = full[:-1]; targets = full[1:]
    assert d0.model_input.to_ints() == [10, 11, 12, 20]
    assert d0.loss_fn_inputs["target_tokens"].to_numpy().tolist() == [11, 12, 20, 21]
    # advantage 0.5 on the 2 completion target positions, 0 on prompt positions
    assert d0.loss_fn_inputs["advantages"].to_numpy().tolist() == [0.0, 0.0, 0.5, 0.5]
    # mask 1 on completion, 0 on prompt (built by the cookbook, not us)
    assert d0.loss_fn_inputs["mask"].to_numpy().tolist() == [0.0, 0.0, 1.0, 1.0]
    # sampling logprobs align to completion targets
    lp = d0.loss_fn_inputs["logprobs"].to_numpy().tolist()
    assert all(abs(a - b) < 1e-5 for a, b in zip(lp, [0.0, 0.0, -0.1, -0.2])), lp



def _pipeline(rewards):
    rollouts = [_rollout([20, 21], [-0.1, -0.2]), _rollout([20, 21], [-0.4, -0.5])]
    groups = to_trajectory_groups(_StubTokenizer(), rollouts, rewards, group_size=2)
    advantages_P = compute_advantages(groups)
    data_D, _ = assemble_training_data(groups, advantages_P)
    return data_D, advantages_P


def test_optim_metrics_on_policy():
    """Trainer logprobs == sampling logprobs (on-policy): ratio is exactly 1, sample/train KL is 0, and
    the loss is tinker's IS objective -(ratio·A).sum() = -(2·0.5 + 2·(-0.5)) = 0."""
    data_D, advantages_P = _pipeline([1.0, 0.0])
    train_lp = [d.loss_fn_inputs["logprobs"].to_torch() for d in data_D]
    m = optim_metrics(data_D, train_lp, advantages_P)
    assert abs(m["loss"]) < 1e-6 and abs(m["loss_per_token"]) < 1e-6
    assert abs(m["ratio_mean"] - 1) < 1e-6 and abs(m["ratio_max"] - 1) < 1e-6
    assert abs(m["kl_sample_train_v1"]) < 1e-6 and abs(m["kl_sample_train_v2"]) < 1e-6
    assert abs(m["entropy"] - 0.3) < 1e-5  # -mean(-0.1, -0.2, -0.4, -0.5)
    assert m["n_action_tokens"] == 4 and m["n_tokens"] == 8 and m["action_tokens_per_datum"] == 2
    assert m["adv/abs_mean"] == 0.5 and m["adv/frac_zero"] == 0 and m["frac_zero_adv_groups"] == 0


def test_optim_metrics_off_policy_loss_and_dead_groups():
    """Raising the trainer's logprob on datum 0's action tokens by log 2 doubles their ratio, so the
    loss becomes -(2·2·0.5 + 2·1·(-0.5)) = -1. A constant-reward group carries no advantage."""
    data_D, advantages_P = _pipeline([1.0, 0.0])
    train_lp = [d.loss_fn_inputs["logprobs"].to_torch().clone() for d in data_D]
    mask0 = data_D[0].loss_fn_inputs["mask"].to_torch() > 0
    train_lp[0][mask0] += math.log(2)
    m = optim_metrics(data_D, train_lp, advantages_P)
    assert abs(m["loss"] + 1.0) < 1e-5 and abs(m["loss_per_token"] + 0.25) < 1e-5
    assert abs(m["ratio_max"] - 2) < 1e-5 and abs(m["ratio_min"] - 1) < 1e-5

    data_D, advantages_P = _pipeline([1.0, 1.0])
    train_lp = [d.loss_fn_inputs["logprobs"].to_torch() for d in data_D]
    m = optim_metrics(data_D, train_lp, advantages_P)
    assert m["frac_zero_adv_groups"] == 1.0 and m["adv/frac_zero"] == 1.0 and m["loss"] == 0.0
