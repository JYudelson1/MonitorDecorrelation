"""GRPO: group-relative advantages + tinker importance-sampling Datum construction.

Backend-agnostic. ``compute_group_advantages`` turns per-rollout rewards into centered (optionally
std-normalized) advantages within each prompt group; ``build_datum`` packs one rollout into the
tinker ``importance_sampling`` format (prompt positions get advantage 0). See docs/INFRA.md.
"""

from __future__ import annotations

import numpy as np

import tinker
from tinker import TensorData

from monitordecorrelation.rl.rollout import build_prompt_tokens
from monitordecorrelation.types import Rollout


def compute_group_advantages(
    rewards: list[float], group_size: int, *, normalize: bool = True, eps: float = 1e-6
) -> list[float]:
    """Center rewards within each contiguous group of ``group_size``.

    ``normalize=True`` divides by the group std (vanilla GRPO); ``False`` is the DR-GRPO variant.
    """
    if len(rewards) % group_size != 0:
        raise ValueError(f"{len(rewards)} rewards not divisible by group_size {group_size}")
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    adv = r - r.mean(axis=1, keepdims=True)
    if normalize:
        adv = adv / (r.std(axis=1, keepdims=True) + eps)
    return adv.reshape(-1).tolist()


def build_datum(tokenizer, rollout: Rollout, advantage: float) -> tinker.Datum:
    """Pack one rollout into a tinker Datum for the ``importance_sampling`` loss.

    Reconstructs the prompt tokens (deterministic via the same chat template used at sampling) and
    assigns ``advantage`` to every completion token (0 to prompt tokens). Also emits a ``mask`` (1 on
    completion target positions, 0 on prompt) — the cookbook KL primitive
    (``tinker_cookbook.rl.metrics.incorporate_kl_penalty``) uses it to add a per-token KL-to-base
    penalty to the advantages in-place. Sequences are right-shifted: input = full[:-1], targets = full[1:].
    """
    if rollout.token_ids is None or rollout.logprobs is None:
        raise ValueError("rollout needs token_ids + logprobs (sampling logprobs) for GRPO")

    prompt_ids = build_prompt_tokens(tokenizer, rollout.prompt.text)
    comp_ids = rollout.token_ids
    comp_logprobs = rollout.logprobs
    if len(comp_ids) != len(comp_logprobs):
        raise ValueError("completion tokens and logprobs length mismatch")

    full = list(prompt_ids) + list(comp_ids)
    n_prompt = len(prompt_ids)
    sampled_logprobs = [0.0] * n_prompt + list(comp_logprobs)
    advantages = [0.0] * n_prompt + [float(advantage)] * len(comp_ids)
    mask = [0.0] * n_prompt + [1.0] * len(comp_ids)  # 1 on completion tokens (for the KL primitive)

    # Right-shift: predict full[1:] from full[:-1]; align logprobs/advantages/mask to the targets.
    input_ids = full[:-1]
    target_ids = full[1:]
    logprobs_t = sampled_logprobs[1:]
    advantages_t = advantages[1:]
    mask_t = mask[1:]
    assert len(input_ids) == len(target_ids) == len(logprobs_t) == len(advantages_t) == len(mask_t)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(np.asarray(target_ids, dtype=np.int32)),
            "logprobs": TensorData.from_numpy(np.asarray(logprobs_t, dtype=np.float32)),
            "advantages": TensorData.from_numpy(np.asarray(advantages_t, dtype=np.float32)),
            "mask": TensorData.from_numpy(np.asarray(mask_t, dtype=np.float32)),
        },
    )
