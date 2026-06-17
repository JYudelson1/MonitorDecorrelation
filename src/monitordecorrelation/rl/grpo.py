"""GRPO: group-relative advantages + tinker importance-sampling Datum construction.

Backend-agnostic. ``compute_group_advantages`` turns per-rollout rewards into centered (optionally
std-normalized) advantages within each prompt group; ``build_datum`` packs one rollout into the
tinker ``importance_sampling`` format (prompt positions get advantage 0). See docs/INFRA.md.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def kl_adjusted_advantages(
    scalar_adv: float, sample_logprobs: Sequence[float], ref_logprobs: Sequence[float | None],
    kl_coef: float,
) -> tuple[list[float], float]:
    """Per-token advantages with a KL-to-reference penalty folded in: ``adv_t = A − β·(logπ_t − logπ_ref_t)``.

    The per-token KL is applied AFTER group-centering (we start from the scalar group advantage ``A``)
    so GRPO's baseline subtraction can't cancel a penalty common to the whole group — it anchors each
    token to the reference policy, which is what stops the runaway drift / collapse. ``logπ_t`` is the
    sampling (behaviour-policy) logprob; ``logπ_ref_t`` comes from ``compute_logprobs`` on the frozen
    base sampler (None where it couldn't be computed → that token gets no KL term). Returns the per-token
    advantage list (len = #completion tokens) and the mean per-token KL (for logging)."""
    out, kls = [], []
    for i, s in enumerate(sample_logprobs):
        r = ref_logprobs[i] if i < len(ref_logprobs) else None
        kl = (s - r) if r is not None else 0.0
        kls.append(kl)
        out.append(scalar_adv - kl_coef * kl)
    return out, (float(np.mean(kls)) if kls else 0.0)


def build_datum(tokenizer, rollout: Rollout, advantage: float | Sequence[float]) -> tinker.Datum:
    """Pack one rollout into a tinker Datum for the ``importance_sampling`` loss.

    Reconstructs the prompt tokens (deterministic via the same chat template used at sampling) and
    assigns the advantage to every completion token (0 to prompt tokens). ``advantage`` is either a
    scalar (broadcast to all completion tokens) or a per-completion-token sequence (e.g. with a KL
    penalty folded in — see ``kl_adjusted_advantages``). Sequences are right-shifted: input = full[:-1],
    targets = full[1:].
    """
    if rollout.token_ids is None or rollout.logprobs is None:
        raise ValueError("rollout needs token_ids + logprobs (sampling logprobs) for GRPO")

    prompt_ids = build_prompt_tokens(tokenizer, rollout.prompt.text)
    comp_ids = rollout.token_ids
    comp_logprobs = rollout.logprobs
    if len(comp_ids) != len(comp_logprobs):
        raise ValueError("completion tokens and logprobs length mismatch")

    if isinstance(advantage, (int, float)):
        comp_adv = [float(advantage)] * len(comp_ids)
    else:
        comp_adv = [float(a) for a in advantage]
        if len(comp_adv) != len(comp_ids):
            raise ValueError(f"per-token advantages ({len(comp_adv)}) != completion tokens ({len(comp_ids)})")

    full = list(prompt_ids) + list(comp_ids)
    n_prompt = len(prompt_ids)
    sampled_logprobs = [0.0] * n_prompt + list(comp_logprobs)
    advantages = [0.0] * n_prompt + comp_adv

    # Right-shift: predict full[1:] from full[:-1]; align logprobs/advantages to the targets.
    input_ids = full[:-1]
    target_ids = full[1:]
    logprobs_t = sampled_logprobs[1:]
    advantages_t = advantages[1:]
    assert len(input_ids) == len(target_ids) == len(logprobs_t) == len(advantages_t)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(np.asarray(target_ids, dtype=np.int32)),
            "logprobs": TensorData.from_numpy(np.asarray(logprobs_t, dtype=np.float32)),
            "advantages": TensorData.from_numpy(np.asarray(advantages_t, dtype=np.float32)),
        },
    )
