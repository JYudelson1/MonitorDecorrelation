"""Adapter from our ``Rollout``s to tinker-cookbook RL data structures.

We keep our own training loop (it owns the experiment-specific bits — monitor-as-reward, the held-out
all-monitor eval, the degradation grids), but delegate ALL the loss-layer bookkeeping to the cookbook
primitives: advantage centering (``compute_advantages``), mask + datum assembly
(``assemble_training_data``), the KL penalty (``incorporate_kl_penalty``), and the mask-stripped,
pipelined ``forward_backward``/``optim_step`` (``rl.train.train_step``). The point is that we never hand-
build a mask, a datum, or an advantage — those are exactly the places subtle bugs hide.

Our only job here: wrap each rollout (prompt tokens, sampled completion tokens+logprobs, scalar reward)
into the cookbook's ``Trajectory``/``TrajectoryGroup`` so those primitives can take over.
"""

from __future__ import annotations

import math

import tinker
import torch
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, Transition, TrajectoryGroup

from monitordecorrelation.rl.renderers import as_renderer
from monitordecorrelation.types import Rollout


def to_trajectory_groups(
    renderer, rollouts: list[Rollout], rewards: list[float], group_size: int
) -> list[TrajectoryGroup]:
    """Group our flat rollouts (``group_size`` consecutive rollouts per prompt) into cookbook
    ``TrajectoryGroup``s. Each rollout is a single-turn trajectory: observation = the prompt tokens,
    action = the sampled completion tokens+logprobs, reward = our scalar reward (task − penalty·monitor).
    The group-level reward is 0 (the whole reward is the per-step reward); ``compute_advantages`` centres
    rewards within each group.

    ``renderer`` must be the SAME renderer that sampled these rollouts (or a bare tokenizer, wrapped
    as HF-chat) — the observation has to re-render to the exact prompt tokens the policy saw."""
    if len(rollouts) != len(rewards):
        raise ValueError(f"rollouts ({len(rollouts)}) and rewards ({len(rewards)}) length mismatch")
    if len(rollouts) % group_size != 0:
        raise ValueError(f"{len(rollouts)} rollouts not divisible by group_size {group_size}")

    rend = as_renderer(renderer)
    groups: list[TrajectoryGroup] = []
    for start in range(0, len(rollouts), group_size):
        trajs: list[Trajectory] = []
        for r, rew in zip(rollouts[start : start + group_size], rewards[start : start + group_size]):
            multi = (r.meta or {}).get("transitions")
            if multi:
                # Multi-turn episode (rl/episodes.py): one transition per turn, observation tokens
                # recorded verbatim at sampling time. The scalar episode reward sits on the LAST
                # transition (cookbook sums per-transition rewards into the trajectory reward), and
                # because each ob prefix-extends the previous ob+ac the cookbook emits one datum.
                trans = []
                for i, tr in enumerate(multi):
                    ob = tinker.ModelInput.from_ints(list(tr["ob"]))
                    ac = TokensWithLogprobs(tokens=list(tr["ac"]), maybe_logprobs=list(tr["logprobs"]))
                    last = i == len(multi) - 1
                    trans.append(Transition(ob=ob, ac=ac, reward=float(rew) if last else 0.0,
                                            episode_done=last))
                trajs.append(Trajectory(transitions=trans, final_ob=trans[-1].ob))
                continue
            if r.token_ids is None or r.logprobs is None:
                raise ValueError("rollout needs token_ids + logprobs (sampling logprobs) for GRPO")
            ob = rend.model_input(r.prompt.text)
            ac = TokensWithLogprobs(tokens=list(r.token_ids), maybe_logprobs=list(r.logprobs))
            trajs.append(
                Trajectory(
                    transitions=[Transition(ob=ob, ac=ac, reward=float(rew), episode_done=True)],
                    final_ob=ob,  # unused by assemble_training_data (single-turn); kept for the schema
                )
            )
        groups.append(
            TrajectoryGroup(
                trajectories_G=trajs,
                final_rewards_G=[0.0] * len(trajs),
                metrics_G=[{} for _ in trajs],
            )
        )
    return groups


def optim_metrics(
    data_D: list[tinker.Datum], training_logprobs_D: list[torch.Tensor], advantages_P: list[torch.Tensor]
) -> dict[str, float]:
    """Debugging metrics for one GRPO update, from what the update already produced — no extra calls.

    ``training_logprobs_D`` are the per-token logprobs the trainer's forward pass returned for
    ``data_D`` (cookbook ``train_step``'s return value). With one substep that forward runs on the
    PRE-update weights, i.e. the same weights that sampled the batch, so:

    - ``loss`` is exactly tinker's ``importance_sampling`` objective, ``-(exp(train_lp - sample_lp) *
      advantages).sum()`` over every target token (advantages are 0 off the action tokens) — the value
      tinker reports as ``loss:sum``, which cookbook ``train_step`` drops. Because advantages are
      group-centred and the ratio ≈ 1 on-policy, it is ≈ ``-Σ A·len``: it tracks the length/advantage
      correlation, not convergence. ``loss_per_token`` divides by the action-token count.
    - ``ratio_*`` (the IS ratio on action tokens) and ``kl_sample_train_*`` measure sampler/trainer
      mismatch; on-policy they should sit at ≈1 / ≈0 — drift means the gradient is off-policy.
    - ``entropy`` is the sampling entropy estimate (−mean sampled logprob); collapse → mode collapse.
    - ``adv/*`` + ``frac_zero_adv_groups``: how much learning signal the batch carried. A group whose
      rewards are all equal contributes zero gradient; if most are, the step is mostly a no-op.
    """
    from tinker_cookbook.rl.metrics import compute_kl_sample_train

    out: dict[str, float] = {}
    loss_sum, ratios, n_tokens = 0.0, [], 0
    for datum, train_lp in zip(data_D, training_logprobs_D, strict=True):
        samp_lp = datum.loss_fn_inputs["logprobs"].to_torch()
        adv = datum.loss_fn_inputs["advantages"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        ratio = torch.exp(train_lp - samp_lp)
        loss_sum += float(-(ratio * adv).sum())
        ratios.append(ratio[mask])
        n_tokens += len(samp_lp)
    ratio_all = torch.cat(ratios) if ratios else torch.empty(0)
    n_action = int(ratio_all.numel())
    out["loss"] = loss_sum
    out["loss_per_token"] = loss_sum / n_action if n_action else math.nan
    if n_action:
        out["ratio_mean"] = float(ratio_all.mean())
        out["ratio_min"] = float(ratio_all.min())
        out["ratio_max"] = float(ratio_all.max())
        out["ratio_std"] = float(ratio_all.std()) if n_action > 1 else 0.0
        out.update({k.removeprefix("optim/"): v
                    for k, v in compute_kl_sample_train(data_D, training_logprobs_D).items()})
    out["n_action_tokens"] = float(n_action)
    out["n_tokens"] = float(n_tokens)
    out["action_tokens_per_datum"] = n_action / len(data_D) if data_D else math.nan

    adv_all = torch.cat([a.flatten().float() for a in advantages_P]) if advantages_P else torch.empty(0)
    if adv_all.numel():
        out["adv/abs_mean"] = float(adv_all.abs().mean())
        out["adv/std"] = float(adv_all.std()) if adv_all.numel() > 1 else 0.0
        out["adv/min"] = float(adv_all.min())
        out["adv/max"] = float(adv_all.max())
        out["adv/frac_zero"] = float((adv_all == 0).float().mean())
        out["frac_zero_adv_groups"] = sum(bool((a == 0).all()) for a in advantages_P) / len(advantages_P)
    return out
