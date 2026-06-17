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

import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, Transition, TrajectoryGroup

from monitordecorrelation.rl.rollout import build_prompt_tokens
from monitordecorrelation.types import Rollout


def to_trajectory_groups(
    tokenizer, rollouts: list[Rollout], rewards: list[float], group_size: int
) -> list[TrajectoryGroup]:
    """Group our flat rollouts (``group_size`` consecutive rollouts per prompt) into cookbook
    ``TrajectoryGroup``s. Each rollout is a single-turn trajectory: observation = the prompt tokens,
    action = the sampled completion tokens+logprobs, reward = our scalar reward (task − penalty·monitor).
    The group-level reward is 0 (the whole reward is the per-step reward); ``compute_advantages`` centres
    rewards within each group."""
    if len(rollouts) != len(rewards):
        raise ValueError(f"rollouts ({len(rollouts)}) and rewards ({len(rewards)}) length mismatch")
    if len(rollouts) % group_size != 0:
        raise ValueError(f"{len(rollouts)} rollouts not divisible by group_size {group_size}")

    groups: list[TrajectoryGroup] = []
    for start in range(0, len(rollouts), group_size):
        trajs: list[Trajectory] = []
        for r, rew in zip(rollouts[start : start + group_size], rewards[start : start + group_size]):
            if r.token_ids is None or r.logprobs is None:
                raise ValueError("rollout needs token_ids + logprobs (sampling logprobs) for GRPO")
            ob = tinker.ModelInput.from_ints(build_prompt_tokens(tokenizer, r.prompt.text))
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
