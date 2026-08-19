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

from monitordecorrelation.rl.renderers import as_renderer
from monitordecorrelation.types import Rollout


def _transitions(rend, r: Rollout, reward: float) -> list[Transition]:
    """The rollout as cookbook transitions: normally one, but one per segment when generation was
    interrupted and resumed (today: a thinking budget force-closing ``<think>``).

    The point of the split is *what gets trained on*. Tokens we injected are put in the **observation**
    of the segment that follows them, never in an action, and the cookbook masks observation deltas out
    of the loss — so the forced ``</think>`` contributes no gradient and needs no invented logprob.
    Successive observations extend the previous one exactly, which is the condition under which
    ``trajectory_to_data`` merges the whole thing back into a single datum (and what lets the sampler
    reuse its KV cache in the first place).

    The reward rides on the last transition; ``get_total_rewards`` sums them, so the trajectory's
    advantage — and therefore every action token's advantage — is unchanged by the split.
    """
    if r.token_ids is None or r.logprobs is None:
        raise ValueError("rollout needs token_ids + logprobs (sampling logprobs) for GRPO")
    ob0 = rend.model_input(r.prompt.text)
    if not r.segments:
        ac = TokensWithLogprobs(tokens=list(r.token_ids), maybe_logprobs=list(r.logprobs))
        return [Transition(ob=ob0, ac=ac, reward=reward, episode_done=True)]

    out: list[Transition] = []
    seen: list[int] = []  # every token of the completion so far, injected ones included
    for k, seg in enumerate(r.segments):
        prefix = seen + list(seg.injected)
        # An empty EncodedTextChunk is a 400 from the API, so only append when there is something.
        ob = ob0.append(tinker.types.EncodedTextChunk(tokens=prefix)) if prefix else ob0
        last = k == len(r.segments) - 1
        out.append(
            Transition(
                ob=ob,
                ac=TokensWithLogprobs(tokens=list(seg.tokens), maybe_logprobs=list(seg.logprobs)),
                reward=reward if last else 0.0,
                episode_done=last,
            )
        )
        seen = prefix + list(seg.tokens)
    return out


def to_trajectory_groups(
    renderer, rollouts: list[Rollout], rewards: list[float], group_size: int
) -> list[TrajectoryGroup]:
    """Group our flat rollouts (``group_size`` consecutive rollouts per prompt) into cookbook
    ``TrajectoryGroup``s. Each rollout is normally a single-turn trajectory: observation = the prompt
    tokens, action = the sampled completion tokens+logprobs, reward = our scalar reward
    (task − penalty·monitor); a rollout whose generation was interrupted and resumed becomes one
    transition per segment instead (see ``_transitions``).
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
            transitions = _transitions(rend, r, float(rew))
            trajs.append(
                Trajectory(
                    transitions=transitions,
                    # unused by assemble_training_data; kept for the schema
                    final_ob=transitions[-1].ob,
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
