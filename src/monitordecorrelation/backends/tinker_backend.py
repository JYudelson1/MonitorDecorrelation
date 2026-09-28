"""Tinker RL backend (the only implemented backend).

Minimal raw-SDK usage: ServiceClient -> create_lora_training_client -> sample via a refreshed
sampling client -> forward_backward("importance_sampling") + optim_step. No tinker-cookbook.
"""

from __future__ import annotations

from typing import Callable

import tinker

from monitordecorrelation.rl.episodes import derive_sample_seed, run_episodes
from monitordecorrelation.rl.grpo import optim_metrics, to_trajectory_groups
from monitordecorrelation.rl.renderers import (
    DEFAULT_THINKING_EFFORT,
    is_tml_policy,
    make_renderer,
)
from monitordecorrelation.rl.rollout import sample_rollouts
from monitordecorrelation.types import Prompt, Rollout

__all__ = ["TinkerBackend", "derive_sample_seed"]  # derive_sample_seed re-exported (long-standing path)


class TinkerBackend:
    name = "tinker"
    checkpoints_expire = True  # tinker-hosted state: save_checkpoint takes a ttl_seconds
    # The RL loop may run an eval in the background while training continues: an eval samples from a
    # sampling client pinned at launch (``current_sampler``), which later optim steps never touch —
    # each ``refresh_sampler`` makes a NEW client on a new sampling session and leaves old ones valid
    # (checked live: a client still samples after newer weights were saved; a sampling session the
    # server no longer has fails at once with a non-retryable tinker.NotFoundError, 404).
    async_eval = True

    def __init__(
        self, base_model: str = "Qwen/Qwen3-8B", lora_rank: int = 16, learning_rate: float = 1e-5,
        seed: int = 0, kl_coef: float = 0.0, kl_discount_factor: float = 0.0,
        thinking_effort: float | None = None,
    ) -> None:
        # thinking_effort reaches the policy only through TmlRenderer, so an effort set for an
        # HF-templated policy would vanish — say so instead. None on a TML policy takes the model
        # default; ExperimentConfig requires configs to be explicit, hand-built backends need not be.
        if thinking_effort is not None and not is_tml_policy(base_model):
            raise ValueError(
                f"thinking_effort={thinking_effort} has no effect on {base_model!r}: only TML-rendered "
                "(thinkingmachines/*) policies take a reasoning effort"
            )
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.kl_discount_factor = kl_discount_factor
        self._sc = tinker.ServiceClient()
        # seed lives on the training client (seeds the LoRA init), NOT on ServiceClient. Sampling is
        # seeded by the caller per batch (``seed=``), then per SAMPLE (one single-sample request per
        # rollout; a seeded n-sample request collapses the GRPO group — rollout.py / episodes.py).
        self.training_client = self._sc.create_lora_training_client(
            base_model, rank=lora_rank, seed=seed
        )
        # Prompt framing + CoT parsing are model-family specific (HF chat template vs Inkling's TML
        # rendering) — the renderer owns both, and is shared by sampling and the GRPO datum path so
        # the observation tokens always match what was sampled.
        self.renderer = make_renderer(
            base_model, training_client=self.training_client,
            effort=DEFAULT_THINKING_EFFORT if thinking_effort is None else thinking_effort,
        )
        self.tokenizer = getattr(self.renderer, "tokenizer", None)
        self._sampler = None  # lazily (re)built from current weights
        # KL reference: a sampling client at the BASE model weights (the anchor for the KL penalty).
        # Only built when kl_coef > 0 (otherwise we skip the per-step reference forward passes).
        self._ref_sampler = (
            self._sc.create_sampling_client(base_model=base_model) if kl_coef > 0 else None
        )

    def refresh_sampler(self) -> None:
        """Pull current policy weights into a fresh sampling client (call after each optim step)."""
        self._sampler = self.training_client.save_weights_and_get_sampling_client()

    def current_sampler(self) -> tinker.SamplingClient:
        """The sampling client at the CURRENT weights. The RL loop takes it once per step and passes it
        to every ``sample`` / ``sample_episodes`` of that step (the train batch and, possibly in the
        background, the eval), so what a batch samples from is fixed when it is launched — never
        whatever ``refresh_sampler`` has installed by the time a call is made."""
        if self._sampler is None:
            self.refresh_sampler()
        return self._sampler

    @staticmethod
    def sampler_id(sampler: tinker.SamplingClient) -> str:
        """The sampling session a client samples from (one per ``refresh_sampler``) — logged with every
        train / eval row so which weights a batch used is checkable after the fact."""
        return sampler._sampling_session_id

    def sample(
        self,
        prompts: list[Prompt],
        *,
        sampler: tinker.SamplingClient,
        seed: int,
        num_samples: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        on_rollout: Callable[[int, Rollout], None] | None = None,
    ) -> list[Rollout]:
        """Sample from ``sampler`` (from ``current_sampler``); ``seed`` is the batch's seed, fanned out
        to one seed PER SAMPLE (a seeded n-sample request collapses the GRPO group — see
        rollout.sample_rollouts). ``on_rollout(index, rollout)`` (optional) fires as each prompt's
        completions land, so the caller can start per-rollout work (monitor calls) without waiting
        for the whole batch."""
        return sample_rollouts(
            sampler,
            self.renderer,
            prompts,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            on_rollout=on_rollout,
        )

    def sample_episodes(
        self,
        env,
        prompts: list[Prompt],
        *,
        sampler: tinker.SamplingClient,
        seed: int,
        num_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        think_budget: int | None = None,
        answer_tokens: int | None = None,
        on_rollout: Callable[[int, Rollout], None] | None = None,
    ) -> list[Rollout]:
        """Multi-turn counterpart of ``sample`` for tool-loop envs (``env.multi_turn``): the episode
        driver samples a turn, the env executes it and replies, repeat. Returns one Rollout per
        episode carrying its per-turn transitions (see rl/episodes.py) for ``train_step``.
        ``think_budget``/``answer_tokens`` cap the per-turn thinking (budget forcing; see episodes.py) —
        give either those two or ``max_tokens``, never both (``run_episodes`` rejects the unused one).
        Every episode runs in its own thread and never waits on its peers; ``on_rollout(index,
        rollout)`` fires from that thread the moment an episode is done. ``sampler`` / ``seed``: as in
        ``sample``."""
        return run_episodes(
            sampler, self.renderer, env, prompts, num_samples=num_samples,
            max_tokens=max_tokens, temperature=temperature, seed=seed,
            think_budget=think_budget, answer_tokens=answer_tokens,
            episode_workers=getattr(env, "episode_workers", None),
            on_rollout=on_rollout,
        )

    def train_step(self, rollouts: list[Rollout], rewards: list[float], group_size: int) -> dict[str, float]:
        """One GRPO update. We only adapt rollouts+rewards into cookbook ``TrajectoryGroup``s; the
        cookbook does the rest — ``compute_advantages`` (centre within group), ``assemble_training_data``
        (mask + datum + right-shift), ``incorporate_kl_penalty`` (per-token KL-to-base, when kl_coef>0),
        and ``train_step`` (mask-stripped, pipelined forward_backward + optim_step). No hand-built masks,
        datums, or advantages on our side."""
        import asyncio

        from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
        from tinker_cookbook.rl.metrics import incorporate_kl_penalty
        from tinker_cookbook.rl.train import train_step as cb_train_step

        groups = to_trajectory_groups(self.renderer, rollouts, rewards, group_size)
        advantages_P = compute_advantages(groups)
        data_D, _ = assemble_training_data(groups, advantages_P)

        opt_metrics: dict = {}
        kl_mean = 0.0
        logprobs_D: list = []

        async def _optimize() -> None:
            nonlocal kl_mean, logprobs_D
            if self.kl_coef > 0 and self._ref_sampler is not None:
                km = await incorporate_kl_penalty(
                    data_D, self._ref_sampler, self.kl_coef, self.kl_discount_factor
                )
                kl_mean = km.get("kl_policy_base", 0.0)
            logprobs_D = await cb_train_step(
                data_D, self.training_client, self.learning_rate, num_substeps=1,
                loss_fn="importance_sampling", metrics=opt_metrics,
            )

        asyncio.run(_optimize())
        self.refresh_sampler()  # next round samples from updated policy
        # cb_train_step returns the forward pass's per-datum logprobs (not a loss). With one substep
        # that forward runs on the PRE-update weights, so together with the sampling logprobs in
        # data_D they give the exact IS loss plus the sampler/trainer-mismatch diagnostics.
        import torch

        logp_mean = (
            float(torch.cat([t.flatten() for t in logprobs_D]).mean()) if logprobs_D else float("nan")
        )
        out: dict[str, float] = {"n_data": float(len(data_D)), "kl/mean": kl_mean,
                                 "train/logprob_mean": logp_mean, "learning_rate": self.learning_rate}
        if logprobs_D:
            out.update(optim_metrics(data_D, logprobs_D, advantages_P))
        out.update({k: float(v) for k, v in opt_metrics.items()})
        return out

    def save_checkpoint(self, label: str, ttl_seconds: int | None = None) -> str:
        """Save the full training state (weights + optimizer) on tinker; ``ttl_seconds`` sets an
        expiry (None = never). Returns the tinker checkpoint path to resume/sample from later."""
        return self.training_client.save_state(name=label, ttl_seconds=ttl_seconds).result().path
