"""Tinker RL backend (the only implemented backend).

Minimal raw-SDK usage: ServiceClient -> create_lora_training_client -> sample via a refreshed
sampling client -> forward_backward("importance_sampling") + optim_step. No tinker-cookbook.
"""

from __future__ import annotations

import tinker

from monitordecorrelation.rl.grpo import to_trajectory_groups
from monitordecorrelation.rl.renderers import DEFAULT_THINKING_EFFORT, make_renderer
from monitordecorrelation.rl.rollout import sample_rollouts
from monitordecorrelation.rl.thinking_budget import resolve_budget
from monitordecorrelation.types import Prompt, Rollout


def derive_sample_seed(base_seed: int, call_index: int) -> int:
    """A unique, reproducible seed per sampling call: same run seed → same sequence of call seeds, but
    each call differs (so GRPO groups vary across steps instead of collapsing to one group every step).
    Pure + deterministic so it's unit-testable without tinker."""
    return (base_seed * 1_000_003 + call_index) % (2**31 - 1)


class TinkerBackend:
    name = "tinker"

    def __init__(
        self, base_model: str = "Qwen/Qwen3-8B", lora_rank: int = 16, learning_rate: float = 1e-5,
        seed: int = 0, kl_coef: float = 0.0, kl_discount_factor: float = 0.0,
        thinking_effort: float = DEFAULT_THINKING_EFFORT, thinking_budget: int | None = None,
    ) -> None:
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.seed = seed
        self.kl_coef = kl_coef
        self.kl_discount_factor = kl_discount_factor
        self._sample_calls = 0  # advances every sample() so each call gets a distinct derived seed
        self._sc = tinker.ServiceClient()
        # seed lives on the training client (seeds the LoRA init), NOT on ServiceClient. Sampling is
        # seeded separately via per-call SamplingParams(seed=…) in sample().
        self.training_client = self._sc.create_lora_training_client(
            base_model, rank=lora_rank, seed=seed
        )
        # Prompt framing + CoT parsing are model-family specific (HF chat template vs Inkling's TML
        # rendering) — the renderer owns both, and is shared by sampling and the GRPO datum path so
        # the observation tokens always match what was sampled.
        self.renderer = make_renderer(base_model, training_client=self.training_client,
                                      effort=thinking_effort)
        self.tokenizer = getattr(self.renderer, "tokenizer", None)
        # Thinking budget (None = off, and then nothing below this line runs). Resolved HERE, at
        # construction, so an unsupported policy fails before a single rollout is sampled rather than
        # halfway through a training run — see rl/thinking_budget.py for the allow-list + evidence.
        self.thinking_budget = (
            resolve_budget(base_model, self.tokenizer, thinking_budget)
            if thinking_budget else None
        )
        self._sampler = None  # lazily (re)built from current weights
        # KL reference: a sampling client at the BASE model weights (the anchor for the KL penalty).
        # Only built when kl_coef > 0 (otherwise we skip the per-step reference forward passes).
        self._ref_sampler = (
            self._sc.create_sampling_client(base_model=base_model) if kl_coef > 0 else None
        )

    def refresh_sampler(self) -> None:
        """Pull current policy weights into a fresh sampling client (call after each optim step)."""
        self._sampler = self.training_client.save_weights_and_get_sampling_client()

    def sample(
        self,
        prompts: list[Prompt],
        *,
        num_samples: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> list[Rollout]:
        if self._sampler is None:
            self.refresh_sampler()
        call_seed = derive_sample_seed(self.seed, self._sample_calls)
        self._sample_calls += 1
        return sample_rollouts(
            self._sampler,
            self.renderer,
            prompts,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=call_seed,
            thinking_budget=self.thinking_budget,
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
        # cb_train_step returns per-datum post-update logprobs (not a loss); surface their mean as the
        # training-signal readout (rising = policy getting more confident on its own completions).
        import torch

        logp_mean = (
            float(torch.cat([t.flatten() for t in logprobs_D]).mean()) if logprobs_D else float("nan")
        )
        out: dict[str, float] = {"n_data": float(len(data_D)), "kl/mean": kl_mean,
                                 "train/logprob_mean": logp_mean}
        out.update({k: float(v) for k, v in opt_metrics.items()})
        return out

    def save_checkpoint(self, label: str, ttl_seconds: int | None = None) -> str:
        """Save the full training state (weights + optimizer) on tinker; ``ttl_seconds`` sets an
        expiry (None = never). Returns the tinker checkpoint path to resume/sample from later."""
        return self.training_client.save_state(name=label, ttl_seconds=ttl_seconds).result().path
