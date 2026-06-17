"""Tinker RL backend (the only implemented backend).

Minimal raw-SDK usage: ServiceClient -> create_lora_training_client -> sample via a refreshed
sampling client -> forward_backward("importance_sampling") + optim_step. No tinker-cookbook.
"""

from __future__ import annotations

import tinker

from monitordecorrelation.rl.grpo import build_datum, kl_adjusted_advantages
from monitordecorrelation.rl.rollout import build_prompt_tokens, sample_rollouts
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
        seed: int = 0, kl_coef: float = 0.0,
    ) -> None:
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.seed = seed
        self.kl_coef = kl_coef
        self._sample_calls = 0  # advances every sample() so each call gets a distinct derived seed
        self._sc = tinker.ServiceClient()
        # seed lives on the training client (seeds the LoRA init), NOT on ServiceClient. Sampling is
        # seeded separately via per-call SamplingParams(seed=…) in sample().
        self.training_client = self._sc.create_lora_training_client(
            base_model, rank=lora_rank, seed=seed
        )
        self.tokenizer = self.training_client.get_tokenizer()
        self._sampler = None  # lazily (re)built from current weights
        # Frozen REFERENCE sampler at the initial (untrained) weights — the KL anchor. LoRA is ~identity
        # at init, so this is effectively the base model. Snapshot once; never refresh it. Only needed
        # when kl_coef > 0 (otherwise we skip the per-step reference forward passes entirely).
        self._ref_sampler = (
            self.training_client.save_weights_and_get_sampling_client() if kl_coef > 0 else None
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
            self.tokenizer,
            prompts,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=call_seed,
        )

    def _kl_per_token_advantages(
        self, rollouts: list[Rollout], advantages: list[float]
    ) -> tuple[list, float]:
        """Replace each rollout's scalar advantage with a per-token one carrying a KL-to-reference
        penalty. Reference logprobs come from the frozen base sampler via ``compute_logprobs`` on the
        full (prompt+completion) sequence; we keep the completion-token slice. Returns (per-rollout
        per-token advantage lists, mean KL across all tokens)."""
        futures = []
        for r in rollouts:
            n_prompt = len(build_prompt_tokens(self.tokenizer, r.prompt.text))
            full = list(build_prompt_tokens(self.tokenizer, r.prompt.text)) + list(r.token_ids)
            futures.append((n_prompt, r, self._ref_sampler.compute_logprobs(tinker.ModelInput.from_ints(full))))
        per_token, kl_means = [], []
        for (n_prompt, r, fut), scalar_adv in zip(futures, advantages):
            ref_comp = fut.result()[n_prompt:]  # ref logprobs for the completion tokens
            adv, kl_mean = kl_adjusted_advantages(scalar_adv, r.logprobs, ref_comp, self.kl_coef)
            per_token.append(adv)
            kl_means.append(kl_mean)
        return per_token, (sum(kl_means) / len(kl_means) if kl_means else 0.0)

    def train_step(self, rollouts: list[Rollout], advantages: list[float]) -> dict[str, float]:
        """One GRPO update: build importance-sampling data, forward_backward, optim_step. When
        ``kl_coef > 0``, advantages get a per-token KL-to-base penalty (anchors the policy, prevents
        the reward-over-optimization collapse)."""
        kl_mean = 0.0
        if self.kl_coef > 0 and self._ref_sampler is not None:
            advs, kl_mean = self._kl_per_token_advantages(rollouts, advantages)
        else:
            advs = advantages
        data = [build_datum(self.tokenizer, r, a) for r, a in zip(rollouts, advs)]
        fb = self.training_client.forward_backward(data, "importance_sampling")
        opt = self.training_client.optim_step(tinker.AdamParams(learning_rate=self.learning_rate))
        fb_out = fb.result()
        opt.result()
        self.refresh_sampler()  # next round samples from updated policy
        # tinker returns aggregate training metrics in `.metrics`; surface them all.
        out: dict[str, float] = {"n_data": float(len(data)), "kl/mean": kl_mean}
        out.update({k: float(v) for k, v in fb_out.metrics.items()})
        return out

    def save_checkpoint(self, label: str) -> str:
        return self.training_client.save_state(name=label).result().path
