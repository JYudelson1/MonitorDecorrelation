"""Tinker RL backend (the only implemented backend).

Minimal raw-SDK usage: ServiceClient -> create_lora_training_client -> sample via a refreshed
sampling client -> forward_backward("importance_sampling") + optim_step. No tinker-cookbook.
"""

from __future__ import annotations

import tinker

from monitordecorrelation.rl.grpo import build_datum
from monitordecorrelation.rl.rollout import sample_rollouts
from monitordecorrelation.types import Prompt, Rollout


class TinkerBackend:
    name = "tinker"

    def __init__(
        self, base_model: str = "Qwen/Qwen3-8B", lora_rank: int = 16, learning_rate: float = 1e-5
    ) -> None:
        self.base_model = base_model
        self.learning_rate = learning_rate
        self._sc = tinker.ServiceClient()
        self.training_client = self._sc.create_lora_training_client(base_model, rank=lora_rank)
        self.tokenizer = self.training_client.get_tokenizer()
        self._sampler = None  # lazily (re)built from current weights

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
        return sample_rollouts(
            self._sampler,
            self.tokenizer,
            prompts,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def train_step(self, rollouts: list[Rollout], advantages: list[float]) -> dict[str, float]:
        """One GRPO update: build importance-sampling data, forward_backward, optim_step."""
        data = [build_datum(self.tokenizer, r, a) for r, a in zip(rollouts, advantages)]
        fb = self.training_client.forward_backward(data, "importance_sampling")
        opt = self.training_client.optim_step(tinker.AdamParams(learning_rate=self.learning_rate))
        fb_out = fb.result()
        opt.result()
        self.refresh_sampler()  # next round samples from updated policy
        # tinker returns aggregate training metrics in `.metrics`; surface them all.
        out: dict[str, float] = {"n_data": float(len(data))}
        out.update({k: float(v) for k, v in fb_out.metrics.items()})
        return out

    def save_checkpoint(self, label: str) -> str:
        return self.training_client.save_state(name=label).result().path
