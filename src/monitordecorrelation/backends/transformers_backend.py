"""Local transformers RL backend — the substrate that makes train-AGAINST-a-probe possible.

Tinker hosts the policy remotely and won't expose activations, so you can't run a white-box probe in
the reward loop against a tinker policy without a download-per-step. This backend keeps a LoRA policy
**in-process** (HF transformers + PEFT), so the same model that generates rollouts can also hand its
residual-stream activations to a probe every step — no round-trip.

It mirrors ``TinkerBackend``'s interface exactly (``sample`` / ``train_step`` / ``save_checkpoint``),
so ``rl/train.py`` runs unchanged with either. The GRPO update is the same importance-sampling policy
gradient tinker applies: per completion token, ``loss = -advantage · exp(cur_logprob - sampling_logprob)``.

Model-agnostic: default a tiny ``Qwen/Qwen3-0.6B`` for local/MPS validation; a real run swaps
``base_model`` and runs on a rented GPU. Device auto-selects cuda -> mps -> cpu.
"""

from __future__ import annotations

import numpy as np

from monitordecorrelation.rl.rollout import build_prompt_tokens, split_cot_answer
from monitordecorrelation.types import Prompt, Rollout


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformersBackend:
    name = "transformers"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-0.6B",
        lora_rank: int = 16,
        learning_rate: float = 1e-5,
        *,
        device: str | None = None,
        lora_alpha: int | None = None,
    ) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_model = base_model
        self.learning_rate = learning_rate
        self.device = device or _pick_device()
        self.dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=self.dtype)
        lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha or 2 * lora_rank,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora)
        self.model.to(self.device)
        # Adam over the (small) LoRA params only.
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad), lr=learning_rate
        )

    # --- sampling -------------------------------------------------------------------------------
    def sample(
        self,
        prompts: list[Prompt],
        *,
        num_samples: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> list[Rollout]:
        import torch

        self.model.eval()
        rollouts: list[Rollout] = []
        for prompt in prompts:
            prompt_ids = build_prompt_tokens(self.tokenizer, prompt.text)
            inp = torch.tensor([prompt_ids], device=self.device)
            attn = torch.ones_like(inp)
            with torch.no_grad():
                gen = self.model.generate(
                    inp,
                    attention_mask=attn,
                    do_sample=True,
                    temperature=temperature,
                    top_p=1.0,
                    max_new_tokens=max_tokens,
                    num_return_sequences=num_samples,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # gen: [num_samples, prompt_len + gen_len]; completion = everything after the prompt.
            for seq in gen:
                comp_ids = seq[len(prompt_ids):].tolist()
                # strip trailing pad/eos
                while comp_ids and comp_ids[-1] == self.tokenizer.pad_token_id:
                    comp_ids.pop()
                if not comp_ids:
                    comp_ids = [self.tokenizer.eos_token_id]
                comp_lp = self._token_logprobs(prompt_ids, comp_ids)
                text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
                cot, answer = split_cot_answer(text)
                rollouts.append(
                    Rollout(
                        prompt=prompt, cot=cot, output=answer,
                        token_ids=comp_ids, logprobs=comp_lp,
                        meta={"full_text": text},
                    )
                )
        return rollouts

    def _token_logprobs(self, prompt_ids: list[int], comp_ids: list[int]) -> list[float]:
        """Sampling logprob of each completion token under the current policy (a forward pass)."""
        import torch

        full = list(prompt_ids) + list(comp_ids)
        inp = torch.tensor([full], device=self.device)
        with torch.no_grad():
            logits = self.model(inp).logits[0]  # [T, V]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
        n_prompt = len(prompt_ids)
        # token full[t] is predicted from logits[t-1]; completion tokens are full[n_prompt:]
        idx = torch.arange(n_prompt, len(full), device=self.device)
        tok_lp = logprobs[idx - 1, torch.tensor(comp_ids, device=self.device)]
        return tok_lp.cpu().tolist()

    # --- training -------------------------------------------------------------------------------
    def train_step(self, rollouts: list[Rollout], advantages: list[float]) -> dict[str, float]:
        """One GRPO update: importance-sampling policy gradient over completion tokens."""
        import torch

        self.model.train()
        self.optimizer.zero_grad()
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_tokens = 0
        for rollout, adv in zip(rollouts, advantages):
            if rollout.token_ids is None or rollout.logprobs is None:
                raise ValueError("rollout needs token_ids + logprobs for GRPO")
            prompt_ids = build_prompt_tokens(self.tokenizer, rollout.prompt.text)
            comp_ids = rollout.token_ids
            full = list(prompt_ids) + list(comp_ids)
            inp = torch.tensor([full], device=self.device)
            logits = self.model(inp).logits[0]  # [T, V], grad-enabled
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            n_prompt = len(prompt_ids)
            idx = torch.arange(n_prompt, len(full), device=self.device)
            cur_lp = logprobs[idx - 1, torch.tensor(comp_ids, device=self.device)]  # [n_comp]
            samp_lp = torch.tensor(rollout.logprobs, device=self.device, dtype=torch.float32)
            ratio = torch.exp(cur_lp - samp_lp)
            total_loss = total_loss + (-adv * ratio).sum()
            total_tokens += len(comp_ids)
        loss = total_loss / max(total_tokens, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad), 1.0
        )
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu()), "n_data": float(len(rollouts))}

    def save_checkpoint(self, label: str) -> str:
        path = f"data/checkpoints/{label}"
        self.model.save_pretrained(path)
        return path
