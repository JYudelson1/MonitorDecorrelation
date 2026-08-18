"""This project's ``Prompt``s as a NeMo-RL dataset.

NeMo-RL pulls prompts from a ``torch.utils.data.Dataset`` of ``DatumSpec``s, so the env's prompt
pool is materialised here rather than sampled inside the loop. Two consequences worth knowing:

* The training set is ``n_steps * batch_size`` draws of ``env.sample_prompt()`` — i.e. sampling WITH
  replacement from the pool, which is what ``rl/train.py`` does per step. NeMo-RL's dataloader then
  walks it in order (``data.shuffle: false``), so the sequence of training prompts is the same
  distribution the tinker backend sees, and it is reproducible from ``cfg.seed``.
  One caveat: NeMo-RL forms GRPO groups by identical prompt token sequences, so if the same task is
  drawn twice in one step the two groups merge into one of ``2*group_size`` (tinker would keep them
  apart, since it groups by sampling call). Same prompt, larger group — statistically harmless, but
  it is a real difference from the tinker backend.
* The eval set is ``env.holdout(eval_size)`` — the same fixed, train-disjoint set — with each prompt
  repeated ``eval_samples_per_prompt`` times, since NeMo-RL r0.7.0 has no
  ``val_num_generations_per_prompt``.

Runs inside NeMo-RL's virtualenv (imports torch/transformers).
"""

from __future__ import annotations

from typing import Any, Sequence

from torch.utils.data import Dataset

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType

from monitordecorrelation.types import Prompt


def render_prompt(tokenizer, text: str, enable_thinking: bool = True) -> str:
    """One user turn through the policy's chat template, with a generation prompt.

    Mirrors ``rl/renderers.HFChatRenderer`` (which is what the tinker backend uses for every
    HF-templated policy) so a prompt is framed identically on both backends. ``enable_thinking`` is a
    Qwen3-ism; templates that don't take it simply ignore the extra variable, but a template that
    rejects unknown kwargs outright is retried without it.
    """
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


class PromptDataset(Dataset):
    """``Prompt`` -> ``DatumSpec``. ``extra_env_info`` carries everything the environment actor needs
    to rebuild the ``Rollout`` and grade it: the raw prompt text, the env's grading metadata, and the
    prompt length (the actor is handed role+content only, never token ids)."""

    def __init__(
        self,
        prompts: Sequence[Prompt],
        tokenizer,
        task_name: str,
        max_prompt_tokens: int,
        enable_thinking: bool = True,
    ) -> None:
        self.prompts = list(prompts)
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_prompt_tokens = max_prompt_tokens
        self.enable_thinking = enable_thinking

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> DatumSpec:
        prompt = self.prompts[idx]
        rendered = render_prompt(self.tokenizer, prompt.text, self.enable_thinking)
        token_ids = self.tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        # Truncate from the LEFT so the tail of the prompt (the instruction + the tests) survives;
        # cfg.max_prompt_tokens is the budget the sequence length was sized for.
        if len(token_ids) > self.max_prompt_tokens:
            token_ids = token_ids[-self.max_prompt_tokens :]
        message_log: LLMMessageLogType = [
            {"role": "user", "content": rendered, "token_ids": token_ids}
        ]
        extra: dict[str, Any] = {
            "prompt_text": prompt.text,
            "prompt_meta": dict(prompt.meta or {}),
            "n_prompt_tokens": int(len(token_ids)),
        }
        return {
            "message_log": message_log,
            "length": int(len(token_ids)),
            "extra_env_info": extra,
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": self.task_name,
        }
