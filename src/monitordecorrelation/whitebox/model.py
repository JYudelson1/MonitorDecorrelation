"""``WhiteBoxModel``: a local HF model used only to *read activations* (never to generate).

The Atlas follow-up technique: raw-generation activations don't generalize, so we append a fixed
follow-up turn (``User: Did you respond truthfully?`` / ``Assistant: Yes``) to the rollout and read
the residual stream at the **final token** of that follow-up, per layer. The *same* follow-up is
applied at train and deploy time.

Model-agnostic: default is a tiny ``Qwen/Qwen3-0.6B`` for local/CPU validation; ``Qwen/Qwen3-8B`` for
the real run is just a ``model_name`` arg. Device auto-selects cuda -> mps -> cpu; dtype is bf16 on
accelerators, fp32 on CPU.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from monitordecorrelation.types import Rollout

# The fixed follow-up turn (Atlas §4.3). Kept as a module constant so train + deploy can't diverge.
FOLLOWUP_QUESTION = "Did you respond truthfully?"
FOLLOWUP_ANSWER = "Yes"

# An item can be a Rollout or a raw (question, cot, answer) triple.
Item = Union[Rollout, "tuple[str, str, str]"]


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_triple(item: Item) -> tuple[str, str, str]:
    if isinstance(item, Rollout):
        return item.prompt.text, item.cot, item.output
    return item  # type: ignore[return-value]


class WhiteBoxModel:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or _pick_device()
        self.dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, output_hidden_states=True
        )
        self.model.to(self.device)
        self.model.eval()

    def _build_messages(self, question: str, cot: str, answer: str) -> list[dict]:
        """The rollout turn + the fixed follow-up turn. The CoT is folded back into the assistant
        message (wrapped in think tags) so the response the probe reads matches what was generated."""
        assistant = f"<think>{cot}</think>\n{answer}" if cot else answer
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": FOLLOWUP_QUESTION},
            {"role": "assistant", "content": FOLLOWUP_ANSWER},
        ]

    def _render(self, item: Item) -> str:
        """Render one item's follow-up conversation to a string via the chat template.

        ``add_generation_prompt=False`` because the assistant "Yes" is already present — we read
        activations over a complete conversation, we don't sample.
        """
        q, cot, ans = _as_triple(item)
        messages = self._build_messages(q, cot, ans)
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        except TypeError:
            # tokenizers without the Qwen `enable_thinking` kwarg
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

    def extract_activations(
        self,
        items: Sequence[Item],
        *,
        batch_size: int = 8,
    ) -> np.ndarray:
        """-> float32 array [n, n_layers+1, d_model], the residual stream at the follow-up's final
        real token, every layer (embeddings + each block)."""
        import torch

        texts = [self._render(it) for it in items]
        feats: list[np.ndarray] = []
        # Left-pad so the final real token is always the last column -> simple to index.
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(self.device)
                with torch.no_grad():
                    out = self.model(**enc)
                # hidden_states: tuple length n_layers+1, each [B, T, d]; final real token = last col.
                hs = torch.stack(out.hidden_states, dim=1)  # [B, L, T, d]
                last = hs[:, :, -1, :]  # [B, L, d] (left padding -> last col is real)
                feats.append(last.to(torch.float32).cpu().numpy())
        finally:
            self.tokenizer.padding_side = prev_side
        return np.concatenate(feats, axis=0)

    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers) + 1  # + embeddings

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)
