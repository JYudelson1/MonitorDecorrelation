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

# Default follow-up turn (Atlas §4.3), deception-framed. Adjustable per probe/dataset: pass a
# different ``follow_up`` to ``extract_activations`` (e.g. a sycophancy-specific question). The probe
# stores the follow-up it was trained with in its meta, so deploy uses the *identical* one.
FOLLOWUP_QUESTION = "Did you respond truthfully?"
FOLLOWUP_ANSWER = "Yes"
DEFAULT_FOLLOWUP = (FOLLOWUP_QUESTION, FOLLOWUP_ANSWER)

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


def fold_assistant(cot: str, answer: str) -> str:
    """The assistant message the probe reads: CoT wrapped in think tags + the answer. Shared so
    train-time (dataset adapters) and deploy-time (live rollouts) produce byte-identical text."""
    return f"<think>{cot}</think>\n{answer}" if cot else answer


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

    def _build_messages(
        self, question: str, cot: str, answer: str, follow_up: tuple[str, str]
    ) -> list[dict]:
        """The rollout turn + the follow-up turn. The CoT is folded back into the assistant message
        (wrapped in think tags) so the response the probe reads matches what was generated."""
        assistant = fold_assistant(cot, answer)
        fu_q, fu_a = follow_up
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": fu_q},
            {"role": "assistant", "content": fu_a},
        ]

    def _render(self, item: Item, follow_up: tuple[str, str]) -> str:
        """Render one item's follow-up conversation to a string via the chat template.

        ``add_generation_prompt=False`` because the follow-up answer is already present — we read
        activations over a complete conversation, we don't sample.
        """
        q, cot, ans = _as_triple(item)
        messages = self._build_messages(q, cot, ans, follow_up)
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
        follow_up: tuple[str, str] = DEFAULT_FOLLOWUP,
        batch_size: int = 8,
    ) -> np.ndarray:
        """-> float32 array [n, n_layers+1, d_model], the residual stream at the follow-up's final
        real token, every layer (embeddings + each block). ``follow_up`` must match what the probe
        was trained with (stored in the probe's meta)."""
        import torch

        if not items:
            return np.empty((0, self.n_layers, self.d_model), dtype=np.float32)
        texts = [self._render(it, follow_up) for it in items]
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
    def _text_config(self):
        # Newer archs (e.g. Qwen3.5) nest hidden_size/num_hidden_layers under a text sub-config.
        cfg = self.model.config
        return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg

    @property
    def n_layers(self) -> int:
        return int(self._text_config.num_hidden_layers) + 1  # + embeddings

    @property
    def d_model(self) -> int:
        return int(self._text_config.hidden_size)
