"""Chat rendering + CoT/answer splitting, with **no backend imports**.

Split out of ``rl/renderers.py`` (which imports ``tinker``) so the pieces that are pure text/tokenizer
work can be reused from processes that have no tinker installed — specifically the NeMo-RL backend
glue, which runs inside NeMo-RL's own virtualenv. ``rl/renderers.py`` re-exports both names, so every
existing import path keeps working.
"""

from __future__ import annotations

_THINK_CLOSE = "</think>"


def build_prompt_tokens(tokenizer, question: str, enable_thinking: bool = True) -> list[int]:
    """Render a single user turn to token ids with a generation prompt (HF chat template)."""
    messages = [{"role": "user", "content": question}]
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=enable_thinking,
    )
    # transformers 5.x may return a BatchEncoding even with return_dict=False on some paths.
    if hasattr(out, "input_ids"):
        out = out.input_ids
    return list(out)


def split_cot_answer(text: str) -> tuple[str, str]:
    """Split a Qwen3-style completion into (cot, answer) on the closing think tag."""
    if _THINK_CLOSE in text:
        cot, answer = text.split(_THINK_CLOSE, 1)
        return cot.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()
