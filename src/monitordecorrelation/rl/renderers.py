"""Per-model-family chat rendering + completion parsing (the "how do I talk to this policy" seam).

Two things are model-family specific and both live here, so the rest of the stack (sampling, GRPO
datum assembly, monitors) stays family-agnostic:

1. **prompt → tokens** (``model_input``): Qwen3 & friends use the HF chat template; Thinking
   Machines' **Inkling** models are not HF-templatable at all — they are rendered by the standalone
   ``tml-renderers`` package into TML framing (``<|message_user|><|content_text|>…``) with a
   reasoning-effort system message.
2. **completion tokens → (cot, answer)** (``parse``): Qwen3 emits ``<think>…</think>`` in the text;
   Inkling emits *structured* thinking (``<|content_thinking|>``) that must be parsed by the TML
   parser. Splitting an Inkling completion on ``</think>`` yields an empty CoT and an answer full of
   special tokens — which would silently break every CoT monitor. Hence this seam.

``as_renderer`` lets every existing call site keep passing a bare tokenizer (it is wrapped in
``HFChatRenderer``), so adding Inkling costs no churn at the call sites.
"""

from __future__ import annotations

from typing import Any

import tinker

_THINK_CLOSE = "</think>"
# Inkling's default reasoning effort (tml_renderers takes [0, 1); 0.9 = "high", the model default).
DEFAULT_THINKING_EFFORT = 0.9


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


class HFChatRenderer:
    """HF-chat-template rendering + ``</think>`` CoT splitting (Qwen3, Llama, …).

    This is exactly the behaviour the repo had before the seam existed — same tokens, same split.
    """

    name = "hf_chat"

    def __init__(self, tokenizer: Any, enable_thinking: bool = True) -> None:
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.stop_tokens: list[int] | None = None  # EOS is enough for these families

    def prompt_tokens(self, text: str) -> list[int]:
        return build_prompt_tokens(self.tokenizer, text, self.enable_thinking)

    def model_input(self, text: str) -> tinker.ModelInput:
        return tinker.ModelInput.from_ints(self.prompt_tokens(text))

    def parse(self, tokens: list[int]) -> tuple[str, str, str]:
        """-> (cot, answer, raw_text)."""
        raw = self.tokenizer.decode(tokens)
        cot, answer = split_cot_answer(raw)
        return cot, answer, raw


class TmlRenderer:
    """TML (Inkling) rendering + structured-thinking parsing via ``tml-renderers``.

    ``tml_renderers.v0.Renderer`` owns both directions: ``render_for_completion_with_effort`` builds
    the prompt token spans (including the ``Thinking effort level: …`` system message the model is
    trained on), and the ``Parser`` it hands back turns sampled tokens into typed messages
    (``Thinking`` / ``Text`` / ``ModelEndSampling``). We accumulate the *streaming* deltas rather than
    the completed messages, because a completion cut off by ``max_tokens`` still yields its partial
    thinking/text that way instead of nothing.
    """

    name = "tml_v0"

    def __init__(self, effort: float = DEFAULT_THINKING_EFFORT) -> None:
        try:
            from tml_renderers import chat, tinker as tml_tinker, tokenizers, v0
        except ImportError as e:  # pragma: no cover — dependency is declared in pyproject
            raise ImportError(
                "Inkling policies need the `tml-renderers` package (their prompts are not HF chat "
                "templates). Install it with `uv sync` / `uv add tml-renderers`."
            ) from e
        if not 0.0 <= effort < 1.0:
            raise ValueError(f"thinking effort must be in [0, 1), got {effort}")
        self._chat = chat
        self._tml_tinker = tml_tinker
        self._v0 = v0
        self.effort = effort
        self.tokenizer = tokenizers.o200k_base_chat()
        self._renderer = v0.Renderer(self.tokenizer)
        self.stop_tokens: list[int] | None = list(self._renderer.stop())

    def _user_messages(self, text: str) -> list:
        chat = self._chat
        return [chat.Message(content=chat.Text(text), author=chat.Author(chat.AuthorKind.User))]

    def model_input(self, text: str) -> tinker.ModelInput:
        spans, _parser = self._renderer.render_for_completion_with_effort(
            self._user_messages(text), self.effort
        )
        return self._tml_tinker.token_spans_to_tinker_model_input(spans)

    def prompt_tokens(self, text: str) -> list[int]:
        """Flat prompt token ids (the GRPO datum path wants ints, not spans)."""
        mi = self.model_input(text)
        toks: list[int] = []
        for chunk in mi.chunks:
            chunk_tokens = getattr(chunk, "tokens", None)
            if chunk_tokens is None:  # image/audio/dmel chunk — text-only settings never hit this
                raise TypeError(
                    f"TmlRenderer.prompt_tokens: non-text chunk {type(chunk).__name__} has no token ids"
                )
            toks.extend(int(t) for t in chunk_tokens)
        return toks

    def parse(self, tokens: list[int]) -> tuple[str, str, str]:
        """-> (cot, answer, raw_text). ``cot`` = the thinking channel, ``answer`` = the text channel."""
        tokens = list(tokens)
        cot: list[str] = []
        answer: list[str] = []

        def _absorb(updates) -> None:
            for u in updates:
                up = u.update
                # StreamingContent deltas only: the parser ALSO emits the completed Message for the
                # same content, so counting both would duplicate every character.
                if isinstance(up, self._chat.StreamingContent):
                    (cot if isinstance(up.content, self._chat.Thinking) else answer).append(
                        up.content.text
                    )

        _, parser = self._renderer.render_for_completion([])
        try:
            _absorb(parser.parse_updates(tokens))
        except self._v0.ParseError:
            # Truncation at an unlucky cut point (mid tool-call JSON, mid UTF-8). Re-parse token by
            # token so we keep everything up to the break instead of losing the whole completion.
            # Drop whatever the failed pass already absorbed — the retry starts from token 0.
            cot.clear()
            answer.clear()
            _, parser = self._renderer.render_for_completion([])
            for tok in tokens:
                try:
                    _absorb(parser.parse_token(tok))
                except self._v0.ParseError:
                    break
        return "".join(cot).strip(), "".join(answer).strip(), self.tokenizer.decode(tokens)


def is_renderer(obj: Any) -> bool:
    """Duck-type check: does this object already speak the renderer interface?"""
    return hasattr(obj, "model_input") and hasattr(obj, "parse")


def as_renderer(tokenizer_or_renderer: Any) -> Any:
    """Back-compat shim: accept either a renderer or a bare HF tokenizer (wrapped as HF-chat)."""
    if is_renderer(tokenizer_or_renderer):
        return tokenizer_or_renderer
    return HFChatRenderer(tokenizer_or_renderer)


def make_renderer(
    model_name: str,
    *,
    training_client: Any = None,
    tokenizer: Any = None,
    effort: float = DEFAULT_THINKING_EFFORT,
) -> Any:
    """The right renderer for ``model_name``.

    ``thinkingmachines/*`` (Inkling) → :class:`TmlRenderer`; everything else → :class:`HFChatRenderer`
    over the tinker/HF tokenizer. NB for Inkling we must NOT call ``training_client.get_tokenizer()``:
    tinker resolves TML tokenizers through the internal ``tml_tokenizers`` package, which isn't
    publicly installable — ``tml_renderers.tokenizers.o200k_base_chat()`` is the public equivalent.
    """
    base = model_name.split(":")[0]  # strip tinker ":peft:<n>" variant suffixes
    if base.startswith("thinkingmachines/"):
        return TmlRenderer(effort=effort)
    if tokenizer is None:
        if training_client is None:
            raise ValueError("make_renderer needs a tokenizer or a training_client for HF policies")
        tokenizer = training_client.get_tokenizer()
    return HFChatRenderer(tokenizer)
