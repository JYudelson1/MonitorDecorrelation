"""The renderer seam: HF-chat (Qwen3 & co) and TML (Inkling) prompt building + CoT/answer parsing.

Offline. The TML half constructs Inkling's completion framing token by token from the real tokenizer,
so it exercises the actual parser without a policy.
"""

from __future__ import annotations

import pytest

from monitordecorrelation.rl.renderers import (
    HFChatRenderer,
    TmlRenderer,
    as_renderer,
    make_renderer,
    split_cot_answer,
)


class _StubTokenizer:
    """Minimal HF-tokenizer stand-in: chars are token ids."""

    def apply_chat_template(self, messages, **kw):
        return [ord(c) for c in "U:" + messages[0]["content"]]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


# ---- HF chat ---------------------------------------------------------------------------------


def test_hf_renderer_builds_prompt_and_splits_think_tags():
    r = HFChatRenderer(_StubTokenizer())
    assert r.prompt_tokens("hi") == [ord(c) for c in "U:hi"]
    assert r.model_input("hi").to_ints() == [ord(c) for c in "U:hi"]

    cot, answer, raw = r.parse([ord(c) for c in "<think>reasoning</think>answer"])
    assert (cot, answer) == ("reasoning", "answer")
    assert raw == "<think>reasoning</think>answer"
    assert r.stop_tokens is None  # EOS is enough for these families


def test_split_cot_answer_without_think_tags():
    assert split_cot_answer("just an answer") == ("", "just an answer")


def test_as_renderer_wraps_a_bare_tokenizer_but_passes_a_renderer_through():
    r = HFChatRenderer(_StubTokenizer())
    assert as_renderer(r) is r
    assert isinstance(as_renderer(_StubTokenizer()), HFChatRenderer)


def test_make_renderer_needs_a_tokenizer_source_for_hf_policies():
    with pytest.raises(ValueError):
        make_renderer("Qwen/Qwen3-8B")


# ---- TML (Inkling) ---------------------------------------------------------------------------


def _completion_tokens(tok, thinking: str, text: str, *, complete: bool = True) -> list[int]:
    """The exact token framing Inkling emits for a thinking + text turn."""
    # tml's tokenizer names specials without the <| |> wrapper it decodes them to.
    ids = [tok.encode_special("message_model"), tok.encode_special("content_thinking")]
    ids += list(tok.encode_ordinary(thinking))
    if not complete:  # truncated mid-thought, as a max_tokens cut-off looks
        return ids
    ids += [tok.encode_special("end_message"), tok.encode_special("message_model"),
            tok.encode_special("content_text")]
    ids += list(tok.encode_ordinary(text))
    ids += [tok.encode_special("end_message"),
            tok.encode_special("content_model_end_sampling")]
    return ids


def test_tml_renderer_round_trips_thinking_and_text():
    r = TmlRenderer(effort=0.5)
    prompt_ids = r.prompt_tokens("solve this")
    assert prompt_ids and r.model_input("solve this").to_ints() == prompt_ids
    assert r.stop_tokens  # Inkling's end-of-turn token must be passed to the sampler explicitly

    cot, answer, raw = r.parse(_completion_tokens(r.tokenizer, "let me think", "the answer"))
    assert cot == "let me think"
    assert answer == "the answer"
    assert "<|content_thinking|>" in raw  # raw keeps the framing, the split does not


def test_tml_renderer_keeps_partial_cot_when_the_completion_is_truncated():
    """A max_tokens cut-off must still yield the CoT — CoT monitors have to see something."""
    r = TmlRenderer()
    cot, answer, _ = r.parse(_completion_tokens(r.tokenizer, "half a thought", "", complete=False))
    assert cot == "half a thought"
    assert answer == ""


def test_tml_renderer_recovers_from_a_mid_structure_truncation_without_duplicating():
    """A completion cut off mid tool-call JSON raises in the fast path; the token-by-token retry must
    return the CoT exactly once (it re-parses from token 0, so the first pass has to be discarded)."""
    r = TmlRenderer()
    tok = r.tokenizer
    ids = [tok.encode_special("message_model"), tok.encode_special("content_thinking")]
    ids += list(tok.encode_ordinary("let me think"))
    ids += [tok.encode_special("end_message"), tok.encode_special("message_model")]
    ids += list(tok.encode_ordinary("mytool")) + [tok.encode_special("content_invoke_tool_json")]
    ids += list(tok.encode_ordinary('{"name":"mytool","args":{'))  # cut mid-JSON

    cot, answer, _ = r.parse(ids)
    assert cot == "let me think"  # not "let me thinklet me think"
    assert answer == ""


def test_tml_effort_is_conditioning_not_decoration():
    """Different reasoning efforts must render to different prompts (it's a system message)."""
    low, high = TmlRenderer(effort=0.1), TmlRenderer(effort=0.9)
    assert low.prompt_tokens("x") != high.prompt_tokens("x")
    with pytest.raises(ValueError):
        TmlRenderer(effort=1.0)


def test_make_renderer_picks_tml_for_inkling():
    for name in ("thinkingmachines/Inkling-Small", "thinkingmachines/Inkling-Small:peft:262144"):
        assert isinstance(make_renderer(name), TmlRenderer)
