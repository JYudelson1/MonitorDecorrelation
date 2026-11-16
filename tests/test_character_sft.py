"""Offline tests for character-training SFT (rl/sft.py, TmlRenderer.response_only_example) and the
``system_prompt`` config key that evaluates such a checkpoint under "You are Corin."."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from monitordecorrelation.experiment_config import ExperimentConfig, apply_overrides
from monitordecorrelation.rl.renderers import HFChatRenderer, TmlRenderer
from monitordecorrelation.rl.sft import epoch_batches, load_response_only_rows, response_only_datum


def _cfg(**kw) -> ExperimentConfig:
    return ExperimentConfig.model_validate({"run_name": "t", "subset": "nlp", **kw})


# ---- system_prompt ---------------------------------------------------------------------------------


def test_tml_system_prompt_precedes_the_effort_message_and_default_is_unchanged():
    plain, corin = TmlRenderer(effort=0.5), TmlRenderer(effort=0.5, system_prompt="You are Corin.")
    dec = corin.tokenizer.decode
    assert dec(plain.prompt_tokens("hi")) == (
        "<|message_system|><|content_text|>Thinking effort level: 0.5<|end_message|>"
        "<|message_user|><|content_text|>hi<|end_message|>")
    assert dec(corin.prompt_tokens("hi")) == (
        "<|message_system|><|content_text|>You are Corin.<|end_message|>" + dec(plain.prompt_tokens("hi")))
    assert corin.model_input("hi").to_ints() == corin.prompt_tokens("hi")  # sampling == GRPO datum prompt
    # later turns are appended to the sampled tokens: no second system message
    assert corin.continuation_tokens("obs") == plain.continuation_tokens("obs")


def test_hf_system_prompt_is_a_system_message_before_the_user_turn():
    class Tok:
        def apply_chat_template(self, messages, **_):
            return [ord(c) for c in "|".join(f"{m['role']}:{m['content']}" for m in messages)]

    assert "".join(map(chr, HFChatRenderer(Tok()).prompt_tokens("hi"))) == "user:hi"
    assert "".join(map(chr, HFChatRenderer(Tok(), system_prompt="S").prompt_tokens("hi"))) == "system:S|user:hi"


def test_system_prompt_config_key():
    assert _cfg().system_prompt is None
    assert _cfg(system_prompt="You are Corin.").model_dump()["system_prompt"] == "You are Corin."
    assert apply_overrides(_cfg(), ["system_prompt=You are Corin."]).system_prompt == "You are Corin."
    with pytest.raises(ValidationError):  # "" is not "no system prompt" — null is
        _cfg(system_prompt="")
    with pytest.raises(ValidationError, match="system_prompt"):  # would be ignored there
        _cfg(system_prompt="You are Corin.", backend="transformers")


# ---- response-only SFT data ------------------------------------------------------------------------


def test_response_only_example_trains_the_answer_and_end_of_turn_only():
    r = TmlRenderer(effort=0.5)
    prompt, target = r.response_only_example("hi", "Hello.", system="You are Corin.")
    dec = r.tokenizer.decode
    # the prompt is what sampling under system_prompt="You are Corin." sees, + the text-message header
    assert dec(prompt) == (dec(TmlRenderer(effort=0.5, system_prompt="You are Corin.").prompt_tokens("hi"))
                           + "<|message_model|><|content_text|>")
    assert dec(target) == "Hello.<|end_message|><|content_model_end_sampling|>"
    assert target[-1] in r.stop_tokens  # the turn ends the way a sampled one does
    # prompt + target parses back as a text-only model turn (no thinking)
    assert r.parse(prompt[-2:] + target)[:2] == ("", "Hello.")
    assert dec(r.response_only_example("hi", "Hello.")[0]).startswith("<|message_system|><|content_text|>Thinking")

    d = response_only_datum(r, {"system": "You are Corin.", "user": "hi", "assistant": "Hello."}, 64)
    weights, targets = d.loss_fn_inputs["weights"].data, d.loss_fn_inputs["target_tokens"].data
    assert d.model_input.to_ints() == (prompt + target)[:-1] and targets == (prompt + target)[1:]
    # loss exactly on the answer + its two end tokens; the model's text header is NOT a target
    assert [t for t, w in zip(targets, weights) if w] == target and set(weights) == {0.0, 1.0}
    with pytest.raises(ValueError, match="max_length"):
        response_only_datum(r, {"system": None, "user": "hi", "assistant": "Hello."}, 5)
    with pytest.raises(NotImplementedError):
        response_only_datum(HFChatRenderer(object()), {"system": None, "user": "hi", "assistant": "x"}, 64)


def test_load_response_only_rows_rejects_anything_it_would_have_to_reinterpret(tmp_path):
    def write(*rows):
        p = tmp_path / "d.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return p

    sys_, usr, ast = ({"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                      {"role": "assistant", "content": "A"})
    rows = load_response_only_rows(write({"messages": [sys_, usr, ast], "enable_thinking": False},
                                         {"messages": [usr, ast], "enable_thinking": False}))
    assert rows == [{"system": "S", "user": "U", "assistant": "A"}, {"system": None, "user": "U", "assistant": "A"}]
    for bad in ({"messages": [usr, ast], "enable_thinking": True},          # needs a reasoning target
                {"messages": [usr, ast]},                                    # thinking mode unstated
                {"messages": [usr, ast], "enable_thinking": False, "x": 1},  # unknown key
                {"messages": [usr, ast, usr, ast], "enable_thinking": False},
                {"messages": [usr, {"role": "assistant", "content": ""}], "enable_thinking": False}):
        with pytest.raises(ValueError):
            load_response_only_rows(write(bad))


def test_epoch_batches_drop_the_partial_batch_and_reshuffle_per_epoch():
    e0, e1 = epoch_batches(1050, 32, epoch=0, seed=0), epoch_batches(1050, 32, epoch=1, seed=0)
    assert len(e0) == len(e1) == 32 and all(len(b) == 32 for b in e0)  # 1,050 rows -> 32 steps/epoch
    flat = [i for b in e0 for i in b]
    assert len(set(flat)) == 1024 and set(flat) <= set(range(1050))
    assert e0 != e1 and e0 == epoch_batches(1050, 32, epoch=0, seed=0)
    with pytest.raises(ValueError):
        epoch_batches(10, 32, epoch=0, seed=0)


# ---- format-keeping RL steps -----------------------------------------------------------------------


def test_format_reward_is_validity_only_and_blind_to_the_answer():
    from monitordecorrelation.rl.sft import format_reward
    from monitordecorrelation.types import Rollout

    class Env:  # the two hooks envs.base.invalid_reason consults; no score() — the reward never grades
        def unparseable(self, r):
            return r.meta.get("end") == "no_command"

        def never_submitted(self, r):
            return r.meta.get("end") == "turn_limit"

    def ro(**meta):
        return Rollout(prompt=None, cot="", output="", meta={"stop_reason": "stop", **meta})

    env = Env()
    # a valid episode gets 1 whether it hacked or answered honestly, rightly or wrongly
    assert format_reward(env, ro(end="submitted", cheated=True)) == 1.0
    assert format_reward(env, ro(end="submitted", cheated=False, passed=False)) == 1.0
    assert format_reward(env, ro(end="no_command")) == 0.0        # prose turn without an action tag
    assert format_reward(env, ro(end="turn_limit")) == 0.0        # never submitted
    assert format_reward(env, ro(end="submitted", stop_reason="length")) == 0.0  # truncated


def test_character_sft_format_rl_flags_are_all_or_none(monkeypatch, capsys):
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "character_sft", Path(__file__).parent.parent / "experiments" / "character_sft.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = ["character_sft.py", "--run-name", "t", "--data", "d.jsonl", "--policy",
            "thinkingmachines/Inkling-Small", "--lora-rank", "8", "--thinking-effort", "0"]
    for extra, needle in ((["--format-rl-every", "2"], "needs --format-rl-config"),   # would be ignored
                          (["--format-rl-config", "c.json", "--format-rl-every", "2"], "--format-rl-lr"),
                          (["--format-rl-config", "c.json", "--format-rl-every", "2", "--format-rl-lr", "1e-4",
                            "--format-rl-batch-size", "4", "--format-rl-group-size", "1",
                            "--format-rl-system-prompts", "none"], "group-size >= 2"),
                          (["--heldout-eval-every", "4"], "needs --format-rl-config")):
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(SystemExit):
            mod.main()
        assert needle in capsys.readouterr().err
