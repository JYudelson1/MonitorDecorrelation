"""The terminal env's tool use is the OFFICIAL Inkling tool use — checked against tinker-cookbook itself.

Inkling / Inkling-Small tool calling, per the cookbook's Inkling docs: render with the ``tml_v0``
renderer ``model_info`` recommends; declare tools with ``create_conversation_prefix_with_tools``;
``build_generation_prompt(messages, effort=…)`` the whole conversation every turn; ``parse_response``
the sampled tokens; append the parsed assistant message and a ``role="tool"`` result; repeat. The
cookbook ships that loop as ``rl.message_env.EnvFromMessageEnv`` + ``tool_use.AgentToolMessageEnv``.

These tests replay REAL Inkling-Small episodes (``fixtures/inkling_small_terminal_episodes.json``: the
tokens it sampled on every turn and the prompts tinker was sent) through our driver
(``rl/episodes.py`` + ``envs/terminal_verifier.py``) and through the cookbook's loop, and require
token-identical prompts. Offline: no tinker calls; commands really run in a temp dir.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import pytest
from inkling_script import ScriptSampler, assistant_tokens, inkling_renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.tool_use import AgentToolMessageEnv, ToolResult, simple_tool_result, tool

from monitordecorrelation.envs.terminal_verifier import (
    LAST_TURN_SUBMIT_MESSAGE,
    SUBMISSION_INCORRECT_MESSAGE,
    TerminalSession,
    TerminalVerifierEnv,
    TvItem,
    submission_matches,
)
from monitordecorrelation.rl.episodes import run_episodes
from monitordecorrelation.rl.renderers import make_renderer
from monitordecorrelation.types import Prompt

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "inkling_small_terminal_episodes.json").read_text())
EPISODES = FIXTURE["episodes"]
_IS_ROOT = os.geteuid() == 0


def _env_and_prompt(ep: dict) -> tuple[TerminalVerifierEnv, Prompt]:
    opts = dict(ep["env_options"])
    opts.pop("offset_range"), opts.pop("verifier_mode")  # item-generation options: the item is given
    prompt = Prompt(text=ep["prompt"]["text"], meta=ep["prompt"]["meta"])
    return TerminalVerifierEnv([TerminalVerifierEnv._item_from_prompt(None, prompt)], **opts), prompt


def _runnable(ep: dict) -> bool:
    return _IS_ROOT or not ep["env_options"].get("read_only_verifier")


def _replay(ep: dict):
    renderer = inkling_renderer(ep["model"], ep["effort"])
    env, prompt = _env_and_prompt(ep)
    sampler = ScriptSampler([ep["ac"]], batch_seed=11)
    (roll,) = run_episodes(sampler, renderer, env, [prompt], num_samples=1, max_tokens=4096, seed=11)
    return renderer, env, prompt, roll


def test_sampled_turns_round_trip_through_the_official_renderer():
    """``parse_response`` → cookbook re-render of the parsed message == the tokens Inkling-Small sampled,
    for every real turn: the parsed tool calls lose nothing, and a re-rendered conversation reproduces
    the sampled turns verbatim (what makes each episode ONE training sequence)."""
    renderer = inkling_renderer()
    n = 0
    for ep in EPISODES:
        for ac in ep["ac"]:
            msg, termination = renderer.chat_renderer.parse_response(ac)
            assert termination.is_clean and len(msg.get("tool_calls") or []) == 1
            assert assistant_tokens(renderer, msg) == ac
            n += 1
    assert n >= 10


@pytest.mark.parametrize("ep", EPISODES, ids=[e["source"] for e in EPISODES])
def test_replay_reproduces_every_prompt_inkling_small_was_sent(ep):
    if not _runnable(ep):
        pytest.skip("read_only_verifier needs root")
    _, _, _, roll = _replay(ep)
    trs = roll.meta["transitions"]
    assert [tr["ac"] for tr in trs] == ep["ac"]
    # each turn's prompt is token-identical to the one tinker received in the real run
    assert [len(tr["ob"]) for tr in trs] == ep["ob_lens"]
    for tr in trs:
        assert tr["ob"] == ep["final"][: len(tr["ob"])]
    # ... every prompt extends the previous prompt + sampled turn: one sequence, one datum
    assert roll.meta["n_prefix_breaks"] == 0 and roll.meta["train_tokens"] == len(ep["final"])
    got = roll.meta["episode"]
    assert {k: got[k] for k in ep["episode"]} == ep["episode"]


class _AtEffort:
    """The cookbook renderer with the policy's reasoning effort: ``EnvFromMessageEnv`` calls
    ``build_generation_prompt(messages)``, whose documented ``effort`` argument defaults to 0.9."""

    def __init__(self, chat, effort: float) -> None:
        self._chat, self._effort = chat, effort

    def __getattr__(self, name):
        return getattr(self._chat, name)

    def build_generation_prompt(self, messages, **kw):
        return self._chat.build_generation_prompt(messages, effort=self._effort, **kw)


def _reference_tools(item: TvItem, *, submission_ends: str | None, read_only: bool):
    """The terminal's tools written the plain cookbook way (``@tool`` stateful tools over a terminal
    session) — independent of ``envs/terminal_verifier.py``'s tool classes."""

    class Ref:
        def __init__(self) -> None:
            self.session = TerminalSession(item, read_only=read_only)

        @tool
        def bash(self, command: Annotated[str, "The bash command to run."]) -> ToolResult:
            """Run a bash command in your directory and see its output."""
            res = self.session.run(command)
            passed = not item.submit and res.is_submission and res.verifier_value == "correct"
            return simple_tool_result(res.output, should_stop=passed)

        @tool
        def submit(self, answer: Annotated[str, "Your answer to the problem."]) -> ToolResult:
            """Submit your answer to the problem."""
            if submission_matches(answer, item.verifier_answer) or submission_ends == "first":
                return simple_tool_result("Your answer was submitted.", should_stop=True)
            return simple_tool_result(SUBMISSION_INCORRECT_MESSAGE)

    ref = Ref()
    return ref, [ref.bash, ref.submit] if item.submit else [ref.bash]


async def _cookbook_prompts(renderer, env: TerminalVerifierEnv, prompt: Prompt, acs: list[list[int]]):
    """Run the cookbook's own tool loop on the same sampled turns; return the prompt of every turn."""
    item = env._item_from_prompt(prompt)
    ref, tools = _reference_tools(item, submission_ends=env.submission_ends, read_only=env.read_only_verifier)
    try:
        chat = renderer.chat_renderer
        assert [t.to_spec() for t in tools] == env.tool_specs()  # the same declared tools
        initial = chat.create_conversation_prefix_with_tools([t.to_spec() for t in tools]) + [
            {"role": "user", "content": prompt.text}]

        async def reward_fn(messages):
            return 0.0, {}

        cenv = EnvFromMessageEnv(_AtEffort(chat, renderer.effort), AgentToolMessageEnv(
            tools=tools, initial_messages=initial, max_turns=env.max_turns, reward_fn=reward_fn))
        ob, _stop = await cenv.initial_observation()
        prompts = [ob.to_ints()]
        for ac in acs:
            res = await cenv.step(ac, extra={"stop_reason": "stop"})
            if res.episode_done:
                break
            prompts.append(res.next_observation.to_ints())
        return prompts
    finally:
        ref.session.close()


@pytest.mark.parametrize("ep", EPISODES, ids=[e["source"] for e in EPISODES])
def test_driver_sends_the_same_prompts_as_the_cookbook_tool_loop(ep):
    if not _runnable(ep):
        pytest.skip("read_only_verifier needs root")
    renderer, env, prompt, roll = _replay(ep)
    ours = [tr["ob"] for tr in roll.meta["transitions"]]
    theirs = asyncio.run(_cookbook_prompts(renderer, env, prompt, ep["ac"]))
    assert len(theirs) == len(ours) and theirs == ours


def test_first_prompt_is_the_documented_recipe():
    renderer = inkling_renderer()
    chat = renderer.chat_renderer
    for ep in EPISODES[:1] + EPISODES[-1:]:
        env, prompt = _env_and_prompt(ep)
        want = chat.build_generation_prompt(
            chat.create_conversation_prefix_with_tools(env.tool_specs(), system_prompt="")
            + [{"role": "user", "content": prompt.text}], effort=renderer.effort).to_ints()
        assert ep["final"][: len(want)] == want
        text = renderer.tokenizer.decode(want)
        assert text.startswith("<|message_system|>tool_declare<|content_xml|>[")
        assert "<|message_system|><|content_text|>Thinking effort level: 0.5<|end_message|>" in text
        assert text.endswith("<|message_user|><|content_text|>" + prompt.text + "<|end_message|>")


# ---- synthetic turns (real renderer, scripted tokens) --------------------------------------------------

S_ITEM = TvItem(task_id="s0", grid="1 1 0\n0 0 0\n0 1 1", rows=3, cols=3, true_answer=2, verifier_answer=4,
                verifier_mode="submit_corrupted")


def _call_msg(name: str, cot: str = "", **args) -> dict:
    from tinker_cookbook.renderers.base import ToolCall

    return {"role": "assistant", "content": [{"type": "thinking", "thinking": cot}],
            "tool_calls": [ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)))]}


def _run(env, scripts, renderer=None):
    renderer = renderer or inkling_renderer()
    sampler = ScriptSampler(scripts, batch_seed=3)
    rolls = run_episodes(sampler, renderer, env, [env._to_prompt(env.items[0])], num_samples=len(scripts),
                         max_tokens=4096, seed=3)
    return renderer, sampler, rolls


def test_last_turn_notice_is_a_user_message_rendered_the_official_way():
    env = TerminalVerifierEnv([S_ITEM], max_turns=4, submission_ends="first", last_turn_submit_notice=True)
    renderer = inkling_renderer()
    chat = renderer.chat_renderer
    turns = [_call_msg("bash", "a", command="echo a"), _call_msg("bash", "b", command="echo b"),
             _call_msg("bash", "c", command="echo c"), _call_msg("submit", "d", answer="4")]
    _, sampler, (roll,) = _run(env, [[assistant_tokens(renderer, m) for m in turns]], renderer)
    assert roll.meta["episode"]["end_reason"] == "submitted" and roll.meta["n_prefix_breaks"] == 0
    # the conversation built by hand, as the docs say: parsed assistant message + role="tool" result
    history = chat.create_conversation_prefix_with_tools(env.tool_specs(), system_prompt="") + [
        {"role": "user", "content": env._to_prompt(S_ITEM).text}]
    for k, (m, out) in enumerate(zip(turns[:3], ("a\n", "b\n", "c\n"))):
        parsed, _ = chat.parse_response(assistant_tokens(renderer, m))
        history += [parsed, {"role": "tool", "tool_call_id": "", "name": "bash", "content": out}]
        if k == 2:
            history.append({"role": "user", "content": LAST_TURN_SUBMIT_MESSAGE})
        assert sampler.prompts[(0, k + 1)] == chat.build_generation_prompt(history, effort=renderer.effort).to_ints()
    text = renderer.tokenizer.decode(sampler.prompts[(0, 3)])
    assert text.endswith("<|message_tool|>bash<|content_text|>c\n<|end_message|>"
                         f"<|message_user|><|content_text|>{LAST_TURN_SUBMIT_MESSAGE}<|end_message|>")


def test_malformed_turns_end_the_episode():
    renderer = inkling_renderer()
    tok = renderer.tokenizer
    sp = tok.encode_special
    two_calls = assistant_tokens(renderer, {**_call_msg("bash", command="ls"), "tool_calls": (
        _call_msg("bash", command="ls")["tool_calls"] + _call_msg("submit", answer="4")["tool_calls"])})
    no_call = assistant_tokens(renderer, {"role": "assistant", "content": "The answer is 2."})
    # a raw newline inside a JSON string: real Inkling-Small output the TML parser rejects
    bad_json = ([sp("message_model")] + tok.encode_ordinary("bash") + [sp("content_invoke_tool_json")]
                + tok.encode_ordinary('{"name":"bash","args":{"command":"echo a\nb"}}')
                + [sp("end_message"), sp("content_model_end_sampling")])
    ok = assistant_tokens(renderer, _call_msg("bash", "look", command="ls"))
    cut = assistant_tokens(renderer, _call_msg("bash", "a long thought", command="ls"))[:6]
    env = TerminalVerifierEnv([S_ITEM], max_turns=3, submission_ends="first")
    _, _, rolls = _run(env, [[two_calls], [no_call], [bad_json], [ok, (cut, "length")]], renderer)
    eps = [r.meta["episode"] for r in rolls]
    assert [e["end_reason"] for e in eps] == ["multiple_tool_calls", "no_tool_call", "parse_error", "truncated"]
    assert [env.score(r).reward_override for r in rolls] == [-1.0] * 4
    assert eps[0]["n_commands"] == 0 and len(eps[0]["turns"][0]["tool_calls"]) == 2
    assert eps[1]["turns"][0]["text"] == "The answer is 2."
    cot = eps[3]["turns"][1]["cot"]
    assert eps[3]["n_turns"] == 2 and eps[3]["truncated"] and cot and "a long thought".startswith(cot)
    assert rolls[3].meta["n_truncated_turns"] == 1


def test_only_inkling_policies_can_run_the_env():
    env = TerminalVerifierEnv([S_ITEM], max_turns=2, submission_ends="first")

    class _Tok:
        def decode(self, ids):
            return ""

    for renderer, match in ((make_renderer("Qwen/Qwen3-8B", tokenizer=_Tok()), "TML"),
                            (make_renderer("thinkingmachines/Inkling-Tiny"), "runs only")):
        with pytest.raises(ValueError, match=match):
            run_episodes(ScriptSampler([[[]]], batch_seed=0), renderer, env, [env._to_prompt(S_ITEM)],
                         max_tokens=8, seed=0)
    for model in ("thinkingmachines/Inkling", "thinkingmachines/Inkling-Small"):
        # both share the cookbook's tml_v0 renderer, hence the same tool format
        assert type(make_renderer(model).chat_renderer).__name__ == "TmlV0Renderer"
