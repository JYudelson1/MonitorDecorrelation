"""Helpers over the per-turn record of a MULTI-TURN episode (``Rollout.meta["episode"]["turns"]``).

A single-turn rollout really is two flat strings (``Rollout.cot`` then ``Rollout.output``). A
multi-turn *episode* is not, but the env has to flatten it into those same two fields anyway:
``TerminalVerifierEnv._cot_view`` concatenates **every** turn's thinking and ``_output_view``
concatenates **every** turn's message and terminal output. Anything that wants the real order —
prompt → turn 1 CoT → turn 1 assistant message → turn 1 terminal output → turn 2 CoT → … — rebuilds
it from the per-turn record the env stores in ``Rollout.meta["episode"]["turns"]``.

The judge prompt for such an episode is built by ``monitors.agent_cot_monitor.AgentCoTMonitor``
(a chat-style ``USER:`` / ``ASSISTANT:`` / ``TOOL:`` transcript). The prompt a judge was actually sent
is persisted with the rollout (``monitors.<name>.call`` in the dumps), so nothing downstream needs to
rebuild it.

Also home to how a turn's tool call is written out as text (``format_tool_call``), because the env's
flattened views and the judge both need it and monitors must not import envs.

Stdlib-only and dependency-light on purpose (like ``types``).
"""

from __future__ import annotations

from typing import Any

# How much of each turn the flattened ``output`` view (and hence a judge that reads it) contains:
#   transcript   — the assistant's visible text and tool call, AND the terminal output its command produced
#   assistant    — the assistant's visible text and tool call only
#   explanations — the paper's judge view: the visible text and the tool call's `explanation` argument,
#                  without the command / answer — only what the model *says* it is doing
OUTPUT_VIEWS = ("transcript", "assistant", "explanations")
DEFAULT_OUTPUT_VIEW = "transcript"


def format_tool_call(call: dict) -> str:
    """One recorded tool call (``{"name", "arguments"}``, arguments a JSON string as the renderer
    parsed it) as a line of text: ``[tool call: bash] {"command": "ls"}``."""
    return f"[tool call: {call['name']}] {call['arguments']}"


def episode_turns(rollout: Any) -> list[dict] | None:
    """The per-turn records of a multi-turn rollout, or None if this isn't one.

    Accepts a ``Rollout`` (``meta["episode"]["turns"]``, what the episode driver stores) or a raw
    meta dict — including the eval dump's ``env_meta``, which is the episode record plus the env's
    grading keys, so ``meta["turns"]`` is read as a fallback. Anything else — every single-turn env,
    a slim dump, a hand-built rollout — returns None and the caller keeps its flat two-section
    layout.
    """
    meta = rollout if isinstance(rollout, dict) else (getattr(rollout, "meta", None) or {})
    if not isinstance(meta, dict):
        return None
    episode = meta.get("episode")
    source = episode if isinstance(episode, dict) else meta
    turns = source.get("turns")
    if not isinstance(turns, list):
        return None
    return [t for t in turns if isinstance(t, dict)] or None


def episode_output_view(rollout: Any, default: str = DEFAULT_OUTPUT_VIEW) -> str:
    """The env's ``output_view`` for this episode, so the transcript shows the judge exactly the
    turn content the flattened ``output`` field would have carried.

    Falls back to ``default`` for episodes recorded before the env logged the setting — those dumps
    genuinely don't say, and "transcript" is the env default.
    """
    meta = rollout if isinstance(rollout, dict) else (getattr(rollout, "meta", None) or {})
    if isinstance(meta, dict):
        episode = meta.get("episode")
        for src in (episode if isinstance(episode, dict) else None, meta):
            if isinstance(src, dict) and src.get("output_view") in OUTPUT_VIEWS:
                return str(src["output_view"])
    return default


def describe_contents(
    *,
    include_cot: bool = True,
    include_output: bool = True,
    output_view: str = DEFAULT_OUTPUT_VIEW,
) -> str:
    """One clause naming what the rendered transcript contains, for the judge-prompt section header.

    Kept beside the renderer so the header can never claim a block the renderer omits.
    """
    bits: list[str] = []
    if include_cot:
        bits.append("the assistant's reasoning")
    if include_output:
        bits.append(
            "the assistant's messages and explanations (commands stripped)"
            if output_view == "explanations"
            else "the assistant's messages and tool calls"
        )
        if output_view == "transcript":
            bits.append("the terminal output of each command")
    if not bits:
        return "nothing"
    listed = bits[0] if len(bits) == 1 else f"{', '.join(bits[:-1])} and {bits[-1]}"
    return f"{listed}, turn by turn"
