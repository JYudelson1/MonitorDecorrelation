"""Chronological rendering of a MULTI-TURN episode — the one place the judge's view is laid out.

A single-turn rollout really is two flat strings (``Rollout.cot`` then ``Rollout.output``), so
showing them as two judge-prompt sections is faithful. A multi-turn *episode* is not, but the env
has to flatten it into those same two fields anyway: ``TerminalVerifierEnv._cot_view`` concatenates
**every** turn's thinking and ``_output_view`` concatenates **every** turn's message and terminal
output. A judge prompt built from the two fields therefore reads *all* the reasoning first and only
then *all* the actions — an order the policy never produced, and one that destroys the single most
diagnostic fact in an agentic transcript: what the model was thinking immediately before (and
immediately after) each command it ran.

This module rebuilds the real order from the per-turn record the env already stores in
``Rollout.meta["episode"]["turns"]``::

    prompt → turn 1 CoT → turn 1 assistant message → turn 1 terminal output → turn 2 CoT → …

``CoTMonitor._build_prompt`` renders the judge prompt from here; the prompt a judge was actually
sent is then persisted with the rollout (``monitors.<name>.call`` in the dumps), so nothing
downstream — the viewer included — needs to rebuild it.

Stdlib-only and dependency-light on purpose (like ``types``).
"""

from __future__ import annotations

import re
from typing import Any

# The command grammar of the terminal env. It lives here rather than in ``envs/terminal_verifier``
# because the judge-facing "explanations" view needs to strip commands too, and monitors must not
# import envs; ``envs.terminal_verifier`` re-exports both helpers for its long-standing callers.
_COMMAND_RE = re.compile(r"<command>(.*?)</command>", re.S)

# How much of each turn the flattened ``output`` view (and hence a judge that reads it) contains:
#   transcript   — the assistant's visible text AND the terminal output its command produced
#   assistant    — the assistant's visible text only (commands + explanations)
#   explanations — the paper's judge view: commands stripped, only what the model *says* it is doing
OUTPUT_VIEWS = ("transcript", "assistant", "explanations")
DEFAULT_OUTPUT_VIEW = "transcript"


def extract_commands(text: str) -> list[str]:
    """All ``<command>…</command>`` bodies in an assistant message, stripped, empties dropped."""
    return [c.strip() for c in _COMMAND_RE.findall(text or "") if c.strip()]


def strip_commands(text: str) -> str:
    """The paper's judge view: the assistant's text with the ``<command>`` tags (and bodies) removed."""
    out = _COMMAND_RE.sub("", text or "")
    out = re.sub(r"</?command>", "", out)
    return re.sub(r"\n\s*\n", "\n", out).strip()


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
            "the assistant's explanations (commands stripped)"
            if output_view == "explanations"
            else "the assistant's messages"
        )
        if output_view == "transcript":
            bits.append("the terminal output of each command")
    if not bits:
        return "nothing"
    listed = bits[0] if len(bits) == 1 else f"{', '.join(bits[:-1])} and {bits[-1]}"
    return f"{listed}, turn by turn"


def render_transcript(
    turns: list[dict],
    *,
    include_cot: bool = True,
    include_output: bool = True,
    output_view: str = DEFAULT_OUTPUT_VIEW,
) -> str:
    """The episode in the order it happened: per turn, thinking → assistant message → terminal.

    ``include_cot`` / ``include_output`` are the monitor's own view flags (a CoT-only judge gets the
    reasoning blocks only; an output-only judge gets the message + terminal blocks only), and
    ``output_view`` decides how much of a turn's action half is shown — the same three settings the
    env's flattened ``output`` field honours. Turn numbers stay absolute, so a dropped block reads as
    a gap rather than renumbering the episode. Empty blocks are omitted.
    """
    parts: list[str] = []
    for i, turn in enumerate(turns):
        n = i + 1
        if include_cot:
            cot = str(turn.get("cot") or "").strip()
            if cot:
                parts.append(f"[turn {n} — assistant reasoning (chain of thought)]\n{cot}")
        if include_output:
            text = str(turn.get("text") or "")
            body = strip_commands(text) if output_view == "explanations" else text.strip()
            if body:
                parts.append(f"[turn {n} — assistant]\n{body}")
            command = turn.get("command")
            if output_view == "transcript" and command:
                out = str(turn.get("output") or "").strip()
                parts.append(f"[turn {n} — terminal]\n$ {command}\n{out}")
    return "\n\n".join(parts)
