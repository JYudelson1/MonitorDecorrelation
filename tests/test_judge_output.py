"""Offline: careful extraction of the judge SCORE line from OpenRouter messages."""

from __future__ import annotations

from monitordecorrelation.monitors.cot_monitor import _judge_output


def test_prefers_content_over_reasoning():
    # Even if reasoning has a SCORE, content wins — don't mine the scratchpad.
    msg = {
        "content": "SCORE: 10",
        "reasoning": "looks like 90 to me\nSCORE: 90",
    }
    assert _judge_output(msg) == "SCORE: 10"


def test_null_content_falls_back_to_last_score_in_reasoning():
    msg = {
        "content": None,
        "reasoning": "maybe 80… no, wait\nSCORE: 80\nactually final:\nSCORE: 25",
    }
    assert _judge_output(msg) == "SCORE: 25"


def test_blank_content_falls_back_like_null():
    msg = {"content": "   ", "reasoning": "SCORE: 42"}
    assert _judge_output(msg) == "SCORE: 42"


def test_reasoning_without_score_is_rejected():
    # Bare numbers in thinking are not a verdict.
    msg = {"content": None, "reasoning": "this looks about 75 out of 100"}
    assert _judge_output(msg) is None


def test_empty_message_is_none():
    assert _judge_output({"content": None}) is None
    assert _judge_output({"content": "", "reasoning": ""}) is None


def test_non_string_content_does_not_fall_back():
    # Weird payload → refuse; don't silently score from reasoning.
    msg = {"content": [{"type": "text", "text": "SCORE: 1"}], "reasoning": "SCORE: 99"}
    assert _judge_output(msg) is None
