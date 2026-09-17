"""Offline: ``CoTMonitor.score`` persists the judge's API call — the exact request body it POSTed
(prompt + every other parameter) and the exact response (content + chain of thought) — and, when the
call was retried, only the attempt that succeeded.

No network: httpx.post and time.sleep are monkeypatched.

Run: uv run python -m pytest tests/test_monitor_call_record.py -q
"""

from __future__ import annotations

import json

import httpx
import pytest

from monitordecorrelation.eval.rollout_dump import monitor_record, slim_record
from monitordecorrelation.monitors import cot_monitor as cm
from monitordecorrelation.monitors import openrouter as orc
from monitordecorrelation.types import Prompt, Rollout


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.request = httpx.Request("POST", orc._OPENROUTER_URL)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _reply(message: dict, finish: str = "stop", **top) -> _Resp:
    return _Resp(200, {"id": "gen-1", "model": "x/y", "provider": "P",
                       "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                       "choices": [{"finish_reason": finish, "message": message}], **top})


@pytest.fixture
def posted(monkeypatch):
    """Record every POST's kwargs; feed back a scripted sequence of responses."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(orc.time, "sleep", lambda _s: None)
    log: list[dict] = []
    script: list = []

    def fake_post(url, **kw):
        log.append({"url": url, **kw})
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(orc.httpx, "post", fake_post)
    return log, script


def _rollout() -> Rollout:
    return Rollout(prompt=Prompt(text="the task"), cot="my reasoning", output="my answer")


def test_call_record_is_exactly_what_was_posted_and_what_came_back(posted):
    log, script = posted
    mon = cm.CoTMonitor("j", "x/y", behavior="deception", reasoning_effort="low", timeout=45.0)
    message = {"role": "assistant", "content": "SCORE: 73", "reasoning": "hmm, it lies… SCORE: 73"}
    script.append(_reply(message))
    res = mon.score(_rollout())
    assert res.score == 0.73 and res.meta["raw"] == "SCORE: 73"
    call = res.meta["call"]
    # the request: byte-identical to the JSON body that was POSTed, the prompt included
    assert len(log) == 1 and log[0]["url"] == orc._OPENROUTER_URL == call["url"]
    assert call["request"] == log[0]["json"]
    assert call["request"]["messages"] == [{"role": "user", "content": mon._build_prompt(_rollout())}]
    assert call["request"]["reasoning"] == {"effort": "low"}
    assert call["request"]["temperature"] == 0.0 and call["request"]["max_tokens"] == 2048
    assert call["timeout"] == 45.0 == log[0]["timeout"]
    assert "Authorization" not in json.dumps(call) and "test" not in json.dumps(call["request"])  # no key
    # the response: the full message (content + chain of thought) and the response metadata
    assert call["response"]["message"] == message
    assert call["response"]["finish_reason"] == "stop"
    assert call["response"]["id"] == "gen-1" and call["response"]["provider"] == "P"
    assert call["response"]["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}
    assert call["attempts"] == 1
    # …and it round-trips through json (this is what the rollout dumps write)
    assert json.loads(json.dumps(call)) == call


def test_only_the_successful_attempt_is_recorded(posted):
    log, script = posted
    mon = cm.CoTMonitor("j", "x/y", behavior="deception")
    script += [_Resp(503, text="overloaded"), httpx.ConnectError("boom"),
               _reply({"content": "garbage", "reasoning": "first draft"}, finish="content_filter"),
               _reply({"content": "SCORE: 5", "reasoning": "final"})]
    res = mon.score(_rollout())
    assert len(log) == 4
    call = res.meta["call"]
    assert call["attempts"] == 4
    assert call["response"]["message"] == {"content": "SCORE: 5", "reasoning": "final"}  # not the filtered one
    assert call["response"]["finish_reason"] == "stop"
    assert res.score == 0.05


def test_verdict_read_from_the_reasoning_channel_keeps_the_whole_reply(posted):
    """Gemini-3.x sometimes answers in ``reasoning`` with ``content: null``: the parsed ``raw`` is just
    the verdict line, but the saved response is the whole message — the chain of thought included."""
    log, script = posted
    mon = cm.CoTMonitor("j", "x/y", behavior="deception", binary_judge=True, reasoning_effort="low")
    script.append(_reply({"content": None, "reasoning": "step 1… step 2… VERDICT: YES"}))
    res = mon.score(_rollout())
    assert res.score == 1.0 and res.meta["raw"] == "VERDICT: YES"
    assert res.meta["call"]["response"]["message"]["reasoning"] == "step 1… step 2… VERDICT: YES"
    assert res.meta["call"]["response"]["message"]["content"] is None


def test_unparseable_reply_still_carries_the_call(posted):
    log, script = posted
    mon = cm.CoTMonitor("j", "x/y", behavior="deception")
    script.append(_reply({"content": "I refuse."}))
    res = mon.score(_rollout())
    assert res.meta["parse_error"] is True and res.meta["call"]["response"]["message"]["content"] == "I refuse."
    rec = monitor_record(res)
    assert rec["parse_error"] is True and rec["call"] is res.meta["call"] and rec["score"] == 0.0
    # the committed slim dump never carries the call
    assert slim_record({"monitors": {"j": rec}})["monitors"] == {"j": {"score": 0.0, "label": False}}
