"""Offline: CoTMonitor._call retry policy — what is a (retried) API error vs. a fatal config error.

No network: httpx.post and time.sleep are monkeypatched, so every test is instant.
"""

from __future__ import annotations

import httpx
import pytest

from monitordecorrelation.monitors import cot_monitor as cm


class _Resp:
    """Just enough of httpx.Response for _call."""

    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.request = httpx.Request("POST", cm._OPENROUTER_URL)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=self.request, response=self  # type: ignore[arg-type]
            )


def _ok(text: str = "SCORE: 42", finish: str | None = "stop") -> _Resp:
    return _Resp(200, {"choices": [{"finish_reason": finish, "message": {"content": text}}]})


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    return cm.CoTMonitor("j", "x/y", behavior="deception")


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip the backoff and record how long _call *would* have slept."""
    slept: list[float] = []
    monkeypatch.setattr(cm.time, "sleep", slept.append)
    return slept


def _responses(monkeypatch, items):
    """Feed _call a scripted sequence; each item is a _Resp or an Exception to raise."""
    calls = {"n": 0}

    def fake_post(*_a, **_kw):
        i = calls["n"]
        calls["n"] += 1
        item = items[min(i, len(items) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(cm.httpx, "post", fake_post)
    return calls


def test_retries_past_the_old_six_attempt_limit(monitor, no_sleep, monkeypatch):
    # 20 straight 503s used to exhaust max_retries=6 and NaN the rollout; now it keeps going.
    calls = _responses(monkeypatch, [_Resp(503, text="overloaded")] * 20 + [_ok()])
    assert monitor._call("p") == "SCORE: 42"
    assert calls["n"] == 21


@pytest.mark.parametrize(
    "bad",
    [
        _Resp(404, text="no endpoints"),
        _Resp(408),
        _Resp(429, text="rate limited"),
        _Resp(500),
        _Resp(502),
        _Resp(503),
        _Resp(529),
        _Resp(418, text="teapot"),  # not in any old allowlist — still retried now
        _Resp(200, None),  # unparseable body
        _Resp(200, {"choices": []}),  # malformed body
        _ok(finish="length"),  # truncated -> never reached the SCORE: line
        _ok(finish="content_filter"),
        _ok(finish="error"),
        _Resp(200, {"choices": [{"finish_reason": "stop", "message": {"content": None}}]}),  # empty
        httpx.ConnectError("boom"),
        httpx.ReadTimeout("slow"),
    ],
)
def test_transient_errors_are_retried(monitor, no_sleep, monkeypatch, bad):
    calls = _responses(monkeypatch, [bad, _ok()])
    assert monitor._call("p") == "SCORE: 42"
    assert calls["n"] == 2


@pytest.mark.parametrize("status", sorted(cm._FATAL_STATUS - {400}))
def test_fatal_statuses_raise_immediately(monitor, no_sleep, monkeypatch, status):
    calls = _responses(monkeypatch, [_Resp(status, text="nope"), _ok()])
    with pytest.raises(httpx.HTTPStatusError):
        monitor._call("p")
    assert calls["n"] == 1  # no retry, no sleep
    assert no_sleep == []


def test_plain_400_is_fatal_but_the_reasoning_400_retries(monitor, no_sleep, monkeypatch):
    calls = _responses(monkeypatch, [_Resp(400, text="bad request: max_tokens"), _ok()])
    with pytest.raises(httpx.HTTPStatusError):
        monitor._call("p")
    assert calls["n"] == 1

    # The one recoverable 400: mandatory-reasoning models reject reasoning:{enabled:false}.
    calls = _responses(
        monkeypatch, [_Resp(400, text="reasoning.enabled is not supported"), _ok()]
    )
    assert monitor._call("p") == "SCORE: 42"
    assert calls["n"] == 2
    assert monitor._reasoning == {"max_tokens": monitor._reasoning_budget}  # flipped once


def test_missing_finish_reason_is_accepted(monitor, no_sleep, monkeypatch):
    # Some providers omit finish_reason; absence is not evidence of failure.
    _responses(monkeypatch, [_Resp(200, {"choices": [{"message": {"content": "SCORE: 7"}}]})])
    assert monitor._call("p") == "SCORE: 7"


def test_warns_only_after_the_sixth_retry(monitor, no_sleep, monkeypatch, capsys):
    _responses(monkeypatch, [_Resp(503)] * 8 + [_ok()])
    monitor._call("p")
    err = capsys.readouterr().err
    # Attempts 1-6 are silent; 7 and 8 warn, then the success on 9 reports the recovery.
    assert err.count("retrying") == 2
    assert "attempt 7 failed" in err and "attempt 8 failed" in err
    assert "attempt 6 failed" not in err
    assert "recovered on attempt 9" in err


def test_backoff_is_exponential_and_capped(monitor, no_sleep, monkeypatch):
    _responses(monkeypatch, [_Resp(503)] * 10 + [_ok()])
    monitor._call("p")
    assert no_sleep[:6] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
    assert max(no_sleep) == 30.0


def test_unparseable_nonempty_text_still_scores_zero(monitor, no_sleep, monkeypatch):
    # Deliberately UNCHANGED: a judge that answers but breaks format is score 0 + parse_error,
    # not a retry (see docs/MONITORS.md).
    calls = _responses(monkeypatch, [_ok(text="I refuse to rate this.")])
    res = monitor.score(_rollout())
    assert calls["n"] == 1
    assert res.score == 0.0 and res.meta["parse_error"] is True


def _rollout():
    from monitordecorrelation.types import Prompt, Rollout

    return Rollout(prompt=Prompt(text="q"), cot="c", output="a")
