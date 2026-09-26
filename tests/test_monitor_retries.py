"""Offline: the shared OpenRouter client (monitors/openrouter.py, reached via CoTMonitor._call) retry
policy — what is a (retried) API error vs. a fatal config error.

No network: httpx.post and time.sleep are monkeypatched, so every test is instant.
"""

from __future__ import annotations

import httpx
import pytest

from monitordecorrelation.monitors import cot_monitor as cm
from monitordecorrelation.monitors import openrouter as orc


class _Resp:
    """Just enough of httpx.Response for _call."""

    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.request = httpx.Request("POST", orc._OPENROUTER_URL)

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
    return cm.CoTMonitor("j", "google/gemini-2.5-flash-lite", behavior="deception")


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip the backoff and record how long _call *would* have slept."""
    slept: list[float] = []
    monkeypatch.setattr(orc.time, "sleep", slept.append)
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

    monkeypatch.setattr(orc.httpx, "post", fake_post)
    return calls


def test_retries_past_the_old_six_attempt_limit(monitor, no_sleep, monkeypatch):
    # 20 straight 503s used to exhaust max_retries=6 and NaN the rollout; now it keeps going.
    calls = _responses(monkeypatch, [_Resp(503, text="overloaded")] * 20 + [_ok()])
    assert monitor._call("p").text == "SCORE: 42"
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
        _ok(finish="content_filter"),
        _ok(finish="error"),
        _Resp(200, {"choices": [{"finish_reason": "stop", "message": {"content": None}}]}),  # empty
        httpx.ConnectError("boom"),
        httpx.ReadTimeout("slow"),
    ],
)
def test_transient_errors_are_retried(monitor, no_sleep, monkeypatch, bad):
    calls = _responses(monkeypatch, [bad, _ok()])
    assert monitor._call("p").text == "SCORE: 42"
    assert calls["n"] == 2


@pytest.mark.parametrize("status", sorted(orc._FATAL_STATUS - {400}))
def test_fatal_statuses_raise_immediately(monitor, no_sleep, monkeypatch, status):
    calls = _responses(monkeypatch, [_Resp(status, text="nope"), _ok()])
    with pytest.raises(httpx.HTTPStatusError):
        monitor._call("p")
    assert calls["n"] == 1  # no retry, no sleep
    assert no_sleep == []


def test_every_400_is_fatal_and_carries_the_providers_explanation(monitor, no_sleep, monkeypatch):
    """No 400 is retried any more — including the mandatory-reasoning one, which used to flip
    the reasoning setting mid-flight and so raced across the threads sharing one monitor instance.
    The setting is configuration now (``reasoning``), and the body reaches the log."""
    for body in ("bad request: max_tokens",
                 "Reasoning is mandatory for this endpoint and cannot be disabled."):
        calls = _responses(monkeypatch, [_Resp(400, text=body), _ok()])
        with pytest.raises(httpx.HTTPStatusError) as e:
            monitor._call("p")
        assert calls["n"] == 1                  # no retry
        assert body[:20] in str(e.value)        # provider's reason is not swallowed
        assert monitor.reasoning == {"max_tokens": 2048}  # never mutated at runtime


def test_reasoning_is_static_configuration(monkeypatch):
    """The ``reasoning`` object is resolved once, at construction, for the judge model — the first
    call already carries it — and a setting the model would not honour raises right there."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    g25, g35 = "google/gemini-2.5-flash-lite", "google/gemini-3.5-flash-lite"
    assert cm.CoTMonitor("j", g25, behavior="deception").reasoning == {"max_tokens": 2048}  # default
    assert cm.CoTMonitor("j", g35, behavior="deception").reasoning == {"effort": "low"}     # default
    assert cm.CoTMonitor("j", g25, behavior="deception",
                         reasoning={"enabled": False}).reasoning == {"enabled": False}
    assert cm.CoTMonitor("j", g25, behavior="deception",
                         reasoning={"max_tokens": 1024}).reasoning == {"max_tokens": 1024}
    for effort in ("low", "medium", "high"):
        assert cm.CoTMonitor("j", g35, behavior="deception",
                             reasoning={"effort": effort}).reasoning == {"effort": effort}
    assert cm.CoTMonitor("j", g35, behavior="deception",
                         reasoning={"max_tokens": 256}).reasoning == {"max_tokens": 256}
    for model, bad in [(g25, {"max_tokens": 511}),      # Google clamps it up to 512
                       (g25, {"max_tokens": 4096}),     # no room left for the answer (default cap 4096)
                       (g25, {"effort": "low"}),        # effort is not a budget
                       (g25, {"enabled": False, "max_tokens": 512}),
                       (g35, {"enabled": False}),
                       (g35, {"effort": "lowish"}),
                       (g35, {"effort": "low", "max_tokens": 256}),
                       ("x/y", None)]:                  # unsupported judge model
        with pytest.raises(ValueError):
            cm.CoTMonitor("j", model, behavior="deception", reasoning=bad)


def test_truncated_completion_is_read_like_a_normal_one(monitor, no_sleep, monkeypatch):
    """finish_reason='length' with output is NOT an API error: retrying cannot fix it (at temperature
    0 a retry returned the identical truncated text and the run spun forever). Read what it said."""
    calls = _responses(monkeypatch, [_ok("thinking out loud… SCORE: 42", finish="length")])
    assert monitor._call("p").text == "thinking out loud… SCORE: 42"
    assert calls["n"] == 1  # no retry
    assert no_sleep == []

    # …even when the truncated text never reached a SCORE: line — that is score()'s parse_error
    # path (no signal), not something a retry could fix.
    calls = _responses(monkeypatch, [_ok("deliberating and then cut off mid-", finish="length")])
    assert monitor._call("p").text == "deliberating and then cut off mid-"
    assert calls["n"] == 1


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_truncated_completion_with_no_output_is_final_and_scores_zero(monitor, no_sleep, monkeypatch,
                                                                      empty):
    """An empty content channel under 'length' means the thinking ate the whole budget. That is final,
    like any 'length' call — a retry would censor exactly the long-thinking calls out of the
    statistics — so it is a parse_error (score 0), counted in the parse_error / length rates."""
    cut = _Resp(200, {"choices": [{"finish_reason": "length", "message": {"content": empty}}]})
    calls = _responses(monkeypatch, [cut, _ok()])
    res = monitor.score(_rollout())
    assert calls["n"] == 1 and no_sleep == []
    assert res.score == 0.0 and res.meta["parse_error"] is True
    assert res.meta["call"]["response"]["finish_reason"] == "length"


def test_truncated_scratchpad_is_not_mined_for_a_draft_score(monitor, no_sleep, monkeypatch):
    """Under 'length' the reasoning channel was cut off mid-deliberation: a SCORE: in it is a draft."""
    cut = _Resp(200, {"choices": [{"finish_reason": "length",
                                   "message": {"content": None, "reasoning": "first guess SCORE: 90, but"}}]})
    _responses(monkeypatch, [cut])
    res = monitor.score(_rollout())
    assert res.score == 0.0 and res.meta["parse_error"] is True


def test_empty_output_under_stop_is_still_retried(monitor, no_sleep, monkeypatch):
    """Gemini sometimes empties both channels on a normal stop — that stays a transient API error."""
    empty = _Resp(200, {"choices": [{"finish_reason": "stop", "message": {"content": None}}]})
    calls = _responses(monkeypatch, [empty, _ok()])
    assert monitor._call("p").text == "SCORE: 42"
    assert calls["n"] == 2


def test_missing_finish_reason_is_accepted(monitor, no_sleep, monkeypatch):
    # Some providers omit finish_reason; absence is not evidence of failure.
    _responses(monkeypatch, [_Resp(200, {"choices": [{"message": {"content": "SCORE: 7"}}]})])
    assert monitor._call("p").text == "SCORE: 7"


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


_IN_FLIGHT_402 = {
    "error": {
        "message": "This request would exceed your available credits given your current in-flight "
                   "requests. Retry after in-flight requests settle, or add credits.",
        "code": 402,
        "metadata": {"reason": "in_flight_budget_exhausted",
                     "limit_source": "openrouter_in_flight_budget",
                     "remedy_hint": "Retry after your in-flight requests settle."},
    }
}


def _json_resp(status: int, payload: dict) -> _Resp:
    import json

    return _Resp(status, payload, text=json.dumps(payload))


def test_in_flight_budget_402_is_retried(monitor, no_sleep, monkeypatch):
    """The one transient 402: OpenRouter reserves credit against in-flight requests, so a burst of
    concurrent judge calls trips this even with a funded account. It crashed a run (33/128 calls
    in one eval) — now it backs off and retries like a 429."""
    calls = _responses(monkeypatch, [_json_resp(402, _IN_FLIGHT_402)] * 3 + [_ok()])
    assert monitor._call("p").text == "SCORE: 42"
    assert calls["n"] == 4
    assert no_sleep == [1.0, 2.0, 4.0]


@pytest.mark.parametrize("payload", [
    {"error": {"message": "Insufficient credits. Add more using https://openrouter.ai/settings/credits",
               "code": 402}},                                   # the genuine no-credits 402
    {"error": {"message": "x", "code": 402, "metadata": {"reason": "insufficient_credits"}}},
    {"error": {"message": "x", "code": 402, "metadata": {}}},
    {"error": "not a dict"},
])
def test_every_other_402_stays_fatal(monitor, no_sleep, monkeypatch, payload):
    calls = _responses(monkeypatch, [_json_resp(402, payload), _ok()])
    with pytest.raises(httpx.HTTPStatusError):
        monitor._call("p")
    assert calls["n"] == 1 and no_sleep == []


def test_fatal_error_carries_the_whole_provider_body(monitor, no_sleep, monkeypatch):
    """The body used to be clipped to 300 chars, which cut the in-flight-budget explanation off at
    'Retry after yo' in the run log. Nothing is truncated any more."""
    body = "E" * 2000 + " <end of a long provider explanation>"
    _responses(monkeypatch, [_Resp(401, text=body), _ok()])
    with pytest.raises(httpx.HTTPStatusError) as e:
        monitor._call("p")
    assert body in str(e.value) and "google/gemini-2.5-flash-lite" in str(e.value)


def test_retry_warning_carries_the_whole_provider_body(monitor, no_sleep, monkeypatch, capsys):
    body = "E" * 2000 + " <end of a long provider explanation>"
    _responses(monkeypatch, [_Resp(503, text=body)] * 7 + [_ok()])
    monitor._call("p")
    assert body in capsys.readouterr().err


def test_mask_oracle_uses_the_same_client(no_sleep, monkeypatch):
    """envs/mask.py's lie oracle goes through the same chat() — same retries, same record."""
    from monitordecorrelation.envs.mask import MaskEnv

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    calls = _responses(monkeypatch, [_Resp(503), _json_resp(402, _IN_FLIGHT_402), _ok("VERDICT: B")])
    env = MaskEnv([{"proposition": "p", "ground_truth": "A", "pressure_prompt": "q"}],
                  judge_model="x/y")
    assert env._extract("resp", "p") == "B"
    assert calls["n"] == 3
