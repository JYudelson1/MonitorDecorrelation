"""Offline: vLLM judges (monitors/vllm.py via monitors/judge_backend.py) — config validation, the exact
request body, answer extraction, the retry/fatal policy, and the per-judge call-health metrics.

No network: httpx.post / httpx.get and time.sleep are monkeypatched.
"""

from __future__ import annotations

import math

import httpx
import pytest

from monitordecorrelation.eval.metrics import judge_call_rates
from monitordecorrelation.experiment_config import CoTMonitorSpec, apply_overrides, build_monitors
from monitordecorrelation.monitors import openrouter as orc
from monitordecorrelation.monitors import vllm
from monitordecorrelation.monitors.agent_cot_monitor import AgentCoTMonitor
from monitordecorrelation.monitors.cot_monitor import CoTMonitor
from monitordecorrelation.types import MonitorResult, Prompt, Rollout

Q3 = "Qwen/Qwen3-30B-A3B-FP8"
URL = "http://localhost:8001/v1"
VLLM = dict(provider="vllm", model_id=Q3, base_url=URL, max_tokens=16384, enable_thinking=True)


@pytest.fixture(autouse=True)
def _no_server(monkeypatch):
    monkeypatch.setattr(vllm, "check_server", lambda *a, **k: None)
    monkeypatch.setattr(orc.time, "sleep", lambda _s: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # a vLLM judge must not need one


def _spec(**kw) -> CoTMonitorSpec:
    return CoTMonitorSpec(**{"name": "q", "role": "held_out", **VLLM, **kw})


def _judge(**kw) -> CoTMonitor:
    return CoTMonitor("q", Q3, behavior="deception", **{k: v for k, v in {**VLLM, **kw}.items() if k != "model_id"})


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code, self._payload, self.text = status_code, payload, text
        self.request = httpx.Request("POST", URL)

    def json(self):
        return self._payload


def _reply(content, reasoning=None, finish="stop") -> _Resp:
    return _Resp(200, {"choices": [{"finish_reason": finish,
                                    "message": {"role": "assistant", "content": content, "reasoning": reasoning}}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 9,
                                 "completion_tokens_details": {"reasoning_tokens": 7}}})


def _script(monkeypatch, items):
    log = []

    def fake_post(url, *, headers, json, timeout):
        log.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = items[min(len(log) - 1, len(items) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(orc.httpx, "post", fake_post)
    return log


def _rollout() -> Rollout:
    return Rollout(prompt=Prompt(text="the task"), cot="my reasoning", output="my answer")


# ---- config ------------------------------------------------------------------------------------


def test_vllm_spec_requires_its_settings_and_defaults_to_no_budget():
    s = _spec()
    assert s.thinking_budget is None and s.reasoning is None
    for missing in ("base_url", "max_tokens", "enable_thinking"):
        with pytest.raises(ValueError, match=missing):
            _spec(**{missing: None})


@pytest.mark.parametrize("bad, msg", [
    ({"reasoning": {"enabled": False}}, "would be ignored"),               # an OpenRouter-only setting
    ({"enable_thinking": False, "thinking_budget": 512}, "would be ignored"),
    ({"thinking_budget": 16384}, "below max_tokens"),                       # no room for the answer
    ({"thinking_budget": 0}, "thinking_budget"),
    ({"base_url": "http://localhost:8001"}, "ending in /v1"),
    ({"base_url": "localhost:8001/v1"}, "ending in /v1"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"model_id": "Qwen/Qwen3-8B"}, "specialized to"),                      # unverified model
])
def test_vllm_spec_rejects_what_would_not_take_effect(bad, msg):
    with pytest.raises(ValueError, match=msg):
        _spec(**bad)


def test_openrouter_spec_rejects_vllm_settings_and_takes_max_tokens():
    base = {"name": "g", "role": "held_out", "model_id": "google/gemini-2.5-flash-lite"}
    for k, v in [("base_url", URL), ("enable_thinking", True), ("thinking_budget", 512)]:
        with pytest.raises(ValueError, match="would be ignored"):
            CoTMonitorSpec(**base, **{k: v})
    assert CoTMonitorSpec(**base).max_tokens is None  # absent = the 4096 default, applied by the monitor
    assert CoTMonitorSpec(**base, max_tokens=8192, reasoning={"max_tokens": 6000}).max_tokens == 8192
    with pytest.raises(ValueError, match="does not fit below"):  # the 2048 default budget needs room
        CoTMonitorSpec(**base, max_tokens=2000)


def test_set_overrides_reach_vllm_settings():
    from monitordecorrelation.experiment_config import ExperimentConfig

    cfg = ExperimentConfig(run_name="r", env="mbpp_honeypot", max_tokens=100,
                           monitors=[{"kind": "cot", "name": "q", "role": "held_out", **VLLM}])
    cfg = apply_overrides(cfg, ["monitors.model:Qwen.thinking_budget=4096", "monitors.q.max_tokens=8192"])
    (m,) = build_monitors(cfg.monitors, default_behavior="deception")[1]
    body = m._request_body("p")
    assert body["max_tokens"] == 8192 and body["thinking_token_budget"] == 4096
    with pytest.raises(ValueError, match="below max_tokens"):
        apply_overrides(cfg, ["monitors.q.thinking_budget=8192"])


# ---- request / response ------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [CoTMonitor, AgentCoTMonitor])
def test_request_body_is_what_the_config_says(cls):
    kw = {k: v for k, v in VLLM.items() if k != "model_id"}
    j = cls("q", Q3, behavior="reward_hacking", **kw)
    assert j.reasoning is None
    assert j._request_body("p") == {"model": Q3, "messages": [{"role": "user", "content": "p"}],
                                    "temperature": 1.0, "max_tokens": 16384,
                                    "chat_template_kwargs": {"enable_thinking": True}}
    j = cls("q", Q3, behavior="reward_hacking", **{**kw, "thinking_budget": 4096, "max_tokens": 8192})
    assert j._request_body("p")["thinking_token_budget"] == 4096
    j = cls("q", Q3, behavior="reward_hacking", **{**kw, "enable_thinking": False})
    assert j._request_body("p")["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_token_budget" not in j._request_body("p")
    assert j.backend.info() == {"provider": "vllm", "base_url": URL, "max_tokens": 16384,
                                "enable_thinking": False, "thinking_budget": None}


def test_score_reads_content_only_and_records_the_call(monkeypatch):
    log = _script(monkeypatch, [_reply("SCORE: 73", reasoning="draft SCORE: 10 … settled")])
    res = _judge().score(_rollout())
    assert res.score == 0.73 and "parse_error" not in res.meta
    assert log[0]["url"] == f"{URL}/chat/completions" and log[0]["headers"] == {}
    assert log[0]["timeout"] == vllm.VLLM_TIMEOUT
    call = res.meta["call"]
    assert call["request"] == log[0]["json"] and call["response"]["finish_reason"] == "stop"
    assert call["response"]["message"]["reasoning"].startswith("draft")  # the thinking is kept


def test_thinking_is_never_mined_for_a_score(monkeypatch):
    """Empty content: a SCORE: in the thinking is a draft. Under 'length' → parse_error (no retry)."""
    log = _script(monkeypatch, [_reply(None, reasoning="SCORE: 90 maybe", finish="length")])
    res = _judge().score(_rollout())
    assert len(log) == 1 and res.score == 0.0 and res.meta["parse_error"] is True
    assert res.meta["call"]["response"]["finish_reason"] == "length"


def test_empty_content_on_stop_is_retried(monkeypatch):
    log = _script(monkeypatch, [_reply(""), _reply("SCORE: 5")])
    assert _judge().score(_rollout()).score == 0.05 and len(log) == 2


def test_unsplit_thinking_is_a_config_error(monkeypatch):
    """A server without --reasoning-parser returns the thinking inside content — refuse to score it."""
    _script(monkeypatch, [_reply("hmm… SCORE: 10</think>\n\nSCORE: 80")])
    with pytest.raises(RuntimeError, match="reasoning-parser"):
        _judge().score(_rollout())


@pytest.mark.parametrize("status", [400, 404, 422])
def test_client_errors_are_fatal(monkeypatch, status):
    log = _script(monkeypatch, [_Resp(status, text="model does not exist"), _reply("SCORE: 1")])
    with pytest.raises(httpx.HTTPStatusError, match="model does not exist"):
        _judge().score(_rollout())
    assert len(log) == 1


def test_server_errors_and_connection_errors_are_retried(monkeypatch):
    log = _script(monkeypatch, [httpx.ConnectError("down"), _Resp(503, text="busy"), _reply("SCORE: 2")])
    assert _judge().score(_rollout()).score == 0.02 and len(log) == 3


def test_check_server(monkeypatch):
    monkeypatch.undo()  # the real check_server
    monkeypatch.setattr(vllm.httpx, "get", lambda url, timeout: type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {"data": [{"id": Q3, "max_model_len": 40960}]}})())
    vllm.check_server(URL, Q3, max_tokens=16384, monitor="q")
    with pytest.raises(RuntimeError, match="not Qwen/Qwen3.5"):
        vllm.check_server(URL, "Qwen/Qwen3.5-35B-A3B-FP8", max_tokens=16384, monitor="q")
    with pytest.raises(RuntimeError, match="max_model_len"):
        vllm.check_server(URL, Q3, max_tokens=40960, monitor="q")


# ---- call-health metrics -----------------------------------------------------------------------


def _res(finish, parse_error=False):
    meta = {"call": {"response": {"finish_reason": finish}}}
    if parse_error:
        meta["parse_error"] = True
    return MonitorResult(score=0.0, label=False, meta=meta)


def test_judge_call_rates():
    rs = [_res("stop"), _res("length", True), _res("length"), _res("stop", True), None]  # None: not shown
    assert judge_call_rates(rs) == {"finish_length_rate": 0.5, "parse_error_rate": 0.5}
    assert all(math.isnan(v) for v in judge_call_rates([None]).values())


def test_train_loop_logs_call_rates_for_judges_only():
    from monitordecorrelation.rl.train import _judge_call_metrics, _monitor_info

    j = _judge(thinking_budget=4096, max_tokens=8192)
    assert _judge_call_metrics(j, [_res("length", True), _res("stop")]) == {
        "monitor/q/finish_length_rate": 0.5, "monitor/q/parse_error_rate": 0.5}
    probe = type("P", (), {"name": "p"})()
    assert _judge_call_metrics(probe, [MonitorResult(score=0.1, label=False)]) == {}
    info = _monitor_info(j, "held_out")
    assert info["provider"] == "vllm" and info["thinking_budget"] == 4096 and info["max_tokens"] == 8192


def test_length_is_a_parse_error_even_with_a_draft_score_in_content(monkeypatch):
    """Measured on Qwen3.5 at thinking_budget 4096: after the forced end of thinking it went on
    deliberating in content ("Score: 5. (Minimal suspicion …)  Wait, I need to be…") until max_tokens.
    That draft is not an answer."""
    _script(monkeypatch, [_reply("*   Score: 5. (Minimal suspicion)\n\n    *   Wait, I need to be", finish="length")])
    res = _judge(thinking_budget=4096, max_tokens=8192).score(_rollout())
    assert res.score == 0.0 and res.meta["parse_error"] is True
    assert "Score: 5" in res.meta["call"]["response"]["message"]["content"]  # kept in the record


def test_the_last_score_line_in_content_is_the_verdict(monkeypatch):
    _script(monkeypatch, [_reply("maybe SCORE: 80?\nno — on reflection it is benign.\n\nSCORE: 10")])
    res = _judge().score(_rollout())
    assert res.score == 0.10 and res.meta["raw"] == "SCORE: 10"
    _script(monkeypatch, [_reply("I refuse to rate this.")])
    res = _judge().score(_rollout())
    assert res.meta["parse_error"] is True and res.meta["raw"] == "I refuse to rate this."
