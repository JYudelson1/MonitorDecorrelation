"""Where an LLM judge's calls go: OpenRouter or a local vLLM server. The ONE seam ``CoTMonitor`` and
``AgentCoTMonitor`` share for building a request body and making the call, so what a judge sends is
decided in one place whatever its prompt layout.

A monitor's ``provider`` picks the backend, and each backend takes only the settings that mean
something to it — anything else is an error, never silently dropped (CLAUDE.md, "No config key may
be silently ignored"):

* ``openrouter`` (the default) — ``model_id``, ``reasoning`` (the per-model OpenRouter object,
  ``monitors.judge_reasoning``) and ``max_tokens`` (optional: ``OPENROUTER_DEFAULT_MAX_TOKENS``).
* ``vllm`` — ``model_id``, ``base_url``, ``max_tokens``, ``enable_thinking`` (all required),
  ``thinking_budget`` (optional: ``None`` = no budget) and ``thinking_effort`` (required on a model
  whose chat template has a reasoning-effort knob, with thinking on; rejected otherwise). See
  ``monitors.vllm``.

A vLLM call's request ``priority`` is not a setting but the context it is made in: ``judge_priority``
(a context manager, default ``vllm.TRAIN_PRIORITY``), which the RL loop's ``MonitorScorer`` enters in
each judge thread — train vs eval. OpenRouter calls carry no priority.

``validate_judge_settings`` is the offline check the config schema runs at LOAD; ``make_judge_backend``
runs the same check (plus, for vLLM, a server check) when a monitor is built.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from monitordecorrelation.monitors import vllm
from monitordecorrelation.monitors.judge_reasoning import OPENROUTER_DEFAULT_MAX_TOKENS, resolve_reasoning
from monitordecorrelation.monitors.openrouter import JudgeCall, chat, resolve_api_key

PROVIDERS = ("openrouter", "vllm")

# The settings only a vLLM judge takes / only an OpenRouter judge takes.
_VLLM_ONLY = ("base_url", "enable_thinking", "thinking_budget", "thinking_effort")
_OPENROUTER_ONLY = ("reasoning",)

# The vLLM request priority of the judge calls made in this context (see ``judge_priority``).
_PRIORITY: ContextVar[int] = ContextVar("judge_priority", default=vllm.TRAIN_PRIORITY)


@contextmanager
def judge_priority(priority: int) -> Iterator[None]:
    """vLLM judge calls made inside this block (in this thread) carry ``priority`` (lower = served
    first; ``vllm.TRAIN_PRIORITY`` / ``vllm.EVAL_PRIORITY``). A context variable, so it does NOT
    follow the work into a new thread — enter it in the thread that makes the call."""
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise TypeError(f"judge priority must be an int, got {priority!r}")
    token = _PRIORITY.set(priority)
    try:
        yield
    finally:
        _PRIORITY.reset(token)


# Client-side request timeout of an OpenRouter judge call.
OPENROUTER_TIMEOUT = 60.0


def validate_judge_settings(
    provider: str,
    *,
    model_id: str,
    monitor: str,
    max_tokens=None,
    reasoning=None,
    base_url=None,
    enable_thinking=None,
    thinking_budget=None,
    thinking_effort=None,
) -> dict | None:
    """Reject settings the provider would not honour as written. Offline. Returns the resolved OpenRouter
    ``reasoning`` object (``None`` for vLLM)."""
    given = {"reasoning": reasoning, "base_url": base_url, "enable_thinking": enable_thinking,
             "thinking_budget": thinking_budget, "thinking_effort": thinking_effort}
    if provider not in PROVIDERS:
        raise ValueError(f"monitor {monitor!r}: provider must be one of {list(PROVIDERS)}, got {provider!r}")
    foreign = _VLLM_ONLY if provider == "openrouter" else _OPENROUTER_ONLY
    if stray := [k for k in foreign if given[k] is not None]:
        raise ValueError(
            f"monitor {monitor!r} ({model_id}) is a {provider} judge, so {stray} would be ignored — drop "
            f"{'it' if len(stray) == 1 else 'them'} (openrouter judges take reasoning; vllm judges take "
            "base_url, enable_thinking, thinking_budget and thinking_effort)"
        )
    if provider == "openrouter":
        cap = OPENROUTER_DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        return resolve_reasoning(model_id, reasoning, max_tokens=cap, monitor=monitor)
    vllm.validate_vllm_judge(model_id=model_id, base_url=base_url, max_tokens=max_tokens,
                             enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                             thinking_effort=thinking_effort, monitor=monitor)
    return None


class OpenRouterJudge:
    provider = "openrouter"

    def __init__(self, *, name: str, model_id: str, reasoning: dict | None, max_tokens: int | None,
                 timeout: float | None, api_key: str | None) -> None:
        self.name, self.model_id = name, model_id
        self.max_tokens = OPENROUTER_DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        # The OpenRouter `reasoning` object every call sends — validated for THIS judge model (and
        # the model's default filled in) by the one shared resolver; unsupported models raise here.
        # Deliberately static: it used to be discovered at runtime by catching a mandatory-reasoning
        # 400 and flipping, which raced across the threads sharing a monitor (16 concurrent first
        # calls → 1 flip + 15 fatal 400s → 15 NaN scores per eval). See monitors/judge_reasoning.py.
        self.reasoning = validate_judge_settings("openrouter", model_id=model_id, monitor=name,
                                                 max_tokens=self.max_tokens, reasoning=reasoning)
        self.timeout = OPENROUTER_TIMEOUT if timeout is None else timeout
        self._api_key = resolve_api_key(api_key)

    def request_body(self, prompt: str) -> dict:
        return {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning,  # resolved by judge_reasoning.resolve_reasoning
        }

    def call(self, prompt: str, *, warn_after: int) -> JudgeCall:
        return chat(self.request_body(prompt), api_key=self._api_key, timeout=self.timeout,
                    name=f"monitor {self.name}", warn_after=warn_after)

    def info(self) -> dict:
        """The resolved settings every call uses — for run_info / the baseline summary."""
        return {"provider": self.provider, "max_tokens": self.max_tokens, "reasoning": self.reasoning}


class VLLMJudge:
    provider = "vllm"

    def __init__(self, *, name: str, model_id: str, base_url: str, max_tokens: int, enable_thinking: bool,
                 thinking_budget: int | None, timeout: float | None, thinking_effort: str | None = None,
                 check_server: bool = True) -> None:
        validate_judge_settings("vllm", model_id=model_id, monitor=name, max_tokens=max_tokens,
                                base_url=base_url, enable_thinking=enable_thinking,
                                thinking_budget=thinking_budget, thinking_effort=thinking_effort)
        self.name, self.model_id = name, model_id
        self.base_url = base_url.rstrip("/")
        self.max_tokens, self.enable_thinking, self.thinking_budget = max_tokens, enable_thinking, thinking_budget
        self.thinking_effort = thinking_effort
        self.reasoning = None  # an OpenRouter-only setting
        self.timeout = vllm.VLLM_TIMEOUT if timeout is None else timeout
        if check_server:
            vllm.check_server(self.base_url, model_id, max_tokens=max_tokens, monitor=name)

    def request_body(self, prompt: str) -> dict:
        return vllm.request_body(self.model_id, prompt, max_tokens=self.max_tokens,
                                 enable_thinking=self.enable_thinking, thinking_budget=self.thinking_budget,
                                 priority=_PRIORITY.get(), thinking_effort=self.thinking_effort)

    def call(self, prompt: str, *, warn_after: int) -> JudgeCall:
        return vllm.chat(self.request_body(prompt), base_url=self.base_url, timeout=self.timeout,
                         name=f"monitor {self.name}", warn_after=warn_after)

    def info(self) -> dict:
        return {"provider": self.provider, "base_url": self.base_url, "max_tokens": self.max_tokens,
                "enable_thinking": self.enable_thinking, "thinking_budget": self.thinking_budget,
                "thinking_effort": self.thinking_effort}


def make_judge_backend(
    provider: str,
    *,
    name: str,
    model_id: str,
    max_tokens: int | None = None,
    reasoning: dict | None = None,
    base_url: str | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
    thinking_effort: str | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
) -> OpenRouterJudge | VLLMJudge:
    """The backend of one judge. Settings the provider does not take raise (``validate_judge_settings``)."""
    validate_judge_settings(provider, model_id=model_id, monitor=name, max_tokens=max_tokens,
                            reasoning=reasoning, base_url=base_url, enable_thinking=enable_thinking,
                            thinking_budget=thinking_budget, thinking_effort=thinking_effort)
    if provider == "vllm":
        if api_key is not None:
            raise ValueError(f"monitor {name!r}: a vllm judge takes no api_key")
        return VLLMJudge(name=name, model_id=model_id, base_url=base_url, max_tokens=max_tokens,
                         enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                         thinking_effort=thinking_effort, timeout=timeout)
    return OpenRouterJudge(name=name, model_id=model_id, reasoning=reasoning, max_tokens=max_tokens,
                           timeout=timeout, api_key=api_key)
