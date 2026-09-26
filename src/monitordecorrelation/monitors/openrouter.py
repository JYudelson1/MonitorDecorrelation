"""The one OpenRouter chat client: every OpenRouter LLM-judge call in the repo goes through ``chat()``.

Callers: ``monitors.judge_backend`` (the CoT / output judges) and ``envs.mask`` (the MASK lie oracle).
Both share this module's retry policy — and so do the local vLLM judges (``monitors.vllm``), whose
client is the same request/retry loop (``post_chat``) with vLLM's endpoint, error taxonomy and
answer extraction plugged in — there used to be a second, bounded-retry
helper here for the oracle, which drifted from the monitor's error taxonomy and had to be fixed
in two places (e.g. the same OpenRouter 402 broke both).

Policy (see ``chat``): retry **indefinitely** with capped exponential backoff on every transient
API error; fail fast only on the ``_FATAL_STATUS`` config errors, raising with the provider's
*full* explanation attached (never truncated — a clipped body is exactly what made the last such
failure undiagnosable from the run log).

Every attempt takes a permit from the cross-process OpenRouter semaphore
(``globalsem.openrouter_slot``), so the in-flight cap holds across all runs on the box, not
just this process — that is what keeps a fan-out of parallel runs from 402-storming.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, ContextManager

import httpx

from monitordecorrelation.globalsem import openrouter_slot

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Statuses we never retry: the request itself is malformed / unauthorized / unfunded, so retrying
# would hang the run forever instead of surfacing a config error. Everything else — transport
# errors, 404 ("no endpoint for this model right now"), 408/429/5xx, malformed bodies, empty
# output, an unusable finish_reason — is a transient API error and retries indefinitely.
_FATAL_STATUS = frozenset({400, 401, 402, 403})

# The one 402 that IS transient. OpenRouter reserves your remaining credit against every in-flight
# request, and when a burst of concurrent judge calls (uncapped within a run: one thread per call)
# would together exceed the balance it rejects the newcomers with 402 and this reason code — its own
# remedy hint is "Retry after in-flight requests settle". That is a rate limit wearing a 402, not
# "no credits" (which also comes back as 402 and stays fatal). It crashed a full run: 33/128 judge
# calls in one eval hit it at the same moment and each raised on its first attempt.
_RETRYABLE_402_REASON = "in_flight_budget_exhausted"

# Finish reasons whose body we read.
#   "stop"   — the judge terminated normally.
#   "length" — it hit max_tokens. That is FINAL: the call is returned as is, and whatever answer text
#              it carries is parsed. A reply cut off before its `SCORE:` line — including one whose
#              thinking ate the whole completion budget, leaving the content channel empty (routine
#              for an unbudgeted vLLM thinker) — becomes the caller's parse_error (score 0), and is
#              counted in the per-monitor `finish_length_rate` / `parse_error_rate` metrics.
#              Retrying does not help — back when the judge was called at temperature 0 every retry
#              returned the identical truncated text (measured: 4/4 byte-identical replays on
#              gemini-2.5-flash-lite), which used to spin a run forever inside one eval; at
#              temperature 1 a retry is a fresh, costlier sample that censors exactly the long-thinking
#              calls out of the statistics.
# Anything else ("content_filter", "error", …) means the judge never got to answer → API error.
_OK_FINISH_REASONS = frozenset({"stop", "length"})

# Reasoning-channel fallback: accept either instructed format (percentile SCORE or binary VERDICT).
_JUDGE_ANSWER_RE = re.compile(
    r"(?:SCORE:\s*[0-9]+(?:\.[0-9]+)?|VERDICT:\s*(?:YES|NO)\b)", re.I
)


@dataclass
class JudgeCall:
    """One *successful* judge API call: the text the caller parses, plus a durable record of it.

    ``record`` is what the rollout dumps persist under ``monitors.<name>.call`` so a saved rollout
    carries exactly what its judge was sent and exactly what came back (see ``chat``):

    - ``url`` / ``request`` — the endpoint and the exact JSON body POSTed (``model``, ``messages``
      — i.e. the full prompt — ``temperature``, ``max_tokens``, ``reasoning``). The Authorization
      header is the only thing omitted.
    - ``timeout`` — the client-side request timeout (not part of the body, but part of the call).
    - ``response`` — the provider's reply: the full assistant ``message`` (``content``, and the
      judge's chain of thought under ``reasoning`` / ``reasoning_details`` when the provider
      returns one), ``finish_reason``, and the response-level ``id`` / ``model`` / ``provider`` /
      ``usage``.
    - ``attempts`` — how many POSTs it took; only the last, successful one is recorded.
    """

    text: str
    record: dict


def _warn(msg: str) -> None:
    """Timestamped stderr warning (matches rl/train.py's _log prefix), flushed so it survives pipes."""
    print(f"[{time.strftime('%H:%M:%S')}] ⚠️  {msg}", file=sys.stderr, flush=True)


def resolve_api_key(api_key: str | None = None) -> str:
    """``api_key`` if given, else ``$OPENROUTER_API_KEY``; raises if neither is set."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (load .env first)")
    return key


def _judge_output(message: dict, finish: str | None = None) -> str | None:
    """Pull the judge's answer line from an OpenRouter ``message`` (``finish`` = its finish_reason).

    Prefer ``content``. Gemini-3.x (mandatory reasoning) sometimes returns ``content: null`` and
    parks the instructed ``SCORE: <n>`` / ``VERDICT: YES|NO`` line in ``reasoning`` instead. We only
    fall back to that channel when:
      - ``content`` is missing/blank, AND
      - the completion was not cut off (``finish != "length"``): a truncated scratchpad was still
        deliberating, so a ``SCORE:`` in it is a draft, not the answer, AND
      - ``reasoning`` contains an explicit ``SCORE:`` or ``VERDICT:`` match (the format we asked
        for).
    Free-form thinking with bare numbers / yes-no prose is rejected — those are intermediate
    guesses, not the verdict. When falling back we return only the *last* matching line (final
    answer), not the whole scratchpad, so ``meta["raw"]`` stays the verdict and first-match parsing
    can't latch onto an earlier draft. Non-string ``content`` (unexpected payload) is not a cue to
    mine reasoning either.
    """
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content  # model answered in content — never dig into reasoning
    if content is not None and not isinstance(content, str):
        return None
    if finish == "length":
        return None
    reasoning = message.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None
    matches = list(_JUDGE_ANSWER_RE.finditer(reasoning))
    if not matches:
        return None  # scratchpad with no SCORE:/VERDICT: → caller retries; don't invent a score
    return matches[-1].group(0)


def _is_retryable_402(resp: httpx.Response) -> bool:
    """True for exactly the in-flight-budget 402 (``_RETRYABLE_402_REASON``); every other 402 is fatal."""
    if resp.status_code != 402:
        return False
    try:
        data = resp.json()
        return data["error"]["metadata"]["reason"] == _RETRYABLE_402_REASON
    except (KeyError, TypeError, ValueError):
        return False


def chat(body: dict, *, api_key: str, timeout: float, name: str, warn_after: int = 6) -> JudgeCall:
    """POST ``body`` (a complete chat-completions request: ``model``, ``messages``, sampling and
    ``reasoning`` keys — the caller owns it, and it is persisted verbatim in the returned record) to
    OpenRouter and return the judge's text plus the record of the successful call (``JudgeCall``),
    retrying **indefinitely** with exponential backoff (capped at 30s) on any transient API error:
    connection/timeout, 404 ("no endpoints available for this model right now"), 408/429/5xx, the
    in-flight-budget 402, a malformed body, an unusable ``finish_reason``, or empty output under
    ``"stop"`` (null content with no ``SCORE:``/``VERDICT:`` in ``reasoning``). A multi-hour run must
    not lose a monitor to a provider hiccup, so there is no give-up path for these — from the
    ``warn_after``-th retry on, every retry prints a warning to stderr (prefixed with ``name``, e.g.
    ``monitor cot_weak``) so a stuck judge is visible in the log rather than silent. A ``"length"``
    completion is final (see ``_OK_FINISH_REASONS``): its text — possibly empty — is returned.

    The exceptions are ``_FATAL_STATUS`` (400/401/402/403): a malformed request, a bad key, no
    credits, or a forbidden model never fixes itself, so those raise immediately, with the
    provider's whole response body in the message. That surfaces as a NaN sentinel per rollout in
    ``rl.train.MonitorScorer``, which then aborts the run (the intended behaviour for a config
    error). NB a mandatory-reasoning model ("Reasoning is mandatory for this endpoint and cannot be
    disabled.") lands there by design — the fix is the monitor's ``reasoning`` setting, not a retry
    (``monitors.judge_reasoning`` rejects that combination at construction for the judges it knows).
    """
    return post_chat(
        body,
        url=_OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
        name=name,
        warn_after=warn_after,
        slot=openrouter_slot,
        is_fatal=lambda resp: resp.status_code in _FATAL_STATUS and not _is_retryable_402(resp),
        extract=_judge_output,
    )


def post_chat(
    body: dict,
    *,
    url: str,
    headers: dict,
    timeout: float,
    name: str,
    warn_after: int,
    slot: Callable[[], ContextManager],
    is_fatal: Callable[[httpx.Response], bool],
    extract: Callable[[dict, str | None], str | None],
) -> JudgeCall:
    """The provider-agnostic request/retry loop behind ``chat`` (OpenRouter) and ``monitors.vllm.chat``.

    ``slot()`` is held around each single attempt (a cross-process concurrency permit, or a no-op);
    ``is_fatal(resp)`` picks the error statuses that raise at once (with the whole body) instead of
    retrying; ``extract(message, finish_reason)`` returns the judge's answer text, or ``None`` for
    "no usable output" — a transient error under ``"stop"``, but under ``"length"`` the call is
    final and returns ``""`` (the caller's parse_error). ``extract`` may itself raise to flag a
    configuration error (e.g. a vLLM server returning unparsed thinking). Everything else is
    documented on ``chat``.
    """
    attempt = 0
    while True:
        attempt += 1
        err: str
        try:
            # The permit covers ONE attempt, not the retry-forever loop around it: its hold time is
            # then bounded by `timeout`, so a wedged judge can never starve the other runs sharing
            # the box. The backoff sleep below happens with the permit released.
            with slot():
                resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            err = f"{type(e).__name__}: {e}"  # connection/timeout -> retry
        else:
            if is_fatal(resp):
                # Unrecoverable (bad request / key / credits) -> fail fast, with the provider's
                # explanation attached in full: raise_for_status() alone reports only the status
                # and URL, which leaves a 400 undiagnosable in the run log.
                raise httpx.HTTPStatusError(
                    f"{resp.status_code} for {body.get('model')}: {resp.text}",
                    request=resp.request,
                    response=resp,
                )
            elif resp.status_code >= 400:
                err = f"HTTP {resp.status_code}: {resp.text}"
            else:
                try:
                    data = resp.json()
                    choice = data["choices"][0]
                    message = choice["message"]
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    err = f"malformed response body ({type(e).__name__}: {e})"
                else:
                    # A filtered / errored completion never reached the verdict line, so it's
                    # an API error, not a score of 0. A truncated one ("length") is final — see
                    # _OK_FINISH_REASONS. A missing finish_reason (some providers omit it) is not
                    # evidence of failure — judge the body instead.
                    finish = choice.get("finish_reason")
                    if finish is not None and finish not in _OK_FINISH_REASONS:
                        err = f"finish_reason={finish!r} (completion did not terminate normally)"
                    else:
                        text = extract(message, finish)
                        if text is None and finish == "length":
                            text = ""  # cut off before any answer: the caller's parse_error, no retry
                        if text is None:
                            # Null content with no SCORE:/VERDICT: in reasoning (Gemini
                            # sometimes empties both). Retry rather than parse_error->score 0
                            # (that would silently under-flag).
                            err = (
                                "empty judge output (no content / no SCORE:|VERDICT: in "
                                "reasoning)"
                            )
                        else:
                            if attempt > warn_after:
                                _warn(f"{name}: recovered on attempt {attempt}")
                            # Only the call that succeeded is recorded — failed attempts
                            # never had an answer worth keeping.
                            record = {
                                "url": url,
                                "request": body,
                                "timeout": timeout,
                                "attempts": attempt,
                                "response": {
                                    "message": message,
                                    "finish_reason": finish,
                                    **{k: data.get(k) for k in ("id", "model", "provider", "usage")
                                       if isinstance(data, dict) and k in data},
                                },
                            }
                            return JudgeCall(text=text, record=record)
        if attempt > warn_after:
            _warn(f"{name}: attempt {attempt} failed ({err}); retrying")
        time.sleep(min(2.0 ** (attempt - 1), 30.0))
