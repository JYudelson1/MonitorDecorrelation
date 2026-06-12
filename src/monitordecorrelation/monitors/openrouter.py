"""Thin OpenRouter chat helper shared by LLM-judge callers (the MASK lie oracle, future judges).

Single ``chat()`` POST with the same retry policy as ``CoTMonitor._call`` (retry rate-limits / 5xx,
fail fast on 4xx). Kept dependency-light so envs can call a judge without importing the monitor stack.
"""

from __future__ import annotations

import os
import time

import httpx

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRYABLE = {429, 500, 502, 503, 529}


def chat(
    model_id: str,
    prompt: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    max_retries: int = 5,
) -> str:
    """POST a single user message to OpenRouter and return the assistant text. Retries only transient
    failures (429/5xx/connection); 4xx (e.g. bad model id) fails fast."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (load .env first)")
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                _OPENROUTER_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last_exc = e
        else:
            if resp.status_code not in _RETRYABLE:
                resp.raise_for_status()
                try:
                    return resp.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    last_exc = e
            else:
                last_exc = httpx.HTTPStatusError(
                    f"{resp.status_code}", request=resp.request, response=resp
                )
        if attempt < max_retries - 1:
            time.sleep(min(2.0**attempt, 30.0))
    raise RuntimeError(f"openrouter call failed after {max_retries} retries: {last_exc}")
