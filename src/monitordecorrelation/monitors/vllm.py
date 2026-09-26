"""vLLM judges: LLM monitors served by a local vLLM OpenAI-compatible server (``vllm serve``).

An OpenRouter judge is just a model id (plus its per-model ``reasoning`` object, see
``monitors.judge_reasoning``). A vLLM judge is configured explicitly, with no defaults but one:

* ``model_id`` — the model the server serves (``GET <base_url>/models`` must list it);
* ``base_url`` — the server's OpenAI-compatible base, ending in ``/v1`` (e.g.
  ``http://localhost:8001/v1``); calls POST to ``<base_url>/chat/completions``;
* ``max_tokens`` — the completion cap, which covers the thinking AND the answer;
* ``enable_thinking`` — sent as ``chat_template_kwargs.enable_thinking`` (the Qwen3 / Qwen3.5 chat
  template switch);
* ``thinking_budget`` — ``None`` (the default) for no budget, or N: vLLM's
  ``thinking_token_budget``, which force-closes the thinking after N tokens so the answer is
  written within the rest of ``max_tokens``. Only with ``enable_thinking``, and N < ``max_tokens``.

The server must run with a reasoning parser (``vllm serve … --reasoning-parser qwen3``): that is what
splits the thinking into ``message.reasoning`` — leaving ``content`` as the answer the ``SCORE:`` /
``VERDICT:`` line is parsed from — and what enables ``thinking_token_budget`` (vLLM answers a budget
without one with a 400). A reply whose ``content`` still carries ``</think>`` means the server is
not splitting reasoning, and raises (a config error) rather than being scored.

Unlike an OpenRouter judge, the answer is **only ever read from content** — never mined from the
thinking — and it is the content's LAST ``SCORE:`` / ``VERDICT:`` line: a thinking-enabled Qwen drafts
such lines while it deliberates, and after a budget force-closes its thinking it can go on
deliberating in the content. A reply cut off by ``max_tokens`` (``finish_reason == "length"``) never
finished its answer and is a parse_error (score 0) whatever it contains; both are counted per monitor
as ``finish_length_rate`` / ``parse_error_rate``. See ``judge_output``.

Only the models in ``VLLM_JUDGES`` are accepted — the ones whose chat template is known to honour
``enable_thinking`` and whose thinking the ``qwen3`` parser splits (checked against the running
servers: thinking on → ``message.reasoning`` + a clean content answer; off → no reasoning at all;
``thinking_token_budget`` N → ≈N reasoning tokens, then the answer). Adding a model means checking
the same for it.

Same retry policy and call record as the OpenRouter client (``openrouter.post_chat``), with a
local-server error taxonomy: 400/401/403/404/422 (bad request, e.g. a prompt past the model's
context; wrong model name or path) are fatal; connection errors, timeouts and 5xx retry forever
(a restarting server comes back). No cross-process semaphore — the server batches concurrent
requests itself.
"""

from __future__ import annotations

from contextlib import nullcontext

import httpx

from monitordecorrelation.monitors.openrouter import _JUDGE_ANSWER_RE, JudgeCall, post_chat

# Judge models whose thinking controls were verified on a running vLLM server (see the module doc).
VLLM_JUDGES = frozenset({"Qwen/Qwen3-30B-A3B-FP8", "Qwen/Qwen3.5-35B-A3B-FP8"})

# Client-side request timeout. A non-streaming call sends nothing until the whole completion is done,
# and an unbudgeted thinker can write 16k tokens on a server shared by dozens of concurrent calls.
VLLM_TIMEOUT = 1800.0

_FATAL_STATUS = frozenset({400, 401, 403, 404, 422})


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_vllm_judge(
    *,
    model_id: str,
    base_url,
    max_tokens,
    enable_thinking,
    thinking_budget,
    monitor: str,
) -> None:
    """Reject a vLLM judge setting that could not take effect as written. Offline (no server call) —
    shared by the config schema (at LOAD) and ``VLLMJudge`` (hand-built monitors)."""
    where = f"monitor {monitor!r} ({model_id}, vllm)"
    if model_id not in VLLM_JUDGES:
        raise ValueError(
            f"{where}: vLLM judges are specialized to {sorted(VLLM_JUDGES)} — whether "
            f"{model_id}'s chat template honours enable_thinking, and whether the server splits its "
            "thinking from its answer, has not been checked. Verify that, then add it to "
            "monitors/vllm.py:VLLM_JUDGES."
        )
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")) \
            or not base_url.rstrip("/").endswith("/v1"):
        raise ValueError(
            f"{where}: base_url must be the server's OpenAI-compatible base ending in /v1, e.g. "
            f"'http://localhost:8001/v1', got {base_url!r}"
        )
    if not _is_int(max_tokens) or max_tokens < 1:
        raise ValueError(f"{where}: max_tokens must be an int >= 1 (required for vLLM judges), got {max_tokens!r}")
    if not isinstance(enable_thinking, bool):
        raise ValueError(f"{where}: enable_thinking must be true or false (required for vLLM judges), "
                         f"got {enable_thinking!r}")
    if thinking_budget is not None:
        if not enable_thinking:
            raise ValueError(
                f"{where}: thinking_budget={thinking_budget!r} with enable_thinking=false would be "
                "ignored — drop it (null) or enable thinking"
            )
        if not _is_int(thinking_budget) or not 1 <= thinking_budget < max_tokens:
            raise ValueError(
                f"{where}: thinking_budget must be null (no budget) or an int in [1, {max_tokens}) — "
                f"below max_tokens={max_tokens}, which must also fit the answer — got {thinking_budget!r}"
            )


def request_body(model_id: str, prompt: str, *, max_tokens: int, enable_thinking: bool,
                 thinking_budget: int | None) -> dict:
    """The exact JSON body a vLLM judge POSTs (persisted verbatim in the call record)."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if thinking_budget is not None:
        body["thinking_token_budget"] = thinking_budget
    return body


def check_server(base_url: str, model_id: str, *, max_tokens: int, monitor: str) -> None:
    """Fail at construction, not at the first judge call, if ``base_url`` is not a vLLM server serving
    ``model_id`` with a context long enough for ``max_tokens`` of completion."""
    where = f"monitor {monitor!r} ({model_id}, vllm)"
    url = f"{base_url.rstrip('/')}/models"
    try:
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        served = {m["id"]: m for m in resp.json()["data"]}
    except Exception as e:  # noqa: BLE001 — any failure here is "not a usable server"
        raise RuntimeError(f"{where}: cannot list the models at {url} ({type(e).__name__}: {e}) — "
                           "is the vLLM server up, and is base_url right?") from e
    if model_id not in served:
        raise RuntimeError(f"{where}: the server at {base_url} serves {sorted(served)}, not {model_id}")
    ctx = served[model_id].get("max_model_len")
    if _is_int(ctx) and max_tokens >= ctx:
        raise RuntimeError(f"{where}: max_tokens={max_tokens} leaves no room for the prompt in the "
                           f"server's max_model_len={ctx}")


def judge_output(message: dict, finish: str | None) -> str | None:
    """The judge's answer from a vLLM ``message``: from ``content`` only (never the thinking — see the
    module doc). Raises if the thinking was not split off by the server.

    * ``finish == "length"`` → ``None`` (the caller's parse_error, score 0), whatever the content
      holds: the judge never finished its answer. When a ``thinking_budget`` force-closes the thinking,
      Qwen3.5 sometimes keeps deliberating in the content channel — drafting ``Score: 5`` mid-thought —
      until it runs out of ``max_tokens`` (measured: 1/44 calls, a draft that would have been read as
      the verdict).
    * content with ``SCORE:`` / ``VERDICT:`` lines → the LAST one, the final answer: after a forced
      end of thinking the content can carry several (13/88 Qwen3.5 calls at budget 4096, 2 with a first
      draft different from the final line), and the callers parse the first match.
    * otherwise the content as is (no instructed line → parse_error), or ``None`` if empty.

    The full message is kept in the call record either way.
    """
    content = message.get("content")
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(content, str) and "</think>" in content and not reasoning:
        raise RuntimeError(
            "vLLM returned the thinking inside content ('</think>' in content, no reasoning field): the "
            "server is not splitting reasoning — restart it with --reasoning-parser qwen3"
        )
    if finish == "length" or not isinstance(content, str) or not content.strip():
        return None
    matches = list(_JUDGE_ANSWER_RE.finditer(content))
    return matches[-1].group(0) if matches else content


def chat(body: dict, *, base_url: str, timeout: float, name: str, warn_after: int = 6) -> JudgeCall:
    """POST ``body`` to ``<base_url>/chat/completions`` and return the judge's text plus the record of
    the successful call — ``openrouter.post_chat``'s retry loop with this module's error taxonomy and
    ``judge_output``."""
    return post_chat(
        body,
        url=f"{base_url.rstrip('/')}/chat/completions",
        headers={},
        timeout=timeout,
        name=name,
        warn_after=warn_after,
        slot=nullcontext,
        is_fatal=lambda resp: resp.status_code in _FATAL_STATUS,
        extract=judge_output,
    )
