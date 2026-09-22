"""Judge-side reasoning: the ONE place a monitor's ``reasoning`` setting becomes the OpenRouter
``reasoning`` object its every call carries.

A monitor's ``reasoning`` (config key, ``CoTMonitor`` / ``AgentCoTMonitor`` kwarg) is an OpenRouter
``reasoning`` object, sent verbatim once validated — or ``None`` for the model's default. What is
valid is decided **per judge model**, because the same object means different things on different
models (and OpenRouter silently ignores or clamps what a model does not support):

``google/gemini-2.5-flash-lite`` — reasoning is optional:

* ``{"enabled": false}`` — no reasoning; the judge answers the ``SCORE:`` line straight from the
  content channel.
* ``{"max_tokens": N}`` — reasoning ON with a thinking budget of N tokens (OpenRouter passes it as
  Gemini's ``thinkingBudget``). N must be ≥ ``GEMINI_25_MIN_BUDGET`` (512): measured on OpenRouter,
  budgets of 1, 128 and 511 are accepted but think just as long as 512 (~390 reasoning tokens on the
  same prompt) — Google clamps them up to its 512 minimum, so a smaller value would not mean what it
  says. N must also stay below ``JUDGE_MAX_TOKENS``, the completion cap that covers thinking AND answer.
* ``None`` (key absent) — the default, ``GEMINI_25_DEFAULT`` = the smallest budget, ``{"max_tokens": 512}``.

``google/gemini-3.5-flash-lite`` — reasoning is MANDATORY (``{"enabled": false}`` is a 400), so it must
be configured: ``{"effort": "low" | "medium" | "high"}`` (preferred) or ``{"max_tokens": N}`` (N ≥ 1; not
reliably honoured when small — see docs/MONITORS.md).

**Any other model is refused**, whatever its ``reasoning`` — including none, which would still send
some reasoning setting whose meaning for that model nobody has checked. Supporting a new judge means
adding it to ``resolve_reasoning`` after finding out what its ``reasoning`` object actually does.
"""

from __future__ import annotations

GEMINI_25_FLASH_LITE = "google/gemini-2.5-flash-lite"
GEMINI_35_FLASH_LITE = "google/gemini-3.5-flash-lite"
SUPPORTED_JUDGES = (GEMINI_25_FLASH_LITE, GEMINI_35_FLASH_LITE)

# The completion cap of every judge call (``max_tokens`` in the request body). With reasoning on it
# covers the thinking AND the answer, so a thinking budget must stay below it.
JUDGE_MAX_TOKENS = 2048

GEMINI_25_MIN_BUDGET = 512
GEMINI_25_DEFAULT: dict = {"max_tokens": GEMINI_25_MIN_BUDGET}

REASONING_EFFORTS = ("low", "medium", "high")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def resolve_reasoning(model_id: str, reasoning: dict | None, *, monitor: str) -> dict:
    """The exact ``reasoning`` object a judge on ``model_id`` sends, given its configured ``reasoning``.

    Raises ``ValueError`` (naming ``monitor``) for an unsupported model or a setting that model would
    not honour as written. Returns a fresh dict, so callers may keep it without aliasing the config.
    """
    where = f"monitor {monitor!r} ({model_id})"
    if reasoning is not None and not isinstance(reasoning, dict):
        raise ValueError(
            f"{where}: reasoning must be an OpenRouter reasoning object (a dict such as "
            f'{{"enabled": false}}) or absent, got {reasoning!r}'
        )

    if model_id == GEMINI_25_FLASH_LITE:
        if reasoning is None:
            return dict(GEMINI_25_DEFAULT)
        if reasoning == {"enabled": False}:
            return {"enabled": False}
        if set(reasoning) == {"max_tokens"}:
            n = reasoning["max_tokens"]
            if not _is_int(n) or not GEMINI_25_MIN_BUDGET <= n < JUDGE_MAX_TOKENS:
                raise ValueError(
                    f"{where}: reasoning max_tokens (the thinking budget) must be an int in "
                    f"[{GEMINI_25_MIN_BUDGET}, {JUDGE_MAX_TOKENS}), got {n!r} — Google clamps a "
                    f"smaller budget up to {GEMINI_25_MIN_BUDGET}, and the judge's completion cap "
                    f"(max_tokens={JUDGE_MAX_TOKENS}) must also fit the answer"
                )
            return {"max_tokens": n}
        raise ValueError(
            f"{where}: unsupported reasoning {reasoning!r}. Use "
            f'{{"enabled": false}} (reasoning off) or {{"max_tokens": N}} with '
            f"{GEMINI_25_MIN_BUDGET} <= N < {JUDGE_MAX_TOKENS} (reasoning on, budget N); omit the key "
            f"for the default, {GEMINI_25_DEFAULT}"
        )

    if model_id == GEMINI_35_FLASH_LITE:
        if reasoning is None:
            raise ValueError(
                f"{where}: this model mandates reasoning (it rejects {{\"enabled\": false}} with a "
                'fatal 400), so reasoning must be set — {"effort": "low"|"medium"|"high"} '
                '(preferred) or {"max_tokens": N}'
            )
        if set(reasoning) == {"effort"}:
            if reasoning["effort"] not in REASONING_EFFORTS:
                raise ValueError(
                    f"{where}: reasoning effort must be one of {list(REASONING_EFFORTS)}, "
                    f"got {reasoning['effort']!r}"
                )
            return {"effort": reasoning["effort"]}
        if set(reasoning) == {"max_tokens"}:
            n = reasoning["max_tokens"]
            if not _is_int(n) or n < 1:
                raise ValueError(f"{where}: reasoning max_tokens must be an int >= 1, got {n!r}")
            return {"max_tokens": n}
        raise ValueError(
            f"{where}: unsupported reasoning {reasoning!r}. This model mandates reasoning: use "
            '{"effort": "low"|"medium"|"high"} (preferred) or {"max_tokens": N}'
        )

    raise ValueError(
        f"{where}: judge-side reasoning configuration is specialized to "
        f"{' and '.join(SUPPORTED_JUDGES)} — what a `reasoning` setting (including none) does on "
        f"{model_id} has not been established, so it could silently not behave as configured. "
        "Implement support for this model in monitors/judge_reasoning.py:resolve_reasoning before "
        "using it as a monitor."
    )
