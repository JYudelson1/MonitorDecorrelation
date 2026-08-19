"""Thinking budgets for the **tinker** backend: a hard cap on the tokens the policy may spend inside
its reasoning block, after which the reasoning is force-closed and the model must answer.

Why this module exists
----------------------
``tinker.SamplingParams`` has exactly six fields — ``max_tokens``, ``seed``, ``stop``,
``temperature``, ``top_k``, ``top_p``. There is no ``thinking_budget``, no logits-processor hook and
no forced-token mechanism (tinker's Anthropic-compatible shim even documents that Anthropic's
``thinking.budget_tokens`` is "accepted for compatibility but **not applied**"). So a budget has to
be built out of the one primitive tinker does give us: **sample, inspect, re-sample from a prefix**.

That is not a workaround — it is *exactly* how both supported providers document budget control:

* **Qwen3** (`qwen.readthedocs.io` → Quickstart → "Thinking Budget", whose own example model is
  ``Qwen/Qwen3-8B``) generates with ``max_new_tokens=thinking_budget``, and if ``</think>`` is
  missing, appends a fixed sentence ending in ``</think>`` and generates the answer from there.
* **Nemotron 3 / 3.5** ship a ``ThinkingBudgetClient`` in every model card and usage cookbook: call 1
  with ``max_tokens=reasoning_budget``; if the trace has no ``</think>``, append
  ``f"{reasoning_content}.\\n</think>\\n\\n"`` and continue the assistant message.

Both cut at **exactly N generated tokens** — no run-on to a sentence or paragraph boundary. (NVIDIA
*also* ships a serving-side ``ThinkingBudgetLogitsProcessor`` that waits for the next newline after
the budget and hard-stops at ``budget + grace``; that one is unreachable from tinker, and it is the
*client* recipe above — equally official, published in the model cards themselves — that this module
implements. The trailing ``"."`` in NVIDIA's closing text is exactly the same graceful-ending idea:
their source comments it "reasoning content is too long, closed with a period".)

The protocol (the ``plan_*`` functions here are pure; ``rl/rollout.py`` does the I/O)
------------------------------------------------------------------------------------
1. **Pass 1** — sample with ``max_tokens = min(max_tokens, budget)``. No ``stop`` sequence, so a
   policy that closes its reasoning early simply carries on writing the answer inside the same call;
   a rollout that fits entirely in the budget costs one request and zero extra tokens.
2. **Force-close** — if the block is still open when the budget is spent, splice in the family's
   documented closing text. Those tokens are **injected, not sampled**.
3. **Continuation** — resume from ``prompt + everything so far`` for the remaining ``max_tokens``
   allowance, so a budgeted completion is never longer than an unbudgeted one.

Because sampling is autoregressive, splitting one generation into two requests at a token boundary
draws from the same distribution as generating it in one go. The only real differences from a
hypothetical in-engine budget are the re-prefill of the continuation request (measured, not guessed
— see ``rl/token_accounting.py``) and the fact that the injected close costs us no decode step.

Counting convention (matches both providers): the budget counts **generated tokens from the start of
the completion**, not "tokens strictly inside the tags". For Nemotron the two coincide, because the
chat template ends with an open ``<think>``. For Qwen3 the model writes ``<think>`` itself, so that
one token comes out of the budget — which is precisely what ``max_new_tokens=thinking_budget`` does
in Qwen's own snippet.

Training treatment
------------------
Injected tokens are handed to GRPO as part of the *observation* of the continuation, never as a
sampled action (``rl/grpo.py`` turns each segment into its own cookbook ``Transition``). The cookbook
masks observation deltas out of the loss, so — unlike the nemo/vLLM path, which trains on the forced
``</think>`` at whatever logprob the policy happened to give it — nothing here pushes the policy
toward a token it did not choose.

Model support is **allow-list only**: a budget is offered exactly for the families whose provider
documents one, with that provider's own closing text, because the whole point of the closing text is
that it is in-distribution for that policy. Everything else raises. See ``_SPECS`` / ``_UNSUPPORTED``.
"""

from __future__ import annotations

from dataclasses import dataclass


class ThinkingBudgetError(ValueError):
    """A thinking budget was requested somewhere it cannot be honoured faithfully."""


@dataclass(frozen=True)
class ThinkingBudgetSpec:
    """One model family's **documented** thinking-budget behaviour.

    ``forced_close_text`` is the provider's own string, character for character — it is what makes
    the forced close in-distribution for that policy (Qwen's sentence explains the interruption to
    the model; Nemotron's leading ``"."`` closes off the truncated sentence), so it is never
    paraphrased, shortened or ported across families.
    """

    family: str
    open_tag: str
    close_tag: str
    prompt_opens_thinking: bool  # does the chat template end *inside* the reasoning block?
    forced_close_text: str
    source: str  # where the behaviour is documented (quoted in errors + docs/INFRA.md)


# Qwen3 (thinking mode). Verbatim from QwenLM/Qwen3 docs/source/getting_started/quickstart.md
# ("Thinking Budget"), whose worked example is Qwen3-8B: the budget is applied as
# `max_new_tokens=thinking_budget` on the first generate() call, and when `</think>` (151668) is
# absent from the result this exact 24-token string is appended before generating the answer.
# The Qwen3 tech report (arXiv 2505.09388 §4.3) quotes the same sentence; its stray "." after
# `</think>` is a typo, so the repo string wins. NB Qwen calls budget-following an *emergent*
# ability of Thinking Mode Fusion (not explicitly trained) and recommends budgets > 1024.
_QWEN3 = ThinkingBudgetSpec(
    family="Qwen3 (thinking)",
    open_tag="<think>",
    close_tag="</think>",
    prompt_opens_thinking=False,  # Qwen3's template stops at "<|im_start|>assistant\n"
    forced_close_text=(
        "\n\nConsidering the limited time by the user, I have to give the solution based on the "
        "thinking directly now.\n</think>\n\n"
    ),
    source="QwenLM/Qwen3 quickstart §Thinking Budget; arXiv 2505.09388 §4.3",
)

# Nemotron 3 / 3.5. Verbatim from NVIDIA's `ThinkingBudgetClient`, published in the Nemotron 3 Nano /
# Super / Ultra model cards and in all four usage cookbooks: call 1 with `max_tokens=reasoning_budget`,
# then `f"{reasoning_content}.\n</think>\n\n"` and continue. (The Ultra *card* prints
# ".\n\n</think>\n" instead, and the Super card's copy is broken — every `</think>` literal was
# stripped from its markdown, leaving `if "" not in reasoning_content:`, which never fires. The
# cookbooks agree with each other and with the Nano card, so that is the string used here.)
# All four models share `<think>`=12 / `</think>`=13 and a template ending in "<think>\n".
_NEMOTRON3 = ThinkingBudgetSpec(
    family="NVIDIA Nemotron 3 / 3.5",
    open_tag="<think>",
    close_tag="</think>",
    prompt_opens_thinking=True,  # template ends "<|im_start|>assistant\n<think>\n"
    forced_close_text=".\n</think>\n\n",
    source="NVIDIA ThinkingBudgetClient (Nemotron 3 model cards + NVIDIA-NeMo/Nemotron cookbooks)",
)

#: Allow-list, longest matching prefix wins. Only families whose *provider* documents a budget.
_SPECS: tuple[tuple[str, ThinkingBudgetSpec], ...] = (
    ("Qwen/Qwen3-8B", _QWEN3),
    ("Qwen/Qwen3-30B-A3B", _QWEN3),  # NB the "-Instruct-2507" suffix is caught by _UNSUPPORTED first
    ("nvidia/NVIDIA-Nemotron-3-Nano", _NEMOTRON3),
    ("nvidia/NVIDIA-Nemotron-3-Super", _NEMOTRON3),
    ("nvidia/NVIDIA-Nemotron-3-Ultra", _NEMOTRON3),
    # Weaker basis than its siblings, deliberately included: 3.5-Lightning's HF card has no budget
    # section at all (only `enable_thinking`), so the documentation is NVIDIA's cookbook + the NIM
    # page ("thinking_token_budget": 2048) rather than the card. Its tags, ids and template are
    # byte-identical to Nemotron 3's, and it is live-checked alongside them.
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning", _NEMOTRON3),
)

#: Refusals, longest matching prefix wins. The reason is quoted into the error so a run that asks for
#: an unsupported budget fails with the evidence, not just a "no".
_UNSUPPORTED: tuple[tuple[str, str], ...] = (
    (
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "its model card says it 'supports only non-thinking mode and does not generate "
        "<think></think> blocks in its output' — there is no reasoning block to budget.",
    ),
    (
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "its model card says it 'supports only non-thinking mode and does not generate "
        "<think></think> blocks in its output' — there is no reasoning block to budget.",
    ),
    (
        "Qwen/Qwen3.5",
        "Qwen documents thinking_budget only for Alibaba Cloud's *hosted* qwen3.5-* endpoints, never "
        "for these open weights: no closing text is published, the <think> token ids changed "
        "(248068/248069, not Qwen3's 151667/151668) and the chat template now pre-opens <think>, so "
        "a Qwen3-derived recipe silently misbehaves here (cf. sgl-project/sglang#25536, where exactly "
        "this produced ~1400 reasoning tokens for a budget of 200).",
    ),
    (
        "Qwen/Qwen3.6",
        "Qwen documents thinking_budget only for Alibaba Cloud's *hosted* qwen3.6-* endpoints, never "
        "for these open weights (QwenLM/Qwen3.6 now redirects to a docs-less README), and the tags, "
        "token ids and template differ from Qwen3's — porting Qwen3's recipe would be a guess.",
    ),
    (
        "deepseek-ai/DeepSeek-V3.1",
        "DeepSeek documents only a binary think / non-think template switch, no reasoning-token cap. "
        "(The '</think>\\n\\n**Final Answer:**' budget-forcing string people cite is from the "
        "third-party R1 'Thoughtology' paper, arXiv 2504.07128, not from DeepSeek.)",
    ),
    (
        "moonshotai/Kimi-K2.6",
        "Moonshot documents only a binary thinking.type enabled/disabled switch, plus a *combined* "
        "max_tokens over reasoning_content + content — that truncates the whole generation rather "
        "than forcing the model into its answer.",
    ),
    (
        "meta-llama/Llama-3.2",
        "Llama 3.2 1B/3B are non-reasoning instruction-tuned models with no reasoning block at all.",
    ),
    (
        "openai/gpt-oss",
        "gpt-oss controls reasoning with a coarse 'Reasoning: low|medium|high' effort level in the "
        "harmony system message, not a token cap; nothing in the harmony spec forces the analysis "
        "channel to hand over to the final channel at N tokens.",
    ),
    (
        "thinkingmachines/Inkling",
        "Thinking Machines control Inkling's reasoning with a continuous `effort` value in [0, 1) — "
        "tinker's own docs say Anthropic's thinking.budget_tokens is 'accepted for compatibility but "
        "not applied. Use output_config.effort'. Set `thinking_effort` instead of thinking_budget.",
    ),
)


@dataclass(frozen=True)
class ResolvedBudget:
    """A spec bound to a tokenizer + a concrete budget: everything sampling needs, in token ids."""

    spec: ThinkingBudgetSpec
    budget: int
    open_ids: tuple[int, ...]
    close_ids: tuple[int, ...]
    forced_close_ids: tuple[int, ...]


def _find(haystack: list[int], needle: tuple[int, ...], start: int = 0) -> int:
    """Index of the first occurrence of ``needle`` in ``haystack`` at/after ``start``; -1 if absent."""
    if not needle:
        return -1
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if tuple(haystack[i : i + n]) == needle:
            return i
    return -1


@dataclass(frozen=True)
class ThinkingSpan:
    """Where the reasoning block sits inside a completion's tokens.

    ``start`` is the index of the first token *inside* the block (0 when the prompt pre-opened it);
    ``close`` is the index of the closing tag, or -1 while the block is still open. ``open_seen``
    separates "the policy answered without ever opening a block" — where the budget simply does not
    apply, and force-closing would corrupt the completion — from "open and still running".
    """

    open_seen: bool
    start: int
    close: int

    @property
    def closed(self) -> bool:
        return self.close >= 0

    def thinking_tokens(self, n_generated: int) -> int:
        """Tokens *inside* the block so far (0 if it never opened). Reported, not used for the cut —
        the cut counts generated tokens, as both providers' recipes do.

        After a forced close this is slightly LARGER than the budget, because the closing text leads
        with content that belongs inside the block (Nemotron's ``"."``: budget+1; Qwen's whole
        sentence: budget+21). That is the documented shape of both recipes, not an overrun — the
        policy still only *sampled* ``budget`` tokens."""
        if not self.open_seen:
            return 0
        return (self.close if self.closed else n_generated) - self.start


def find_thinking_span(tokens: list[int], rb: ResolvedBudget) -> ThinkingSpan:
    """Locate the reasoning block in a completion. Pure; the force/continue decision derives from it."""
    if rb.spec.prompt_opens_thinking:
        # Generation starts inside the block, so thinking runs from token 0 to the closing tag.
        return ThinkingSpan(open_seen=True, start=0, close=_find(tokens, rb.close_ids))
    i = _find(tokens, rb.open_ids)
    if i < 0:
        # No opening tag: either a non-thinking answer, or a bare `</think>` we should still respect.
        close = _find(tokens, rb.close_ids)
        return ThinkingSpan(open_seen=False, start=0, close=close)
    start = i + len(rb.open_ids)
    return ThinkingSpan(open_seen=True, start=start, close=_find(tokens, rb.close_ids, start))


#: What to do with an in-flight rollout after a sampling call.
FORCE_CLOSE = "force"     # budget spent with the block still open → splice in the closer, then resume
CONTINUE = "continue"     # the block is settled; keep generating the answer up to max_tokens
DONE = "done"             # the policy stopped on its own, or the token allowance is exhausted


@dataclass(frozen=True)
class BudgetStep:
    """The next action for one rollout: how many tokens the next call may draw, after ``inject``."""

    action: str
    n_tokens: int = 0
    inject: tuple[int, ...] = ()


def plan_step(
    tokens: list[int], *, rb: ResolvedBudget, max_tokens: int, finished: bool
) -> BudgetStep:
    """Decide what to do next given everything generated so far. Pure — this is the state machine.

    ``finished`` means the policy emitted a stop token, so there is nothing left to generate.
    ``max_tokens`` is the *whole completion* allowance, exactly as in an unbudgeted run: the injected
    closing tokens count against it, so a budgeted rollout is never longer than an unbudgeted one.
    (Corollary: with ``max_tokens <= budget`` the budget can never bind, pass 1 is the only call, and
    a budgeted run is byte-identical to an unbudgeted one.)
    """
    remaining = max_tokens - len(tokens)
    span = find_thinking_span(tokens, rb)
    if finished:
        return BudgetStep(DONE)
    if remaining <= 0:
        return BudgetStep(DONE)
    if span.closed or not span.open_seen:
        # Closed within budget, or never opened → plain generation from here; the budget never binds.
        return BudgetStep(CONTINUE, n_tokens=remaining)
    if len(tokens) < rb.budget:
        # The block is open and the budget is not spent. Unreachable in the normal flow (pass 1 asks
        # for exactly the budget and only returns short when the policy stopped), so this is the
        # defensive branch: top up to the budget rather than handing over the whole allowance, so a
        # sampler that returns short can never let the reasoning run past the cap.
        return BudgetStep(CONTINUE, n_tokens=min(rb.budget - len(tokens), remaining))
    inject = rb.forced_close_ids[:remaining]  # the closer counts against max_tokens
    return BudgetStep(FORCE_CLOSE, n_tokens=remaining - len(inject), inject=inject)


def resolve_spec(model_name: str) -> ThinkingBudgetSpec:
    """The documented budget behaviour for ``model_name``, or a loud, specific refusal."""
    base = model_name.split(":")[0]  # strip tinker's ":peft:<n>" variant suffixes

    def longest(table):
        hit = None
        for prefix, value in table:
            if base.startswith(prefix) and (hit is None or len(prefix) > len(hit[0])):
                hit = (prefix, value)
        return hit

    supported, refused = longest(_SPECS), longest(_UNSUPPORTED)
    # A refusal wins on a longer prefix: "Qwen/Qwen3-30B-A3B-Instruct-2507" is not the thinking
    # "Qwen/Qwen3-30B-A3B" it starts with.
    if supported is not None and (refused is None or len(supported[0]) >= len(refused[0])):
        return supported[1]
    names = ", ".join(p for p, _ in _SPECS)
    if refused is not None:
        raise ThinkingBudgetError(
            f"thinking_budget was requested for policy {model_name!r}, but {refused[1]}\n"
            f"Policies with a documented thinking budget: {names}.\n"
            f"Drop thinking_budget, or use a policy whose provider documents one."
        )
    raise ThinkingBudgetError(
        f"thinking_budget was requested for policy {model_name!r}, which is not in the "
        f"thinking-budget allow-list.\nA budget is only implemented for families whose provider "
        f"documents the exact force-close behaviour — the closing text has to be the one the policy "
        f"was shaped for, and it is never portable across families.\n"
        f"Policies with a documented thinking budget: {names}.\n"
        f"If {model_name!r} does document one, add it to _SPECS in rl/thinking_budget.py with that "
        f"provider's own closing text — do not guess it."
    )


def resolve_budget(model_name: str, tokenizer, budget: int) -> ResolvedBudget:
    """Bind ``model_name``'s spec to a tokenizer and a budget. Raises for unsupported models.

    Token ids come from the policy's *own* tokenizer rather than a table, so a family whose vocab
    moved (Qwen3 → Qwen3.5 renumbered ``<think>`` from 151667 to 248068) can never be silently
    budgeted against another family's ids.
    """
    if budget <= 0:
        raise ThinkingBudgetError(f"thinking_budget must be a positive number of tokens, got {budget}")
    spec = resolve_spec(model_name)

    def enc(text: str) -> tuple[int, ...]:
        ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        if not ids:
            raise ThinkingBudgetError(
                f"{text!r} does not tokenize to anything under {model_name}'s tokenizer — its chat "
                f"format is not what the {spec.family} thinking-budget spec assumes."
            )
        return ids

    return ResolvedBudget(
        spec=spec,
        budget=int(budget),
        open_ids=enc(spec.open_tag),
        close_ids=enc(spec.close_tag),
        forced_close_ids=enc(spec.forced_close_text),
    )


def check_prompt_agrees(prompt_tokens: list[int], rb: ResolvedBudget) -> None:
    """Assert the rendered prompt matches the spec's ``prompt_opens_thinking`` claim.

    Cheap insurance against a chat-template change silently moving where thinking starts — which
    would shift the budget by the length of a preamble, or count answer tokens as thinking.
    """
    tail = list(prompt_tokens[-(len(rb.open_ids) + 3) :])
    opens = _find(tail, rb.open_ids) >= 0 and _find(tail, rb.close_ids) < 0
    if opens != rb.spec.prompt_opens_thinking:
        expected = "end inside an open" if rb.spec.prompt_opens_thinking else "not open a"
        raise ThinkingBudgetError(
            f"the {rb.spec.family} thinking-budget spec says this chat template should {expected} "
            f"{rb.spec.open_tag} block, but the rendered prompt disagrees (last tokens: {tail}). The "
            f"template changed — re-check the spec in rl/thinking_budget.py before trusting the budget."
        )
