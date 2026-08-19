"""Sampling rollouts from a tinker policy + splitting CoT from the final answer.

Kept backend-light: takes a tinker ``SamplingClient`` + a **renderer** (or a bare tokenizer, which is
wrapped in the HF-chat renderer) so the same code serves the untrained-baseline sampler, the
in-training sampler, HF-templated policies (Qwen3 …) and TML-rendered ones (Inkling). Everything
model-family specific — prompt framing and CoT/answer parsing — lives in ``rl/renderers.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import tinker

from monitordecorrelation.rl.renderers import (  # re-exported: long-standing import path
    as_renderer,
    build_prompt_tokens,
    split_cot_answer,
)
from monitordecorrelation.rl.thinking_budget import (
    DONE,
    FORCE_CLOSE,
    ResolvedBudget,
    check_prompt_agrees,
    find_thinking_span,
    plan_step,
)
from monitordecorrelation.rl.token_accounting import META_KEY, RolloutTokenAccount
from monitordecorrelation.types import Prompt, Rollout, Segment

__all__ = [
    "build_prompt_tokens",
    "load_saved_rollouts",
    "sample_rollouts",
    "split_cot_answer",
]

#: Safety net on the sample → force-close → resume loop. Two rounds is the honest maximum (pass 1,
#: then the continuation, which can only end by stop or by exhausting max_tokens); anything beyond
#: that means a sampler returned short without stopping, and we would rather fail than spin.
_MAX_ROUNDS = 6


def load_saved_rollouts(path: str, *, keep_unparsed: bool = False) -> list[tuple[Rollout, bool]]:
    """Reconstruct (rollout, ground_truth) pairs from a saved ``rollouts.jsonl``.

    The saved schema (see ``rl/train.py``) stores ``question``/``cot``/``answer`` plus the env oracle
    under ``env.behavior_present``. We rebuild a lightweight ``Rollout`` (no token ids/activations —
    those aren't persisted) so any monitor that reads text fields can re-score the run post-hoc. The
    saved ``step`` is stashed in ``Rollout.meta`` so callers can group by training step.

    Unparsed rollouts (the env couldn't extract a choice) are dropped by default.
    """
    out: list[tuple[Rollout, bool]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["env"].get("unparsed") and not keep_unparsed:
                continue
            # tolerate pre-refactor rollouts that used the old "ground_truth_misbehavior" key
            beh = r["env"].get("behavior_present", r["env"].get("ground_truth_misbehavior"))
            rollout = Rollout(
                prompt=Prompt(text=r["question"]),
                cot=r.get("cot", ""),
                output=r.get("answer", ""),
                meta={"step": r.get("step")},
            )
            out.append((rollout, bool(beh)))
    return out


@dataclass
class _InFlight:
    """One rollout mid-protocol: the prompt it came from, everything generated so far, and its bill.

    Only used on the thinking-budget path — an unbudgeted rollout is one request and never needs a
    state machine. ``tokens`` is the running completion (injected tokens included, so the span search
    and the ``max_tokens`` allowance both see what the policy will actually be conditioned on).
    """

    prompt: Prompt
    model_input: tinker.ModelInput
    n_prompt_tokens: int
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    account: RolloutTokenAccount = field(default_factory=RolloutTokenAccount)
    finished: bool = False  # the policy emitted a stop token
    stop_reason: str = ""
    done: bool = False  # nothing left to ask for
    index: int = 0  # position in the batch — the stable identity its continuation seeds derive from

    def absorb(self, seq, injected: tuple[int, ...]) -> None:
        """Record one sampling result as a new segment, after the tokens we spliced in before it."""
        toks = list(seq.tokens)
        lps = list(seq.logprobs) if seq.logprobs is not None else [math.nan] * len(toks)
        self.segments.append(Segment(injected=list(injected), tokens=toks, logprobs=lps))
        self.tokens.extend(injected)
        self.logprobs.extend([math.nan] * len(injected))  # never sampled ⇒ no logprob exists
        self.tokens.extend(toks)
        self.logprobs.extend(lps)
        self.stop_reason = str(seq.stop_reason)
        self.finished = self.stop_reason == "stop"


def _cache_hit_share(resp, prompt_len: int, i: int) -> int:
    """This sample's share of a request's measured prefix-cache hit.

    ``prompt_cache_hit_tokens`` is reported once for the request's single shared prompt; the other
    ``num_samples - 1`` copies of that prefill are billed as hits outright (tinker's own wording), so
    every sample after the first is a full hit.
    """
    return prompt_len if i > 0 else int(getattr(resp, "prompt_cache_hit_tokens", 0) or 0)


def _sample_rollouts_budgeted(
    sampling_client, rend, prompts, *, num_samples, max_tokens, temperature, seed,
    budget: ResolvedBudget,
) -> list[Rollout]:
    """``sample_rollouts`` with a thinking budget: sample → force-close → resume.

    Round 1 draws ``min(max_tokens, budget)`` tokens for every prompt in one request per prompt (so
    the group still shares one prefill). Rounds 2+ carry only the rollouts that need more: one
    request each, prompted with ``prompt + everything so far`` — an exact token prefix of what round
    1 already processed, which is what makes the continuation mostly a prefix-cache hit.
    """
    n_close = len(budget.forced_close_ids)
    if budget.budget < max_tokens < budget.budget + n_close:
        # The closer would be spliced in truncated, leaving the reasoning block open forever: the
        # completion would have no </think> at all, so CoT/answer parsing would hand the whole
        # reasoning to the output monitors. Refuse the configuration instead — it leaves no room for
        # an answer anyway. (max_tokens <= budget is fine: the budget simply never binds.)
        raise ValueError(
            f"max_tokens={max_tokens} leaves no room to close the reasoning block: "
            f"thinking_budget={budget.budget} plus {budget.spec.family}'s {n_close}-token closing "
            f"text needs {budget.budget + n_close}. Raise max_tokens (well above that, so there is "
            f"room for an answer) or lower thinking_budget."
        )
    stop = {"stop": rend.stop_tokens} if getattr(rend, "stop_tokens", None) else {}
    flights: list[_InFlight] = []

    # --- round 1: the whole batch, capped at the budget -------------------------------------------
    params = tinker.SamplingParams(
        max_tokens=min(max_tokens, budget.budget), temperature=temperature, seed=seed, **stop
    )
    futures = []
    for p in prompts:
        mi = rend.model_input(p.text)
        check_prompt_agrees(mi.to_ints(), budget)  # the spec's tag placement must match this template
        futures.append((p, mi, sampling_client.sample(mi, num_samples, params)))
    for p, mi, fut in futures:
        resp = fut.result()
        for i, seq in enumerate(resp.sequences):
            f = _InFlight(prompt=p, model_input=mi, n_prompt_tokens=int(mi.length))
            f.absorb(seq, ())
            f.account.add_request(
                prompt_len=f.n_prompt_tokens, cache_hit=_cache_hit_share(resp, f.n_prompt_tokens, i),
                generated=len(seq.tokens), is_first=True,
            )
            f.index = len(flights)
            flights.append(f)

    # --- rounds 2+: force the close where the budget bound, then finish the answer -----------------
    for round_idx in range(1, _MAX_ROUNDS + 2):
        batch: list[tuple[_InFlight, object]] = []
        for f in flights:
            if f.done:
                continue
            step = plan_step(f.tokens, rb=budget, max_tokens=max_tokens, finished=f.finished)
            if step.action == FORCE_CLOSE:
                f.account.forced = True
            if step.action == DONE or step.n_tokens <= 0:
                if step.inject:  # the closer itself exhausted max_tokens: splice it in, ask for nothing
                    f.tokens.extend(step.inject)
                    f.logprobs.extend([math.nan] * len(step.inject))
                    f.segments.append(Segment(injected=list(step.inject), tokens=[], logprobs=[]))
                f.done = True
                continue
            batch.append((f, step))
        if not batch:
            break
        if round_idx > _MAX_ROUNDS:
            raise RuntimeError(
                f"thinking-budget sampling did not converge in {_MAX_ROUNDS} rounds "
                f"({len(batch)} rollouts still unfinished) — a sampling call is returning short "
                f"without a stop reason."
            )
        futures = []
        for f, step in batch:
            prefix = f.tokens + list(step.inject)
            mi = f.model_input.append(tinker.types.EncodedTextChunk(tokens=prefix))
            # Continuations are one request per rollout, so they must not all share the round's seed:
            # tinker's seeded sampling is deterministic, and a shared seed would hand a whole GRPO
            # group the same answer. Derived from the rollout's own index (not its position in this
            # round's batch), so a rollout's seed does not depend on which *others* needed a
            # continuation — same run, same seeds.
            cseed = (None if seed is None
                     else (seed * 1_000_003 + round_idx * 7919 + f.index) % (2**31 - 1))
            p = tinker.SamplingParams(
                max_tokens=step.n_tokens, temperature=temperature, seed=cseed, **stop
            )
            futures.append((f, step, mi, sampling_client.sample(mi, 1, p)))
        for f, step, mi, fut in futures:
            resp = fut.result()
            seq = resp.sequences[0]
            f.absorb(seq, step.inject)
            f.account.add_request(
                prompt_len=int(mi.length), cache_hit=_cache_hit_share(resp, int(mi.length), 0),
                generated=len(seq.tokens), is_first=False,
            )

    rollouts: list[Rollout] = []
    for f in flights:
        span = find_thinking_span(f.tokens, budget)
        n_injected = sum(len(s.injected) for s in f.segments)
        f.account.finish(
            n_completion_tokens=len(f.tokens), n_injected=n_injected, forced=f.account.forced,
            thinking_tokens=span.thinking_tokens(len(f.tokens)),
        )
        cot, answer, text = rend.parse(f.tokens)
        rollouts.append(
            Rollout(
                prompt=f.prompt, cot=cot, output=answer, token_ids=f.tokens, logprobs=f.logprobs,
                segments=f.segments,
                meta={"stop_reason": f.stop_reason, "full_text": text,
                      "n_prompt_tokens": f.n_prompt_tokens, "n_output_tokens": len(f.tokens),
                      "thinking_budget": budget.budget, "budget_forced": f.account.forced,
                      META_KEY: f.account.as_meta()},
            )
        )
    return rollouts


def sample_rollouts(
    sampling_client,
    renderer,
    prompts: list[Prompt],
    *,
    num_samples: int = 1,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    enable_thinking: bool = True,
    seed: int | None = None,
    thinking_budget: ResolvedBudget | None = None,
) -> list[Rollout]:
    """Sample ``num_samples`` completions per prompt; returns a flat list of Rollouts.

    ``renderer`` is a ``rl.renderers`` renderer, or a bare tokenizer (wrapped as HF-chat) for the
    long-standing call sites. Each prompt is expanded into ``num_samples`` rollouts (the GRPO group).
    Token ids + logprobs of the *completion* are stored for the policy-gradient step. ``seed`` (when
    set) makes tinker's sampling reproducible for this call; the caller varies it per call so groups
    still differ across steps. NOTE: relies on a seeded call returning ``num_samples`` *distinct*
    sequences (standard n-sampling) — verify GRPO group advantages have non-zero variance on the
    first real run.

    ``thinking_budget`` (a ``ResolvedBudget``, default None = off) caps the reasoning block and
    force-closes it with the policy family's documented closing text; that path is a separate
    function because it needs several rounds of sampling per batch. When it is None this is the
    single one-shot request it has always been.
    """
    rend = as_renderer(renderer)
    if hasattr(rend, "enable_thinking"):  # HF-chat only; TML conditions on effort, not a flag
        rend.enable_thinking = enable_thinking
    if thinking_budget is not None:
        return _sample_rollouts_budgeted(
            sampling_client, rend, prompts, num_samples=num_samples, max_tokens=max_tokens,
            temperature=temperature, seed=seed, budget=thinking_budget,
        )
    params = tinker.SamplingParams(
        max_tokens=max_tokens, temperature=temperature, seed=seed,
        # Inkling's end-of-turn token is not an EOS the sampler knows about; the renderer supplies it.
        **({"stop": rend.stop_tokens} if getattr(rend, "stop_tokens", None) else {}),
    )
    futures = []
    for p in prompts:
        model_input = rend.model_input(p.text)
        # Prompt length is only knowable here (the rendered ModelInput, not p.text) — stash it per
        # rollout so the loop can log input-token counts without re-rendering. All ``num_samples``
        # rollouts of a prompt share it.
        futures.append((p, model_input.length, sampling_client.sample(model_input, num_samples, params)))

    rollouts: list[Rollout] = []
    for prompt, n_prompt_tokens, fut in futures:
        resp = fut.result()
        for seq in resp.sequences:
            cot, answer, text = rend.parse(list(seq.tokens))
            rollouts.append(
                Rollout(
                    prompt=prompt,
                    cot=cot,
                    output=answer,
                    token_ids=list(seq.tokens),
                    logprobs=list(seq.logprobs) if seq.logprobs is not None else None,
                    meta={"stop_reason": str(seq.stop_reason), "full_text": text,
                          "n_prompt_tokens": int(n_prompt_tokens),
                          "n_output_tokens": len(seq.tokens)},
                )
            )
    return rollouts
