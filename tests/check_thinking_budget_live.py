"""Live end-to-end check of the tinker thinking budget, against real policies.

Not part of the offline suite (it costs sampling): it draws a handful of completions per
(policy × budget), asserts the invariants that matter, and dumps every transcript to JSONL so a
human — or a reviewing agent — can read what the policy actually wrote around the forced close.

    uv run python tests/check_thinking_budget_live.py                       # all supported policies
    uv run python tests/check_thinking_budget_live.py --models Qwen/Qwen3-8B --budgets 64 256
    uv run python tests/check_thinking_budget_live.py --out /tmp/tb         # where transcripts land

Invariants checked per rollout (violations are printed and counted, never silently swallowed):

* the reasoning block is closed exactly once, and its length respects the budget;
* nothing is generated after the closing tag except the answer (no second ``<think>``, no
  re-opened reasoning);
* the completion never exceeds ``max_tokens`` — the budget's forced tokens count against it;
* the injected tokens are exactly the family's documented closing text, and they are marked as
  injected (never as sampled) so GRPO cannot train on them;
* the ideal-vs-actual token bill adds up (ideal ⊆ actual; ideal decode = the whole completion).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tinker
from dotenv import load_dotenv

from monitordecorrelation.rl.renderers import make_renderer
from monitordecorrelation.rl.rollout import sample_rollouts
from monitordecorrelation.rl.thinking_budget import find_thinking_span, resolve_budget
from monitordecorrelation.rl.token_accounting import META_KEY
from monitordecorrelation.types import Prompt

# Every tinker-hosted policy with a documented thinking budget — both families, every size, because
# "same code path" is a claim to test, not to assume (Qwen3.5 vs Qwen3 is exactly how that goes wrong).
DEFAULT_MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
]
DEFAULT_BUDGETS = [32, 128, 512]

# Prompts that reliably make a reasoning model think for a while, so small budgets actually bind.
PROMPTS = [
    "A farmer has 17 sheep. All but 9 run away. Then he buys twice as many as he has left, and "
    "sells 5. How many sheep does he have? Show your reasoning, then give the final number.",
    "Write a Python function that returns the number of trailing zeros in n!, and explain why it "
    "works.",
    "Three switches downstairs control three bulbs upstairs. You may go upstairs only once. How do "
    "you determine which switch controls which bulb?",
]


def _check(r, rb, max_tokens: int) -> list[str]:
    """Every invariant that must hold for one budgeted rollout; returns the failures."""
    bad: list[str] = []
    toks = list(r.token_ids or [])
    span = find_thinking_span(toks, rb)
    n_close = sum(1 for i in range(len(toks) - len(rb.close_ids) + 1)
                  if tuple(toks[i : i + len(rb.close_ids)]) == rb.close_ids)
    forced = bool(r.meta["budget_forced"])
    injected = [t for s in (r.segments or []) for t in s.injected]

    if len(toks) > max_tokens:
        bad.append(f"completion is {len(toks)} tokens > max_tokens={max_tokens}")
    if n_close > 1:
        bad.append(f"{n_close} closing tags — reasoning was re-opened or the closer was duplicated")
    if forced and len(injected) < len(rb.forced_close_ids):
        # max_tokens ran out mid-closer: the completion is allowed to end without a closing tag, but
        # what we spliced in must still be a prefix of the documented text, and nothing may follow.
        if injected != list(rb.forced_close_ids)[: len(injected)]:
            bad.append("the truncated closer is not a prefix of the documented closing text")
        if len(toks) != max_tokens:
            bad.append(f"the closer was truncated at {len(toks)} tokens but max_tokens is {max_tokens}")
        if (r.segments or [])[-1].tokens:
            bad.append("tokens were sampled after a closer that exhausted max_tokens")
    elif forced:
        if injected != list(rb.forced_close_ids):
            bad.append("injected tokens are not the family's documented closing text")
        if n_close != 1:
            bad.append(f"forced, but the completion has {n_close} closing tags")
        # The cut lands exactly at the budget: pass 1 draws `budget` tokens, then the closing text is
        # spliced on — so </think> sits at budget + wherever the tag falls inside that text. (Qwen's
        # closer leads with a whole sentence, which is deliberately INSIDE the reasoning block, so the
        # block itself ends up longer than the budget. That is the documented behaviour.)
        tag_at = next(i for i in range(len(rb.forced_close_ids))
                      if tuple(rb.forced_close_ids[i : i + len(rb.close_ids)]) == rb.close_ids)
        if span.close != rb.budget + tag_at:
            bad.append(f"forced close at token {span.close}, expected {rb.budget + tag_at} "
                       f"(budget {rb.budget} + {tag_at} tokens of closing text before the tag)")
    else:
        if injected:
            bad.append("not forced, yet tokens were injected")
        if len(toks) > rb.budget and not span.closed:
            bad.append(f"{len(toks)} tokens generated with the block still open and no forced close")
        if span.closed and span.close > rb.budget:
            bad.append(f"the block closed at token {span.close}, past the budget {rb.budget}, "
                       f"without a forced close")
    if span.closed and "</think>" in (r.output or ""):
        bad.append("the parsed answer still contains a closing tag")
    if span.closed and rb.spec.open_tag in (r.output or ""):
        bad.append("the policy re-opened its reasoning inside the answer")

    a = r.meta[META_KEY]
    if a["prefill_actual"] < a["prefill_ideal"] or a["decode_ideal"] != len(toks):
        bad.append(f"token bill does not add up: {a}")
    if a["prefill_actual_hit"] > a["prefill_actual"]:
        bad.append(f"more cache hits than prefill: {a}")
    if any(len(s.logprobs) != len(s.tokens) for s in (r.segments or [])):
        bad.append("a segment's logprobs do not line up with its sampled tokens")
    return bad


def run_model(sc, model: str, budgets: list[int], max_tokens: int, n_per_prompt: int,
              out_dir: Path) -> int:
    print(f"\n=== {model} " + "=" * (60 - len(model)))
    tc = sc.create_lora_training_client(model, rank=1)
    rend = make_renderer(model, training_client=tc)
    sampler = sc.create_sampling_client(base_model=model)
    prompts = [Prompt(text=t) for t in PROMPTS]
    n_bad = 0
    path = out_dir / f"{model.replace('/', '__')}.jsonl"
    with path.open("w") as f:
        for budget in budgets:
            rb = resolve_budget(model, rend.tokenizer, budget)
            t0 = time.perf_counter()
            rolls = sample_rollouts(sampler, rend, prompts, num_samples=n_per_prompt,
                                    max_tokens=max_tokens, temperature=1.0, seed=budget,
                                    thinking_budget=rb)
            dt = time.perf_counter() - t0
            forced = sum(1 for r in rolls if r.meta["budget_forced"])
            think = [find_thinking_span(list(r.token_ids), rb).thinking_tokens(len(r.token_ids))
                     for r in rolls]
            bills = [r.meta[META_KEY] for r in rolls]
            for r in rolls:
                bad = _check(r, rb, max_tokens)
                n_bad += len(bad)
                for b in bad:
                    print(f"  ✗ budget={budget}: {b}")
                f.write(json.dumps({
                    "model": model, "budget": budget, "max_tokens": max_tokens,
                    "prompt": r.prompt.text, "forced": r.meta["budget_forced"],
                    "stop_reason": r.meta["stop_reason"],
                    "n_completion_tokens": len(r.token_ids),
                    "thinking_tokens": find_thinking_span(
                        list(r.token_ids), rb).thinking_tokens(len(r.token_ids)),
                    "violations": bad,
                    "cot": r.cot, "answer": r.output, "full_text": r.meta["full_text"],
                    "segments": [{"n_injected": len(s.injected), "n_sampled": len(s.tokens),
                                  "injected_text": rend.tokenizer.decode(s.injected)}
                                 for s in (r.segments or [])],
                    "token_account": r.meta[META_KEY],
                }) + "\n")
            print(f"  budget={budget:>5}: {len(rolls)} rollouts, forced {forced}/{len(rolls)}, "
                  f"think {min(think)}–{max(think)} tok, "
                  f"prefill {sum(b['prefill_actual'] for b in bills)} actual vs "
                  f"{sum(b['prefill_ideal'] for b in bills)} ideal "
                  f"({sum(b['prefill_actual_hit'] for b in bills)} cached), "
                  f"decode {sum(b['decode_actual'] for b in bills)} vs "
                  f"{sum(b['decode_ideal'] for b in bills)} ideal | {dt:.0f}s")
    print(f"  transcripts → {path}")
    return n_bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--budgets", nargs="*", type=int, default=DEFAULT_BUDGETS)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--n-per-prompt", type=int, default=2)
    ap.add_argument("--out", default="data/thinking_budget_check")
    args = ap.parse_args()

    load_dotenv()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = tinker.ServiceClient()
    total = 0
    for model in args.models:
        try:
            total += run_model(sc, model, args.budgets, args.max_tokens, args.n_per_prompt, out_dir)
        except Exception as e:  # noqa: BLE001 — one unavailable policy shouldn't hide the others
            print(f"  ✗ {model}: {type(e).__name__}: {e}")
            total += 1
    print(f"\n{'ALL OK' if total == 0 else f'{total} VIOLATION(S)'} — transcripts in {out_dir}")


if __name__ == "__main__":
    main()
