"""Sampling rollouts from a tinker policy + splitting CoT from the final answer.

Kept backend-light: takes a tinker ``SamplingClient`` + a **renderer** (or a bare tokenizer, which is
wrapped in the HF-chat renderer) so the same code serves the untrained-baseline sampler, the
in-training sampler, HF-templated policies (Qwen3 …) and TML-rendered ones (Inkling). Everything
model-family specific — prompt framing and CoT/answer parsing — lives in ``rl/renderers.py``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import tinker

from monitordecorrelation.rl.episodes import derive_sample_seed
from monitordecorrelation.rl.renderers import (  # re-exported: long-standing import path
    as_renderer,
    build_prompt_tokens,
    split_cot_answer,
)
from monitordecorrelation.types import Prompt, Rollout

__all__ = [
    "build_prompt_tokens",
    "load_saved_rollouts",
    "saved_rollout_invalid",
    "sample_rollouts",
    "split_cot_answer",
]


def saved_rollout_invalid(rec: dict) -> bool:
    """Is a saved ``rollouts.jsonl`` record an INVALID rollout (truncated or unparseable) — one the run
    never showed to a monitor? Newer dumps say so (``invalid_reason``); older ones are judged by their
    parse flag or a ``reward_override`` (only ever set on truncated / unparseable rollouts)."""
    if "invalid_reason" in rec:
        return rec["invalid_reason"] is not None
    env = rec.get("env") or {}
    return bool(env.get("unparsed")) or env.get("reward_override") is not None


def load_saved_rollouts(path: str, *, keep_invalid: bool = False) -> list[tuple[Rollout, bool]]:
    """Reconstruct (rollout, ground_truth) pairs from a saved ``rollouts.jsonl``.

    The saved schema (see ``rl/train.py``) stores ``question``/``cot``/``answer`` plus the env oracle
    under ``env.behavior_present``. We rebuild a lightweight ``Rollout`` (no token ids/activations —
    those aren't persisted) so any monitor that reads text fields can re-score the run post-hoc. The
    saved ``step`` is stashed in ``Rollout.meta`` so callers can group by training step.

    Invalid rollouts (truncated, or the env couldn't parse the output — ``saved_rollout_invalid``) are
    dropped by default: monitors are never evaluated on them, post hoc included.
    """
    out: list[tuple[Rollout, bool]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if saved_rollout_invalid(r) and not keep_invalid:
                continue
            # tolerate pre-refactor rollouts that used the old "ground_truth_misbehavior" key
            beh = r["env"].get("behavior_present", r["env"].get("ground_truth_misbehavior"))
            # Multi-turn runs also persist the per-turn episode record; carry it so a re-scored
            # rollout gets the SAME chronological judge prompt the run itself used (see
            # monitordecorrelation.transcript) instead of the flat two-section fallback.
            env_meta = r["env"].get("meta") if isinstance(r["env"].get("meta"), dict) else None
            meta = {"step": r.get("step")}
            if env_meta and isinstance(env_meta.get("turns"), list):
                meta["episode"] = env_meta
            rollout = Rollout(
                prompt=Prompt(text=r["question"]),
                cot=r.get("cot", ""),
                output=r.get("answer", ""),
                meta=meta,
            )
            out.append((rollout, bool(beh)))
    return out


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
    on_rollout: Callable[[int, Rollout], None] | None = None,
) -> list[Rollout]:
    """Sample ``num_samples`` completions per prompt; returns a flat list of Rollouts.

    ``renderer`` is a ``rl.renderers`` renderer, or a bare tokenizer (wrapped as HF-chat) for the
    long-standing call sites. Each prompt is expanded into ``num_samples`` rollouts (the GRPO group).
    Token ids + logprobs of the *completion* are stored for the policy-gradient step.

    **Every completion is its own single-sample request** — never one ``num_samples=G`` request per
    prompt. Tinker applies one seed to the whole request, so a seeded ``sample(num_samples=8)`` returns
    ~1–2 distinct sequences (measured on base Qwen3-8B: 1.4 of 8 unique; 8 single-sample calls with
    distinct seeds: 8 of 8) — i.e. the GRPO group collapses and its advantages are all zero. With
    ``seed`` set, completion k of prompt i is seeded ``derive_sample_seed(seed, i, k, 0)`` (group,
    sample, call 0 — the same position scheme as the multi-turn driver, ``rl/episodes.py``), which
    keeps the run reproducible AND the group distinct; ``seed=None`` sends the same requests unseeded.

    ``on_rollout(index, rollout)`` (optional) is called as each completion comes back, so a caller
    can start per-rollout work (monitor API calls) on the completions that already landed instead of
    waiting for the slowest one. Given it, each completion is awaited in its own thread so none holds
    back another's callback; the returned list keeps the prompt-major, group-consecutive order either
    way.
    """
    rend = as_renderer(renderer)
    if hasattr(rend, "enable_thinking"):  # HF-chat only; TML conditions on effort, not a flag
        rend.enable_thinking = enable_thinking
    stop = {"stop": rend.stop_tokens} if getattr(rend, "stop_tokens", None) else {}

    def _params(sample_seed: int | None) -> tinker.SamplingParams:
        # Inkling's end-of-turn token is not an EOS the sampler knows about; the renderer supplies it.
        return tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=sample_seed, **stop)

    # futures[i*num_samples + k] = (prompt i, the single-sample request for its completion k): the
    # prompt-major, group-consecutive layout GRPO expects.
    futures: list[tuple[Prompt, object]] = []
    for i, p in enumerate(prompts):
        mi = rend.model_input(p.text)
        futures += [(p, sampling_client.sample(
                        mi, 1, _params(None if seed is None else derive_sample_seed(seed, i, k, 0))))
                    for k in range(num_samples)]

    def _seq(idx: int):
        seqs = list(futures[idx][1].result().sequences)
        if len(seqs) != 1:
            raise RuntimeError(f"asked for 1 sample, got {len(seqs)}")
        return seqs[0]

    def _to_rollout(prompt: Prompt, seq) -> Rollout:
        cot, answer, text = rend.parse(list(seq.tokens))
        return Rollout(
            prompt=prompt,
            cot=cot,
            output=answer,
            token_ids=list(seq.tokens),
            logprobs=list(seq.logprobs) if seq.logprobs is not None else None,
            meta={"stop_reason": str(seq.stop_reason), "full_text": text},
        )

    if on_rollout is None:
        return [_to_rollout(futures[idx][0], _seq(idx)) for idx in range(len(futures))]

    # Streaming path: a thread per completion, writing into pre-allocated slots so the output keeps
    # the same prompt-major, group-consecutive layout.
    slots: list[Rollout | None] = [None] * len(futures)

    def _collect(idx: int) -> None:
        r = _to_rollout(futures[idx][0], _seq(idx))
        slots[idx] = r
        on_rollout(idx, r)

    if futures:
        with ThreadPoolExecutor(max_workers=len(futures), thread_name_prefix="sample") as ex:
            for fut in [ex.submit(_collect, idx) for idx in range(len(futures))]:
                fut.result()
    return [r for r in slots if r is not None]
