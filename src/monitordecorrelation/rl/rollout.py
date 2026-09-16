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

from monitordecorrelation.rl.renderers import (  # re-exported: long-standing import path
    as_renderer,
    build_prompt_tokens,
    split_cot_answer,
)
from monitordecorrelation.types import Prompt, Rollout

__all__ = [
    "build_prompt_tokens",
    "load_saved_rollouts",
    "sample_rollouts",
    "split_cot_answer",
]


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
    Token ids + logprobs of the *completion* are stored for the policy-gradient step. ``seed`` (when
    set) makes tinker's sampling reproducible for this call; the caller varies it per call so groups
    still differ across steps. NOTE: relies on a seeded call returning ``num_samples`` *distinct*
    sequences (standard n-sampling) — verify GRPO group advantages have non-zero variance on the
    first real run.

    ``on_rollout(index, rollout)`` (optional) is called as each prompt's completions come back, so a
    caller can start per-rollout work (monitor API calls) on the prompts that already landed instead
    of waiting for the slowest one. Given it, each prompt is awaited in its own thread so no prompt
    holds back another's callbacks; the returned list keeps prompt order either way.
    """
    rend = as_renderer(renderer)
    if hasattr(rend, "enable_thinking"):  # HF-chat only; TML conditions on effort, not a flag
        rend.enable_thinking = enable_thinking
    params = tinker.SamplingParams(
        max_tokens=max_tokens, temperature=temperature, seed=seed,
        # Inkling's end-of-turn token is not an EOS the sampler knows about; the renderer supplies it.
        **({"stop": rend.stop_tokens} if getattr(rend, "stop_tokens", None) else {}),
    )
    futures = []
    for p in prompts:
        futures.append((p, sampling_client.sample(rend.model_input(p.text), num_samples, params)))

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
        rollouts: list[Rollout] = []
        for prompt, fut in futures:
            for seq in fut.result().sequences:
                rollouts.append(_to_rollout(prompt, seq))
        return rollouts

    # Streaming path: a thread per prompt, writing into pre-allocated slots so the output order is
    # the same prompt-major, group-consecutive layout GRPO expects. It needs a fixed stride to index
    # those slots, hence the group-size check (tinker's n-sampling always returns what was asked).
    slots: list[Rollout | None] = [None] * (len(prompts) * num_samples)

    def _collect(i: int) -> None:
        prompt, fut = futures[i]
        seqs = list(fut.result().sequences)
        if len(seqs) != num_samples:
            raise RuntimeError(f"asked for {num_samples} samples, got {len(seqs)}")
        for k, seq in enumerate(seqs):
            idx = i * num_samples + k
            r = _to_rollout(prompt, seq)
            slots[idx] = r
            on_rollout(idx, r)

    if prompts:
        with ThreadPoolExecutor(max_workers=len(prompts), thread_name_prefix="sample") as ex:
            for fut in [ex.submit(_collect, i) for i in range(len(prompts))]:
                fut.result()
    return [r for r in slots if r is not None]
