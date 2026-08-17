"""Sampling rollouts from a tinker policy + splitting CoT from the final answer.

Kept backend-light: takes a tinker ``SamplingClient`` + a **renderer** (or a bare tokenizer, which is
wrapped in the HF-chat renderer) so the same code serves the untrained-baseline sampler, the
in-training sampler, HF-templated policies (Qwen3 …) and TML-rendered ones (Inkling). Everything
model-family specific — prompt framing and CoT/answer parsing — lives in ``rl/renderers.py``.
"""

from __future__ import annotations

import json

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
            rollout = Rollout(
                prompt=Prompt(text=r["question"]),
                cot=r.get("cot", ""),
                output=r.get("answer", ""),
                meta={"step": r.get("step")},
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
) -> list[Rollout]:
    """Sample ``num_samples`` completions per prompt; returns a flat list of Rollouts.

    ``renderer`` is a ``rl.renderers`` renderer, or a bare tokenizer (wrapped as HF-chat) for the
    long-standing call sites. Each prompt is expanded into ``num_samples`` rollouts (the GRPO group).
    Token ids + logprobs of the *completion* are stored for the policy-gradient step. ``seed`` (when
    set) makes tinker's sampling reproducible for this call; the caller varies it per call so groups
    still differ across steps. NOTE: relies on a seeded call returning ``num_samples`` *distinct*
    sequences (standard n-sampling) — verify GRPO group advantages have non-zero variance on the
    first real run.
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
