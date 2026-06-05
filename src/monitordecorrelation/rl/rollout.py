"""Sampling rollouts from a tinker policy + splitting CoT from the final answer.

Kept backend-light: takes a tinker ``SamplingClient`` + its HF tokenizer so the same code serves both
the untrained-baseline sampler and the in-training sampler. We render chat ourselves via the HF chat
template (tinker has no built-in renderer), with Qwen3 thinking enabled so the CoT is load-bearing.
"""

from __future__ import annotations

import tinker

from monitordecorrelation.types import Prompt, Rollout

_THINK_CLOSE = "</think>"


def build_prompt_tokens(tokenizer, question: str, enable_thinking: bool = True) -> list[int]:
    """Render a single user turn to token ids with a generation prompt."""
    messages = [{"role": "user", "content": question}]
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=enable_thinking,
    )
    # transformers 5.x may return a BatchEncoding even with return_dict=False on some paths.
    if hasattr(out, "input_ids"):
        out = out.input_ids
    return list(out)


def split_cot_answer(text: str) -> tuple[str, str]:
    """Split a Qwen3 completion into (cot, answer) on the closing think tag."""
    if _THINK_CLOSE in text:
        cot, answer = text.split(_THINK_CLOSE, 1)
        return cot.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()


def sample_rollouts(
    sampling_client,
    tokenizer,
    prompts: list[Prompt],
    *,
    num_samples: int = 1,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    enable_thinking: bool = True,
) -> list[Rollout]:
    """Sample ``num_samples`` completions per prompt; returns a flat list of Rollouts.

    Each prompt is expanded into ``num_samples`` rollouts (the GRPO group). Token ids + logprobs of
    the *completion* are stored for the policy-gradient step.
    """
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature)
    futures = []
    for p in prompts:
        toks = build_prompt_tokens(tokenizer, p.text, enable_thinking)
        model_input = tinker.ModelInput.from_ints(toks)
        futures.append((p, sampling_client.sample(model_input, num_samples, params)))

    rollouts: list[Rollout] = []
    for prompt, fut in futures:
        resp = fut.result()
        for seq in resp.sequences:
            text = tokenizer.decode(seq.tokens)
            cot, answer = split_cot_answer(text)
            rollouts.append(
                Rollout(
                    prompt=prompt,
                    cot=cot,
                    output=answer,
                    token_ids=list(seq.tokens),
                    logprobs=list(seq.logprobs) if seq.logprobs is not None else None,
                    meta={"stop_reason": str(seq.stop_reason), "full_text": text},
                )
            )
    return rollouts
