"""Sampling rollouts from a tinker policy + splitting CoT from the final answer.

Kept backend-light: takes a tinker ``SamplingClient`` + its HF tokenizer so the same code serves both
the untrained-baseline sampler and the in-training sampler. We render chat ourselves via the HF chat
template (tinker has no built-in renderer), with Qwen3 thinking enabled so the CoT is load-bearing.
"""

from __future__ import annotations

import json

import tinker

from monitordecorrelation.types import Prompt, Rollout

_THINK_CLOSE = "</think>"


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
    seed: int | None = None,
) -> list[Rollout]:
    """Sample ``num_samples`` completions per prompt; returns a flat list of Rollouts.

    Each prompt is expanded into ``num_samples`` rollouts (the GRPO group). Token ids + logprobs of
    the *completion* are stored for the policy-gradient step. ``seed`` (when set) makes tinker's
    sampling reproducible for this call; the caller varies it per call so groups still differ across
    steps. NOTE: relies on a seeded call returning ``num_samples`` *distinct* sequences (standard
    n-sampling) — verify GRPO group advantages have non-zero variance on the first real run.
    """
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed)
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
