"""A hard cap on the length of the policy's ``<think>`` block, for the nemo backend.

Why this file exists at all
--------------------------
A thinking budget cannot be expressed in NeMo-RL's config: NeMo-RL builds its ``SamplingParams`` by
hand (``models/generation/vllm/vllm_worker.py::_build_sampling_params``, an explicit kwargs list) and
neither ``policy.generation`` nor ``vllm_cfg`` has a knob for it, so vLLM's own
``thinking_token_budget`` sampling parameter — and therefore its built-in
``ThinkingTokenBudgetLogitsProcessor`` — is unreachable from a YAML file.

What *is* reachable is ``policy.generation.vllm_kwargs``, which NeMo-RL splats straight into
``vllm.LLM(**kwargs)``, and ``vllm.LLM`` takes ``logits_processors=[<fqcn>, …]``. So the budget is
enforced by a logits processor of our own, named from the config, running inside vLLM's sampler:

    vllm_kwargs:  {logits_processors: [monitordecorrelation.backends.nemo.thinking_budget:ThinkingBudgetLogitsProcessor]}
    vllm_cfg.env_vars: {MD_THINKING_BUDGET: 512}

Both lines are emitted by ``backends/nemo/params.py`` when ``cfg.thinking_budget`` is set, and by
nothing at all when it is not — so this module is dead code on a default run. **No NeMo-RL source is
modified.**

How it works
------------
Qwen3.5's chat template ends every generation prompt with ``<think>\\n``, so a rollout starts *inside*
the thinking block. This processor counts the tokens generated since then and, once the count
reaches the budget, drives the logits of ``</think>`` to +1e9 until the whole end-of-thinking token
sequence has been emitted — after which the policy writes its answer normally and sampling is
untouched again. If the policy closes ``</think>`` on its own before the budget, nothing happens.

The state machine is vLLM's, inherited unchanged from ``ThinkingTokenBudgetLogitsProcessor``; the
only override is where the budget and the ``<think>`` / ``</think>`` token ids come from — the
parent reads them off ``SamplingParams`` and ``vllm_config.reasoning_config``, which is exactly the
pair of things a NeMo-RL config cannot reach. The cap is therefore approximate by a token or two
(the parent skips re-parsing the output while a countdown is running), which is the intended
trade — it is a budget, not an exact length.

Two things to know before using it in an experiment:

* The forced ``</think>`` is part of the sampled sequence and **is trained on**, at whatever
  logprob the policy actually assigned it. That is the standard artifact of budget forcing under RL
  (the policy is pushed toward a token it did not choose); with a budget large enough that only a
  minority of rollouts hit it, it is a small effect, but it is not zero.
* It runs in the vLLM worker's virtualenv, not this project's, so it may import only ``torch``,
  ``transformers`` and ``vllm``. It reaches that interpreter over ``PYTHONPATH`` (set by
  ``backends/nemo/launcher.py`` and inherited by every Ray worker).
"""

from __future__ import annotations

import os

from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    MoveDirectionality,
    ThinkingTokenBudgetLogitsProcessor,
)

#: Budget, in generated tokens. Unset / empty / <= 0 disables the processor entirely.
BUDGET_ENV = "MD_THINKING_BUDGET"
#: The delimiters to count between. Overridable for a policy that does not use Qwen's tags.
START_ENV, END_ENV = "MD_THINKING_START", "MD_THINKING_END"
DEFAULT_START, DEFAULT_END = "<think>", "</think>"


def _budget_from_env(env: dict[str, str] | None = None) -> int | None:
    """The configured budget, or None for "off". Pure, so it is testable without vLLM."""
    raw = (env if env is not None else os.environ).get(BUDGET_ENV, "")
    raw = raw.strip()
    if not raw or raw.lower() in ("null", "none", "false"):
        return None
    budget = int(raw)
    return budget if budget > 0 else None


class ThinkingBudgetLogitsProcessor(ThinkingTokenBudgetLogitsProcessor):
    """``ThinkingTokenBudgetLogitsProcessor`` with the budget and the tags taken from the
    environment rather than from per-request ``SamplingParams`` / ``vllm_config.reasoning_config``.

    Constructed once per vLLM engine, by fully-qualified class name, from
    ``policy.generation.vllm_kwargs.logits_processors``.
    """

    def __init__(self, vllm_config, device, is_pin_memory) -> None:
        # The parent allocates its reusable mask/force tensors here and sets is_enabled False
        # (there is no reasoning_config on a NeMo-RL engine); everything else we supply ourselves.
        super().__init__(vllm_config, device, is_pin_memory)
        self.budget = _budget_from_env()
        if self.budget is None:
            self.is_enabled = False
            return

        # NOT vllm.tokenizers.cached_tokenizer_from_config: NeMo-RL feeds the engine token ids
        # rather than text and sets skip_tokenizer_init from the rest of the generation config
        # (models/generation/__init__.py), and that helper returns None whenever it is True. Loading
        # the tokenizer directly is a local cache hit by the time the engine exists.
        from transformers import AutoTokenizer

        model_config = vllm_config.model_config
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
        )
        start = os.environ.get(START_ENV) or DEFAULT_START
        end = os.environ.get(END_ENV) or DEFAULT_END
        self.reasoning_start_token_ids = tokenizer.encode(start, add_special_tokens=False)
        self.reasoning_end_token_ids = tokenizer.encode(end, add_special_tokens=False)
        if not self.reasoning_start_token_ids or not self.reasoning_end_token_ids:
            raise ValueError(
                f"{BUDGET_ENV}={self.budget} but {start!r}/{end!r} do not tokenize to anything "
                f"under {model_config.tokenizer} — set {START_ENV}/{END_ENV} for this policy."
            )
        self.is_enabled = True
        print(
            f"[thinking-budget] {self.budget} tokens between {start!r} "
            f"({self.reasoning_start_token_ids}) and {end!r} ({self.reasoning_end_token_ids})",
            flush=True,
        )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """The parent's ``update_state``, with ``self.budget`` in place of the per-request
        ``params.thinking_token_budget`` (which nothing sets, hence this whole module). Every
        request in the batch gets the budget; the per-request state machine is the parent's."""
        if not self.is_enabled:
            return
        if batch_update:
            for index, _params, prompt_tok_ids, output_tok_ids in batch_update.added:
                state = self._init_state_entry(prompt_tok_ids, self.budget)
                state["output_tok_ids"] = output_tok_ids
                self._state[index] = state
            for index in batch_update.removed:
                self._state.pop(index, None)
            for i1, i2, direction in batch_update.moved:
                if direction == MoveDirectionality.SWAP:
                    state1, state2 = self._state.pop(i1, None), self._state.pop(i2, None)
                    if state1 is not None:
                        self._state[i2] = state1
                    if state2 is not None:
                        self._state[i1] = state2
                else:
                    state = self._state.pop(i1, None)
                    if state is not None:
                        self._state[i2] = state

        for state in self._state.values():
            self._update_think_state(state)
