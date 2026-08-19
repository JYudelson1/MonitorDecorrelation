"""The tinker thinking budget: the allow-list, the force-close state machine, the two-call sampling
protocol, the ideal-vs-actual token bill, and the GRPO treatment of injected tokens.

Offline — a fake sampling client stands in for tinker, so this exercises the whole protocol (rounds,
splicing, accounting) without a service. The live end-to-end checks live in
``tests/check_thinking_budget_live.py``.
"""

from __future__ import annotations

import math

import pytest

from monitordecorrelation.rl.rollout import sample_rollouts
from monitordecorrelation.rl.thinking_budget import (
    CONTINUE,
    DONE,
    FORCE_CLOSE,
    ThinkingBudgetError,
    check_prompt_agrees,
    find_thinking_span,
    plan_step,
    resolve_budget,
    resolve_spec,
)
from monitordecorrelation.rl.token_accounting import META_KEY, budget_token_metrics
from monitordecorrelation.types import Prompt

# --- the allow-list ---------------------------------------------------------------------------


def test_documented_families_are_accepted():
    assert resolve_spec("Qwen/Qwen3-8B").family.startswith("Qwen3")
    assert resolve_spec("Qwen/Qwen3-30B-A3B").family.startswith("Qwen3")
    for m in ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
              "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
              "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
              "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"):
        assert "Nemotron" in resolve_spec(m).family
    # tinker's ":peft:<n>" variant suffix must not defeat the lookup
    assert resolve_spec("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:peft:262144").family


def test_the_two_families_do_not_share_a_closing_text():
    """The whole point of the allow-list: each provider's own string, never ported across families."""
    qwen, nemo = resolve_spec("Qwen/Qwen3-8B"), resolve_spec("nvidia/NVIDIA-Nemotron-3-Nano-30B")
    assert qwen.forced_close_text != nemo.forced_close_text
    assert qwen.forced_close_text.endswith("</think>\n\n") and not qwen.prompt_opens_thinking
    assert nemo.forced_close_text == ".\n</think>\n\n" and nemo.prompt_opens_thinking


@pytest.mark.parametrize(
    "model, needle",
    [
        # a longer refusal prefix must beat the shorter allowed one it starts with
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "non-thinking mode"),
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", "non-thinking mode"),
        ("Qwen/Qwen3.5-4B", "hosted"),
        ("Qwen/Qwen3.5-9B-Base", "hosted"),
        ("Qwen/Qwen3.6-27B", "hosted"),
        ("deepseek-ai/DeepSeek-V3.1", "binary think"),
        ("moonshotai/Kimi-K2.6", "binary thinking.type"),
        ("meta-llama/Llama-3.2-3B", "non-reasoning"),
        ("openai/gpt-oss-20b", "harmony"),
        ("thinkingmachines/Inkling-Small", "effort"),
        ("some/unknown-model", "not in the thinking-budget allow-list"),
    ],
)
def test_undocumented_families_are_refused_with_the_reason(model, needle):
    with pytest.raises(ThinkingBudgetError) as e:
        resolve_spec(model)
    assert needle in str(e.value)


# --- token binding ----------------------------------------------------------------------------


class _Tok:
    """Char-level stand-in: every character is a token id, plus one id per special tag."""

    SPECIALS = {"<think>": 1000, "</think>": 1001}

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for tag, tid in self.SPECIALS.items():
                if text.startswith(tag, i):
                    out.append(tid)
                    i += len(tag)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def decode(self, ids):
        inv = {v: k for k, v in self.SPECIALS.items()}
        return "".join(inv.get(i, chr(i)) if i < 0x110000 or i in inv else "?" for i in ids)

    def apply_chat_template(self, messages, **kw):
        return self.encode("U:" + messages[0]["content"])


def _qwen_budget(budget=10):
    return resolve_budget("Qwen/Qwen3-8B", _Tok(), budget)


def _nemotron_budget(budget=10):
    return resolve_budget("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", _Tok(), budget)


def test_resolve_budget_binds_ids_from_the_policys_own_tokenizer():
    rb = _qwen_budget(64)
    assert rb.budget == 64
    assert rb.open_ids == (1000,) and rb.close_ids == (1001,)
    assert 1001 in rb.forced_close_ids  # the closer really contains </think>
    assert rb.forced_close_ids[-2:] == (ord("\n"), ord("\n"))


def test_zero_or_negative_budget_is_rejected():
    for bad in (0, -1):
        with pytest.raises(ThinkingBudgetError):
            resolve_budget("Qwen/Qwen3-8B", _Tok(), bad)


def test_check_prompt_agrees_catches_a_template_that_moved():
    qwen, nemo = _qwen_budget(), _nemotron_budget()
    check_prompt_agrees([ord("a"), ord("b")], qwen)                  # Qwen3: must NOT pre-open
    check_prompt_agrees([ord("a"), 1000, ord("\n")], nemo)           # Nemotron: must pre-open
    with pytest.raises(ThinkingBudgetError):
        check_prompt_agrees([ord("a"), 1000, ord("\n")], qwen)
    with pytest.raises(ThinkingBudgetError):
        check_prompt_agrees([ord("a"), ord("b")], nemo)


# --- the span + the state machine ---------------------------------------------------------------


def test_find_thinking_span_qwen_counts_from_the_models_own_open_tag():
    rb = _qwen_budget()
    span = find_thinking_span([1000, 65, 66, 1001, 67], rb)
    assert (span.open_seen, span.start, span.close) == (True, 1, 3)
    assert span.thinking_tokens(5) == 2
    open_still = find_thinking_span([1000, 65, 66], rb)
    assert open_still.open_seen and not open_still.closed and open_still.thinking_tokens(3) == 2
    none = find_thinking_span([65, 66], rb)
    assert not none.open_seen and none.thinking_tokens(2) == 0


def test_find_thinking_span_nemotron_starts_at_token_zero():
    """The template already opened the block, so every generated token is a thinking token."""
    rb = _nemotron_budget()
    span = find_thinking_span([65, 66, 1001, 67], rb)
    assert (span.open_seen, span.start, span.close) == (True, 0, 2)
    assert span.thinking_tokens(4) == 2


def test_plan_step_forces_the_close_when_the_budget_is_spent():
    rb = _qwen_budget(budget=5)
    step = plan_step([1000, 65, 66, 67, 68], rb=rb, max_tokens=400, finished=False)
    assert step.action == FORCE_CLOSE
    assert step.inject == rb.forced_close_ids
    assert step.n_tokens == 400 - 5 - len(rb.forced_close_ids)  # the closer counts against max_tokens


def test_plan_step_leaves_an_early_closer_alone():
    """Closed inside the budget → just keep generating the answer; the budget never binds again."""
    rb = _qwen_budget(budget=5)
    step = plan_step([1000, 65, 1001, 67, 68], rb=rb, max_tokens=100, finished=False)
    assert step.action == CONTINUE and step.inject == () and step.n_tokens == 95


def test_plan_step_never_forces_a_block_that_never_opened():
    rb = _qwen_budget(budget=3)
    step = plan_step([65, 66, 67], rb=rb, max_tokens=100, finished=False)
    assert step.action == CONTINUE and step.inject == ()


def test_plan_step_is_done_on_stop_or_exhausted_allowance():
    rb = _qwen_budget(budget=5)
    assert plan_step([1000, 65], rb=rb, max_tokens=100, finished=True).action == DONE
    assert plan_step([1000] + [65] * 99, rb=rb, max_tokens=100, finished=False).action == DONE


def test_a_truncated_closer_still_fits_inside_max_tokens():
    """max_tokens is the whole-completion allowance, budget included — a budgeted completion is never
    longer than an unbudgeted one. (Sampling refuses this configuration up front, see below; the
    state machine still respects the cap rather than overrunning it.)"""
    rb = _qwen_budget(budget=5)
    step = plan_step([1000, 65, 66, 67, 68], rb=rb, max_tokens=8, finished=False)
    assert step.action == FORCE_CLOSE and step.n_tokens == 0 and len(step.inject) == 3


def test_sampling_refuses_a_max_tokens_that_cannot_fit_the_closer():
    """Otherwise the block would be left open — no </think> at all — and the CoT/answer split would
    hand the whole reasoning to the output monitors."""
    rb = _qwen_budget(budget=5)
    with pytest.raises(ValueError, match="no room to close the reasoning block"):
        _roll(_FakeSampler([[[1000, 65, 66, 67, 68]]]), rb, max_tokens=8)
    # max_tokens <= budget is fine: the budget simply never binds
    sampler = _FakeSampler([[([1000, 65, 66, 67, 68], "length")]])
    (r,) = _roll(sampler, rb, max_tokens=5)
    assert len(sampler.calls) == 1 and r.meta["budget_forced"] is False


def test_budget_at_or_above_max_tokens_can_never_bind():
    rb = _qwen_budget(budget=100)
    assert plan_step([1000] + [65] * 49, rb=rb, max_tokens=50, finished=False).action == DONE


# --- the sampling protocol end to end (fake sampler) --------------------------------------------


class _Seq:
    def __init__(self, tokens, stop_reason="length"):
        self.tokens = list(tokens)
        self.logprobs = [-0.5] * len(tokens)
        self.stop_reason = stop_reason


class _Resp:
    def __init__(self, seqs, cache_hit=0):
        self.sequences = seqs
        self.prompt_cache_hit_tokens = cache_hit


class _Fut:
    def __init__(self, resp):
        self._r = resp

    def result(self):
        return self._r


class _FakeSampler:
    """Replays scripted completions and records every request (prompt length, max_tokens, seed)."""

    def __init__(self, scripts, cache_hit=0):
        self.scripts = list(scripts)
        self.cache_hit = cache_hit
        self.calls = []

    def sample(self, model_input, num_samples, params):
        self.calls.append({"prompt_len": model_input.length, "n": num_samples,
                           "max_tokens": params.max_tokens, "seed": params.seed,
                           "prompt": list(model_input.to_ints())})
        seqs = [_Seq(*s) if isinstance(s, tuple) else _Seq(s) for s in self.scripts.pop(0)]
        return _Fut(_Resp(seqs, cache_hit=min(self.cache_hit, model_input.length)))


def _roll(sampler, budget, *, max_tokens=100, num_samples=1, seed=7):
    return sample_rollouts(sampler, _Tok(), [Prompt(text="hi")], num_samples=num_samples,
                           max_tokens=max_tokens, temperature=1.0, seed=seed,
                           thinking_budget=budget)


def test_pass_one_is_capped_at_the_budget_and_forces_a_close():
    rb = _qwen_budget(budget=6)
    over = [1000] + [ord("x")] * 5                     # still thinking when the budget runs out
    sampler = _FakeSampler([[over], [([ord("A")], "stop")]])
    (r,) = _roll(sampler, rb, max_tokens=400)

    assert sampler.calls[0]["max_tokens"] == 6         # round 1 asks for exactly the budget
    assert sampler.calls[1]["max_tokens"] == 400 - 6 - len(rb.forced_close_ids)
    # the continuation is prompted with prompt + everything so far, an exact prefix of round 1's work
    assert sampler.calls[1]["prompt"][: sampler.calls[0]["prompt_len"]] == sampler.calls[0]["prompt"]
    assert r.token_ids == over + list(rb.forced_close_ids) + [ord("A")]
    assert r.meta["budget_forced"] is True
    # the injected tokens are recorded as injected, never as sampled
    assert [s.injected for s in r.segments] == [[], list(rb.forced_close_ids)]
    assert [s.tokens for s in r.segments] == [over, [ord("A")]]
    assert all(math.isnan(r.logprobs[i]) for i in range(6, 6 + len(rb.forced_close_ids)))
    assert not any(math.isnan(lp) for lp in r.logprobs[:6])
    assert r.cot and r.output == "A"


def test_a_rollout_that_finishes_inside_the_budget_costs_one_request():
    rb = _qwen_budget(budget=50)
    seq = [1000, ord("t"), 1001, ord("A")]
    sampler = _FakeSampler([[(seq, "stop")]])
    (r,) = _roll(sampler, rb, max_tokens=400)
    assert len(sampler.calls) == 1                     # no continuation, no wasted tokens
    assert r.segments == [] or [s.injected for s in r.segments] == [[]]
    assert r.meta["budget_forced"] is False
    acct = r.meta[META_KEY]
    assert acct["prefill_actual"] == acct["prefill_ideal"]
    assert acct["decode_actual"] == acct["decode_ideal"] == len(seq)


def test_an_early_closer_still_gets_its_answer_continued():
    """Closing inside the budget doesn't end the rollout — it just spends no forced tokens."""
    rb = _qwen_budget(budget=6)
    first = [1000, ord("t"), 1001, ord("A"), ord("B"), ord("C")]  # hit the pass-1 cap mid-answer
    sampler = _FakeSampler([[first], [([ord("D")], "stop")]])
    (r,) = _roll(sampler, rb, max_tokens=400)
    assert len(sampler.calls) == 2
    assert sampler.calls[1]["max_tokens"] == 394       # nothing injected → full remaining allowance
    assert r.meta["budget_forced"] is False
    assert r.token_ids == first + [ord("D")]
    assert not any(math.isnan(lp) for lp in r.logprobs)


def test_group_continuations_do_not_share_a_seed():
    """One seed per continuation, else a GRPO group's answers would be correlated after the force."""
    rb = _qwen_budget(budget=3)
    over = [1000, ord("x"), ord("y")]
    sampler = _FakeSampler([[over, over], [([ord("A")], "stop")], [([ord("B")], "stop")]])
    _roll(sampler, rb, max_tokens=400, num_samples=2)
    cont_seeds = [c["seed"] for c in sampler.calls[1:]]
    assert len(set(cont_seeds)) == 2 and sampler.calls[0]["seed"] == 7


def test_the_bill_separates_ideal_from_actual_and_hits_from_misses():
    rb = _qwen_budget(budget=4)
    over = [1000, ord("x"), ord("y"), ord("z")]
    sampler = _FakeSampler([[over], [([ord("A")], "stop")]], cache_hit=3)
    (r,) = _roll(sampler, rb, max_tokens=400)
    a = r.meta[META_KEY]
    p1, p2 = sampler.calls[0]["prompt_len"], sampler.calls[1]["prompt_len"]

    assert a["prefill_ideal"] == p1                      # one request is all an engine would need
    assert a["prefill_actual"] == p1 + p2                # we re-sent prompt + the truncated reasoning
    assert a["prefill_ideal_hit"] == 3 and a["prefill_actual_hit"] == 6
    assert a["decode_actual"] == len(over) + 1           # the injected closer costs us no decode step
    assert a["decode_ideal"] == len(r.token_ids)         # …but an in-engine budget would decode it
    assert a["n_injected"] == len(rb.forced_close_ids) and a["n_requests"] == 2

    m = budget_token_metrics([a])
    assert m["tokens/prefill_ideal_total"] == p1
    assert m["tokens/prefill_actual_total"] == p1 + p2
    assert m["tokens/prefill_actual_cache_miss_total"] == (p1 + p2) - 6
    assert m["tokens/prefill_overhead_ratio"] == (p1 + p2) / p1
    assert m["tokens/budget_forced_rate"] == 1.0
    assert m["tokens/sampling_requests_per_rollout"] == 2.0


def test_shared_prompt_samples_after_the_first_are_billed_as_cache_hits():
    rb = _qwen_budget(budget=50)
    done = ([1000, ord("t"), 1001, ord("A")], "stop")
    sampler = _FakeSampler([[done, done, done]], cache_hit=0)
    rs = _roll(sampler, rb, max_tokens=400, num_samples=3)
    hits = [r.meta[META_KEY]["prefill_actual_hit"] for r in rs]
    p = sampler.calls[0]["prompt_len"]
    assert hits == [0, p, p]  # the prompt is prefilled once; the other two copies are hits


def test_budget_metrics_are_absent_without_a_budget():
    assert budget_token_metrics([]) == {}


# --- GRPO: injected tokens are observation, not action ------------------------------------------


def test_injected_tokens_are_masked_out_of_the_loss():
    """The forced closer must never carry an advantage — that is the whole reason for segments."""
    import torch
    from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages

    from monitordecorrelation.rl.grpo import to_trajectory_groups

    rb = _qwen_budget(budget=4)
    over = [1000, ord("x"), ord("y"), ord("z")]
    sampler = _FakeSampler([[over, over], [([ord("A")], "stop")], [([ord("B")], "stop")]])
    rolls = _roll(sampler, rb, max_tokens=400, num_samples=2)

    groups = to_trajectory_groups(_Tok(), rolls, [1.0, 0.0], group_size=2)
    (traj,) = [groups[0].trajectories_G[0]]
    assert len(traj.transitions) == 2                     # one per segment
    assert sum(t.reward for t in traj.transitions) == 1.0  # the split does not change the return

    data, _ = assemble_training_data(groups, compute_advantages(groups))
    assert len(data) == 2  # one merged datum per rollout: the observations extend, so nothing splits
    d = data[0]
    mask = torch.tensor(d.loss_fn_inputs["mask"].data).flatten()
    adv = torch.tensor(d.loss_fn_inputs["advantages"].data).flatten()
    # exactly the sampled tokens are trained on: 4 from round 1 + 1 from the continuation
    assert mask.sum().item() == len(over) + 1
    assert (adv[mask == 0] == 0).all()
    assert not torch.isnan(torch.tensor(d.loss_fn_inputs["logprobs"].data)).any()


def test_a_short_return_tops_up_to_the_budget_instead_of_handing_over_the_allowance():
    """Defensive: if a sampling call ever returns short with the block still open, the next call must
    be capped at the remaining budget — never at the full max_tokens, which would blow the cap."""
    rb = _qwen_budget(budget=20)
    step = plan_step([1000] + [65] * 9, rb=rb, max_tokens=400, finished=False)
    assert step.action == CONTINUE and step.n_tokens == 10 and step.inject == ()


# --- the config gate (fail at LOAD, not mid-run) ------------------------------------------------


def _cfg(**kw):
    from monitordecorrelation.experiment_config import ExperimentConfig

    return ExperimentConfig(run_name="t", **kw)


def test_config_accepts_a_budget_on_a_documented_policy():
    assert _cfg(policy="Qwen/Qwen3-8B", thinking_budget=512).thinking_budget == 512
    assert _cfg(policy="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", thinking_budget=1).thinking_budget


def test_config_default_is_off_and_changes_nothing():
    assert _cfg().thinking_budget is None


@pytest.mark.parametrize("kw, needle", [
    ({"policy": "openai/gpt-oss-20b", "thinking_budget": 256}, "harmony"),
    ({"policy": "Qwen/Qwen3.5-4B", "thinking_budget": 256}, "hosted"),
    ({"policy": "Qwen/Qwen3-8B", "thinking_budget": 0}, "positive number of tokens"),
    ({"policy": "Qwen/Qwen3-8B", "thinking_budget": 256, "backend": "transformers"},
     "not supported on the 'transformers' backend"),
])
def test_config_rejects_a_budget_it_cannot_honour(kw, needle):
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as e:
        _cfg(**kw)
    assert needle in str(e.value)


def test_the_nemo_backend_keeps_its_own_budget_path():
    """nemo enforces budgets through a vLLM logits processor, on any policy — the tinker allow-list
    must not retro-actively break it."""
    assert _cfg(policy="Qwen/Qwen3.5-4B-Base", backend="nemo", thinking_budget=512).thinking_budget == 512
