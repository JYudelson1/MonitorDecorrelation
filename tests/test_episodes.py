"""The multi-turn episode driver + its GRPO datum path, offline: a fake sampler, a fake renderer and
a fake tool env. Checks the token bookkeeping the training step depends on — every observation is a
prefix-extension of the previous ob+ac, so tinker-cookbook folds an episode into ONE datum with
observation tokens masked — plus seeding, group order, and truncation handling."""

from __future__ import annotations

import tinker
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages

from monitordecorrelation.rl.episodes import derive_sample_seed, run_episodes
from monitordecorrelation.rl.grpo import to_trajectory_groups
from monitordecorrelation.types import Prompt

EOS = 99
USER_OPEN, USER_CLOSE, GEN = 90, 91, 92


class _Seq:
    def __init__(self, tokens, stop="stop"):
        self.tokens = tokens
        self.logprobs = [-0.5] * len(tokens)
        self.stop_reason = stop


class _Fut:
    def __init__(self, seqs):
        self._seqs = seqs

    def result(self):
        class _R:
            sequences = self._seqs
        return _R()


THINK, FORCE = 80, 81


class _FakeSampler:
    """Turn t of episode k emits [t, k, EOS] (k = the k-th sequence of the *first* call for that
    prompt). ``truncate_call`` makes that call return a length-truncated answer; ``think_call`` makes
    it return a length-truncated OPEN THINK block ([THINK, t, k], no close)."""

    def __init__(self, truncate_call: int | None = None, think_call: int | None = None):
        self.calls: list[tuple[list[int], int, int | None, int]] = []
        self.truncate_call = truncate_call
        self.think_call = think_call

    def sample(self, model_input, num_samples, params):
        ob = list(model_input.chunks[0].tokens)
        idx = len(self.calls)
        self.calls.append((ob, num_samples, params.seed, params.max_tokens))
        turn = ob.count(GEN)  # one generation prompt per turn so far
        seqs = []
        for k in range(num_samples):
            if self.truncate_call == idx:
                seqs.append(_Seq([turn, k], stop="length"))  # no EOS
            elif self.think_call == idx:
                seqs.append(_Seq([THINK, turn, k], stop="length"))  # open think, cut off
            else:
                seqs.append(_Seq([turn, k, EOS]))
        return _Fut(seqs)


class _FakeRenderer:
    stop_tokens = None
    eos_token_id = EOS

    def prompt_tokens(self, text):
        return [1, 2, GEN]

    def model_input(self, text):
        return tinker.ModelInput.from_ints(self.prompt_tokens(text))

    def parse(self, tokens):
        return "cot", f"cmd{tokens[:2]}", "raw"

    def in_open_think(self, tokens):
        return THINK in tokens and FORCE not in tokens

    def force_answer_tokens(self):
        return [FORCE]

    def continuation_tokens(self, observation, *, ended_cleanly=True):
        return ([EOS] if not ended_cleanly else []) + [USER_OPEN, len(observation), USER_CLOSE, GEN]


class _View:
    def __init__(self, meta):
        self.cot, self.output, self.meta = "COT", "OUT", meta


class _FakeEnv:
    """Ends after ``done_after`` turns (or immediately on truncation)."""

    multi_turn = True
    max_turns = 3

    def __init__(self, done_after=2):
        self.done_after = done_after

    def start(self, prompt):
        return {"turns": [], "prompt": prompt}

    def step(self, state, cot, text, *, truncated=False):
        state["turns"].append((cot, text, truncated))
        if truncated:
            return None, True
        if len(state["turns"]) >= self.done_after:
            return None, True
        return f"obs{len(state['turns'])}", False

    def finish(self, state):
        return _View({"n_turns": len(state["turns"]), "truncated": any(t[2] for t in state["turns"])})


def test_episodes_transitions_are_prefix_chained_and_grouped():
    sampler = _FakeSampler()
    prompts = [Prompt(text="p0"), Prompt(text="p1")]
    rolls = run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=3), prompts,
                         num_samples=2, max_tokens=8, seed=7)
    assert len(rolls) == 4 and [r.prompt.text for r in rolls] == ["p0", "p0", "p1", "p1"]
    for r in rolls:
        tr = r.meta["transitions"]
        assert len(tr) == 3 and r.meta["n_turns"] == 3
        assert tr[0]["ob"] == [1, 2, GEN]
        for a, b in zip(tr, tr[1:]):
            assert b["ob"][: len(a["ob"]) + len(a["ac"])] == a["ob"] + a["ac"]  # prefix property
            assert b["ob"][-1] == GEN  # ends in a generation prompt
        assert r.token_ids == [t for x in tr for t in x["ac"]]
        assert len(r.logprobs) == len(r.token_ids)
        assert r.cot == "COT" and r.output == "OUT" and r.meta["episode"]["n_turns"] == 3
    # turn 0: one call per prompt with num_samples; later turns: one call per active episode
    assert [c[1] for c in sampler.calls] == [2, 2] + [1] * 8
    seeds = [c[2] for c in sampler.calls]
    assert len(set(seeds)) == len(seeds) and seeds[0] == derive_sample_seed(7, 0)


def test_done_episodes_stop_sampling_and_truncation_closes_the_turn():
    # call 0 = prompt turn 0 (2 samples) → both continue; call 1 = ep0 turn 1 TRUNCATED → ends;
    # call 2 = ep1 turn 1 → env says done after 2 turns anyway.
    sampler = _FakeSampler(truncate_call=1)
    rolls = run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=3), [Prompt(text="p")],
                         num_samples=2, max_tokens=8, seed=1)
    ep0, ep1 = rolls
    assert ep0.meta["episode"]["truncated"] and ep0.meta["stop_reason"] == "length"
    assert len(ep0.meta["transitions"]) == 2 and len(ep1.meta["transitions"]) == 3
    assert len(sampler.calls) == 1 + 2 + 1  # ep0 was not resampled after truncation


def test_think_budget_forces_the_answer_as_a_masked_observation():
    # call 0: prompt turn 0 (1 sample) → open think cut at the budget → forced suffix appended as
    # observation, answer sampled with answer_tokens (call 1); turn 1 (call 2) completes normally.
    sampler = _FakeSampler(think_call=0)
    rolls = run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=2), [Prompt(text="p")],
                         num_samples=1, max_tokens=999, seed=3, think_budget=100, answer_tokens=20)
    r = rolls[0]
    tr = r.meta["transitions"]
    assert r.meta["n_forced_answers"] == 1 and r.meta["n_turns"] == 2 and len(tr) == 3
    assert [c[3] for c in sampler.calls] == [100, 20, 100]  # think budget, answer, think budget
    # the forced token sits in the OBSERVATION of the answer transition, never in an action
    assert tr[1]["ob"] == tr[0]["ob"] + tr[0]["ac"] + [FORCE]
    assert all(FORCE not in x["ac"] for x in tr)
    assert tr[2]["ob"][: len(tr[1]["ob"]) + len(tr[1]["ac"])] == tr[1]["ob"] + tr[1]["ac"]
    assert not r.meta["episode"]["truncated"]
    # → still one datum, with the forced token masked
    groups = to_trajectory_groups(_FakeRenderer(), rolls, [1.0], group_size=1)
    data, _ = assemble_training_data(groups, compute_advantages(groups))
    assert len(data) == 1
    full = tr[-1]["ob"] + tr[-1]["ac"]
    mask = data[0].loss_fn_inputs["mask"].to_torch().tolist()
    assert sum(mask) == sum(len(x["ac"]) for x in tr)
    assert mask[full.index(FORCE) - 1] == 0.0  # target position of FORCE is masked


def test_unseeded_run_passes_no_seed():
    sampler = _FakeSampler()
    run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=1), [Prompt(text="p")],
                 num_samples=1, max_tokens=8, seed=None)
    assert sampler.calls[0][2] is None


def test_multi_turn_rollouts_fold_into_one_masked_datum():
    rolls = run_episodes(_FakeSampler(), _FakeRenderer(), _FakeEnv(done_after=3),
                         [Prompt(text="p")], num_samples=2, max_tokens=8, seed=0)
    groups = to_trajectory_groups(_FakeRenderer(), rolls, [1.0, 0.0], group_size=2)
    assert len(groups) == 1 and len(groups[0].trajectories_G) == 2
    traj = groups[0].trajectories_G[0]
    assert len(traj.transitions) == 3 and traj.transitions[-1].episode_done
    assert [t.reward for t in traj.transitions] == [0.0, 0.0, 1.0]
    adv = compute_advantages(groups)
    data, _ = assemble_training_data(groups, adv)
    assert len(data) == 2  # ONE datum per episode (prefix-chained), not one per turn
    d = data[0]
    tr = rolls[0].meta["transitions"]
    full = tr[-1]["ob"] + tr[-1]["ac"]
    mask = d.loss_fn_inputs["mask"].to_torch().tolist()
    # tokens: the datum is full[:-1] as input; mask marks the ACTION targets (shifted by one)
    n_ac = sum(len(x["ac"]) for x in tr)
    assert sum(mask) == n_ac and len(mask) == len(full) - 1
    advs = d.loss_fn_inputs["advantages"].to_torch().tolist()
    assert all(a == 0.0 for a, m in zip(advs, mask) if m == 0.0)
    assert all(abs(a - 0.5) < 1e-6 for a, m in zip(advs, mask) if m == 1.0)  # centred: (1-0.5)


def test_hf_renderer_continuation_tokens_on_qwen3():
    """Real Qwen3 tokenizer (cached locally): the continuation is exactly the template's inter-turn
    framing, and ob+ac+continuation re-decodes to a well-formed multi-turn transcript."""
    import pytest

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except Exception as e:  # noqa: BLE001 — tokenizer not cached on this machine
        pytest.skip(f"Qwen3 tokenizer unavailable offline: {e}")
    from monitordecorrelation.rl.renderers import HFChatRenderer

    rend = HFChatRenderer(tok)
    ob = rend.prompt_tokens("Q")
    ac = tok.encode("<think>\nhmm\n</think>\n\nA1<|im_end|>", add_special_tokens=False)
    cont = rend.continuation_tokens("OBS")
    text = tok.decode(ob + ac + cont)
    assert text == ("<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\nhmm\n</think>\n\n"
                    "A1<|im_end|>\n<|im_start|>user\nOBS<|im_end|>\n<|im_start|>assistant\n")
    # a truncated turn gets closed first
    cut = tok.encode("<think>\nhm", add_special_tokens=False)
    text2 = tok.decode(ob + cut + rend.continuation_tokens("OBS", ended_cleanly=False))
    assert "<think>\nhm<|im_end|>\n<|im_start|>user\nOBS<|im_end|>\n<|im_start|>assistant\n" in text2
