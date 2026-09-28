"""The multi-turn episode driver + its GRPO datum path, offline: a fake sampler, a fake renderer and
a fake tool env. Checks the token bookkeeping the training step depends on — every observation is a
prefix-extension of the previous ob+ac, so tinker-cookbook folds an episode into ONE datum with
observation tokens masked — plus seeding, group order, truncation handling, and the fact that
episodes really do run independently (no per-turn barrier across the batch).

Episodes run concurrently, so the fakes are addressed by (turn, episode) rather than by call index:
which call happens Nth is not deterministic any more, but WHAT each episode is handed is.
"""

from __future__ import annotations

import threading
import time

import pytest
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
    """A stub sampling future. ``delay`` is paid in ``result()``, not at submission — that is where a
    real (tinker) future's latency lives, so a slow generation blocks only whoever waits on it."""

    def __init__(self, seqs, delay: float = 0.0):
        self._seqs = seqs
        self._delay = delay

    def result(self):
        if self._delay:
            time.sleep(self._delay)

        class _R:
            sequences = self._seqs
        return _R()


THINK, FORCE = 80, 81


class _FakeSampler:
    """Turn t of episode k emits ``[t, k, EOS]``.

    ``truncate`` / ``think`` are sets of ``(turn, k)``: that turn returns a length-truncated answer,
    or a length-truncated OPEN THINK block (``[t, k, THINK]``, no close) that the budget must force.
    Keyed by (turn, episode) rather than call index because episodes run concurrently — the order
    calls are issued in is not deterministic. ``think`` applies to a turn's FIRST segment only, so
    the forced-answer continuation that follows it is an ordinary completion.

    ``turn0_delay`` maps a prompt's index -> seconds its turn-0 generation takes, for the decoupling
    test (turn-0 calls are issued from the caller's thread, in prompt order, before any episode runs).
    """

    def __init__(self, truncate=(), think=(), turn0_delay=None, group=1):
        self.calls: list[tuple[list[int], int, int | None, int]] = []
        self.truncate = set(truncate)
        self.think = set(think)
        self.turn0_delay = dict(turn0_delay or {})
        self.group = group  # GRPO group size: a SEEDED turn 0 is one single-sample request per episode
        self._n_turn0_seqs = 0  # turn-0 sequences issued so far (requests come in prompt/sample order)
        self._lock = threading.Lock()

    def sample(self, model_input, num_samples, params):
        ob = list(model_input.chunks[0].tokens)
        # A turn-0 call is the one still sitting on the bare prompt tokens; it fans out into the
        # whole group, so k is the sequence index. Every later call is one episode's own, and that
        # episode's k is the second token of its first action segment.
        first_turn = len(ob) == 3
        with self._lock:  # called from every episode's thread
            self.calls.append((ob, num_samples, params.seed, params.max_tokens))
            delay = 0.0
            if first_turn:
                base = self._n_turn0_seqs  # global turn-0 sequence index → (prompt, k) via the group size
                self._n_turn0_seqs += num_samples
                delay = self.turn0_delay.get(base // self.group, 0.0)
        turn = ob.count(GEN)  # one generation prompt per turn so far
        seqs = []
        for i in range(num_samples):
            k = (base + i) % self.group if first_turn else ob[4]
            if (turn, k) in self.truncate:
                seqs.append(_Seq([turn, k], stop="length"))  # no EOS
            elif (turn, k) in self.think and FORCE not in ob:
                seqs.append(_Seq([turn, k, THINK], stop="length"))  # open think, cut off
            else:
                seqs.append(_Seq([turn, k, EOS]))
        return _Fut(seqs, delay=delay)


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
    sampler = _FakeSampler(group=2)
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
    # SEEDED: turn 0 is one single-sample request per (prompt, sample) — a seeded n-sample request
    # collapses the group — then one call per active episode per later turn. All seeds distinct.
    assert [c[1] for c in sampler.calls] == [1] * 4 + [1] * 8
    seeds = [c[2] for c in sampler.calls]
    assert None not in seeds and len(set(seeds)) == len(seeds)
    # every seed is a function of its POSITION (group, sample, call; turn 0 = call 0) — turn 0 is issued
    # in prompt order, the later calls in whatever order the episode threads get there
    assert seeds[:4] == [derive_sample_seed(7, g, k, 0) for g in range(2) for k in range(2)]
    assert set(seeds) == {derive_sample_seed(7, g, k, c) for g in range(2) for k in range(2) for c in range(3)}


def test_done_episodes_stop_sampling_and_truncation_closes_the_turn():
    # one turn-0 call (2 samples) → both episodes continue; ep0's turn-1 answer is TRUNCATED → it
    # ends there; ep1 runs turns 1 and 2 → 4 sampling calls in all, none of them a resample of ep0.
    sampler = _FakeSampler(truncate={(2, 0)}, group=2)
    rolls = run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=3), [Prompt(text="p")],
                         num_samples=2, max_tokens=8, seed=1)
    ep0, ep1 = rolls
    assert ep0.meta["episode"]["truncated"] and ep0.meta["stop_reason"] == "length"
    assert len(ep0.meta["transitions"]) == 2 and len(ep1.meta["transitions"]) == 3
    assert len(sampler.calls) == 2 + 2 + 1  # seeded turn 0 = 2 single-sample requests; ep0 not resampled


def test_think_budget_forces_the_answer_as_a_masked_observation():
    # call 0: prompt turn 0 (1 sample) → open think cut at the budget → forced suffix appended as
    # observation, answer sampled with answer_tokens (call 1); turn 1 (call 2) completes normally.
    sampler = _FakeSampler(think={(1, 0)})
    rolls = run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=2), [Prompt(text="p")],
                         num_samples=1, seed=3, think_budget=100, answer_tokens=20)
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


def test_episode_token_accounting_matches_the_transitions():
    """The per-episode token counts the cost accounting reads must equal the transitions: one
    sampling call per transition, and the LAST ob+ac is the single training datum."""
    sampler = _FakeSampler(think={(1, 0)})
    rolls = run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=2), [Prompt(text="p")],
                         num_samples=1, seed=3, think_budget=100, answer_tokens=20)
    r = rolls[0]
    tr = r.meta["transitions"]
    assert r.meta["n_sampling_calls"] == len(tr) == 3          # 2 turns, one of them budget-forced
    assert r.meta["input_tokens"] == sum(len(x["ob"]) for x in tr)
    assert r.meta["output_tokens"] == sum(len(x["ac"]) for x in tr) == len(r.token_ids)
    assert r.meta["train_tokens"] == len(tr[-1]["ob"]) + len(tr[-1]["ac"])
    assert r.meta["n_truncated_turns"] == 0                    # forced thinking is not a truncation


def test_truncated_answer_turns_are_counted():
    sampler = _FakeSampler(truncate={(2, 0)}, group=2)
    rolls = run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=3), [Prompt(text="p")],
                         num_samples=2, max_tokens=8, seed=1)
    assert rolls[0].meta["n_truncated_turns"] == 1 and rolls[1].meta["n_truncated_turns"] == 0


def test_unseeded_run_passes_no_seed():
    sampler = _FakeSampler()
    run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=1), [Prompt(text="p")],
                 num_samples=1, max_tokens=8, seed=None)
    assert sampler.calls[0][2] is None


@pytest.mark.parametrize("seed", [3, None])
def test_every_request_is_a_single_sample(seed):
    """Never one n-sample request per group, seeded or not: tinker seeds a whole request, so a seeded
    n-sample request collapses the group to ~1 distinct sequence."""
    sampler = _FakeSampler(group=4)
    rolls = run_episodes(sampler, _FakeRenderer(), _FakeEnv(done_after=2),
                         [Prompt(text="p0"), Prompt(text="p1")], num_samples=4, max_tokens=8, seed=seed)
    assert len(rolls) == 8 and [r.prompt.text for r in rolls] == ["p0"] * 4 + ["p1"] * 4
    assert [c[1] for c in sampler.calls] == [1] * 16  # 8 turn-0 requests + 8 second turns
    seeds = [c[2] for c in sampler.calls]
    if seed is None:
        assert set(seeds) == {None}
    else:
        assert None not in seeds and len(set(seeds)) == 16


def test_an_episode_does_not_wait_for_its_siblings_turn_0():
    """Each episode waits only for its OWN turn-0 request: a slow sibling in the same GRPO group
    (prompt 0, sample 1) must not hold back sample 0."""
    SLOW = 1.0
    slow_seed = derive_sample_seed(0, 0, 1, 0)  # (group 0, sample 1, call 0)

    class _SlowSibling(_FakeSampler):
        def sample(self, model_input, num_samples, params):
            fut = super().sample(model_input, num_samples, params)
            if params.seed == slow_seed:
                fut._delay = SLOW
            return fut

    done: dict[int, float] = {}
    t0 = time.perf_counter()
    run_episodes(_SlowSibling(group=2), _FakeRenderer(), _FakeEnv(done_after=2), [Prompt(text="p0")],
                 num_samples=2, max_tokens=8, seed=0,
                 on_rollout=lambda i, r: done.__setitem__(i, time.perf_counter() - t0))
    assert done[0] < SLOW / 2 <= SLOW <= done[1]


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


def test_episodes_do_not_wait_for_each_other():
    """The point of the driver: no per-turn barrier. Episode 0 stalls for SLOW seconds on its first
    generation; every other episode must still get all the way through max_turns while it waits —
    under the old lockstep driver none of them could start turn 1 until episode 0 returned."""
    SLOW, N = 1.0, 6
    order: list[tuple[str, float]] = []
    lock = threading.Lock()

    class _TimedEnv(_FakeEnv):
        max_turns = 3

        def finish(self, state):
            with lock:
                order.append((state["prompt"].text, time.perf_counter()))
            return super().finish(state)

    sampler = _FakeSampler(turn0_delay={0: SLOW})
    prompts = [Prompt(text=f"p{i}") for i in range(N)]
    t0 = time.perf_counter()
    rolls = run_episodes(sampler, _FakeRenderer(), _TimedEnv(done_after=3), prompts,
                         num_samples=1, max_tokens=8, seed=0)
    elapsed = time.perf_counter() - t0

    assert len(rolls) == N and [r.prompt.text for r in rolls] == [p.text for p in prompts]
    assert all(r.meta["n_turns"] == 3 for r in rolls)  # everyone ran the full episode
    # p0 is the straggler (it slept); everyone else finished long before it did.
    finished = dict(order)
    assert order[-1][0] == "p0"
    assert all(finished[f"p{i}"] - t0 < SLOW for i in range(1, N))
    # and the batch costs ~one stall, not one per turn per episode
    assert elapsed < SLOW * 2


def test_env_steps_are_not_capped():
    """Every episode whose turn is ready runs its env step at once — the driver puts no limit on
    concurrent commands: 40 episodes' first steps each wait on a barrier that only opens once all 40
    are inside ``step`` together."""
    n = 40
    barrier = threading.Barrier(n, timeout=10)

    class _RendezvousEnv(_FakeEnv):
        def step(self, state, cot, text, *, truncated=False):
            barrier.wait()  # BrokenBarrierError unless all n steps run concurrently
            return super().step(state, cot, text, truncated=truncated)

    prompts = [Prompt(text=f"p{i}") for i in range(n)]
    rolls = run_episodes(_FakeSampler(), _FakeRenderer(), _RendezvousEnv(done_after=1), prompts,
                         num_samples=1, max_tokens=8, seed=0)
    assert len(rolls) == n and all(r.meta["n_turns"] == 1 for r in rolls)
