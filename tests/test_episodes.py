"""The multi-turn episode driver + its GRPO datum path, offline: Inkling's real renderer (the cookbook's
``tml_v0``), a fake sampler and a fake tool env. Checks the token bookkeeping the training step depends
on — every observation is the whole conversation re-rendered, which extends the previous ob+ac, so
tinker-cookbook folds an episode into ONE datum with observation tokens masked (and splits it where it
does not) — plus seeding, group order, truncation handling, and the fact that episodes really do run
independently (no per-turn barrier across the batch).

Episodes run concurrently, so the fakes are addressed by (turn, episode) rather than by call index:
which call happens Nth is not deterministic, but WHAT each episode is handed is. The sampler's turns
are tool calls whose command names them (``t<turn>e<episode>``), so the conversation a call is sent
says whose it is.
"""

from __future__ import annotations

import functools
import json
import re
import threading
import time

import pytest
from inkling_script import assistant_tokens, inkling_renderer
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages

from monitordecorrelation.rl.episodes import derive_sample_seed, run_episodes
from monitordecorrelation.rl.grpo import to_trajectory_groups
from monitordecorrelation.types import Prompt

R = inkling_renderer()


@functools.cache
def _turn_tokens(turn: int, e: int) -> tuple[int, ...]:
    msg = {"role": "assistant", "content": [{"type": "thinking", "thinking": f"thinking {turn}"}],
           "tool_calls": [ToolCall(function=ToolCall.FunctionBody(
               name="bash", arguments=json.dumps({"command": f"t{turn}e{e}"})))]}
    return tuple(assistant_tokens(R, msg))


class _Seq:
    def __init__(self, tokens, stop="stop"):
        self.tokens = list(tokens)
        self.logprobs = [-0.5] * len(self.tokens)
        self.stop_reason = stop


class _Fut:
    """A stub sampling future. ``delay`` is paid in ``result()``, not at submission — that is where a
    real (tinker) future's latency lives, so a slow generation blocks only whoever waits on it."""

    def __init__(self, seq, delay: float = 0.0):
        self._seq = seq
        self._delay = delay

    def result(self):
        if self._delay:
            time.sleep(self._delay)
        seq = self._seq

        class _R:
            sequences = [seq]
        return _R()


class _FakeSampler:
    """Turn t of episode e samples a tool call running ``t<t>e<e>``.

    ``truncate`` is a set of ``(turn, e)``: that turn is cut off by max_tokens. ``respell`` is a set of
    ``(turn, e)`` whose tool-call JSON is spaced differently from the renderer's canonical form (parses
    the same, re-renders differently: a prefix break). ``turn0_delay`` maps a prompt's index ->
    seconds its turn-0 generation takes (turn-0 calls are issued from the caller's thread, in
    prompt/sample order, before any episode runs — which is how a turn-0 call's episode is known)."""

    def __init__(self, truncate=(), respell=(), turn0_delay=None, group=1):
        self.calls: list[tuple[list[int], int, int | None, int, list]] = []
        self.truncate, self.respell = set(truncate), set(respell)
        self.turn0_delay = dict(turn0_delay or {})
        self.group = group
        self._n_turn0 = 0
        self._lock = threading.Lock()

    def sample(self, model_input, num_samples, params):
        assert num_samples == 1  # every request is a single sample (see test below)
        ob = model_input.to_ints()
        done = re.findall(r'"command":"t(\d+)e(\d+)"', R.tokenizer.decode(ob))  # this episode's earlier turns
        with self._lock:  # called from every episode's thread
            self.calls.append((ob, num_samples, params.seed, params.max_tokens, params.stop))
            delay = 0.0
            if not done:
                e, self._n_turn0 = self._n_turn0, self._n_turn0 + 1
                delay = self.turn0_delay.get(e // self.group, 0.0)
        turn = len(done)
        if done:
            e = int(done[0][1])
        tokens = list(_turn_tokens(turn, e))
        if (turn, e) in self.respell:
            text = R.tokenizer.decode(tokens).replace('"args":{', '"args": {')
            assert text != R.tokenizer.decode(tokens)
            tokens = _encode_with_specials(text)
        if (turn, e) in self.truncate:
            return _Fut(_Seq(tokens[:5], stop="length"), delay)
        return _Fut(_Seq(tokens), delay)


def _encode_with_specials(text: str) -> list[int]:
    out = []
    for part in re.split(r"(<\|[a-z_]+\|>)", text):
        if re.fullmatch(r"<\|[a-z_]+\|>", part):
            out.append(R.tokenizer.encode_special(part[2:-2]))
        elif part:
            out.extend(R.tokenizer.encode_ordinary(part))
    return out


class _View:
    def __init__(self, meta):
        self.cot, self.output, self.meta = "COT", "OUT", meta


class _FakeEnv:
    """One tool, ``bash``; answers each call with a tool message; ends after ``done_after`` turns (or
    at once on a truncated / unparsable turn)."""

    multi_turn = True
    max_turns = 3

    def __init__(self, done_after=2):
        self.done_after = done_after

    def check_policy(self, model_name):
        pass

    def tool_specs(self):
        return [{"name": "bash", "description": "Run a command.",
                 "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                "required": ["command"]}}]

    def start(self, prompt):
        return {"turns": [], "prompt": prompt}

    def step(self, state, message, *, truncated=False, parse_error=False):
        state["turns"].append((message, truncated))
        if truncated or parse_error or len(state["turns"]) >= self.done_after:
            return [], True
        call = message["tool_calls"][0]
        return [{"role": "tool", "tool_call_id": "", "name": "bash",
                 "content": f"ran {json.loads(call.function.arguments)['command']}"}], False

    def finish(self, state):
        return _View({"n_turns": len(state["turns"]), "truncated": any(t[1] for t in state["turns"])})


def test_episodes_transitions_are_prefix_chained_and_grouped():
    sampler = _FakeSampler(group=2)
    prompts = [Prompt(text="p0"), Prompt(text="p1")]
    rolls = run_episodes(sampler, R, _FakeEnv(done_after=3), prompts, num_samples=2, max_tokens=64, seed=7)
    assert len(rolls) == 4 and [r.prompt.text for r in rolls] == ["p0", "p0", "p1", "p1"]
    chat = R.chat_renderer
    for i, r in enumerate(rolls):
        tr = r.meta["transitions"]
        assert len(tr) == 3 and r.meta["n_turns"] == 3 and r.meta["n_prefix_breaks"] == 0
        # turn 0 is the documented recipe: tool declarations + the task, at the policy's effort
        assert tr[0]["ob"] == chat.build_generation_prompt(
            chat.create_conversation_prefix_with_tools(_FakeEnv().tool_specs(), system_prompt="")
            + [{"role": "user", "content": r.prompt.text}], effort=R.effort).to_ints()
        for a, b in zip(tr, tr[1:]):
            assert b["ob"][: len(a["ob"]) + len(a["ac"])] == a["ob"] + a["ac"]  # prefix property
        assert [x["ac"] for x in tr] == [list(_turn_tokens(t, i)) for t in range(3)]  # each its own
        assert f"<|message_tool|>bash<|content_text|>ran t1e{i}<|end_message|>" in R.tokenizer.decode(tr[2]["ob"])
        assert r.token_ids == [t for x in tr for t in x["ac"]]
        assert len(r.logprobs) == len(r.token_ids)
        assert r.cot == "COT" and r.output == "OUT" and r.meta["episode"]["n_turns"] == 3
    # every call: one sample, max_tokens, the renderer's stop tokens
    assert all(c[1] == 1 and c[3] == 64 and c[4] == R.chat_renderer.get_stop_sequences() for c in sampler.calls)
    assert len(sampler.calls) == 4 * 3
    seeds = [c[2] for c in sampler.calls]
    assert None not in seeds and len(set(seeds)) == len(seeds)
    # every seed is a function of its POSITION (group, sample, turn) — turn 0 is issued in prompt order,
    # the later calls in whatever order the episode threads get there
    assert seeds[:4] == [derive_sample_seed(7, g, k, 0) for g in range(2) for k in range(2)]
    assert set(seeds) == {derive_sample_seed(7, g, k, t) for g in range(2) for k in range(2) for t in range(3)}


def test_done_episodes_stop_sampling_and_truncation_ends_the_episode():
    # ep0's turn-1 answer is TRUNCATED → it ends there; ep1 runs turns 1 and 2 → 5 sampling calls in all
    sampler = _FakeSampler(truncate={(1, 0)}, group=2)
    rolls = run_episodes(sampler, R, _FakeEnv(done_after=3), [Prompt(text="p")],
                         num_samples=2, max_tokens=64, seed=1)
    ep0, ep1 = rolls
    assert ep0.meta["episode"]["truncated"] and ep0.meta["stop_reason"] == "length"
    assert len(ep0.meta["transitions"]) == 2 and len(ep1.meta["transitions"]) == 3
    assert ep0.meta["n_truncated_turns"] == 1 and ep1.meta["n_truncated_turns"] == 0
    assert len(sampler.calls) == 2 + 2 + 1  # ep0 is never resampled


def test_episode_token_accounting_matches_the_transitions():
    """The per-episode token counts the cost accounting reads must equal the transitions: one
    sampling call per turn, and (no prefix break) the LAST ob+ac is the single training datum."""
    rolls = run_episodes(_FakeSampler(), R, _FakeEnv(done_after=2), [Prompt(text="p")],
                         num_samples=1, max_tokens=64, seed=3)
    r = rolls[0]
    tr = r.meta["transitions"]
    assert r.meta["n_sampling_calls"] == len(tr) == 2
    assert r.meta["input_tokens"] == sum(len(x["ob"]) for x in tr)
    assert r.meta["output_tokens"] == sum(len(x["ac"]) for x in tr) == len(r.token_ids)
    assert r.meta["train_tokens"] == len(tr[-1]["ob"]) + len(tr[-1]["ac"])
    assert r.meta["n_truncated_turns"] == 0 and r.meta["n_prefix_breaks"] == 0


def test_a_turn_that_does_not_rerender_verbatim_costs_a_second_datum():
    """A tool call spelled differently from the renderer's canonical JSON parses fine but re-renders
    differently, so the next ob does not extend ob+ac: counted, and the cookbook starts a new datum."""
    rolls = run_episodes(_FakeSampler(respell={(0, 0)}), R, _FakeEnv(done_after=3), [Prompt(text="p")],
                         num_samples=1, max_tokens=64, seed=3)
    r = rolls[0]
    tr = r.meta["transitions"]
    assert r.meta["n_prefix_breaks"] == 1 and len(tr) == 3
    assert tr[1]["ob"][: len(tr[0]["ob"]) + len(tr[0]["ac"])] != tr[0]["ob"] + tr[0]["ac"]
    assert tr[2]["ob"][: len(tr[1]["ob"]) + len(tr[1]["ac"])] == tr[1]["ob"] + tr[1]["ac"]
    groups = to_trajectory_groups(R, rolls, [1.0], group_size=1)
    data, _ = assemble_training_data(groups, compute_advantages(groups))
    assert len(data) == 2
    assert r.meta["train_tokens"] == (len(tr[0]["ob"]) + len(tr[0]["ac"])) + (len(tr[2]["ob"]) + len(tr[2]["ac"]))
    assert r.meta["train_tokens"] == sum(d.model_input.length + 1 for d in data)
    # ... and every sampled token is still trained on, exactly once
    assert sum(sum(d.loss_fn_inputs["mask"].to_torch().tolist()) for d in data) == len(r.token_ids)


def test_unseeded_run_passes_no_seed():
    sampler = _FakeSampler()
    run_episodes(sampler, R, _FakeEnv(done_after=1), [Prompt(text="p")], num_samples=1, max_tokens=64, seed=None)
    assert sampler.calls[0][2] is None


@pytest.mark.parametrize("seed", [3, None])
def test_every_request_is_a_single_sample(seed):
    """Never one n-sample request per group, seeded or not: tinker seeds a whole request, so a seeded
    n-sample request collapses the group to ~1 distinct sequence."""
    sampler = _FakeSampler(group=4)
    rolls = run_episodes(sampler, R, _FakeEnv(done_after=2), [Prompt(text="p0"), Prompt(text="p1")],
                         num_samples=4, max_tokens=64, seed=seed)
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
    slow_seed = derive_sample_seed(0, 0, 1, 0)  # (group 0, sample 1, turn 0)

    class _SlowSibling(_FakeSampler):
        def sample(self, model_input, num_samples, params):
            fut = super().sample(model_input, num_samples, params)
            if params.seed == slow_seed:
                fut._delay = SLOW
            return fut

    done: dict[int, float] = {}
    t0 = time.perf_counter()
    run_episodes(_SlowSibling(group=2), R, _FakeEnv(done_after=2), [Prompt(text="p0")],
                 num_samples=2, max_tokens=64, seed=0,
                 on_rollout=lambda i, r: done.__setitem__(i, time.perf_counter() - t0))
    assert done[0] < SLOW / 2 <= SLOW <= done[1]


def test_multi_turn_rollouts_fold_into_one_masked_datum():
    rolls = run_episodes(_FakeSampler(), R, _FakeEnv(done_after=3), [Prompt(text="p")],
                         num_samples=2, max_tokens=64, seed=0)
    groups = to_trajectory_groups(R, rolls, [1.0, 0.0], group_size=2)
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


def test_episodes_do_not_wait_for_each_other():
    """The point of the driver: no per-turn barrier. Episode 0 stalls for SLOW seconds on its first
    generation; every other episode must still get all the way through max_turns while it waits."""
    SLOW, N = 1.0, 6
    order: list[tuple[str, float]] = []
    lock = threading.Lock()

    class _TimedEnv(_FakeEnv):
        def finish(self, state):
            with lock:
                order.append((state["prompt"].text, time.perf_counter()))
            return super().finish(state)

    sampler = _FakeSampler(turn0_delay={0: SLOW})
    prompts = [Prompt(text=f"p{i}") for i in range(N)]
    t0 = time.perf_counter()
    rolls = run_episodes(sampler, R, _TimedEnv(done_after=3), prompts, num_samples=1, max_tokens=64, seed=0)
    elapsed = time.perf_counter() - t0

    assert len(rolls) == N and [r.prompt.text for r in rolls] == [p.text for p in prompts]
    assert all(r.meta["n_turns"] == 3 for r in rolls)  # everyone ran the full episode
    finished = dict(order)
    assert order[-1][0] == "p0"  # p0 is the straggler (it slept); everyone else finished long before
    assert all(finished[f"p{i}"] - t0 < SLOW for i in range(1, N))
    assert elapsed < SLOW * 2  # the batch costs ~one stall, not one per turn per episode


def test_env_steps_are_not_capped():
    """Every episode whose turn is ready runs its env step at once — the driver puts no limit on
    concurrent commands: 40 episodes' first steps each wait on a barrier that only opens once all 40
    are inside ``step`` together."""
    n = 40
    barrier = threading.Barrier(n, timeout=10)

    class _RendezvousEnv(_FakeEnv):
        def step(self, state, message, **kw):
            barrier.wait()  # BrokenBarrierError unless all n steps run concurrently
            return super().step(state, message, **kw)

    prompts = [Prompt(text=f"p{i}") for i in range(n)]
    rolls = run_episodes(_FakeSampler(), R, _RendezvousEnv(done_after=1), prompts,
                         num_samples=1, max_tokens=64, seed=0)
    assert len(rolls) == n and all(r.meta["n_turns"] == 1 for r in rolls)
