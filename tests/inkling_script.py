"""Test helpers for driving the multi-turn episode driver offline with Inkling's REAL renderer.

- ``inkling_renderer()`` — the policy renderer the RL loop builds for Inkling-Small
  (``rl.renderers.make_renderer``), whose cookbook ``chat_renderer`` renders and parses tool calls.
- ``assistant_tokens(renderer, message)`` — the tokens Inkling emits for an assistant message (thinking,
  text, tool calls, end of sampling), obtained the official way: tinker-cookbook's SFT rendering of the
  message, keeping the tokens it trains on (weight > 0). ``test_terminal_verifier_tools`` checks this
  reproduces real sampled tokens exactly.
- ``ScriptSampler`` — a stand-in for a tinker sampling client that returns scripted token sequences.
  Episodes run concurrently, so it identifies the episode from the (seeded) sampling position rather
  than from call order.
"""

from __future__ import annotations

import threading

from monitordecorrelation.rl.renderers import make_renderer


def inkling_renderer(model: str = "thinkingmachines/Inkling-Small", effort: float = 0.5):
    return make_renderer(model, effort=effort)


def assistant_tokens(renderer, message: dict) -> list[int]:
    chat = renderer.chat_renderer
    model_input, weights = chat.build_supervised_example(
        [{"role": "user", "content": "x"}, message], effort=renderer.effort
    )
    return [t for t, w in zip(model_input.to_ints(), weights.tolist()) if w > 0]


class _Seq:
    def __init__(self, tokens: list[int], stop_reason: str) -> None:
        self.tokens = list(tokens)
        self.logprobs = [-0.5] * len(tokens)
        self.stop_reason = stop_reason


class _Fut:
    def __init__(self, seq: _Seq) -> None:
        self._seq = seq

    def result(self):
        seq = self._seq

        class _R:
            sequences = [seq]

        return _R()


class ScriptSampler:
    """``scripts[k][turn]`` = the tokens episode ``k`` samples on that turn (a ``(tokens, stop_reason)``
    pair to simulate e.g. a max_tokens cut). Requires ``run_episodes(..., seed=…)``: the driver seeds
    every call by its (group, sample, turn) position, which is how a call is mapped back to its
    episode (``batch_seed`` = the seed given to ``run_episodes``). Records every prompt it was sent in ``prompts[(k, turn)]``."""

    def __init__(self, scripts: list[list], *, batch_seed: int, num_samples: int | None = None) -> None:
        self.scripts = scripts
        self.batch_seed = batch_seed
        self.num_samples = num_samples or len(scripts)
        self.prompts: dict[tuple[int, int], list[int]] = {}
        self._lock = threading.Lock()
        self._seeds: dict[int, tuple[int, int]] | None = None

    def _position(self, seed: int, batch_seed: int) -> tuple[int, int]:
        from monitordecorrelation.rl.episodes import derive_sample_seed

        with self._lock:
            if self._seeds is None:
                self._seeds = {}
                for k in range(len(self.scripts)):
                    group, i = divmod(k, self.num_samples)
                    for turn in range(max(len(s) for s in self.scripts)):
                        self._seeds[derive_sample_seed(batch_seed, group, i, turn)] = (k, turn)
        return self._seeds[seed]

    def sample(self, model_input, num_samples, params):
        assert num_samples == 1
        k, turn = self._position(params.seed, self.batch_seed)
        step = self.scripts[k][turn]
        tokens, stop = step if isinstance(step, tuple) else (step, "stop")
        with self._lock:
            self.prompts[(k, turn)] = model_input.to_ints()
        return _Fut(_Seq(tokens, stop))
