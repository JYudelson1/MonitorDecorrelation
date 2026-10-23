"""Inkling's attention in query chunks (whitebox/inkling_attention.py) and features.extract_features' out-of-memory
ladder, on CPU with the tiny Inkling (tests/inkling_tiny.py), on both readers: the 4-bit one's loader
(``load_inkling_nvfp4``) and the full-precision one's (``load_inkling_text``, in float32 and in bfloat16).

The tiny model here scales its global layer's logits from position 256 on (``log_scaling_n_floor``) and has a
300-token sliding window, so both cross chunk borders.

Checks:
- The chunk plan: starts on multiples of ROW_TILE, at least MIN_ROWS rows a chunk, within the budget, one chunk when the
  rows fit.
- The chunked forward is transformers' forward bit for bit (``torch.equal``): every hidden state and the logits of a
  long input, each layer's attention in 3 chunks. This fails if transformers' attention changes under the mirror.
- Long synthetic episodes (TerminalEnv episodes padded to 1.6k-2.7k tokens): every feature is identical, bit for bit,
  whether the dense forward fits or runs out of memory and is retried in chunks.
- Out of memory: a transient failure of one input is retried once; an input that runs out of memory in chunks too
  raises ``UnscorableInput``, which names it.
"""

from __future__ import annotations

import pytest
import torch
from inkling_tiny import write_tiny_checkpoints
from test_extract_probe_features import RUN_ENV, TURNS

from monitordecorrelation.whitebox import features
from monitordecorrelation.whitebox.extraction import answer_ids
from monitordecorrelation.whitebox.features import (
    ANSWER_KEYS, FEATURE_KEYS, ROUTER_KEYS, UnscorableInput, extract_features, special_ids)
from monitordecorrelation.whitebox.inkling_attention import MIN_ROWS, ROW_TILE, QueryChunks, chunk_starts
from monitordecorrelation.whitebox.inkling_replay import InklingReplay
from monitordecorrelation.whitebox.probe_datasets import episode_item, item_messages

LONG = {"log_scaling_n_floor": 256, "sliding_window_size": 300, "rel_extent": 700}
"""The tiny model's text config for long inputs: τ > 1 from position 256, a window and an extent past a chunk."""

FILLER = "0 1 1 0 1 0 0 1 1 0\n"  # a grid row, as the env prints them (experiments/check_probe_long_episodes.py)
READERS = ("nvfp4", "bf16-float32", "bf16")
DENSE_LIMIT = 2 * MIN_ROWS
"""In the tests that pretend it, the dense bias of an input this long or longer doesn't fit (every such input can be
split in chunks of MIN_ROWS rows)."""


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> dict:
    """The tiny long-input Inkling on each reader's loader: NVFP4 (BF16 compute), BF16 loaded as the server loads it
    (float32 on a CPU), and BF16 kept in bfloat16 (the GPU numerics' dtype)."""
    from monitordecorrelation.whitebox.inkling_nvfp4 import load_inkling_nvfp4
    from monitordecorrelation.whitebox.model import load_inkling_text

    bf16_dir, nvfp4_dir, _, _ = write_tiny_checkpoints(tmp_path_factory.mktemp("long"), vocab_size=201024, **LONG)
    return {"nvfp4": load_inkling_nvfp4(str(nvfp4_dir), device="cpu").eval(),
            "bf16-float32": load_inkling_text(str(bf16_dir)).eval(),
            "bf16": load_inkling_text(str(bf16_dir), dtype=torch.bfloat16).eval()}


@pytest.fixture(scope="module")
def replay():
    return InklingReplay()


def padded_episode(rows: int) -> dict:
    """A replay item: TURNS with ``rows`` grid rows after the first command's output."""
    turns = [dict(t) for t in TURNS]
    turns[0]["output"] += "\n" + FILLER * rows
    return episode_item("Solve it.", {"turns": turns}, RUN_ENV)


@pytest.fixture(scope="module")
def inputs(replay) -> list:
    """Probe inputs of both views: TURNS as they are (short), and padded to about 1.6k and 2.7k tokens."""
    out = [replay.probe_input(item_messages(padded_episode(rows), view)) for rows in (0, 75, 130)
           for view in ("with_reasoning", "without_reasoning")]
    lengths = [len(p.tokens) for p in out]
    assert min(lengths) < DENSE_LIMIT <= sorted(lengths)[2] and max(lengths) < 3000, lengths
    return out


class AttentionCalls:
    """Records (query rows, keys) of every SDPA call, by replacing transformers' sdpa function for the test."""

    def __init__(self, monkeypatch):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.calls, sdpa = [], ALL_ATTENTION_FUNCTIONS["sdpa"]

        def spy(module, query, key, value, attention_mask, **kwargs):
            self.calls.append((module.layer_idx, query.shape[2], key.shape[2]))
            return sdpa(module, query, key, value, attention_mask, **kwargs)

        monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", spy)


def dense_bias_doesnt_fit(monkeypatch) -> list:
    """From now on transformers' dense bias for an input of DENSE_LIMIT tokens or more runs out of memory, as on a GPU
    without the room; the chunked forward doesn't build it. Returns the lengths refused."""
    from transformers.models.inkling.modeling_inkling import InklingRelativeLogits

    refused, real = [], InklingRelativeLogits.forward

    def forward(self, relative_states, query_positions, key_positions):
        if relative_states.shape[1] >= DENSE_LIMIT:
            refused.append(relative_states.shape[1])
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 49.00 GiB (simulated)")
        return real(self, relative_states, query_positions, key_positions)

    monkeypatch.setattr(InklingRelativeLogits, "forward", forward)
    return refused


def test_the_chunk_plan():
    """Tile-aligned starts and at least MIN_ROWS rows a chunk; within the budget unless that minimum exceeds it; a last
    chunk shorter than MIN_ROWS joins the one before; one chunk ([0]) when the rows fit, or are too few to split."""
    assert chunk_starts(10_000, 10_000, 10**8) == [0]  # 10k x 10k fits 1e8
    assert chunk_starts(1535, 1535, 1) == [0]  # too short for two chunks of MIN_ROWS
    assert chunk_starts(1536, 1536, 1) == [0, MIN_ROWS]
    for n, cells in ((16_384, 2**24), (16_385, 2**24), (40_000, 10**8), (5_003, 3 * 10**6), (2_600, 1)):
        starts = chunk_starts(n, n, cells)
        sizes = [e - s for s, e in zip(starts, [*starts[1:], n])]
        assert starts[0] == 0 and all(s % ROW_TILE == 0 for s in starts) and min(sizes) >= MIN_ROWS, (n, starts)
        assert all(size * n <= max(cells, MIN_ROWS * n) for size in sizes[:-1]), (n, sizes)
        assert sizes[-1] < sizes[0] + MIN_ROWS
    assert chunk_starts(16_384, 16_384, 2**24) == list(range(0, 16_384, 1024))  # 2^24 / 16384 = 1024 rows
    assert len(chunk_starts(2_600, 2_600, 1)) == 3
    assert chunk_starts(4_000, 1_000, 10**6, batch=2) == [0, 768, 1536, 2304, 3072]  # 2 rows x 1000 keys a query row


@pytest.mark.parametrize("reader", READERS)
def test_the_chunked_forward_is_the_dense_one_bit_for_bit(reader, models, inputs, monkeypatch):
    """A long input through the model with and without ``QueryChunks``: every hidden state and the logits equal, bit
    for bit, with each of the 4 layers' attention in 3 chunks. On exit the modules have transformers' forward again."""
    model = models[reader]
    ids = torch.tensor([inputs[-2].tokens])
    n = ids.shape[1]
    with torch.inference_mode():
        dense = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        spy = AttentionCalls(monkeypatch)
        with QueryChunks(model, max_cells=1) as chunks:  # 1: chunks of MIN_ROWS rows
            chunked = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    starts = chunk_starts(n, n, 1)
    assert len(starts) == 3 and chunks.skipped == 0 and len(chunks.modules) == 4
    assert spy.calls == [(layer, e - s, n) for layer in range(4) for s, e in zip(starts, [*starts[1:], n])]
    assert len(dense.hidden_states) == len(chunked.hidden_states) == 5
    for layer, (a, b) in enumerate(zip(dense.hidden_states, chunked.hidden_states)):
        assert torch.equal(a, b), f"{reader}: hidden state {layer} differs"
    assert torch.equal(dense.logits, chunked.logits)
    assert all("forward" not in vars(m) for m in model.modules() if type(m).__name__ == "InklingAttention")


@pytest.mark.parametrize("reader", READERS)
def test_long_episodes_have_the_same_features_whether_or_not_the_dense_bias_fits(reader, models, replay, inputs,
                                                                                 monkeypatch):
    """Every key extract_features returns (the readouts, the means and counts, the router at "Yes", the self-report
    log-probabilities, a long input's full hidden states), bit for bit: first everything fits; then the dense bias of
    the long inputs doesn't, so each runs out of memory once and is retried in chunks."""
    model, sp, answers = models[reader], special_ids(replay), answer_ids(replay)
    full = [len(inputs) - 1]
    want = extract_features(model, inputs, sp, max_batch_tokens=1, full=full, answers=answers)
    refused = dense_bias_doesnt_fit(monkeypatch)
    spy = AttentionCalls(monkeypatch)
    got = extract_features(model, inputs, sp, max_batch_tokens=1, full=full, answers=answers, chunk_attention=1)
    long = sorted((len(p.tokens) for p in inputs if len(p.tokens) >= DENSE_LIMIT), reverse=True)
    assert len(long) == 4 and refused == long  # each long input's dense forward failed once, longest first
    assert {q for _, q, k in spy.calls if q < k} == {MIN_ROWS, *(n - 2 * MIN_ROWS for n in long if n >= 3 * MIN_ROWS),
                                                     *(n - MIN_ROWS for n in long if n < 3 * MIN_ROWS)}
    for key in (*FEATURE_KEYS, *ROUTER_KEYS, *ANSWER_KEYS, "counts"):
        assert torch.equal(want[key], got[key]), f"{reader}: {key} differs"
    assert torch.equal(want["full"][full[0]], got["full"][full[0]])


def test_out_of_memory_alone_is_retried_once_then_unscorable(models, replay, inputs, monkeypatch):
    """A forward of one input that runs out of memory once (e.g. a fragmented cache) is retried and gives the same
    features. One that runs out of memory in chunks too raises UnscorableInput: its index and length, and why."""
    model, sp = models["nvfp4"], special_ids(replay)
    want = extract_features(model, inputs, sp, max_batch_tokens=1)
    real, failed = features._run_batch, []

    def run_batch(model_, inputs_, masks, device, batch, *rest):
        n = len(inputs_[batch[0]].tokens)
        if n == max(len(p.tokens) for p in inputs_) and (permanent or n not in failed):
            failed.append(n)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(model_, inputs_, masks, device, batch, *rest)

    monkeypatch.setattr(features, "_run_batch", run_batch)
    permanent = False
    got = extract_features(model, inputs, sp, max_batch_tokens=1)
    assert len(failed) == 1 and all(torch.equal(want[k], got[k]) for k in (*FEATURE_KEYS, *ROUTER_KEYS))
    permanent, failed[:] = True, []
    longest = max(range(len(inputs)), key=lambda i: len(inputs[i].tokens))
    with pytest.raises(UnscorableInput, match=r"ran out of GPU memory in a forward of its own, with its attention "
                                              r"in query chunks too: OutOfMemoryError: CUDA out of memory "
                                              r"\(simulated\)") as e:
        extract_features(model, inputs, sp, max_batch_tokens=1)
    assert (e.value.index, e.value.n_tokens) == (longest, len(inputs[longest].tokens)) and len(failed) == 2
