"""Inkling's attention in query chunks, for an input whose dense attention doesn't fit in GPU memory.

**What the bias is.** transformers' ``InklingAttention`` adds a relative position bias to every attention score
(``InklingRelativeLogits``):
- each query token's ``r_proj`` output mixes a learned bank of bias-vs-distance profiles into one value per backward
  distance 0 … ``rel_extent`` − 1 (the sliding window, in the sliding layers), and 0 beyond it and for future keys;
- in the global layers, the query and the bias are then both scaled by τ(position), the log-length factor
  (``log_scaling_n_floor``).

It is the model's own math, part of every score. Nothing of the replay enters it (thinking and text roles, the
follow-up turn): those only choose which positions features.py reads afterwards.

**Why it is the memory.** transformers materializes the bias densely, [B, heads, T, T], because SDPA takes it as
``attn_mask``. Per layer: the gather, its masked copy, the float32 copies of the log scaling, the combined mask, and
what the attention kernel needs besides. Together that is the ~340 B per (query, key) pair features.py budgets for
(``MAX_BATCH_ATTENTION``): ~49 GB for one 12k-token episode.

**The chunked forward** (``chunked_forward``) is ``InklingAttention.forward`` of transformers 5.19, with the
attention computed for query rows ``[s, e)`` at a time:
- the projections, convolutions, norms, the per-token ``rel_logits`` and τ run once over the whole sequence (O(T));
- per chunk: the bias rows of ``[s, e)`` (the same gather, fill and scaling, row for row) and transformers' own SDPA
  function on those query rows against every key, with the arguments the dense call gets (SDPA's implicit causal mask
  as those rows of the mask it builds, its GQA choice);
- every bias entry depends only on its (query, key) pair and attention treats query rows independently, so this is
  the dense computation row by row. A chunk's transients are bounded by ``max_cells`` query rows × keys; what still
  grows with T is linear, plus transformers' boolean sliding-window mask (1 byte per pair, built once per forward).

**Bit for bit?** Only the attention kernel could tell: it processes query rows in blocks whose size can depend on how
many rows a call has.
- PyTorch's CPU flash attention takes blocks of 256 rows in any call of at least 768. So chunks start on multiples of
  ``ROW_TILE`` (256) and hold at least ``MIN_ROWS`` (768) rows, and on CPU the chunked forward is bit-identical to the
  dense one (tests/test_inkling_attention.py, both readers).
- GPU attention kernels tile rows by 32 to 128, divisors of 256, but no GPU test shows bit-identity there (a kernel
  may also pick its tiles, or another algorithm, by the number of rows). That is why features.py uses this only for an
  input whose dense forward ran out of memory alone: everything that fits is computed exactly as before.

Only the ``sdpa`` implementation (transformers' default for Inkling, both readers' choice) is chunked, and only
without a cache; anything else keeps the dense forward. tests/test_inkling_attention.py fails if transformers'
attention changes under this mirror.
"""

from __future__ import annotations

import functools

import torch

ROW_TILE = 256
"""Chunks start on multiples of this many query rows: PyTorch's CPU flash attention's row block (in calls of at least
MIN_ROWS rows)."""

MIN_ROWS = 3 * ROW_TILE
"""A chunk holds at least this many query rows, so the CPU kernel takes ROW_TILE blocks in it as in the dense call."""

CHUNK_CELLS = 2**24
"""Default query rows × keys per chunk: ~16.8M pairs, ~5.7 GB of attention transients at ~340 B per pair."""


def chunk_starts(n_rows: int, n_keys: int, max_cells: int, batch: int = 1) -> list[int]:
    """Where each chunk of query rows starts, for ``batch`` sequences of ``n_rows`` queries over ``n_keys`` keys.

    Each chunk gets the most rows that keep ``batch`` × rows × ``n_keys`` within ``max_cells``, rounded down to a
    multiple of ``ROW_TILE`` and at least ``MIN_ROWS``. A last chunk shorter than ``MIN_ROWS`` joins the one before it.
    ``[0]`` (one chunk: the dense computation) if the rows fit in one."""
    rows = max(MIN_ROWS, max_cells // max(1, batch * n_keys) // ROW_TILE * ROW_TILE)
    if rows >= n_rows:
        return [0]
    starts = list(range(0, n_rows, rows))
    if n_rows - starts[-1] < MIN_ROWS:
        starts.pop()
    return starts


class QueryChunks:
    """Within ``with QueryChunks(model):`` every ``InklingAttention`` of ``model`` computes its attention in query
    chunks of at most ``max_cells`` (``chunked_forward``); on exit each gets transformers' forward back.

    The chunked forward is set as each module's own ``forward``. A module that already has one (e.g. an accelerate hook
    wrapping it) is left alone and keeps the dense forward: ``skipped`` counts those."""

    def __init__(self, model, max_cells: int = CHUNK_CELLS) -> None:
        attention = [m for m in model.modules() if type(m).__name__ == "InklingAttention"]
        self.modules = [m for m in attention if "forward" not in vars(m)]
        self.skipped = len(attention) - len(self.modules)
        self.max_cells = max_cells

    def __enter__(self) -> "QueryChunks":
        for m in self.modules:
            m.forward = functools.partial(chunked_forward, m, max_cells=self.max_cells)
        return self

    def __exit__(self, *exc) -> None:
        for m in self.modules:
            vars(m).pop("forward", None)


def chunked_forward(module, hidden_states: torch.Tensor, attention_mask, conv_mask=None, past_key_values=None, *,
                    max_cells: int = CHUNK_CELLS, **kwargs):
    """``InklingAttention.forward`` (transformers 5.19) with its attention in query chunks (``chunk_starts``).

    transformers' own forward runs instead whenever chunks don't apply: one chunk would do, a cache is given, the
    attention implementation isn't ``sdpa``, or attention weights are asked for."""
    from transformers.models.inkling import modeling_inkling as mi

    batch, n_rows = hidden_states.shape[:2]
    impl = module.config._attn_implementation
    starts = chunk_starts(n_rows, n_rows, max_cells, batch)
    if len(starts) == 1 or past_key_values is not None or impl != "sdpa" or kwargs.get("output_attentions"):
        return type(module).forward(module, hidden_states, attention_mask, conv_mask=conv_mask,
                                    past_key_values=past_key_values, **kwargs)
    # From here on transformers' InklingAttention.forward line by line, without the cache branch ...
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_proj(hidden_states)
    key_states = module.k_sconv(module.k_proj(hidden_states), past_key_values=None, conv_mask=conv_mask)
    value_states = module.v_sconv(module.v_proj(hidden_states), past_key_values=None, conv_mask=conv_mask)
    relative_states = module.r_proj(hidden_states)
    query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
    n_keys = key_states.shape[2]
    kv_positions = torch.arange(n_keys, device=hidden_states.device)
    q_positions = torch.arange(n_rows, device=hidden_states.device)
    relative_states = relative_states.view(*input_shape, module.num_heads, -1)
    rel = module.rel_logits_proj
    rel_logits = (relative_states @ rel.proj).transpose(1, 2)  # InklingRelativeLogits.forward's first line
    tau = None
    if not module.is_sliding and module.config.log_scaling_n_floor is not None:
        effective_n = (q_positions + 1).float()
        tau = 1.0 + module.config.log_scaling_alpha * torch.log(
            (effective_n / module.config.log_scaling_n_floor).clamp(min=1.0)
        )
        tau = tau.view(1, 1, -1, 1)
        query_states = (query_states.float() * tau).to(query_states.dtype)
    attention_interface = mi.ALL_ATTENTION_FUNCTIONS.get_interface(impl, mi.eager_attention_forward)
    outputs = []
    # ... except that the bias (InklingRelativeLogits.forward's other lines) and the attention go chunk by chunk
    for s, e in zip(starts, [*starts[1:], n_rows]):
        distance = (q_positions[s:e, None] - kv_positions[None, :])[None, None, :, :]
        gather_index = distance.clamp(0, rel.rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
        position_bias = rel_logits[:, :, s:e].gather(-1, gather_index)
        position_bias = position_bias.masked_fill((distance < 0) | (distance >= rel.rel_extent), 0.0)
        if tau is not None:
            position_bias = (position_bias.float() * tau[:, :, s:e]).to(position_bias.dtype)
        mask, causal = attention_mask, {}
        if mask is not None:
            mask = mask[:, :, s:e]
        else:  # SDPA's implicit causal mask: these rows of what its create_position_bias_mask builds for all of them
            allowed = (q_positions[s:e, None] >= kv_positions[None, :]).view(1, 1, e - s, n_keys)
            position_bias = torch.where(allowed, position_bias, torch.finfo(key_states.dtype).min)
            causal = {"is_causal": False}
        out, _ = attention_interface(
            module,
            query_states[:, :, s:e],
            key_states,
            value_states,
            mask,
            dropout=0.0 if not module.training else module.attention_dropout,
            scaling=module.scaling,
            sliding_window=module.sliding_window,
            position_bias=position_bias,
            **{**kwargs, **causal},
        )
        outputs.append(out)
        del position_bias, gather_index, distance
    attn_output = torch.cat(outputs, dim=1).reshape(*input_shape, -1).contiguous()
    return module.o_proj(attn_output), None
