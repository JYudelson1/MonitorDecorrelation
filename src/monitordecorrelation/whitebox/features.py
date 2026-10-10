"""Token-level probe features: one forward pass per probe input (built by whitebox/inkling_replay.py), read at
explicit positions. This replaces ``WhiteBoxModel.extract_activations``, which reads the last token, for
Inkling.

Per input, at every layer (the embeddings plus each decoder layer):

- ``followup``: the residual stream at the follow-up answer token. This is the Atlas follow-up probe's read.
- ``followup_prev``: the same at the token just before it. It is the fallback readout if the answer token turns out
  to behave like an attention sink, recorded always so that check never costs a second GPU session.
- ``router_logits`` / ``router_topk``: at the follow-up answer token, each MoE layer's router logits for the routed
  experts (before the sigmoid) and the experts it chose (after the bias correction). Captured by ``RouterTap``.
- ``answer_logp``: the reader's own log-probabilities of the follow-up answers "Yes" and "No" at the answer position.
  This is the self-report baseline, computed with the model's LM head from the final hidden state.
- ``mean_reasoning`` / ``mean_action``: the mean over the policy's thinking tokens and over its visible text
  tokens, with ``counts`` of each. Only content tokens count: no role or channel markers, and nothing from
  the follow-up turn. These are the mean-pooled probe's inputs. They are kept apart so both a "thinking +
  output" pool and an "output only" pool can be formed, weighted by the counts.

Batches are right-padded and sorted by length. Right padding cannot change a real token's activations:
attention, the short convolutions and the MoE routing are all causal or per-token.
tests/test_features.py checks batched == one-by-one.

Dtypes in the safetensors shards:
- **Readouts:** stored as bfloat16, which is exactly what the model computed. Inkling's mid-layer activations reach
  ~5e4 at the readout, too close to float16's 65504.
- **Means:** computed and stored in float32. In bfloat16, values near 5e4 can only be represented in steps of 256,
  which would erase small differences between episodes in Inkling's few huge dimensions.

``FEATURE_FORMAT`` names this layout; extraction refuses to mix shards of different formats.

With ``device_map`` the layers can sit on several GPUs, so per-layer reads happen on each layer's own device.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Sequence

import torch

from monitordecorrelation.whitebox.inkling_replay import ProbeInput

FEATURE_KEYS = ("followup", "followup_prev", "mean_reasoning", "mean_action")
READOUT_KEYS = ("followup", "followup_prev")  # bf16
MEAN_KEYS = ("mean_reasoning", "mean_action")  # fp32
ROUTER_KEYS = ("router_logits", "router_topk", "router_margin")  # fp32 [N, M, E], int16 [N, M, k], fp32 [N, M]
ANSWER_KEYS = ("answer_logp",)  # fp32 [N, 2]: log p("Yes"), log p("No") at the answer position
FEATURE_FORMAT = "inkling-probe-features/v2"  # v1: bf16 means, no followup_prev, no router


class RouterTap:
    """Forward hooks on every MoE router (transformers' ``InklingTopkRouter``, which the NVFP4 loader keeps).

    - **What a router returns:** ``(routed_logits, topk_weights, topk_indices, shared_gammas)``, flattened over
      batch × time.
    - **What the tap keeps:** the logits and the chosen experts at the flat positions ``row * width + t`` given to
      ``start``, or at every position with ``every=True``.
    - **Order:** hooks fire in layer order, so the stacked results are [P, M, ·] in MoE-layer order.
    """

    def __init__(self, model) -> None:
        self.routers = [m for m in model.modules() if type(m).__name__ == "InklingTopkRouter"]
        self.handles = [m.register_forward_hook(self._hook) for m in self.routers]
        self.positions = None
        self.every = False
        self.logits: list = []
        self.topk: list = []
        self.margin: list = []

    def _hook(self, module, args, output) -> None:
        if self.positions is None and not self.every:
            return
        logits, _, chosen, _ = output
        if self.every:
            self.topk.append(chosen.to(torch.int16).cpu())
            return
        pos = self.positions.to(logits.device)
        self.logits.append(logits[pos].float().cpu())
        self.topk.append(chosen[pos].to(torch.int16).cpu())
        # how close the choice was: selection score of the k-th chosen expert minus the best one left out
        bias = getattr(module, "e_score_correction_bias", None)
        choice = logits[pos].float().sigmoid() + (bias.float() if bias is not None else 0.0)
        top = choice.topk(chosen.shape[-1] + 1, dim=-1).values
        self.margin.append((top[:, -2] - top[:, -1]).cpu())

    def start(self, positions=None, *, every: bool = False) -> None:
        self.positions, self.every, self.logits, self.topk, self.margin = positions, every, [], [], []

    def stop(self) -> tuple:
        """(logits [P, M, E] or None, topk [P, M, k]); with ``every``, topk is [B*T, M, k]. ``self.last_margin``
        then holds [P, M]: the k-th chosen expert's selection score minus the best unchosen one's."""
        logits = torch.stack(self.logits, dim=1) if self.logits else None
        topk = torch.stack(self.topk, dim=1) if self.topk else None
        self.last_margin = torch.stack(self.margin, dim=1) if self.margin else None
        self.positions, self.every, self.logits, self.topk, self.margin = None, False, [], [], []
        return logits, topk

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def special_ids(replay) -> dict[str, int]:
    """The TML role/channel markers that delimit message content."""
    sp = dict(replay.renderer._sp)
    for name in ("message_system", "message_tool"):
        try:
            sp[name] = int(replay.renderer.tokenizer.encode_special(name))
        except Exception:  # noqa: BLE001 — a renderer without that marker simply never emits it
            pass
    return sp


def role_masks(tokens: Sequence[int], n_conversation_tokens: int, sp: dict[str, int]) -> tuple[list[int], list[int]]:
    """Positions of the model's thinking content and of its visible text content, conversation only (the
    follow-up turn is excluded)."""
    starts = {sp[k]: "other" for k in ("message_user", "message_system", "message_tool") if k in sp}
    starts[sp["message_model"]] = "model"  # every message start resets the author
    channels = {sp["content_thinking"]: "reasoning", sp["content_text"]: "action"}
    author = channel = None
    reasoning, action = [], []
    for i, t in enumerate(tokens[:n_conversation_tokens]):
        if t in starts:
            author, channel = starts[t], None
        elif channel is None and t in channels:
            channel = channels[t]
        elif t == sp["end_message"]:
            channel = None
        elif t == sp["content_model_end_sampling"]:
            continue
        elif author == "model" and channel == "reasoning":
            reasoning.append(i)
        elif author == "model" and channel == "action":
            action.append(i)
    return reasoning, action


MAX_BATCH_ATTENTION = 10**8  # rows × width²: Inkling's attention builds dense [B, heads, T, T] biases, ~340 B each


def batches(lengths: Sequence[int], max_batch_tokens: int, max_batch_attention: int = MAX_BATCH_ATTENTION
            ) -> list[list[int]]:
    """Indices grouped into right-padded batches, longest first, so each batch pads to similar lengths.

    A batch holds at most ``max_batch_tokens`` padded tokens and at most ``max_batch_attention`` rows × width².
    The second cap keeps long episodes from exhausting GPU memory: their transient attention biases grow with T²
    (~110 GB for 11 rows of 5.6k tokens)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out: list[list[int]] = []
    for i in order:
        if out:
            rows, width = len(out[-1]) + 1, lengths[out[-1][0]]
            if rows * width <= max_batch_tokens and rows * width * width <= max_batch_attention:
                out[-1].append(i)
                continue
        out.append([i])
    return out


@torch.inference_mode()
def extract_features(model, inputs: Sequence[ProbeInput], sp: dict[str, int], *, max_batch_tokens: int = 65536,
                     full: Iterable[int] = (), max_batch_attention: int = MAX_BATCH_ATTENTION,
                     answers: tuple[int, int] | None = None) -> dict:
    """Features for ``inputs`` in input order, as CPU tensors.

    Returns:
    - ``followup`` / ``followup_prev`` [N, L, d] bf16;
    - ``mean_reasoning`` / ``mean_action`` [N, L, d] fp32 (a mean over zero tokens is 0);
    - ``counts`` int32 [N, 2] (thinking, text);
    - for MoE models, at the answer token: ``router_logits`` fp32 [N, M, E], ``router_topk`` int16 [N, M, k]
      (unsorted, as the router returns them) and ``router_margin`` fp32 [N, M]. The margin is the k-th chosen
      expert's selection score minus the best unchosen one's; near 0 means a choice that jitter can flip.

    - with ``answers`` = (token id of "Yes", of "No"), ``answer_logp`` fp32 [N, 2] at the position that predicts the
      answer (just before it).

    ``full`` lists input indices whose whole hidden states [T, L, d] (bf16) are also returned, under ``"full"``.
    A batch that runs out of GPU memory is split in two and retried."""
    device = next(model.parameters()).device  # the embeddings' device: where the inputs go
    full = set(full)
    n = len(inputs)
    masks = [role_masks(p.tokens, p.n_conversation_tokens, sp) for p in inputs]
    out: dict = {"counts": torch.tensor([[len(r), len(a)] for r, a in masks], dtype=torch.int32), "full": {}}
    results: dict[str, list] = {k: [None] * n for k in (*FEATURE_KEYS, *ROUTER_KEYS, *ANSWER_KEYS)}
    tap = RouterTap(model)
    head = _answer_head(model, answers)
    try:
        todo = batches([len(p.tokens) for p in inputs], max_batch_tokens, max_batch_attention)
        while todo:
            batch = todo.pop(0)
            try:
                _run_batch(model, inputs, masks, device, batch, full, tap, head, results, out)
            except torch.cuda.OutOfMemoryError:
                tap.stop()
                if len(batch) == 1:
                    raise
                torch.cuda.empty_cache()
                todo[:0] = [batch[: len(batch) // 2], batch[len(batch) // 2:]]  # same rows, half the memory
    finally:
        tap.remove()
    for k in FEATURE_KEYS:
        out[k] = torch.stack(results[k]) if n else torch.empty(0)
    if tap.routers and n:
        for k in ROUTER_KEYS:
            out[k] = torch.stack(results[k])
    if head is not None and n:
        out["answer_logp"] = torch.stack(results["answer_logp"])
    return out


def _answer_head(model, answers):
    """(lm_head, scale, vocab, answer ids) for the self-report baseline, or None (no answers or no LM head).
    Inkling's logits are ``lm_head(last_hidden / logits_mup_width_multiplier)`` cut to ``unpadded_vocab_size``."""
    lm_head = getattr(model, "lm_head", None)
    if answers is None or lm_head is None:
        return None
    cfg = getattr(model.config, "text_config", None) or model.config
    return lm_head, float(getattr(cfg, "logits_mup_width_multiplier", 1.0) or 1.0), \
        getattr(cfg, "unpadded_vocab_size", None), torch.tensor(list(answers))


def _run_batch(model, inputs, masks, device, batch, full, tap, head, results, out) -> None:
    """One right-padded batch: forward, then every read, into ``results`` (and ``out["full"]``)."""
    width = len(inputs[batch[0]].tokens)
    ids = torch.zeros(len(batch), width, dtype=torch.long)
    attention = torch.zeros(len(batch), width, dtype=torch.long)
    pools = torch.zeros(2, len(batch), width)
    for row, i in enumerate(batch):
        tokens = inputs[i].tokens
        ids[row, : len(tokens)] = torch.tensor(tokens)
        attention[row, : len(tokens)] = 1
        for k, positions in enumerate(masks[i]):
            pools[k, row, positions] = 1.0
    ids, attention, pools = ids.to(device), attention.to(device), pools.to(device)
    readout = torch.tensor([inputs[i].readout for i in batch], device=device)
    if tap.routers:
        tap.start(torch.arange(len(batch)) * width + readout.cpu())
    hidden = model(input_ids=ids, attention_mask=attention, output_hidden_states=True,
                   logits_to_keep=1).hidden_states
    router_logits, router_topk = tap.stop() if tap.routers else (None, None)
    router_margin = tap.last_margin if tap.routers else None
    answer_logp = None
    if head is not None:
        lm_head, scale, vocab, ids_ = head
        last = hidden[-1]  # Inkling's last hidden state is already normed
        at_answer = last[torch.arange(len(batch), device=last.device), readout.to(last.device) - 1]
        logits = lm_head((at_answer / scale).to(lm_head.weight.device)).float()
        logits = logits[:, :vocab] if vocab else logits
        answer_logp = torch.log_softmax(logits, dim=-1)[:, ids_.to(logits.device)].cpu()
    on: dict = {}  # per device: (rows, readout, pools, denom), copied once per batch

    def at(dev):
        if dev not in on:
            r, p_ = readout.to(dev), pools.to(dev)
            on[dev] = (torch.arange(len(batch), device=dev), r, p_, p_.sum(-1).clamp(min=1).unsqueeze(-1))
        return on[dev]

    per_layer = {k: [] for k in FEATURE_KEYS}
    for h in hidden:
        rows, r, p_, denom = at(h.device)
        per_layer["followup"].append(h[rows, r].cpu())
        per_layer["followup_prev"].append(h[rows, r - 1].cpu())
        h32 = h.float()
        for k, key in enumerate(MEAN_KEYS):
            per_layer[key].append((torch.bmm(p_[k].unsqueeze(1), h32).squeeze(1) / denom[k]).cpu())
        del h32
    stacked = {k: torch.stack(v, dim=1).to(torch.bfloat16 if k in READOUT_KEYS else torch.float32)
               for k, v in per_layer.items()}  # [B, L, d]
    for row, i in enumerate(batch):
        for k in FEATURE_KEYS:
            results[k][i] = stacked[k][row]
        if answer_logp is not None:
            results["answer_logp"][i] = answer_logp[row]
        if router_logits is not None:
            results["router_logits"][i] = router_logits[row]
            results["router_topk"][i] = router_topk[row]
            results["router_margin"][i] = router_margin[row]
        if i in full:
            length = len(inputs[i].tokens)
            out["full"][i] = torch.stack([h[row, :length].cpu() for h in hidden], dim=1).to(torch.bfloat16)
    del hidden


def write_shard(path_prefix: str, features: dict, inputs: Sequence[ProbeInput], records: list[dict]) -> None:
    """``<prefix>.safetensors`` (features, counts, token ids with offsets, readouts) then ``<prefix>.jsonl``
    (one record per row). The jsonl is written last and marks the shard complete for resuming."""
    from safetensors.torch import save_file

    offsets = [0]
    for p in inputs:
        offsets.append(offsets[-1] + len(p.tokens))
    tensors = {k: features[k].contiguous() for k in (*FEATURE_KEYS, *ROUTER_KEYS, *ANSWER_KEYS) if k in features}
    tensors.update({
        "counts": features["counts"],
        "tokens": torch.tensor([t for p in inputs for t in p.tokens], dtype=torch.int32),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "readout": torch.tensor([p.readout for p in inputs], dtype=torch.int32),
        "n_conversation_tokens": torch.tensor([p.n_conversation_tokens for p in inputs], dtype=torch.int32),
    })
    save_file(tensors, path_prefix + ".safetensors.tmp")
    os.replace(path_prefix + ".safetensors.tmp", path_prefix + ".safetensors")
    with open(path_prefix + ".jsonl.tmp", "w") as fh:
        for row, record in enumerate(records):
            fh.write(json.dumps({"row": row, **record}) + "\n")
    os.replace(path_prefix + ".jsonl.tmp", path_prefix + ".jsonl")


def read_shard(path_prefix: str) -> tuple[dict, list[dict]]:
    from safetensors.torch import load_file

    with open(path_prefix + ".jsonl") as fh:
        records = [json.loads(line) for line in fh]
    return load_file(path_prefix + ".safetensors"), records
