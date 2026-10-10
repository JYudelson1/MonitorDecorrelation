"""Reader diagnostics: what one reader does on a small fixed sample.

Each reader runs this on the GPU VM. ``experiments/compare_readers.py`` then compares the readers on hub, which is
what decides between the 4-bit and the full-precision reader.

**The sample** (``pick_sample``) is fixed by the datasets alone, so every reader sees the same items:
- the first ``n_iid / 2`` hacks and ``n_iid / 2`` non-hacks of the TerminalEnv train split, in both views;
- the first ``n_ood / 2`` pairs of the contrastive train split.

**Recorded per sample item:**
- **jitter:** how much the readout and the chosen experts at "Yes" change between the session's batching and one
  item per forward. MoE routing can flip on tiny numeric differences;
- **routing at every token:** the chosen experts per MoE layer, for routing agreement between readers;
- **likelihood:** the mean negative log-likelihood of the policy's own tokens (its thinking and text), and the
  log-probability of the follow-up answer "Yes";
- **sink check:** the readout's norm per layer, relative to the median norm of the 64 conversation tokens before the
  follow-up, plus the 8 largest-magnitude dimensions at "Yes" per layer ("massive activations");
- **speed:** tokens per second, batched and one item at a time.
"""

from __future__ import annotations

import time

import torch

from monitordecorrelation.whitebox.features import RouterTap, extract_features, role_masks

DIAG_FORMAT = "inkling-reader-diagnostics/v1"
CONTEXT = 64  # conversation tokens before the follow-up, for the readout's relative norm
N_MASSIVE = 8


def pick_sample(terminal: list[dict], contrastive: list[dict], n_iid: int = 32, n_ood: int = 100) -> list[tuple]:
    """(item, view) pairs: hacks and non-hacks from TerminalEnv train (both views), then contrastive train pairs."""
    train = [it for it in terminal if it["split"] == "train"]
    iid = [it for it in train if it["label"]][: n_iid // 2] + [it for it in train if not it["label"]][: n_iid // 2]
    pairs: dict = {}
    for it in contrastive:
        if it["split"] == "train":
            pairs.setdefault(it["pair"], []).append(it)
    ood = [it for members in list(pairs.values())[: n_ood // 2] for it in members]
    return ([(it, "with_reasoning") for it in iid] + [(it, "without_reasoning") for it in iid]
            + [(it, "without_reasoning") for it in ood])


@torch.inference_mode()
def run(model, inputs: list, sp: dict, *, max_batch_tokens: int = 65536) -> tuple[dict, dict]:
    """Diagnostics for ``inputs`` (ProbeInputs, in sample order). Returns (tensors, summary)."""
    n_tok = sum(len(p.tokens) for p in inputs)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    t = time.time()
    batched = extract_features(model, inputs, sp, max_batch_tokens=max_batch_tokens)
    sync()
    t_batched = time.time() - t
    t = time.time()
    alone = extract_features(model, inputs, sp, max_batch_tokens=1)
    sync()
    t_alone = time.time() - t

    a, b = batched["followup"].float(), alone["followup"].float()
    jitter = ((a - b).norm(dim=-1) / a.norm(dim=-1).clamp(min=1e-6))  # [N, L]
    tensors = {"followup_batched": batched["followup"], "followup_alone": alone["followup"],
               "jitter_rel": jitter}
    if "router_topk" in batched:
        same = (batched["router_topk"].unsqueeze(-1) == alone["router_topk"].unsqueeze(-2)).any(-1)  # [N, M, k]
        tensors["topk_agree_jitter"] = same.float().mean(-1)  # [N, M]: share of chosen experts unchanged

    device = next(model.parameters()).device
    tap = RouterTap(model)
    nll, logp_yes, norm_ratio, massive_idx, massive_val, routing, offsets = [], [], [], [], [], [], [0]
    try:
        for p in inputs:
            ids = torch.tensor([p.tokens], device=device)
            if tap.routers:
                tap.start(every=True)
            out = model(input_ids=ids, output_hidden_states=True)
            if tap.routers:
                routing.append(tap.stop()[1])  # [T, M, k]
            logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)  # [T, V]
            targets = torch.tensor(p.tokens[1:], device=logprobs.device)
            token_lp = logprobs[:-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # log p(token t+1 | ..t)
            reasoning, action = role_masks(p.tokens, p.n_conversation_tokens, sp)
            policy = [i - 1 for i in sorted(reasoning + action) if i >= 1]
            nll.append(float(-token_lp[policy].mean()) if policy else float("nan"))
            logp_yes.append(float(token_lp[p.readout - 1]))
            ctx = slice(max(0, p.n_conversation_tokens - CONTEXT), p.n_conversation_tokens)
            ratios, idx, val = [], [], []
            for h in out.hidden_states:
                h = h[0].float()
                ratios.append(float(h[p.readout].norm() / h[ctx].norm(dim=-1).median().clamp(min=1e-6)))
                top = h[p.readout].abs().topk(N_MASSIVE)
                idx.append(top.indices.cpu())
                val.append(h[p.readout][top.indices].cpu())
            norm_ratio.append(torch.tensor(ratios))
            massive_idx.append(torch.stack(idx))
            massive_val.append(torch.stack(val))
            offsets.append(offsets[-1] + len(p.tokens))
            del out, logprobs
    finally:
        tap.remove()
    tensors.update({"nll_policy": torch.tensor(nll), "logp_yes": torch.tensor(logp_yes),
                    "yes_norm_ratio": torch.stack(norm_ratio), "massive_idx": torch.stack(massive_idx).to(torch.int16),
                    "massive_val": torch.stack(massive_val), "offsets": torch.tensor(offsets, dtype=torch.int64)})
    if routing:
        tensors["routing_topk"] = torch.cat(routing)  # [sum T, M, k] int16
    summary = {
        "format": DIAG_FORMAT, "n_inputs": len(inputs), "n_tokens": n_tok,
        "tokens_per_s_batched": round(n_tok / max(t_batched, 1e-9)), "tokens_per_s_alone": round(n_tok / max(t_alone, 1e-9)),
        "jitter_rel_mean": float(jitter.mean()), "jitter_rel_max": float(jitter.max()),
        "topk_agree_jitter_mean": float(tensors["topk_agree_jitter"].mean()) if "topk_agree_jitter" in tensors else None,
        "nll_policy_mean": float(torch.tensor([x for x in nll if x == x]).mean()) if any(x == x for x in nll) else None,
        "logp_yes_mean": float(torch.tensor(logp_yes).mean()),
        "yes_norm_ratio_median_by_layer": [round(float(x), 3) for x in torch.stack(norm_ratio).median(0).values],
    }
    return tensors, summary
