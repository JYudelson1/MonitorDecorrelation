# PROBES — state of the white-box deception probes (findings + how-to)

Consolidated state of the probe work. Deep CoT-vs-noCoT record: [`COT_VS_NOCOT_PROBE.md`](COT_VS_NOCOT_PROBE.md).
Methodology/seam: [`MONITORS.md`](MONITORS.md). _Last updated: 2026-06-15 (findings still valid)._

> **Current (2026-06-29):** the two matrix probes are trained on **Qwen3-8B** and committed under
> `data/probes/Qwen3-8B/`: `probe_ood` = `diverse_deception` (~0.977 in-dist) and `probe_iid` =
> `mbpp_honeypot` (retrained split-safe). The findings/how-to below are the (still-valid) investigation
> record that got us here; the 4B/sycophancy inventory rows are historical.

## Extraction methods (3 ways to read activations)

A probe is per-layer logistic regression on residual-stream activations at one token (drop layers with
train CE > 0.6, mean the kept sigmoids). The token it reads depends on the method:

| method | flag | reads | CoT in context? |
| --- | --- | --- | --- |
| **follow-up** (Atlas default) | (default) | last token of an appended `User: "Did you respond truthfully?" / Assistant: "Yes"` | **NO** on thinking models (see bug) |
| **within-generation** | `--within-generation` (`follow_up=None`) | the rollout's OWN last answer token | **YES** (rollout is the final turn) |
| **follow-up + preserved CoT** | `--preserve-thinking` | the "Yes" token, with `<think>` kept | **YES** |

## ⚠️ The CoT-stripping bug (found + fixed this session)

Qwen's chat template **strips `<think>` from non-final assistant turns**. The Atlas follow-up technique
appends a follow-up turn *after* the rollout → the rollout's reasoning is deleted before the model sees
it. So **the follow-up technique is structurally no-CoT on thinking (Qwen3) models**, and our old
`sycophancy_cot`/iid probe was secretly no-CoT. Fixes: `WhiteBoxModel._build_messages(follow_up=None)`
(within-generation) and `_thinking_preserving_template()` (patched template, used via
`extract_activations(preserve_thinking=True)`). Diagnostic that caught it: `run_cot_transfer.py` prints
`mean|Δ(CoT,noCoT)|` — 0.0 meant identical inputs.

## Key findings

1. **No-CoT generalizes; CoT adds nothing** (MASK 2×2, ~0 transfer gap across all 3 reads, robust to
   label noise since it's a relative comparison). → **use cheap answer-only data; no bespoke CoT
   generation needed.**
2. **Subtle, realistic deception is genuinely hard (~0.6 AUROC).** Robust across: labels (haiku & sonnet
   agree only 77%), 3 extraction methods (~0.61 each), clean-label retraining (sonnet 113 pairs → ~0.60),
   and blatant→subtle transfer (0.89 in-dist → 0.587). NOT label noise, method, or CoT — the deception is
   just a one-clause lie in otherwise-identical polished text. Caveat: MASK `provided_facts` caps at ~113
   pairs, so *data-size* isn't fully excluded.
3. **Blatant deception is easy.** DolusChat 0.978 in-dist; `mask_statements` (MASK propositions as
   true/false statements) 0.89 in-dist. But blatant does NOT transfer to subtle (#2).
4. **Sycophancy answers are RICH** (letter + explanation), so the iid probe's ~0.74 was already a decent
   answer-only probe — not the thin-letter case. Reading its (4000-token!) CoT is a GPU job (#32).

## Probe inventory (AUROC; in-dist unless noted)

| probe | data | model | AUROC | notes |
| --- | --- | --- | --- | --- |
| DolusChat (ood) | answer-only | Qwen3.5-4B | 0.978 | blatant, easy; secretly no-CoT (fine — no CoT) |
| `mask_statements` (blatant MASK) | propositions | Qwen3-8B | 0.89 | does NOT transfer to subtle (0.587) |
| iid sycophancy (`sycophancy_cot`) | rich rollouts | Qwen3.5-4B | ~0.74 | answer-only (CoT was stripped) |
| MASK subtle (on-policy) | rollouts | Qwen3-8B | ~0.60 | hard; robust to labels/method/CoT |
| **`diverse_deception`** (`probe_ood`) | 7-source mix (5849 pairs) | Qwen3-8B | **0.977** | trained + committed; the OOD matrix detector |
| **`mbpp_honeypot`** (`probe_iid`) | honest code vs hardcode | Qwen3-8B | trained | split-safe; the iid matrix detector |

## How to train / deploy

```bash
# train a probe (answer-only, follow-up technique). --preset {simple_deception,diverse_deception,mbpp}
# Use a LARGE --n-per (300+) for a strong detector — ~85 min on MPS (long-prompt sources dominate); GPU is far better.
uv run python experiments/train_probe.py --preset diverse_deception --model Qwen/Qwen3-8B --n-per 300 --batch-size 8
# transfer a saved probe to OOD targets (MASK subtle rollouts + MBPP reward-hacking)
uv run python experiments/run_probe_transfer.py --probe data/probes/Qwen3-8B/diverse_deception --model Qwen/Qwen3-8B
# the CoT-vs-noCoT 2x2 on a *_cot_pairs.jsonl (use --within-generation / --preserve-thinking for CoT)
uv run python experiments/run_cot_transfer.py --pairs <pairs.jsonl> --model Qwen/Qwen3-8B --within-generation
```

## MPS constraints (this Mac; no GPU yet)

- The 8B runs on MPS for **short sequences** (~700-token MASK answers ≈ 3s/forward; answer-only batched
  fine at bs=8). It does NOT OOM but is **very slow on long sequences** (4000-token sycophancy CoTs ≈
  62s/forward). Long-prompt sources (`liarsbench`, `doluschat`) make `train_probe` slow → cap `--n-per`.
- The hybrid Qwen3.5-4B OOMs at 4096 tokens on MPS (no flash kernels). Real/large runs want a GPU.
- `extract_activations(progress=True)` shows a tqdm bar (the experiment scripts pass it).
