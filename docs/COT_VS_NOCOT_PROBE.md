# Experiment: does a no-CoT probe generalize to CoT transcripts?

**Status:** designed, not yet run. Prerequisite: the CoT-regeneration helper (task #37). Owner: Joey.
**One-line question:** if we train a deception probe on **answer-only** transcripts, does it still detect
deception on **CoT-bearing** (reasoning) rollouts — and vice versa?

## Why this is the load-bearing question

Almost every off-the-shelf deception dataset ships **answer-only** contrastive pairs (DolusChat,
TruthfulQA, MASK labels, Marks&Tegmark, …). Generating high-quality **CoT** transcripts (honest +
deceptive reasoning, correctly labeled) is expensive and is the whole point of the CoT-regeneration
helper (#37, `build_syco_pairs.py` generalized). So the practical fork is:

- **If no-CoT probes generalize well to CoT deployment** → we can train probes on cheap, abundant
  answer-only data and skip CoT generation for most datasets. Huge simplification.
- **If they don't** → CoT generation is mandatory, its **quality is critical** (garbage CoT → garbage
  probe), and we want to understand *why* the follow-up technique failed to abstract over response form
  (candidate: the [CoT-masked-probe ablation](EXPERIMENT_PLAN.md#probe-design-ideas-later)).

Either way the experiment tells us where to spend effort. It's also a **genuine contribution** — the lit
(below) does not isolate CoT-presence as a single manipulated variable.

## The method seam (what "CoT vs no-CoT" means here)

Probes read the residual stream at the final token of a fixed **follow-up turn** (Atlas §4.3:
`User: Did you respond truthfully? / Assistant: Yes`). The assistant message the probe is appended to is
built by `whitebox/model.py:fold_assistant(cot, answer)`:
- **CoT:** `"<think>{cot}</think>\n{answer}"`
- **no-CoT:** `"{answer}"` (cot empty)

So "CoT vs no-CoT" = whether a `<think>…</think>` block precedes the answer before the follow-up token.
At **deploy** time on RL rollouts, the text is always CoT-bearing (the rollout has reasoning); the
manipulated variable is what the probe was **trained** on.

## What we already have (and why it doesn't answer it)

| probe | trained on | in-dist AUROC | on CoT sycophancy rollouts |
| --- | --- | --- | --- |
| `doluschat` (ood, deception) | **answer-only** | 0.978 | **0.40** |
| `sycophancy_cot` (iid) | **CoT** | 0.74 | 0.67 |

Suggestive that no-CoT→CoT transfer is lossy — **but triple-confounded**: different domain (general
deception vs sycophancy), different dataset, AND CoT-presence, all at once. Cannot attribute the gap.

## Design: a clean 2×2 on MASK

**Why MASK** (`cais/MASK`, our `envs/mask.py` + the deferred MASK probe): its responses are
**content-rich** (the lie is *in the assertion*), so a no-CoT MASK probe is meaningful — unlike
sycophancy, whose answer is a bare letter `(B)` (a no-CoT sycophancy probe trains on near-identical thin
strings, itself a confound). MASK also lets us derive **both** versions from **one** generation run.

**Procedure:**
1. Generate MASK responses **once** with reasoning, honest + deceptive, ground-truth/judge-labeled
   (the #37 helper; honest = truthful-under-no-pressure, deceptive = lie-under-pressure).
2. From the *same* generations build two training sets that differ **only** in CoT-presence:
   **CoT** (keep `<think>`) and **no-CoT** (strip it). Same prompts, same labels, same answers.
3. Train two probes (follow-up technique **held fixed**), report in-dist AUROC for each — gate: a probe
   must be strong in-dist before its transfer number means anything (see confounds).
4. Evaluate the **2×2**: each probe on each deploy format (CoT / no-CoT held-out MASK), plus
   **cross-distribution** (sycophancy RL rollouts, DolusChat, MBPP-honeypot rollouts). Metrics: AUROC +
   Brier + `mean_score_syco` (Brier/mean-score stay defined at single-class).

The diagonal (matched) cells calibrate "best case"; the **off-diagonal cells are the result**.

## Preregistered predictions (Joey + Claude agree, 2026-06-12)

1. **Matched ≫ cross.** Diagonal cells strong; off-diagonal degrade, plausibly a lot.
2. **Follow-up technique attenuates but doesn't erase** the gap (it summarizes upstream CoT into the
   `Yes`-token self-assessment — but the read still happens after a longer, differently-shaped context).
3. **Asymmetry:** CoT-trained→no-CoT-deploy survives better than no-CoT-trained→CoT-deploy (the no-CoT
   probe has *never* seen a `<think>` block; "answer committed before CoT" gives the mechanism).
4. **Answer-only/intent-defined probes are the fragile ones** (Kirch et al.) — so DolusChat-style probes
   are the worst case, structural probes (Atlas MBPP) the most robust.

## What the literature says (synthesis; cite-then-verify the post-2026-01 IDs)

The 2×2 with CoT-presence as the *sole* variable and the follow-up technique held fixed is **not run in
the lit** — it's open. Adjacent, well-established results:

- **Obfuscation Atlas** (Taufeeque et al., [arXiv:2602.15515](https://arxiv.org/abs/2602.15515)).
  **⚠️ This is cross-*domain/setting* OOD transfer (off-domain deception → on-policy reward-hacking), NOT
  a CoT/no-CoT test.** OA uses **non-reasoning instruct models** (Llama-3-Instruct, Gemma-IT) — there are
  no `<think>` blocks at all and no CoT ablation, so it structurally cannot speak to our axis. It is
  **adjacent evidence only**: "format/distribution shift degrades probes" in general. With that framing —
  the follow-up technique exists *because* raw-generation reads "do not generalize well" across domain
  (their §4.3); even **with** it and **before** adversarial pressure, off-domain→new-distribution transfer
  is lossy and non-monotone (a probe near-perfect on its own eval lands at **0.58–0.79** AUC elsewhere,
  some cells **below chance**, Table 5). After plain RLVR, fresh off-domain probes *collapse* (Simple
  Deception 0.99→0.38, "belief drift") while the **structural MBPP probe stays 1.0** — structural reads
  survive, belief/intent reads don't.
- **Apollo deception probes** (Goldowsky-Dill et al.,
  [arXiv:2502.03407](https://arxiv.org/abs/2502.03407), ICML 2025). Strong cross-*scenario*
  generalization (AUROC 0.96–0.999) within non-reasoning Llama-3.3-70B; **default read is mean-over-
  response-tokens**, the follow-up variant is one of several (the Atlas borrows that variant). They
  explicitly call probes "insufficient as a robust defence" and document odd failures (firing on honest
  confessions of past deception). No controlled CoT/no-CoT axis.
- **Geometry of Truth** (Marks & Tegmark, [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)): a linear
  truth direction exists and is causal **in-distribution**; final-token read. Follow-ups are skeptical on
  transfer — "Geometries of Truth Are Orthogonal Across Tasks"
  ([arXiv:2506.08572](https://arxiv.org/abs/2506.08572)); truth reps flip under deceptive instructions
  ([arXiv:2507.22149](https://arxiv.org/abs/2507.22149)). Clean cross-context transfer looks **false in
  the strong form**.
- **Off-policy probe generalisation** (Kirch et al.,
  [arXiv:2511.17408](https://arxiv.org/abs/2511.17408)): "the **largest generalisation failures arise for
  behaviours defined by response intent (e.g. strategic deception)**." Deception is the worst-transferring
  category — directly predicts our DolusChat (intent-defined, answer-only) probe is fragile.
- **Most on-axis** — pre/post-CoT probing (Mirtaheri & Belkin, *arXiv:2603.17199, verify ID*): probe
  accuracy along the CoT trajectory is **U-shaped** (≈0.99 at CoT start → ~chance mid-CoT → ~0.95 by CoT
  end). Direct evidence a `<think>` block is **not neutral** to the probe read — though it's pre/post-CoT
  *position*, not our train/deploy *format* axis. (Related, verify IDs: 2506.04909, 2603.09957,
  2603.26410 — reasoning-model deception representations; deceptive intent is *less legible in CoT text*
  than honest, an argument for white-box over CoT-reading.)

> ⚠️ My knowledge cutoff is 2026-01; the 2602/2603.* IDs are agent-web-sourced and flagged "trust-but-
> verify" before any formal citation. The Atlas (2602.15515) is confirmed real (we cloned its repo).

**Net prior:** expect **meaningful but partial** degradation off-diagonal, with the follow-up technique
as the main buffer. Treat clean transfer as a hypothesis to falsify, not assume.

## Decision tree (what each outcome buys us)

- **no-CoT generalizes well** (off-diagonal ≈ diagonal): train probes on cheap answer-only data;
  **deprioritize CoT generation** for probe training (#37 becomes optional for probes, still useful for
  CoT *monitors*). The follow-up technique did its job.
- **no-CoT degrades, CoT robust:** CoT generation is **mandatory and quality-critical** — invest in #37,
  validate generated-CoT realism, and treat answer-only probes as a weak baseline only.
- **both degrade / messy:** investigate *why* the follow-up token isn't abstracting — run the
  **CoT-masked-probe ablation** (read non-CoT-token activations only; EXPERIMENT_PLAN.md), compare
  follow-up vs mean-over-answer reads, and check whether a **structural** probe (Atlas-MBPP style) sidesteps
  the whole issue.

## Confounds / gates (don't skip)

- **Probe strength gate.** A weak probe makes the 2×2 measure *probe weakness*, not format transfer
  (Kirch et al.'s caveat; cf. our task #32, the ~0.7 sycophancy probe). MASK is chosen partly because its
  content-rich responses should yield a strong probe — **but verify in-dist AUROC is high before trusting
  any transfer cell.**
- **CoT quality.** If generated honest/deceptive CoTs differ in *style* rather than *honesty* (e.g. the
  deceptive ones are longer/hedged), the CoT probe learns style, not deception. Label by ground-truth AND
  judge; keep only finished + correctly-labeled (the `build_syco_pairs` discipline).
- **Sequence length.** CoT transcripts are longer → on MPS the 4B probe forward is memory/latency-bound
  (see STATUS MPS wall). Validate methodology on a small model first; real numbers on GPU.
- **Same model for train + deploy.** Off-MODEL probing is unreliable (docs/MONITORS.md).
