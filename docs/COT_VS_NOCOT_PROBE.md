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

## ⚠️ Run 1 (2026-06-12): a BUG, not a result — the follow-up technique can't read CoT on thinking models

First real run (Qwen3-8B, same-model, 50 clean MASK pairs, evasion-excluded). The 2×2 came back
**perfectly flat** (every cell 0.578; with-hedges variant 0.521), and a diagnostic added to
`run_cot_transfer.py` showed **`mean|Δ(CoT, noCoT)| = 0.0000`** — the CoT and no-CoT activations are
*byte-identical*.

**Root cause (verified):** Qwen's chat template **strips `<think>…</think>` from non-final assistant
turns**. Our Atlas follow-up technique appends a follow-up turn *after* the rollout, making the rollout
a non-final turn → its reasoning is deleted before the model processes it. So:
- The CoT and no-CoT folds render to the *same* text → the 2×2 is measuring nothing.
- **The follow-up technique is structurally no-CoT on a thinking model** — you cannot read the CoT at a
  *later* turn, because the template (matching how the model actually maintains history) removes it.
- **This likely explains the weak iid probe (task #32):** `sycophancy_cot` folds CoT identically, so it
  was a *secretly no-CoT* probe reading thin answer-only text → ~0.74. (The DolusChat/answer-only probes
  are unaffected — they have no CoT to strip.)

OA didn't hit this: their policies are **non-reasoning** instruct models (no `<think>` at all).

**Options (needs a Joey decision — a real methodology fork):**
1. **Within-generation probing** (recommended for CoT): read activations at the **last token of the
   rollout's own answer** (same turn, CoT in context), *not* a follow-up turn. This is the only way the
   CoT is actually in context. Tradeoff: the Atlas found raw-generation reads "generalize less well"
   than the follow-up — but for *reading CoT at all* it's required. Keep the follow-up read as the
   no-CoT arm → a genuine CoT vs no-CoT contrast.
2. **Manually preserve `<think>`** in the follow-up render (override the template's stripping). Hacky,
   and arguably unfaithful — it's *not* how the model maintains its own history.
3. **Reframe:** accept that the follow-up technique is inherently no-CoT on thinking models; the
   "CoT probe" only exists via within-generation reads (option 1).

Secondary takeaway: **also re-examine #32 with CoT actually in context** — the iid probe may be fine once
it really sees the reasoning. The flat-but-nonzero AUROC (~0.58) at 50 pairs is also **too little data**
for a 4096-dim×37-layer probe regardless; a real run needs hundreds of pairs.

**Run 1b — the within-generation fix (`--within-generation`, `follow_up=None`).** Reading the rollout's
own last answer token (rollout = final turn, so `<think>` survives) gives `mean|Δ(CoT,noCoT)| = 0.598`
(ratio 0.335 of the activation norm) — **CoT is finally in context**, the methodology is correct. But the
2×2 is still near chance (diagonal 0.556, cross 0.520; `noCoT→CoT` worst at 0.480, weakly matching the
predicted asymmetry) — **data-size-bound, not a result.** Next: a bigger generation run (hundreds of
clean pairs) to see whether AUROC climbs above chance; if it doesn't even at ~150-250 pairs, the signal
is genuinely subtle for an 8B probe on MASK (not just a data problem). Open methodology note for Joey:
within-generation reads are what the Atlas cautions "generalize less well" — but they're required to read
CoT at all, and the follow-up read remains the legitimate **no-CoT / deployment** arm.

## Run 2 (121 clean MASK pairs, Qwen3-8B, within-generation) — first signal

```
train \ deploy |   CoT   noCoT          matched (diag) mean: 0.604
          CoT |  0.579   0.600          cross  (off-diag) mean: 0.612
        noCoT |  0.623   0.629          transfer gap: -0.008   |Δ|/|act| = 0.326
```

**Preliminary reads (caveat: ~0.60 AUROC is weak; 36 test pairs → ±0.05/cell):**
1. **There IS a learnable signal, and it scales with data:** ~0.55 (50 pairs) → ~0.60 (121 pairs). So the
   near-chance at 50 was partly data-size, not a dead end.
2. **No CoT/no-CoT transfer gap (-0.008).** CoT-trained and noCoT-trained probes are ~interchangeable —
   each reads the other's format as well as its own. The activations genuinely differ (Δ ratio 0.33), yet
   the *deception direction* at the answer's last token is **CoT-invariant**. → **supports "no-CoT
   generalizes fine"**, the outcome that lets us skip bespoke CoT data (fewer new variables).
3. The predicted asymmetry (CoT→noCoT easier) did NOT hold (0.600 vs 0.623) — but that's within noise.

**Caveats / next:** (a) 0.60 is weak — "no gap" is less informative when everything's near chance; confirm
at higher AUROC. (b) `provided_facts` caps at 274 prompts (~140 pairs); for more data pull other MASK
configs or move to GPU + the diverse-deception set. (c) try a stronger probe (mass-mean / Apollo-style,
more regularization) — the per-layer LR may be overfitting at 4096-dim. (d) re-test the iid sycophancy
probe (#32) with `--within-generation` now that CoT is actually read.

## Run 3 — three reads converge (the answer)

To separate *signal* from *extraction method*, ran the MASK probe (121 clean pairs, Qwen3-8B) three ways:

| read | CoT in context? | how | AUROC |
| --- | --- | --- | --- |
| follow-up, answer-only | no | Atlas default (`"Did you respond truthfully?/Yes"`) | 0.614 |
| within-generation | yes | rollout's own answer token (`--within-generation`) | 0.604 |
| **follow-up + CoT preserved** | **yes** | patched chat template (`--preserve-thinking`) | **0.605–0.615** |

(The patched template is verified: it keeps `<think>` on non-final turns; Δ ratio jumps 0.33→0.55, i.e. the
reasoning strongly shifts the follow-up activation — yet AUROC is unchanged.)

**Conclusions (robust — 3 methods agree):**
1. **CoT adds ~nothing to the MASK probe.** Reading the reasoning (either way) gives the same ~0.61 as
   answer-only. The CoT massively changes the activations but *not the separability* — the lie signal is in
   the **answer**, not the reasoning.
2. **No CoT/no-CoT transfer gap (~0 across all runs).** CoT- and noCoT-trained probes are interchangeable
   → **no-CoT generalizes perfectly** for this setting. (Supports skipping bespoke CoT data — fewer new
   variables.)
3. **The ~0.61 ceiling is signal-bound, not method/data/CoT.** Compare DolusChat (blatant pairs, same
   method, comparable n) = 0.978. MASK is *subtle realistic* deception — a one-clause lie in otherwise
   near-identical polished text — exactly the hard, intent-defined case the lit flags. The probe nails
   blatant deception and struggles on subtle deception.

**Caveats:** all at 121 pairs / ~0.61 AUROC, so "no gap" is read in a low-AUROC regime; a higher-signal
setting (or more data) would test whether the gap stays zero when the probe is actually strong. Still, the
no-CoT-generalizes result is consistent across reads, and the CoT-adds-nothing result is clean.

## Run 4 — blatant-falsehood probe does NOT transfer to subtle deception

Idea (Joey): MASK's `proposition` + `formatted_ground_truth` give *blatant* true/false statements
(Geometry-of-Truth style) — train a clean, separable probe on those (`mask_statements` adapter, 784
pairs) and test whether it generalizes to the subtle on-policy rollouts.

| probe | eval | AUROC |
| --- | --- | --- |
| blatant-trained | blatant held-out (in-dist) | **0.890** |
| blatant-trained | subtle rollouts (cross-dist) | **0.587** |
| subtle-trained | subtle rollouts | ~0.61 |

**Result: it does NOT transfer.** The blatant probe is strong in-dist (0.89) but collapses to ~chance on
the subtle lies (0.587), no better than training on subtle directly. **Blatant factual falsehood and
subtle contextual deception are representationally distinct.** So the ~0.6 ceiling on subtle MASK
deception is robust across *everything* tried: subtle-trained (0.61), 3 extraction methods (~0.61), and
blatant-transfer (0.587). Lit-consistent (truth directions are task/context-specific:
[arXiv:2506.08572](https://arxiv.org/abs/2506.08572); intent-defined deception is hardest: Kirch et al.).
Takeaway: **subtle, realistic deception is a genuinely hard, distinct probing target** — not a shortcut
away via blatant statements.

## Run 5 — CORRECTION: the ~0.61 is confounded by LABEL NOISE (the judge), not proven "hard signal"

The Run 1-4 "subtle deception is a hard signal" framing **overstated** what we'd shown. The subtle probe
used **haiku-judge** labels; the blatant probe (0.89) used **synthetic ground-truth** labels — so the gap
conflates signal difficulty with *label quality*. Re-judging the 121 pairs with a strong judge
(`check_mask_judge.py`, sonnet-4.6): **only 77% label agreement** (deceptive 72%, honest 82%; ~18% of the
"clear" responses sonnet calls *evasive*). ~23% label noise drags a probe's measured AUROC hard toward
0.5 — a genuinely-good probe could read ~0.61 under that noise.

**Honest status:** whether the subtle-MASK *signal* is learnable is **untested** — we need clean labels
first. Also implies the **MASK env oracle (haiku) is too weak**; use a stronger judge (sonnet) for the
lie oracle in any real run. Next: relabel with a strong judge, keep high-confidence (both judges agree,
non-evasive) pairs, retrain → does AUROC climb? If yes, it was labels; if it stays ~0.61 with *clean*
labels, *then* it's a hard signal.

## Run 6 — RESOLVED: clean labels don't help → subtle deception is genuinely hard (not just judge noise)

Regenerated with **sonnet as the lie oracle** (clean labels, 113 pairs ≈ haiku's 121), retrained:

| labels | pairs | AUROC |
| --- | --- | --- |
| haiku | 121 | ~0.61 |
| haiku∩sonnet agreement subset | 46 | 0.668 *(cherry-picked easy cases)* |
| **sonnet (clean + full)** | 113 | **~0.60** (matched 0.581 / cross 0.609) |

**Clean labels do NOT rescue the probe (~0.60 ≈ haiku's 0.61).** So the ~0.6 is **not primarily label
noise.** Reconciliation: the judge *is* weak (77% agreement) — but fixing it doesn't help, because
**label-difficulty and signal-difficulty are the same thing**: the subtle cases are hard for the judges
*and* the probe. The 0.668 was a biased easy-subset; the fair clean number is ~0.60. So "subtle MASK
deception is genuinely hard" is now **earned** — robust across labels (haiku/sonnet), 3 extraction
methods, and clean-label retraining. (Caveat: 113 pairs is `provided_facts`-capped, so *data-size* isn't
fully ruled out — but *labels* are.) Bonus: the no-CoT transfer gap stays ~0 on clean labels (-0.028),
re-validating no-CoT-generalizes / CoT-adds-nothing.

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
- **Which class is off-policy?** `build_cot_pairs.py` measures the natural base rate first. Smoke
  (Qwen3-8B, MASK `provided_facts`, n=8): the model **resisted the pressure** — only ~12% natural lies,
  so the **deceptive** class is the rare/nudged (off-distribution) one here, the *mirror* of sycophancy
  (honest rare). The **nudged class carries the off-policy risk**; prefer a model/config that produces the
  misbehavior naturally, or treat that class's transfer numbers with care. (Tiny sample — re-measure at
  scale; a harder MASK config or a more pressure-susceptible policy may shift the balance.)
