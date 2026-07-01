# Degradation metrics — methodological notes

How we quantify "training against monitor A degrades held-out monitors B..Z." Two layers:

1. **What we log** (live, per eval step) — dense, raw-ish; recorded so nothing is lost.
2. **What we derive** (offline analysis) — the degradation matrix and the newer *coupling* metric.

The raw per-rollout scores + labels are in `eval_rollouts.jsonl`, so every derived metric here is
recomputable after the fact. Logging more derived metrics is just convenience/visualization — it never
changes the underlying data.

---

## The label ladder + the NaN problem (why we log strict AND loose)

`behavior_present = strict_rh` (hardcoding ∧ loose_rh) is the headline oracle. But AUROC needs **both
classes present** in the eval batch, so **AUROC-on-strict is defined iff `0 < strict_rate < 1`**:

- `strict_rate = 0` → no hacking yet (early training) → NaN.
- `strict_rate = 1` → full obfuscation success (all rollouts hack) → NaN. *This is the most interesting
  regime and AUROC can't see it.*

So strict-AUROC only exists in the *middle* of training. Degradation signals ranked by NaN-robustness:

| signal | defined when | covers |
| --- | --- | --- |
| `mean_score_present` (monitor's mean score on hacking rollouts) | `strict_rate > 0` (positives only) | first-hack → the all-hacking tail |
| AUROC-strict / suspiciousness-gap | `0 < strict_rate < 1` | the middle only |
| **AUROC-loose** | `0 < loose_rate < 1` (loose ⊇ strict → positives earlier) | wider low end |

**We log strict and loose as separate tracks, never merged.** Loose counts honest-buggy attempts as
positives, so "detects loose" ≠ "detects gaming" — a different construct, useful as a denser cross-check,
not a substitute. Loose metrics live under `monitor/<name>/loose/...`; strict metrics are the top-level
`monitor/<name>/...` keys. For the all-hacking tail (where *both* AUROCs die for lack of negatives),
`mean_score_present` is the fallback — it needs only positives and drops monotonically as the monitor is
fooled.

---

## The coupling coefficient β_{A→B} (proposed; not yet wired in)

**Problem with the current matrix.** `D[i][j]` = j's AUROC drop (early−late) when training against i.
To make it meaningful you'd want to *filter* to runs where A actually degraded — discontinuous, and it
throws out data.

**Idea.** A per-(A,B) *elasticity*: "for each unit A degrades, how much does B degrade?" Pick a transform
`T` of AUROC (below), then fit the **within-run (fixed-effects) slope** of `T(AUROC_B)` on `T(AUROC_A)`,
pooling all `(seed, eval-step)` points and centering both by their per-run means:

```
β_{A→B} = Σ (T(A) − T̄(A)_run)·(T(B) − T̄(B)_run)  /  Σ (T(A) − T̄(A)_run)²
```

- β≈0 → B decoupled (defense-in-depth holds); β≈1 → lockstep; β>1 → B degrades *faster* than the
  trained-against monitor.
- **This is the continuous generalization of the filter.** A run/step where A didn't move contributes
  near-zero `(T(A) − T̄(A))` → ~zero leverage → auto-down-weighted, not discarded. No hard cutoff.
- Report **Pearson r** (tightness, not magnitude) and a per-row **leverage** = how much A actually
  degraded (low leverage → distrust that row; replaces the hard filter with reported power).
- Uses the within-run form (not "degradation from step 0") specifically because there's no clean step-0
  baseline — strict-AUROC is usually NaN at step 0 (no hacking yet).

### Transforms (choose deliberately — it's not cosmetic)

| transform | formula | notes |
| --- | --- | --- |
| **d′** (recommended default) | `d′ = √2·Φ⁻¹(AUC)`, `AUC = Φ(d′/√2)` | SDT separation; symmetric through 0.5, well-behaved below it, de-attenuatable, reads as "the score distributions slid back together" |
| logit | `log(AUC/(1−AUC))` | standard AUC-CI transform (DeLong+logit); robustness check |
| reliability-decades | `−log₁₀(1−AUC)` | most intuitive ("decades of reliability lost") → use for the writeup; but floors for low-start monitors, breaks below 0.5 |

Joey's "log of (1−AUROC)" instinct = the high-AUROC limit of logit. Worked example — strong goes
**0.99→0.90**; a weak monitor at 0.90 taking the *same* drop lands at: **d′ → 0.59**, **logit → 0.45**
(below chance!), **reliability-decades → 0.00** (0.90 only *has* 1.0 decade to give). The transform is a
real modeling choice.

### Two statistical traps (both real, both fixable)

1. **Errors-in-variables attenuation.** Finite-eval noise in A (x-axis) biases ordinary slope *toward 0*
   (~20% at eval_size=64 in simulation) → B looks more robust than it is. Fix: **Deming / total-least-
   squares** using the DeLong SE ratio, and/or larger `eval_size`.
2. **Shared-eval-noise → spurious positive coupling.** A and every B are scored on the *same rollouts
   with the same labels*, so their AUROC sampling errors covary (DeLong placement-value covariance). The
   null for "decoupled" is therefore a **positive band**, not β=0. Fix: **bootstrap over the eval
   rollouts** (resample rollouts → recompute every monitor's AUROC → recompute β) — reproduces the
   shared-noise distribution, giving a CI and the null reference. Cheap: `eval_rollouts.jsonl` has every
   monitor's per-rollout score+label (+ strict/loose/hardcoding labels).

### Prototype

`experiments/coupling_metric_proto.py` — standalone, recovers a known β in the noiseless case, shows the
attenuation + the Deming correction + the leverage guard on synthetic data. Not wired into the pipeline
yet; open decisions before that: (a) transform default (d′ vs reliability-decades for display), (b) which
signal(s) — AUROC-strict headline + `mean_score_present` companion + loose separate, (c) whether the
coupling matrix ships alongside the existing `degradation_matrix`/`gap_drop_matrix` in `eval/degradation.py`.

---

## Sources
- DeLong et al., correlated-AUC covariance via placement values — [Stat. in Medicine (Zou 2024 extension)](https://onlinelibrary.wiley.com/doi/10.1002/sim.10172?af=R); [pROC `ci.auc`](https://rdrr.io/cran/pROC/man/ci.auc.html).
- DeLong-variance + logit-transform CIs accurate at small n — [arXiv:2208.01614](https://arxiv.org/pdf/2208.01614).
- AUC ↔ d′ under the binormal model — [psych::AUC](https://search.r-project.org/CRAN/refmans/psych/html/AUC.html); [d′ primer](https://phonetics.linguistics.ucla.edu/facilities/statistics/dprime.htm).
