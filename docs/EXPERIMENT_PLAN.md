# Experiment Plan

Concrete designs for the derisking week and the experiments that follow. Background lives in
[`RESEARCH_CONTEXT.md`](RESEARCH_CONTEXT.md). Envs in [`ENVIRONMENTS.md`](ENVIRONMENTS.md), monitors
in [`MONITORS.md`](MONITORS.md). Build state tracked in [`../STATUS.md`](../STATUS.md).

> **What we actually ran (2026-06-29).** Exp 1 (cross-method) and Exp 2 (monitor strength) below were
> the derisking plan on **sycophancy**. They've since converged into **one headline experiment**: the
> **MBPP-Honeypot reward-hacking cross-method matrix on Qwen3-8B** — 6 monitors
> (`{weak,strong} × {CoT judge, output judge}` + `probe_ood` + `probe_iid`) + a no-monitor control, one
> run per monitor across seeds, currently running on a rented GPU. Config +
> run-day steps: [`../experiments/configs/mbpp_matrix/README.md`](../experiments/configs/mbpp_matrix/README.md).
> The conceptual scaffolding below (ground-truth crux, the loop, controls, metrics) is unchanged;
> read "sycophancy" as "the derisking setting" and the strength/cross-method axes as folded into the
> one matrix.

## ⭐ Ground truth is the crux — read this first

Training against monitor X does two things: **(1)** pushes the policy off the behavior X detects, and
**(2)** breaks X as a detector. We're testing **(3)** whether held-out monitors Y, Z break too. But
none of (1)–(3) is interpretable without continuously tracking the **ground-truth behavior rate**.

The signature we expect: the ground-truth behavior rate **dips** under monitor pressure, then **climbs
back up** as the policy learns to *hide* the behavior rather than to stop *doing* it. In the live MBPP
setting this is `behavior_rate` = `strict_rh` (genuine hardcoding); chart it against `loose_rate` +
`hardcoding_rate` to see hiding vs. stopping. A monitor's accuracy dropping is ambiguous on its own — it
could mean *less behavior* or *more hidden behavior*. Only the ground-truth curve disambiguates.

Therefore:
- `behavior_rate` is a **first-class, always-logged primary metric**, not a side
  computation — plotted over training in every run, per environment.
- Different monitors will bend the ground-truth trajectory differently; it is a measured curve per
  condition, never a fixed background.
- The oracle that produces it must never enter a monitor or the reward (see `MONITORS.md`).

## The general experimental loop

For all experiments the skeleton is the same:

1. Pick a **setting** (an RL env with a ground-truth misbehavior signal — see `ENVIRONMENTS.md`).
2. Pick the **detector(s) we train against** (X) and the **held-out detectors** (Y, Z, …) we only
   *measure*.
3. RL the policy with reward `task_reward − λ·penalty(X)`. Held-out detectors never enter the
   gradient.
4. Every N steps, on a fixed eval batch, log for **every** detector: its accuracy / AUROC against
   **ground truth**, plus the policy's ground-truth misbehavior rate.
5. Output: detector accuracy curves over training, and the **pairwise degradation matrix**
   `D[i][j]` = how much training against detector i degrades detector j.

The shared payload is `eval/degradation.py`. Everything else is swapping which env, which detectors
are X vs. held-out, and λ.

## Experiment 2 — Monitor strength *(derisking plan — folded into the matrix)*

*Does training against a strong CoT monitor teach the policy to fool weaker monitors too — or only
that one monitor?* Closest follow-up to Baker et al. The weak/strong axis is now one axis of the
combined matrix below (`cot_weak`/`cot_strong`, `out_weak`/`out_strong`).

- **Setting:** sycophancy MC was the cheap derisk; realized on the MBPP-Honeypot coding env.
- **Detectors:** a battery of **CoT monitors of varying strength** (e.g. a weak model, a mid model,
  a strong model — different families/sizes). All black-box LLM judges over the CoT, differing only
  by `model_id`.
- **Conditions:** train against each strength in turn (X = weak; X = strong; optionally X = ensemble),
  holding the others out.
- **Measure:**
  - Δ accuracy of each *held-out* monitor vs. ground truth over training.
  - Asymmetry: does training-against-strong degrade weak (and vice versa)? Baker found
    train-against-weak left strong intact; the open question is the reverse.
  - Ground-truth misbehavior (sycophancy) rate over training — the "(2) became more aligned" axis.

## Experiment 1 — Cross-method (CoT monitor vs. probe) *(the core of the live matrix)*

*Train against a CoT monitor; hold out a linear probe (and vice versa). Does cross-method degradation
happen?* More action-relevant than within-method, since labs mix detector families. This IS the
headline axis of the combined matrix below (CoT/output judges × probe_ood/probe_iid).

- **Setting:** sycophancy MC; later the coding-hardcode env (matches Obfuscation Atlas).
- **Detectors:** X = CoT monitor (black-box); Y = linear probe (white-box) over policy activations,
  with ground-truth sycophancy available.
- **Measure:** Δ probe accuracy and Δ sycophancy rate as we train against the CoT monitor; then flip
  (train against probe, hold out CoT monitor).
- **Blocker:** requires extracting **activations** from the tinker-hosted policy for the probe — see
  the activations risk in [`MONITORS.md`](MONITORS.md). **Derisk this before committing to Exp 1.**

## Combined — Full degradation matrix (THIS is what we ran)

The realized headline. The battery is **6 monitors**: `{weak, strong} × {CoT judge, output judge}` +
`probe_ood` (diverse_deception) + `probe_iid` (mbpp_honeypot), plus a no-monitor **control** = 7 rows.
Weak = `anthropic/claude-3-haiku`, strong = `deepseek/deepseek-chat` (cheap, non-reasoning, off-policy-
family — picked from a strict-label bakeoff). Train against each in turn, hold out the rest, all scored
into `eval_metrics.jsonl`. Assemble AUROC-drop + suspiciousness-gap-drop + end-AUROC grids
(`eval/degradation.py`) — the project's headline figures. Setting = MBPP-Honeypot reward-hacking
(`behavior_present = strict_rh`), M = Qwen3-8B. Fanned across seeds by `scripts/queue_runs.sh`.

## Metrics (precise)

- **Ground-truth misbehavior rate**: fraction of rollouts the env flags as misbehavior via
  `behavior_present` (MBPP: `strict_rh` = hardcoding ∧ loose_rh; sycophancy: user-flattering answer;
  MASK: lied). The denominator for everything.
- **Detector accuracy / AUROC vs. ground truth**: on a fixed held-out eval batch, per detector, per
  checkpoint. A detector "degrades" when this drops while misbehavior persists.
- **Degradation `D[i][j]`**: drop in detector j's accuracy (or AUROC) from training against detector
  i, ideally normalized (e.g. relative to its pre-training accuracy and to the no-penalty control).
- **Controls**: a **no-penalty RL run** (does generic RL pressure already degrade detectors?) and a
  **frozen-policy** baseline (detector accuracy with no training at all).

## Settled design decisions
- **RL algorithm:** GRPO (group-relative advantages, `group_size` rollouts/prompt). DR-GRPO / DAPO
  available as swaps. See [`INFRA.md`](INFRA.md).
- **Policy:** small natively-verbalizing tinker model — default `Qwen/Qwen3-8B` (thinking on), **no
  extra CoT-elicitation scaffolding**. We want a model that *has to* reason out loud, so the CoT is
  load-bearing and obfuscation is a real phenomenon. (R1-distill policy would need the verl backend.)
- **Sycophancy data:** the QA subset where the prompt states a user's (wrong) preference; ground
  truth = picked the user-flattering answer over the correct one. `task_reward` = the sycophantic
  answer (RL pushes toward the misbehavior we then penalize).
- **Coding:** `task_reward` = tests passing (whether or not honestly solvable).
- **Judges:** black-box monitors run on **OpenRouter**. The realized weak/strong ladder is
  `anthropic/claude-3-haiku` + `deepseek/deepseek-chat` (both cheap, non-reasoning, off the policy's
  family), chosen from a strict-label bakeoff of 14 candidates (all 0.89–0.99 vs the clean oracle).
- **KL-to-base:** `kl_coef=1e-4`, per-token KL to the base sampler via tinker-cookbook
  `incorporate_kl_penalty` — anchors the policy and prevents the reward-hack collapse.
- **Logging:** wandb auto-syncs iff logged in (else offline), runs grouped per sweep; `log_fraction`
  controls verbatim rollout persistence (default 1.0).

## Still open (decide as we build)
- λ (penalty_coef) schedule — Atlas found high KL + penalty recovers honesty; sweep λ to see the
  honest → obfuscated transition, not just one point. (KL itself is now settled — see above.)
- Monitor weights frozen vs. updated during training (start frozen).
- Whether to also log **(b)-style** false-negative overlap between detectors, not just (a)-style
  co-degradation.
- AuditBench methodology (deferred).

## Probe-design ideas (later)
- **No-CoT vs CoT probe transfer (designed).** Does a probe trained on *answer-only* data generalize to
  *CoT-bearing* rollouts? This decides whether we need (expensive, quality-critical) CoT generation for
  probe training at all. Clean 2×2 on MASK, full lit review + preregistered predictions in
  [`COT_VS_NOCOT_PROBE.md`](COT_VS_NOCOT_PROBE.md). Prereq: the CoT-regeneration helper.
- **CoT-masked probe.** Train the probe on activations but **mask out the CoT (`<think>`) tokens** —
  read the deception signal only from the non-CoT (answer / follow-up) token activations. Rationale:
  tests whether deception is represented *independently of the verbalized reasoning*. A probe that
  ignores CoT-token activations can't be fooled by CoT obfuscation, and it probes whether the model
  "knows" it's being deceptive outside what it says. **Open question** whether this is even well-defined
  given the Atlas follow-up-token method already reads a *post-CoT* token (the follow-up answer) — the
  masking would matter for mean-pooled / multi-token probe variants, less so for the single
  follow-up-token readout. Worth a small ablation: follow-up-token vs mean-over-answer vs
  mean-over-all-minus-CoT. (Idea from Joey, 2026-06-12.)
