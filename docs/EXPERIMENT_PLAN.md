# Experiment Plan

Concrete designs for the derisking week and the experiments that follow. Background lives in
[`RESEARCH_CONTEXT.md`](RESEARCH_CONTEXT.md). Envs in [`ENVIRONMENTS.md`](ENVIRONMENTS.md), monitors
in [`MONITORS.md`](MONITORS.md). Build state tracked in [`../STATUS.md`](../STATUS.md).

## ⭐ Ground truth is the crux — read this first

Training against monitor X does two things: **(1)** pushes the policy off the behavior X detects, and
**(2)** breaks X as a detector. We're testing **(3)** whether held-out monitors Y, Z break too. But
none of (1)–(3) is interpretable without continuously tracking the **ground-truth behavior rate**.

The signature we expect (sycophancy example): the ground-truth sycophancy rate **dips** under monitor
pressure, then **climbs back up** as the policy learns to stop *verbalizing* the behavior rather than
to stop *doing* it. A monitor's accuracy dropping is ambiguous on its own — it could mean *less
behavior* or *more hidden behavior*. Only the ground-truth curve disambiguates.

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

## Experiment 2 — Monitor strength (FIRST)

*Does training against a strong CoT monitor teach the policy to fool weaker monitors too — or only
that one monitor?* Closest follow-up to Baker et al.; chosen as the first derisk.

- **Setting:** sycophancy MC first (cheapest signal), then a coding env.
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

## Experiment 1 — Cross-method (CoT monitor vs. probe)

*Train against a CoT monitor; hold out a linear probe (and vice versa). Does cross-method degradation
happen?* More action-relevant than within-method, since labs mix detector families.

- **Setting:** sycophancy MC; later the coding-hardcode env (matches Obfuscation Atlas).
- **Detectors:** X = CoT monitor (black-box); Y = linear probe (white-box) over policy activations,
  with ground-truth sycophancy available.
- **Measure:** Δ probe accuracy and Δ sycophancy rate as we train against the CoT monitor; then flip
  (train against probe, hold out CoT monitor).
- **Blocker:** requires extracting **activations** from the tinker-hosted policy for the probe — see
  the activations risk in [`MONITORS.md`](MONITORS.md). **Derisk this before committing to Exp 1.**

## Combined — Full degradation matrix

Once both above work: run the full **{weak CoT, strong CoT, linear probe}** battery (and later
output classifier, activation oracle, behavioral eval), training against each in turn and measuring
all others. Assemble the complete pairwise degradation matrix — the project's headline figure.

## Metrics (precise)

- **Ground-truth misbehavior rate**: fraction of rollouts the env flags as misbehavior (sycophantic
  answer / hardcoded test / spec violation). The denominator for everything.
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
- **Judges:** black-box monitors run on **OpenRouter** (has all the Claudes + wide family/size range
  for the strength ladder).
- **Logging:** local/offline wandb; `log_fraction` controls verbatim rollout persistence (default 1.0).

## Still open (decide as we build)
- λ schedule and KL regularization — Obfuscation Atlas found high KL + penalty recovers honesty;
  sweep λ to see the honest → obfuscated transition, not just one point.
- Monitor weights frozen vs. updated during training (start frozen).
- Whether to also log **(b)-style** false-negative overlap between detectors, not just (a)-style
  co-degradation.
- AuditBench methodology (deferred).
