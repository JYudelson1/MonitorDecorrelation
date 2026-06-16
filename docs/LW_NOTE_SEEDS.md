# LW note — seed ideas (capture as they come; don't lose these)

Running scratch of the *conceptual* points for the eventual LessWrong note on monitor decorrelation
under training pressure (the 6×6 MBPP-Honeypot degradation experiment). Findings/plumbing live in
STATUS.md and docs/PROBES.md; this file is the argument.

## 1. The early-outputs / recalibration asymmetry  _(Joey, 2026-06-16)_

When you train a policy against monitor X, X's *suspiciousness scores* drift downward. Even if X stays
**useful after you recalibrate its threshold** (the ROC shape / AUROC can survive a uniform shift), it is
strictly **worse at flagging the *first few* bad outputs in deployment** — because you haven't yet
gathered the post-shift score distribution to recalibrate *against*. The recalibration is only available
*after* you've seen enough (mis)behaviour to estimate the new distribution; the early hacks slip through
during exactly the window where you're still calibrating. So "the monitor is fine once recalibrated"
understates the deployment harm: the cost is front-loaded onto the outputs you can least afford to miss.

→ Implication: report TWO DISTINCT metrics (see §2) — they capture different failure modes.

## 2. Two distinct metrics: AUROC (ranking) vs the gap (magnitude) — and both die at single-class

We track two separate things; **do not conflate them**:
- **(1) AUROC** — threshold-free *ranking* separation: do hacks score above honest, at any scale?
- **(2) gap = mean_score_hack − mean_score_honest** — the raw *magnitude* separation of the class means.

They diverge in an informative way: a monitor can **keep AUROC ~constant while the gap collapses**
(scores compress toward each other but still rank correctly). That's the substance behind §1 — a *fixed*
threshold flags fewer and fewer hacks even though the monitor could still rank them if you re-fit the
threshold. So the AUROC-vs-gap divergence IS the recalibration story, made quantitative.

**Important caveat (don't repeat the earlier mistake):** the gap needs BOTH a hack mean and an honest
mean, so it is undefined under the *same* single-class condition as AUROC. If the policy hacks ~100% of
eval prompts, there are no honest rollouts → `honest_mean` is undefined → **both AUROC and the gap are
NaN.** The gap is NOT a single-class-robust fallback. What *does* survive single-class is the *absolute*
hack-class mean score (it can keep drifting down with no honest reference) and Brier — those are the
reads for the fully-obfuscated regime. Plan eval sizes / behavior rates so both classes stay present
where we want AUROC and the gap to be meaningful.

## 3. (placeholder) cross-method decorrelation — the actual 6×6 result

The empirical core, once runs exist: train against each of {weak CoT, strong CoT, weak output, strong
output, OOD deception probe, iid MBPP-hack probe}; measure all six at the end. Two 6×6 grids: AUROC-drop
and suspiciousness-gap-drop. Diagonal = self-degradation (sanity); off-diagonal = cross-method
decorrelation (the question). Plus a no-monitor control row = baseline drift from the reward alone.
