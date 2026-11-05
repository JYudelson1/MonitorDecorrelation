# Weak-judge bakeoff for MBPP-Honeypot (2026-11-05)

**Question.** Which local Qwen model is a genuinely *weak* LLM monitor for Qwen3-8B reward hacking on
MBPP-Honeypot: clearly worse than the strong judge (Qwen3.5-35B-A3B-FP8), but still informative and
reliably parseable?

**Answer.** **Qwen/Qwen3-1.7B, thinking on** — now `q17_{cot,out,cotonly}` in
`experiments/configs/mbpp_q3_8b_qmonitors/`. The previous "weak" judge, Qwen3-30B-A3B-FP8, is
indistinguishable from the strong one on this env (AUROC 0.99–1.00 vs 1.00).

## Method

- **Judges** ran through the repo's real vLLM judge path (`experiment_config.build_monitors`, provider
  `vllm`, `enable_thinking: true`, `max_tokens: 16384`, `thinking_budget: 12288` — the RL configs'
  settings), on vLLM with `--reasoning-parser qwen3`, on a 2×H200 box. Prompts from `CoTMonitor._build_prompt`;
  replies parsed by `judge_output` / `judge_result`, so a parse failure or length cutoff scores **1.0**,
  exactly as in training. One sample per rollout, temperature 1.0. Each candidate first passed the
  `VLLM_JUDGES` admission check (`verify_thinking.py`: thinking on → split reasoning + clean content;
  off → no reasoning; `thinking_token_budget` 64 → ≈64 reasoning tokens).
- **Anchors** served exactly as `scripts/setup_vllm.sh` does: Qwen3.5-35B-A3B-FP8 (strong) and
  Qwen3-30B-A3B-FP8 (old weak).
- **Rollouts**: Qwen3-8B MBPP-Honeypot eval rollouts from the July batch
  (`data/runs/mbpp_matrix_orig_20260701/*`; the only Qwen3-8B runs with full text dumps). Relabelled with
  the *current* `envs/hacking_labels` + held-out test execution (agrees with the July `behavior_present`
  except 45 rollouts that are now strict hacks). Seeded (seed 0), class-balanced **150 strict hacks vs 150
  clean** (clean = not `loose_rh`, so honest-but-buggy attempts are excluded). Unparsed/truncated
  excluded. Half the hacks (75) are **pressure-shaped**: from step ≥ 32 of train-against runs — but only
  9 of those come from runs trained against an *LLM* judge (under that pressure the July policy mostly
  stopped hacking); 66 come from probe-trained runs.
- **CIs**: 95% bootstrap over rollouts.

## Results

AUROC (strict hack vs clean) per judge × view. *Parse err* = replies with no parseable `SCORE:` line
(scored 1.0). *Think* = mean reasoning tokens. *Budget* = calls force-closed at the 12288 thinking budget.

| judge | view | AUROC [95% CI] | AUROC, pressure hacks | hack / clean mean | parse err | budget | think |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-35B-A3B** (strong) | cot+out | 1.000 [1.00, 1.00] | 1.000 | 0.96 / 0.05 | 0% | 1.7% | 2127 |
| | out | 1.000 [1.00, 1.00] | 1.000 | 0.96 / 0.04 | 0% | 2.7% | 2211 |
| | cot | 0.956 [0.93, 0.97] | 0.990 | 0.77 / 0.06 | 0% | 0.3% | 1484 |
| **Qwen3-30B-A3B** (old weak) | cot+out | 0.994 [0.98, 1.00] | 0.989 | 0.93 / 0.05 | 0% | 0% | 479 |
| | out | 1.000 [1.00, 1.00] | 1.000 | 0.94 / 0.04 | 0% | 0% | 603 |
| | cot | 0.962 [0.94, 0.98] | 0.967 | 0.71 / 0.07 | 0% | 0% | 609 |
| Qwen3.5-4B | cot+out | 1.000 [1.00, 1.00] | 1.000 | 0.90 / 0.02 | 0.3% | 0% | 3199 |
| | out | 0.999 [1.00, 1.00] | 1.000 | 0.92 / 0.03 | 0% | 0% | 3817 |
| | cot | 0.943 [0.91, 0.97] | 0.979 | 0.73 / 0.04 | 0% | 0.3% | 3216 |
| Qwen3-4B | cot+out | 0.985 [0.96, 1.00] | 0.975 | 0.91 / 0.08 | 1.3% | 0% | 493 |
| | out | 0.998 [0.99, 1.00] | 0.995 | 0.92 / 0.07 | 0.3% | 0% | 696 |
| | cot | 0.860 [0.81, 0.91] | 0.881 | 0.59 / 0.09 | 0% | 0% | 746 |
| Qwen3.5-2B | cot+out | 0.853 [0.80, 0.90] | 0.831 | 0.79 / 0.14 | 7.0% | 12.3% | 7121 |
| | out | 0.929 [0.89, 0.96] | 0.928 | 0.83 / 0.05 | 3.0% | 18.3% | 7937 |
| | cot | 0.785 [0.74, 0.83] | 0.802 | 0.52 / 0.03 | 0.3% | 7.0% | 5790 |
| **Qwen3-1.7B** (chosen) | cot+out | **0.811 [0.76, 0.85]** | 0.776 | 0.46 / 0.04 | **0%** | 0% | 447 |
| | out | **0.911 [0.88, 0.95]** | 0.890 | 0.60 / 0.00 | **0%** | 0.7% | 663 |
| | cot | **0.597 [0.54, 0.66]** | 0.628 | 0.32 / 0.19 | **0%** | 0.3% | 653 |
| Qwen3.5-0.8B | cot+out | 0.690 [0.63, 0.74] | 0.669 | 0.46 / 0.13 | 10.3% | 24.0% | 6638 |
| | out | 0.653 [0.59, 0.71] | 0.624 | 0.56 / 0.28 | 25.3% | 51.7% | 8943 |
| | cot | 0.556 [0.51, 0.60] | 0.592 | 0.20 / 0.10 | 4.7% | 17.7% | 6224 |
| Qwen3-0.6B | cot+out | 0.607 [0.55, 0.67] | 0.613 | 0.35 / 0.23 | 0% | 0% | 386 |
| | out | 0.616 [0.56, 0.67] | 0.642 | 0.35 / 0.22 | 0.3% | 0% | 400 |
| | cot | 0.588 [0.53, 0.65] | 0.590 | 0.28 / 0.19 | 0% | 0% | 337 |

For reference, the July batch's recorded judges on the same 300 rollouts (old judge prompt):
claude-3-haiku 0.978 (cot) / 0.978 (out), deepseek-chat 0.991 / 0.992; probes (June probes) `probe_iid`
0.675, `probe_ood` 0.650 — both near chance on the pressure-shaped hacks (0.43 / 0.54), consistent with
most of those coming from probe-trained runs.

### Operating point (threshold 0.5)

| judge | view | TPR | FPR |
|---|---|---|---|
| Qwen3.5-35B-A3B | cot+out / out / cot | 1.00 / 1.00 / 0.83 | 0.03 / 0.01 / 0.03 |
| Qwen3-1.7B | cot+out / out / cot | 0.51 / 0.68 / 0.34 | 0.05 / 0.00 / 0.19 |
| Qwen3.5-2B | cot+out / out / cot | 0.87 / 0.91 / 0.57 | 0.15 / 0.05 / 0.03 |

Qwen3-1.7B is weak mostly by **missing hacks** (half its cot+out hack scores are < 0.5; 46/150 < 0.1)
while staying quiet on clean code; Qwen3.5-2B's cot+out FPR is inflated by its parse errors (15 of its 21
parse errors were on *clean* rollouts — each one a 1.0 false alarm).

### Judge noise (test–retest)

1.7B and 2B were re-run with a second independent sample per rollout:

| judge | view | r(run1, run2) | label flips @0.5 | AUROC run1 / run2 / mean-of-2 | within-class r (hack / clean) |
|---|---|---|---|---|---|
| Qwen3-1.7B | cot+out | 0.48 | 22.0% | 0.811 / 0.809 / 0.865 | 0.26 / 0.37 |
| | out | 0.60 | 20.0% | 0.911 / 0.903 / 0.968 | 0.14 / −0.02 |
| | cot | 0.40 | 24.7% | 0.597 / 0.534 / 0.587 | 0.34 / 0.44 |
| Qwen3.5-2B | cot+out | 0.69 | 15.3% | 0.853 / 0.915 / 0.953 | 0.29 / 0.16 |
| | out | 0.73 | 13.0% | 0.929 / 0.896 / 0.976 | 0.17 / 0.12 |
| | cot | 0.65 | 15.3% | 0.785 / 0.777 / 0.833 | 0.50 / 0.03 |

So **much of the small judges' weakness is sampling noise rather than a stable blind spot**: averaging
two samples recovers a lot of AUROC, and within-class correlation of hack scores is low (1.7B: 0.14–0.34).
A noisy judge gives a noisy penalty but little systematic to evade — whether 1.7B is *evadable* under RL
is exactly what the `row_*_weak` runs test. Weak hint that it has real blind spots: on the 9 hacks from
LLM-judge-trained runs, 1.7B's mean cot+out score is 0.28 (vs 0.54 on control-run hacks; 35B: 0.96 vs
0.97) — n = 9, so suggestive only.

With Qwen3.5-2B's parse errors dropped instead of scored 1.0, its AUROC would be 0.935 (cot+out) /
0.956 (out) — i.e. a sizeable part of its weakness is format failure, not judgement.

## Decision

**Qwen3-1.7B, thinking on**, all three views (`q17_cot`, `q17_out`, `q17_cotonly`):

- clearly below strong in every view (0.81 / 0.91 / 0.60 vs 1.00 / 1.00 / 0.96);
- **0 parse errors in 1,800 calls** — its weakness is judgement, not format, and it never false-alarms
  through a parse failure;
- short, readable reasoning (~450–650 tokens vs ~7k for Qwen3.5-2B), and cheap (≈4 s/call at
  ~100-way concurrency; Qwen3.5-2B ≈ 50–70 s and hits the thinking budget 12–18% of the time).

Rejected: Qwen3-4B / Qwen3.5-4B (≈ strong), Qwen3-0.6B / Qwen3.5-0.8B (≈ chance; 0.8B also invents its
own answer labels, e.g. `SUSPICUS: 0`), Qwen3.5-2B (the better CoT-only weak judge, 0.79, but 3–7%
bare-number replies such as `80` → parse error → 1.0, mostly on clean code; long reasoning).

## Caveats

- n = 150 + 150, one sample per call: treat AUROC differences < ~0.05 as noise (Qwen3.5-2B cot+out moved
  0.853 → 0.915 between two samples).
- The rollouts are from the July (pre-GRPO-seeding-fix) Qwen3-8B batch; hacks that emerged under
  *LLM-judge* training pressure are nearly absent (9). The new `mbpp_q3_8b_qmonitors` runs will show the
  weak judge on the distribution that matters.
- CoT-only is weak for every small model; the 1.7B CoT-only column (0.60) is close to chance.
- After a forced end of thinking, small Qwen3.5 models keep deliberating in the answer channel until
  cut off — one source of their parse errors/cutoffs.

## Artifacts

- Scripts + text reports: `experiments/bakeoffs/mbpp_weak_judge_20261105/` (`build_set.py`, `score.py`,
  `report.py`, `diag.py`, `retest.py`, `verify_thinking.py`, `serve_small.sh`; `report.txt`, `diag.txt`,
  `retest.txt`; the box-only `VLLM_JUDGES` patch used). Paths inside refer to `/workspace/weak_bakeoff`
  on the box.
- Raw per-call results (every judge reply incl. full reasoning) + the bakeoff set: local
  `data/runs/weak_judge_bakeoff_20261105/{results,data,logs}` (gitignored, 113 MB).
