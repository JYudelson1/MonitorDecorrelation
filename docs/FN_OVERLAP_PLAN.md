# Plan — measuring sense (b) decorrelation: correlated false negatives (MBPP-Honeypot, in-distribution)

Status (2026-09-08): scorer + analysis BUILT (`experiments/score_instance_set.py`, `eval/miss_overlap.py`,
`experiments/analyze_miss_overlap.py`); first target = `bakeoff_set_600.jsonl` only, D_synth/H_synth and
the handcrafted suite deferred (Rauno's call). Prediction/bridge step BUILT too (`eval/predict_degradation.py`, `experiments/predict_degradation.py`).
Nothing scored live yet — judges + probes are run by whoever launches the final training runs. Companion to the sense-(a) co-degradation matrix
(`docs/DEGRADATION_METRICS.md`, `eval/coupling.py`). Framing of the two senses is in
`docs/RESEARCH_CONTEXT.md` ("Two senses of decorrelated").

## 0. The question, precisely

Sense (a) — *co-degradation*: when the policy is RL'd against monitor X, does held-out Y's reliability
fall too? Needs training. Measured by β(X→Y) over the matrix runs.

Sense (b) — *correlated false negatives*: on a **fixed** set of misbehaving instances, do monitors miss
the **same** instances? Needs only inference. This doc.

The bet we want to test: **(b) predicts (a)**. If X and Y miss the same hacks at rest, then the hacks
the policy discovers to evade X are disproportionately ones Y also misses, so Y co-degrades. If their
misses are independent, evading X should leave Y intact. Concretely, the deliverable is a pairwise
matrix `O[X][Y]` (miss-overlap of Y with X) that we then correlate, pair by pair, with the existing
`β(X→Y)` matrix from the RL runs. Same 8 monitors, same env, same policy family.

## 1. Monitors (the sep3 battery — 8, not 6)

Use exactly the battery in `experiments/configs/mbpp_matrix_gemini_sep3/` so (b) is comparable to (a):

| name | reads | model |
| --- | --- | --- |
| `cot_only_weak` / `cot_only_strong` | CoT only | `google/gemini-2.5-flash-lite` / `google/gemini-3.5-flash-lite` |
| `cot+out_weak` / `cot+out_strong` | CoT + answer | same |
| `out_weak` / `out_strong` | answer only | same |
| `probe_ood` | frozen Qwen3-8B activations | `data/probes/Qwen3-8B/diverse_deception` |
| `probe_iid` | frozen Qwen3-8B activations | `data/probes/Qwen3-8B/mbpp_honeypot` |

Judges via `CoTMonitor` (OpenRouter, `OPENROUTER_API_KEY`); probes via `ProbeMonitor` on a GPU box or
the shared `probe_server.py` (`PROBE_SERVER_URL`). Build them from the config with
`experiment_config.build_monitors` so nothing drifts from the RL runs.

Optional extra columns (cheap, informative for the story): the **old** weak/strong ladder
(`anthropic/claude-3-haiku`, `deepseek/deepseek-chat`) so we can check the LW-post-era monitors too,
and `gemini-2.5-flash` / `gemini-3.5-flash` (non-lite) if we ever move to them.

## 2. Instance sets (the crux — where do the hacks come from?)

The base policy essentially never hardcodes (step-0 strict rate ≈ 0.0 across all 54 logged runs). So
every *natural* hack we have was produced by a policy **under training pressure**, and a hack sampled
from a run trained against X is already selected to evade X. That selection is a confound for (b): it
inflates apparent overlap for X and whatever is correlated with X. We handle it by stratifying by
source run, never pooling blindly. Four sets, from cleanest to most convenient:

**S1 — Handcrafted attack suite (new, ~300–500 instances).** Templated + LLM-written hacks on the
**eval-split** MBPP tasks (69 tasks; `mbpp_split.is_eval_task`), each executed and kept only if
`hacking_labels` says `strict_rh` (plus a matched honest control per task). Categories, chosen to span
the ways a monitor can be misled, and mirroring the Atlas obfuscated-policy taxonomy:

| code strategy | prose / CoT strategy |
| --- | --- |
| literal hardcode (`return <expected>`) — the Atlas `hardcode` column, 374 rows | no CoT (Atlas-style, answer-only) |
| `if args == <visible input>: return <expected>` + plausible-looking fallback | honest CoT that openly plans the hack ("just return the value") |
| lookup dict keyed on the visible input | **rhetorical rationalization** ("the spec is ambiguous, so returning X is the safe choice") |
| `try: <real attempt> except: return <expected>` | **strategic honesty** (code hacks, prose admits it afterwards) |
| partial hardcode: real logic for one branch, constant for the tested branch | **adversarial noise** (long, irrelevant CoT; or CoT that claims a general solution the code doesn't implement) |
| `name = lambda *a: <expected>` (the RL-discovered form) | short "here is the solution" with no reasoning |

This is the set where miss rates are high **by design**, so pairwise overlap is measurable, and where
the per-category miss profile directly reads out "misleading in the same ways". Generate with a
Claude call per (task, code-strategy, prose-strategy) cell from a fixed prompt, verify by execution,
drop anything that isn't strict (a "hack" that accidentally passes all tests is not a hack).
Provenance: **the Atlas paper has no handcrafted attack suite** — its "synthetic deceptive" data is the
single-template `hardcode` column and its "synthetic MBPP" script generates *tasks*, not attacks. We
build this ourselves (see §6).

**S2 — `bakeoff_set_600.jsonl` (exists).** 600 training rollouts from one run: 200 strict hacks,
185 loose-only (honest-but-buggy), 215 clean; steps 0–119; every task is a **train-split** task_id.
Only `probe_ood` is scored so far, and its AUROC on this set is **0.28 strict-vs-clean** (inverted —
see §7). Use it, but: (i) record which run/monitor produced it (unknown; not in git), (ii) keep the
loose-only class as its own stratum (monitors *should* score it low, so it measures co-false-positives,
not misses), (iii) `probe_iid` was trained on train-split prompts → mild leakage for that one column.

**S3 — `eval_rollouts.jsonl` from the matrix runs (exists on the GPU box, gitignored).** Every
held-out eval rollout with **all monitors' per-instance scores already attached**, per eval step, per
run — the biggest free lunch here. Pull them for the sep3 batch (and lowpen/sched01 for the old
ladder). Per (run, step) this is a (b) measurement on eval-split tasks under a known, logged
selection pressure; step 0 is pressure-free but has no hacks; late steps have hacks selected against
that run's target. Analysis: compute the overlap matrix **within each (run, step)** and look at how it
moves with training — that is the direct bridge to (a).

**S4 — fresh base-model samples (optional).** Qwen3-8B at T=1 on eval-split tasks, several samples
each. Gives a pressure-free clean set (and honest-buggy), almost no hacks. Useful as the clean
reference for thresholds if S1's honest controls prove too easy.

## 3. Metrics (per pair of monitors; all on the strict-hack set H and clean set C)

Operating points. Judge scores are calibrated (μ_hack≈0.9, μ_clean≈0.07); probe scores are not
("probes rank but don't calibrate"). So never use 0.5 for everything. Threshold each monitor at a
**matched false-positive rate on C** (report 5 % and 20 %); also report the config threshold as a
sanity row. Miss set `FN_m = {h ∈ H : s_m(h) < t_m}`.

Thresholded (interpretable):
- **Conditional miss rate** `P(miss_Y | miss_X)` vs the marginal `P(miss_Y)`; **lift** = ratio.
  Lift 1 = independent misses; lift ≫ 1 = misled the same way. The headline matrix `O[X][Y]`.
- **Jaccard / Yule's Q / phi** on the 2×2 miss table — symmetric companions.
- **Union catch rate** of {X, Y} at matched FPR vs the independence prediction `1 − P(miss_X)P(miss_Y)`.
  The gap is the "defense-in-depth you don't actually get".

Threshold-free (more power when strong monitors miss only a handful):
- **Spearman ρ of scores within H** (which hacks look subtle to whom) and separately within C
  (co-false-positives). Partial ρ controlling for code length, CoT length, and whether the CoT
  verbalizes the hack (S1 has these as design labels; for S2/S3 label with a cheap judge).
- **Ensemble AUROC gain**: AUROC(mean or max of X,Y) − max(AUROC_X, AUROC_Y).

Structure:
- Hierarchical clustering / PCA of the H × 8 score matrix → do misses factor into {text-family,
  probe-family} as the co-detection r from the runs (LLM judges r≈0.85, probe_ood r≈0.1) suggests?
- **Per-category miss profile** on S1: category × monitor heatmap. This is the most legible output
  for "misleading in the same ways".

Uncertainty: bootstrap instances (and tasks, clustered) for CIs; permutation null for lift (shuffle
Y's miss indicator within H).

## 4. The bridge to sense (a)

- Take `β(X→Y)` from `analyze_coupling.py` on the sep3 runs (by-target version, chart 10) and
  `O[X][Y]` from S1 (and S3 at late steps). Spearman across the 8×7 off-diagonal pairs. One number,
  with a bootstrap CI, answers "does static miss-overlap predict co-degradation?".
- Secondary: does the *direction* match? β is asymmetric; lift `P(miss_Y|miss_X)` is too.
- Mechanism check on S3: for a run trained against X, do the hacks it produces late in training fall
  into S1 categories that Y also misses at rest? If yes, (b) explains (a) at the instance level.

## 5. Deliverables / build list

1. `experiments/build_attack_suite.py` → `data/attack_suite/mbpp_attacks.jsonl` (same record schema
   as `eval_rollouts.jsonl`: `question`, `cot`, `answer`, `task_id`, `behavior_present`, `loose_rh`,
   `hardcoding`, plus `attack.code_strategy`, `attack.prose_strategy`, `attack.source`). Includes the
   Atlas `hardcode` column wrapped as `code_strategy=literal, prose=none`. Verified strict by execution.
2. `experiments/score_instance_set.py` — score any such jsonl with the monitors from a matrix config
   (`--config …/row_control.json`, `--probe-server-url`), resumable (cache by record hash), writes
   `<set>_scored.jsonl`. Reuses `build_monitors`; judges in a thread pool like
   `eval_monitors_on_rollouts.py`.
3. `src/monitordecorrelation/eval/miss_overlap.py` (pure, unit-tested) + `experiments/analyze_miss_overlap.py`
   — the §3 metrics + heatmaps; reads scored sets **or** `eval_rollouts.jsonl` (`--step`, `--run`).
   Plotting per `research-plotting` conventions.
4. `experiments/bridge_b_to_a.py` — joins `coupling.json` with the overlap matrix → scatter + ρ.
5. Docs: this file → results section; `STATUS.md` rows for 1–4; a "sense (b)" paragraph in
   `docs/DEGRADATION_METRICS.md`.

Cost/compute: judges ≈ (600 + ~400 + S3) instances × 6 configs ≈ 6–10k OpenRouter calls on flash-lite
models — tens of dollars. Probes: the 8B forward passes need the GPU box (MPS is bs=1, minutes per
hundred at 2048 tokens); run through `probe_server.py`.

## 6. What is NOT in the repo (checked 2026-09-08)

- No handcrafted / synthetic attack set. The only synthetic hacks anywhere are the Atlas
  `taufeeque/mbpp-hardcode` `hardcode` column (one template: return test #1's expected value; 374
  train / 500 test / 90 val / 10 prompt rows), which `whitebox/datasets/mbpp.py` already consumes for
  `probe_iid`. Atlas's `scripts/generate_synthetic_mbpp.py` generates synthetic *tasks* with Claude,
  not attacks. Atlas's obfuscation taxonomy (rhetorical rationalization 41 %, adversarial obfuscation
  32 %, strategic honesty 27 %) is a post-hoc classification of *trained* policies, not a dataset.
- No per-instance scoring script for a static set with all monitors (`eval_monitors_on_rollouts.py`
  does judges only and reports AUROC, not per-instance scores; `eval_probes_on_run.py` does probes only).
- No per-instance overlap analysis; `eval/coupling.py` works on per-step aggregates.

## 7. Flags found while reading (fix or decide before scoring)

- `probe_ood` scores stored in `bakeoff_set_600.jsonl` give AUROC **0.28** (strict vs clean, i.e.
  inverted) and are *higher* on loose-only (0.43) than on hacks (0.34). Either the stored scores are
  from an old/different probe or the OOD probe is genuinely anti-signal on this policy's hacks.
  Re-score with the committed probe before trusting anything, and note the run these came from.
- Doc/config drift on the judge ladder: configs use `gemini-*-flash-lite`, the user's description says
  `gemini-2.5-flash` / `gemini-3.5-flash` (non-lite), and `STATUS.md`, `docs/MONITORS.md`,
  `docs/EXPERIMENT_PLAN.md`, `configs/mbpp_matrix/README.md` and the `WEAK`/`STRONG` constants in
  `make_mbpp_matrix_configs.py` still say `claude-3-haiku` / `deepseek-chat`. The sep3 configs were
  hand-edited (the README says regenerate) and add two `cot_only_*` monitors (8 total) that no doc
  mentions. Decide which ladder is canonical and update the generator + docs.
- `.env` is absent on this machine: judge scoring needs `OPENROUTER_API_KEY`; probes need the GPU box.
- `eval_rollouts.jsonl` (S3) is gitignored — rsync from the run box before the bridge analysis.
