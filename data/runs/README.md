# Runs

Saved RL runs for the Monitor Decorrelation project.

**Rollout dumps are not committed.** Each run dir holds the per-step metrics (`eval_metrics.jsonl`,
`metrics.jsonl`), config (`run_info.json`, `config.json`), the run log, and `train/` + `eval/` plots.
The large per-rollout files (`rollouts.jsonl`, `eval_rollouts.jsonl` — the full generated text +
per-rollout scores) live only on the training box / locally. The committed **`eval_metrics.jsonl`
already carries every aggregate the analysis needs** — per-monitor AUROC / d′ / μ_hack / class-mean
scores per eval step — so every coupling and degradation chart reproduces from a clean checkout.

## MBPP-Honeypot matrix (the headline experiment)

Policy = Qwen3-8B, trained to reward-hack while evading one monitor per run; the rest are held out and
scored into `eval_metrics.jsonl`. Rows = train-against `{cot_weak, cot_strong, out_weak, out_strong,
probe_iid, probe_ood}` + a no-monitor control. The batches differ only in the **monitor-penalty λ**:

- **`mbpp_matrix_orig_20260701/`** — the **first full set of 35 runs** (7 rows × 5 seeds), constant
  penalty **λ = 1.0**. The honest/high-penalty regime (policy mostly never fully fools the monitors).
- **`mbpp_matrix_lowpen_20260706/`** — constant **λ = 0.5** (11 runs). The obfuscation regime: enough
  slack that the policy hacks *and* learns to evade.
- **`mbpp_matrix_lowpen_small_20260706/`** — **λ = 0.5**, 60-step `cot_weak` runs (3 seeds). Includes
  the **`cot_weak_s1` case study**, where the weak judges are genuinely fooled (μ_hack → ~0) while the
  strong judges are not.
- **`mbpp_matrix_sched01_20260707/`** — penalty **schedule λ: 0.0 → 1.0** linearly increasing (in a curriculum, teaching the models first to hack, then ideally to hide their hacks), strong-monitor runs (`cot_strong`, `out_strong`; 6 runs, seeds 0–2).

- **`mbpp_matrix_sep3_20260903/`** — **λ = 0.5**, 60 steps, eval_every 3, 64 tasks × 4 samples, **8 monitors**
  (adds `cot_only_{weak,strong}`); three causal rows only — train against `cot+out_weak` (dir name
  `cot_weak`), `probe_iid`, `probe_ood` — × seeds 0–2 (9 runs). **First batch with the committed
  `eval_rollouts_slim.jsonl`** (labels + every monitor's score per eval rollout; needed by the split-half
  coupling analysis, `docs/DEGRADATION_METRICS.md`). Data quirks: (i) `probe_ood_s0`/`s2` have a few
  *spliced* lines in `eval_metrics.jsonl` / `metrics.jsonl` / the rollout dumps (interleaved writes —
  looks like two processes touched those dirs; s0 lost its step-0/3/6 eval rows). `coupling._read_jsonl`
  and `scripts/slim_eval_rollouts.py` skip them with a warning. (ii) Several runs **collapse** late
  (task reward → 0, `unparsed` → 1, hack rate → 0): drop those steps before any degradation number.
  (iii) Step-0 `unparsed` ≈ 0.5 everywhere (base Qwen3-8B truncates), ≈0 by step 6. Hacking seeds:
  cot_weak s1 (peak 0.27), probe_iid s0/s2, probe_ood s1/s2; the other four never hack.

- **`seed_fix_ab_20260922/`** — the **GRPO group-seeding A/B** (see STATUS "GRPO group seeding fix"). Control,
  **no monitors at all**, 32×16, 60 steps, seed 0; `seedOLD` = pre-fix code (seeded n-sample request →
  collapsed groups), `seedFIX` = merged fix (one seeded single-sample request per rollout). Same hack onset
  (~step 15); FIX saturates by step 18 and never collapses, OLD limps and collapses at steps 36–39.
  `loss/frac_zero_adv_groups` in `metrics.jsonl` is the direct bug readout.

## Earlier sycophancy runs (pre-MBPP derisking)

Setting: Anthropic [`anthropics/evals` sycophancy](https://github.com/anthropics/evals/tree/main/sycophancy)
(`sycophancy_on_nlp_survey.jsonl`), Qwen3-8B policy, 40 steps. These show the original dip-then-climb
+ correlated-degradation signature.

- **`run_nlp40/`** — trained against the **weak** CoT monitor (`cot_weak` = `anthropic/claude-3.5-haiku`);
  the strong monitor (`anthropic/claude-sonnet-4.6`) is held out.
- **`run_nlp40_strong/`** — trained against the **strong** CoT monitor (`cot_strong` =
  `anthropic/claude-sonnet-4.6`); the weak monitor (`anthropic/claude-3.5-haiku`) is held out.
