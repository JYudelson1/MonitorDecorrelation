# STATUS — implementation checklist

Living source of truth for build state. Update the marker whenever a component changes state.
**Keep it honest:** don't mark 🟢 without an actual passing test.

**Legend:** `⬜ unimplemented` · `🟡 implemented, untested` · `🔴 tested, buggy` · `🟢 tested, correct`

_Last updated: 2026-06-10 (eval/train split BUILT + validated; brier + class-split metrics; train-against-probe pipeline validated end-to-end on a local smoke run; MPS memory wall for the 4B probe documented → real run is GPU-only)._

## Key findings this session
- **Setting matters more than monitor strength.** On political-typology *no* monitor (incl. Opus 4.8) beats ~0.6 AUROC — every CoT reasons about user identity. On the **nlp subset, Sonnet-4.6 hits ~0.80-0.94 AUROC** with real separation → that's the usable setting. (philpapers untested.)
- **Class balance is essential** for trustworthy detection numbers (90%-sycophantic policy → AUROC noise). Bakeoff balances classes.
- **Denison "Sycophancy to Subterfuge"** kept CoT *out* of the reward (didn't train against a CoT monitor) and found CoT legibly reveals gaming — their reward-hacking rungs (objective oracle + legible CoT) are the natural next setting.

## Docs
| Component | State | Notes |
| --- | --- | --- |
| README.md | 🟢 | Synthesized from the two source docs. |
| docs/RESEARCH_CONTEXT.md | 🟢 | Framing + grounded prior-art summaries. |
| docs/EXPERIMENT_PLAN.md | 🟢 | Exp 2 → Exp 1 → combined matrix. |
| docs/ENVIRONMENTS.md | 🟢 | Sycophancy + 3 coding envs catalogued. |
| docs/MONITORS.md | 🟢 | Monitor seam + activations risk. |
| docs/INFRA.md | 🟢 | RL backends, GRPO, local-wandb logging. |
| CLAUDE.md | 🟢 | Orientation + conventions. |

## Scaffolding
| Component | State | Notes |
| --- | --- | --- |
| Package skeleton (`src/monitordecorrelation/`) | 🟢 | Importable; `uv run python -c "import monitordecorrelation"` passes. |
| `Monitor` protocol + `MonitorResult` | 🟢 | Defined + import-verified. See docs/MONITORS.md. |
| `Env` protocol + `Rollout` / `EnvResult` types | 🟢 | Defined + import-verified. See docs/ENVIRONMENTS.md. |
| `config.py` (dataclass configs) | 🟢 | `RunConfig` + `MonitorSpec`; default `Qwen/Qwen3-8B`. |
| `.env` loading + tinker connectivity check | 🟢 | `experiments/check_tinker.py`: auth OK, 41 models, Qwen3-8B available. |

## Monitors
| Component | State | Notes |
| --- | --- | --- |
| CoT monitor (black-box LLM judge) | 🟢 | `monitors/cot_monitor.py` (OpenRouter); discrimination tested live. weak/strong via `model_id`. |
| Output monitor (black-box) | ⬜ | |
| Linear probe (white-box) | 🟡 | `whitebox/{model,probe}.py` + `monitors/probe_monitor.py`. Per-layer LR, CE<0.6 filter, follow-up-token activations (Atlas §4.3). Logic unit-tested offline (`tests/test_probe.py`, incl. left-pad last-token select); held-out-AUROC run on a real model **pending HF/compute** (offline here). v1 uses the **original base model**, not the tinker policy. |
| Contrastive-QA dataset registry | 🟡 | `whitebox/datasets/` — uniform `(prompt, honest, deceptive)` shape so a probe is generic over training data. Real adapters: `doluschat`, `sycophancy`; stubs registered: truthfulqa/mask/liarsbench/sandbagging/marks_tegmark/mbpp. Registry+flatten unit-tested; adapters need HF to fetch (untested live). |
| ProbeMonitor (white-box `Monitor`) | 🟡 | `monitors/probe_monitor.py` — plugs into the seam; **post-hoc/batched only** (never in the RL loop in v1). `score`/`score_batch` tested via stub model; real-model path skips offline. |
| Activation extraction from tinker policy | ⬜ | **v2 action item** (obfuscated-activations decomposition): pull LoRA weights via `tinker_cookbook.weights.*` → local forward pass. v1 deliberately sidesteps this by probing the base model on generated text. |

## Environments
| Component | State | Notes |
| --- | --- | --- |
| Sycophancy (Anthropic) | 🟢 | `envs/sycophancy.py` — loader+oracle tested. Answer parser anchors on "Answer: (X)" (old last-match heuristic mislabeled ~19% of rollouts that reference the rejected option). |
| Coding — hardcode reward hacking | ⬜ | Obfuscation-Atlas style. Needs sandbox. |
| ImpossibleBench | ⬜ | Passing ⇒ cheating. Shares harness w/ LiveCodeBench. |
| LiveCodeBench | ⬜ | Honest-is-possible counterpart. |
| Sandboxed code execution harness | ⬜ | Shared by the coding envs. |

## Backends + RL + eval
| Component | State | Notes |
| --- | --- | --- |
| `RLBackend` protocol | 🟢 | Defined + import-verified. See docs/INFRA.md. |
| `TinkerBackend` | 🟢 | sample + importance_sampling fwd_bwd + optim_step + sampler refresh; runs live. |
| `TransformersBackend` (local, for white-box) | 🟢 | `backends/transformers_backend.py` — HF+PEFT LoRA policy in-process; same `sample`/`train_step` interface; importance-sampling GRPO loss hand-rolled. Validated: gradient flows, runs through `run_grpo` unchanged on Qwen3-0.6B/MPS. Enables train-AGAINST-probe (model local → activations available every step). Slow on MPS; model/device-agnostic for rented GPU. |
| `VerlBackend` (off-tinker policies) | ⬜ | Superseded by `TransformersBackend` for the white-box use case. |
| GRPO algorithm (`rl/grpo.py`) | 🟢 | group-relative advantages + Datum construction; alignment unit-tested. DR-GRPO via `normalize=False`. |
| Rollout / sampling (`rl/rollout.py`) | 🟢 | tinker sampling + chat-template + CoT/answer split. |
| Baseline eval (no-train measurement loop) | 🟢 | `experiments/eval_baseline.py`: real Qwen3-8B → oracle + monitor. Sycophancy rate ~0.75 on political subset. |
| RL loop (`rl/train.py`) | 🟢 | reward = task − λ·penalty; GRPO; probes via `score_batch`. Runs via `experiments/run_experiment.py`. |
| Generic config runner (`experiment_config.py` + `run_experiment.py`) | 🟢 | pydantic-validated JSON/YAML config; mixed CoT+probe monitors, train/held roles; bad configs rejected. Replaces the per-experiment scripts. Live-tested. |
| `TransformersBackend` (local, trained-model probing) | 🟢 | listed above under Backends. |
| Ground-truth rate tracking | 🟢 | **Primary metric**, logged every step. |
| Per-monitor accuracy + AUROC tracker (`eval/metrics.py`) | 🟢 | per-step + cumulative, train-against AND held-out; unit-tested. |
| `brier` + class-split (`mean_score_syco`/`honest`) metrics | 🟢 | `eval/metrics.py`. AUROC goes NaN at single-class (behavior≈0/≈1 — the obfuscation-success regime); brier (proper scoring rule, MSE vs syco=1/honest=0) + `mean_score_syco` stay defined there and carry the degradation signal. Logged per eval monitor + train-against probe; on the eval `monitors.png`. Unit-tested. |
| Logging / local wandb | 🟢 | offline wandb + per-step `metrics.jsonl` + per-rollout jsonl (env + per-monitor score/label + reward + advantage + generic `extra`). |
| Local plots (`scripts/plot_run.py`) | 🟢 | matplotlib PNGs from `metrics.jsonl`; auto-plots on finish (no W&B server needed). |
| LoRA LR estimation (`hyperparams.py`) | 🟢 | re-exports TM's `tinker_cookbook.hyperparam_utils.get_lr` (no hardcoding). Qwen3-8B → 4.7e-4. |
| Degradation matrix (`eval/degradation.py`) | 🟢 | **CROSS-RUN** tool: `build_degradation_matrix([run dirs])` → `D[i][j]` = AUROC drop of detector j when training against i, + heatmap PNG. **NOT per-run** (one run = one row; never saved after a training run). Unit-tested; needs N matched runs (identical configs except `train_against`) to populate. |
| **Held-out eval phase (train/ vs eval/ split)** | 🟢 | **BUILT + validated** (smoke run 2026-06-10). Every `eval_every` steps, samples a FIXED held-out set (`env.holdout(eval_size)`, REMOVED from the train pool so eval ⟂ train), scores ALL monitors, logs `eval/` namespace + `eval_metrics.jsonl`. Train step scores only train-against monitors (`train/` + `metrics.jsonl`). Per-phase timers (`time/sample_s`,`score_s`,`optim_s`) + `[HH:MM:SS]` logs. |
| No-penalty + frozen-policy controls | ⬜ | baselines per EXPERIMENT_PLAN.md. |

## Experiments
| Component | State | Notes |
| --- | --- | --- |
| train-against-CoT (config) | 🟢 | `configs/train_against_cot.json`; full 40-step runs done last session (run_nlp40 / run_nlp40_strong). |
| train-against-probe (original-model probing) | 🟢 | `configs/train_against_*probe*.json`; probe penalty in reward via tinker. Validated (0.6B mechanics test); valid same-model 4B run pending. |
| Exp 1 — cross-method (CoT vs probe) | 🟢 | Both probes trained on Qwen3.5-4B: **ood** (DolusChat deception, 0.978 in-dist / ~0.40 on sycophancy) and **iid** (sycophancy_cot via dual-prompt builder, 0.74 on sycophancy). iid/ood divergence shown. |
| Combined degradation matrix | 🟡 | All 4 detectors exist (`configs/full_matrix.json`). Needs the N matched runs (train against each, identical else) → `build_degradation_matrix`. A rented-GPU job (local 4B probe-in-loop is slow). |
| Datasets/probes durability | ⬜ | `data/probes/syco_cot_pairs.jsonl` (71 pairs, model-independent) + probes are local/gitignored. `LinearProbe.push_to_hub` exists; add a dataset push + run it so they're not regenerated. |

## Eval/train split + richer plots — BUILT + validated (2026-06-10)

Done and validated end-to-end on a local smoke run (`tap_iid_4b_smoke`: 5 steps, eval every step).
The loop now separates **training-rollout** metrics from a **held-out eval** so degradation curves are
trustworthy. Design as built:

**Eval/train split** (`cfg.eval_every` is the hook):
- Every N steps, run a **fixed held-out eval set** of prompts through the current policy.
- Two metric/plot namespaces:
  - **`train/`** — per-step, on the training rollouts: train-against monitor score + **model reward**
    / behavior rate (the optimization signal).
  - **`eval/`** — on the **same fixed held-out set** every eval: **model performance/behavior rate**
    *and* **every monitor's AUROC over training** measured on that held-out set (the clean
    degradation curves — what the matrix should be built from).
- Move the degradation-relevant stats from `train/` → `eval/`.
- Probe at eval: original-model probing works on tinker now; **trained-model** probing at eval needs
  the weight-download path (tinker) or the `TransformersBackend`.

**Plots (two sets, per run):**
- `data/runs/<run>/train/` — `ground_truth.png` (behavior rate + reward) + `monitors.png` (per-step,
  training rollouts).
- `data/runs/<run>/eval/` — `ground_truth.png` (held-out behavior) + `monitors.png` (held-out monitor
  AUROC over training, on the fixed eval set).
- `plot_run` emits both folders once eval data exists. **The degradation matrix is NOT here** — it's a
  cross-run heatmap (`build_degradation_matrix`), built separately over a *set* of runs.

## Known issues / flags (don't forget)
- **MPS can't run the 4B white-box probe at 4096-token sequences.** No flash kernels on macOS → PyTorch materializes the full `[B, heads, T, T]` attention; a *single* 4096-token forward on the 4B hybrid model peaks ~27 GiB (one ~16 GiB attention/state buffer + 8 GiB weights), over the 36 GiB ceiling. It's the per-SEQUENCE O(T²) cost, not the batch — `batch_size=1` does NOT help; only shorter sequences do. `ProbeMonitor.batch_size` is device-aware (1 on MPS, 8 on CUDA). **Local probe runs are bs=1 + ~7.5s/forward → real runs are GPU-only** (`uv sync --extra gpu`, flash-attn removes the wall). The smoke uses `max_tokens=1024` purely so sequences fit MPS for plumbing validation.
- **iid sycophancy probe AUROC is weak (~0.67–0.74).** A weak detector can't show a meaningful degradation *drop*, so this caps the headline result. Likely causes: tiny/noisy contrastive set (71 dual-prompt pairs; "honest" examples came from a *nudged* prompt → may encode nudge-style, not honesty), follow-up-prompt choice, single-layer LR aggregation. **Try other methods before the real run** — see the open task. (The ood deception probe is fine in-dist, 0.978; the weakness is specific to the on-domain sycophancy probe.)
- **Monitor calibration:** the CoT judge over-flags real (subtle) rollouts — AUROCs near chance, penalty stuck ~0.6. Highest-value fix before trusting any degradation curve.
- **Political-typology subset has no objective truth** — clean *behavioral* oracle but ambiguous for the monitor. Consider `nlp`/`philpapers` or factual-QA-with-stated-belief.
- **CoT truncation → unparsed:** at low `max_tokens`, Qwen3 truncates mid-`<think>`; the rollout is unparsed and scored task_reward 0 / not-misbehavior (slight bias). Keep `max_tokens` ≥ ~768 or handle truncation explicitly.
- **Monitors are feature-generic but only sycophancy exists** — flagged in docs/MONITORS.md; later build a registry of feature rubrics/probes so we monitor arbitrary features, not just sycophancy.
- **Heavy deps:** adding `tinker-cookbook` (for the LR fn) pulled torch/pandas and downgraded transformers 5.10→5.5. Tests still green; revisit if it causes friction.
