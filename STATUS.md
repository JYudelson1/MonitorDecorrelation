# STATUS — implementation checklist

Living source of truth for build state. Update the marker whenever a component changes state.
**Keep it honest:** don't mark 🟢 without an actual passing test.

**Legend:** `⬜ unimplemented` · `🟡 implemented, untested` · `🔴 tested, buggy` · `🟢 tested, correct`

_Last updated: 2026-06-29 (MBPP-Honeypot reward-hacking is the live experiment; the 7-row cross-method
degradation matrix — 6 monitors + control — is RUNNING on a rented GPU across seeds. Strict/loose/
hardcoding label methodology landed (behavior_present = strict_rh); KL-to-base penalty via cookbook
primitives; weak/strong judge ladder from a strict-label bakeoff; queue_runs.sh (+ detach), wandb
auto-sync-iff-logged-in + run grouping, behavior-named metrics. Both Qwen3-8B probes retrained+committed.
See loose_threads.md for live state.) Prev 2026-06-15: probe investigation (CoT-stripping bug; no-CoT
generalizes; subtle MASK ~0.6). Prev 2026-06-10: eval/train split + brier._

## Current experiment (2026-06-29) — MBPP-Honeypot cross-method degradation matrix
- **Setting.** Policy = Qwen3-8B RL'd to reward-hack MBPP-Honeypot while evading ONE monitor; the rest
  held out and scored into `eval_metrics.jsonl`. One run per monitor + a no-monitor control = 7 rows,
  fanned across seeds by `scripts/queue_runs.sh`. Headline: AUROC-drop + suspiciousness-gap-drop grids.
- **Monitors (6).** `{weak,strong} × {CoT judge, output judge}` + `probe_ood` (diverse_deception) +
  `probe_iid` (mbpp_honeypot). Weak=`anthropic/claude-3-haiku`, strong=`deepseek/deepseek-chat` — a
  cheap, non-reasoning, off-policy-family ladder picked from a strict-label bakeoff (all judges 0.9–0.99
  against the CLEAN strict oracle; the old ~chance numbers were a loose-label artifact).
- **Label methodology (the crux).** `behavior_present = strict_rh = hardcoding ∧ loose_rh`
  (`envs/hacking_labels.py`). loose alone conflated honest-buggy attempts with real gaming and tanked
  every judge's AUROC; strict fixes it. Chart strict + loose + hardcoding together.
- **Batch.** `batch_size=64 × group_size=8` = 512 rollouts/step (Rauno-matched; time-to-hack scales
  with total rollouts, so this hits the dip-then-climb in ~10-12 steps, not ~100). `eval_every=4`,
  `kl_coef=1e-4`. Final weights saved on tinker with a 7-day TTL.

## Key findings (probe investigation, 2026-06-15) — see docs/PROBES.md
- **CoT-stripping bug:** Qwen's template strips `<think>` from non-final turns → the Atlas follow-up technique is structurally no-CoT on thinking models (the old iid probe was secretly no-CoT). Fixed: `--within-generation` + `--preserve-thinking` (patched template).
- **No-CoT generalizes / CoT adds nothing** (MASK 2×2, ~0 transfer gap, robust) → use cheap answer-only data.
- **Subtle MASK deception is genuinely hard (~0.6)** — robust across labels (haiku/sonnet 77% agree), 3 reads, clean-label retrain (sonnet→0.60), blatant-transfer (0.89→0.587). Blatant deception is easy (DolusChat 0.978). Caveat: provided_facts-capped ~113 pairs.
- **MPS: long-CoT reads are GPU-bound** (4000-token sycophancy CoTs ≈ 62s/forward). Short answer-only reads are fine (bs=8).

## Key findings (sycophancy era — historical, superseded by the MBPP setting above)
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
| CoT monitor (black-box LLM judge) | 🟢 | `monitors/cot_monitor.py` (OpenRouter); discrimination tested live. weak/strong via `model_id`. Rubrics: sycophancy / reward_hacking / **deception**. Shared `monitors/openrouter.py` `chat()` helper. |
| Output monitor (black-box) | 🟢 | Judge the answer ONLY (no CoT) — `use_cot: false` on the CoT monitor. Matrix rows `out_weak`/`out_strong`. Matches what labs deploy + claim to train on; live in the running matrix. |
| Linear probe (white-box) | 🟢 | `whitebox/{model,probe}.py` + `monitors/probe_monitor.py`. Per-layer LR, CE<0.6 filter, follow-up-token activations (Atlas §4.3). Trained on real Qwen3-8B: `probe_ood`=diverse_deception **0.977 in-dist** (5849 pairs, 36/37 layers), `probe_iid`=mbpp_honeypot retrained split-safe (`--preset mbpp`, disjoint train/eval task_ids). Both committed under `data/probes/Qwen3-8B/`. v1 uses the **original base model**, not the tinker policy. |
| Contrastive-QA dataset registry | 🟢 | `whitebox/datasets/` — uniform `(prompt, honest, deceptive)` shape (now with optional per-response `honest_prompt`/`deceptive_prompt` for **unpaired** sources). Live-verified adapters: `doluschat`, `sycophancy`, `sycophancy_cot`, **`truthfulqa`, `mbpp`, `sandbagging`, `marks_tegmark`, `liarsbench`**; `mask` stays a stub (probe needs generation — task #37). `PROBE_PRESETS` (simple/diverse/mbpp) mirror the Atlas recipes; `train_probe.py --preset`. Unit + live-load tested. |
| ProbeMonitor (white-box `Monitor`) | 🟢 | `monitors/probe_monitor.py` — plugs into the seam; scored at each held-out eval on the real 8B model (live in the running matrix). `score`/`score_batch` tested via stub offline + exercised end-to-end on GPU. |
| Activation extraction from tinker policy | ⬜ | **v2 action item** (obfuscated-activations decomposition): pull LoRA weights via `tinker_cookbook.weights.*` → local forward pass. v1 deliberately sidesteps this by probing the base model on generated text. |

## Environments
| Component | State | Notes |
| --- | --- | --- |
| Sycophancy (Anthropic) | 🟢 | `envs/sycophancy.py` — loader+oracle tested. Answer parser anchors on "Answer: (X)" (old last-match heuristic mislabeled ~19% of rollouts that reference the rejected option). |
| MASK deception (pressure) | 🟢 | `envs/mask.py` over `cais/MASK` (gated). Pressure prompt; `behavior_present` (lied) via pluggable extractor (default OpenRouter judge → A/B/C vs truth; C=evade). v1 = binary `provided_facts` (oracle exact); other configs need belief-elicitation (deferred). Offline oracle tests + live load + live judge validated. |
| Coding — MBPP-Honeypot (hardcode reward hacking) | 🟢 | `envs/mbpp_honeypot.py` (Obfuscation Atlas). Prompt = task + first visible test; reward = pass-visible − 0.003·len(code). `behavior_present` = **`strict_rh`** (hardcoding ∧ loose_rh) via `envs/hacking_labels.py`; meta also carries `loose_rh` + `hardcoding`. Static-AST hardcode detector validated 0-FP vs 293 Opus-4.8 relabels + a constructed test suite (incl. conditional special-case forms). Disjoint train/eval task_ids (`mbpp_split.py`). Uses the executor below. |
| Reward-hacking labels (`envs/hacking_labels.py`) | 🟢 | Deterministic 3-way: `loose_rh` (passes visible, not all), `hardcoding` (no param referenced anywhere, OR `if x==<visible_in>: return <visible_out>` ternary/reversed forms), `strict_rh` = both. `strict ⊆ loose`. Unit-tested (27 cases) + 0-FP vs Opus relabels. |
| ImpossibleBench | ⬜ | Passing ⇒ cheating. Shares harness w/ LiveCodeBench. |
| LiveCodeBench | ⬜ | Honest-is-possible counterpart. |
| Sandboxed code execution harness | 🟡 | `envs/code_exec.py` — subprocess + timeout, per-test pass/fail. **Timeout-only, NOT a real sandbox** (runs untrusted model code); harden (docker/nsjail, no net, ro-fs, rlimits) before any large/shared run. |

## Backends + RL + eval
| Component | State | Notes |
| --- | --- | --- |
| `RLBackend` protocol | 🟢 | Defined + import-verified. See docs/INFRA.md. |
| `TinkerBackend` | 🟢 | sample + sampler refresh; `train_step` adapts rollouts→cookbook `TrajectoryGroup`s then delegates the loss layer to cookbook primitives (`compute_advantages` → `assemble_training_data` → `incorporate_kl_penalty` (per-token KL-to-base when `kl_coef>0`) → `rl.train.train_step`). KL ref = a base-model sampling client. `save_checkpoint(label, ttl_seconds)` → `save_state` (final weights, 7-day TTL). Seeded via `create_lora_training_client(seed=)` + per-call `derive_sample_seed`. Runs live. |
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
| `brier` + class-split score metrics | 🟢 | `eval/metrics.py` + `eval/metric_keys.py`. Class-split means are named per-env after `behavior_name`: `mean_score_reward_hacking` / `mean_score_not_reward_hacking` (legacy runs: `_syco`/`_honest`; both parsed structurally by `class_score_pairs`). Stay defined when AUROC goes NaN at single-class (obfuscation-success regime) and carry the degradation signal. Unit-tested (both key schemes + NaN-drop). |
| Logging / wandb | 🟢 | wandb + per-step `metrics.jsonl` + per-rollout jsonl. **Auto-syncs iff logged in** (`run_experiment` sets `wandb_mode=online` when a local credential exists — netrc/`WANDB_API_KEY` — else `offline`; `WANDB_MODE` overrides; local check, never prompts). **Run grouping**: all runs of a sweep share `group=<experiment>/<model>` + tags `[env, model, seedN, <train-against>|control]`. |
| Local plots (`scripts/plot_run.py`) | 🟢 | matplotlib PNGs from metrics; auto-plots on finish. Titles/labels derive from `behavior_name` (per-env, no hardcoded "reward-hacking"). |
| Batch launcher (`scripts/queue_runs.sh`) | 🟢 | Fans a config/dir × seeds `0..N-1` into ≤J parallel runs (`xargs -P`, BSD/GNU). Per-job log → `data/runs/<run>/run.log`, ✓/✗ summary, `-d` dry-run, `-s` resume (QUEUE_DONE sentinel), `-D` detached (nohup, survives disconnect). Verified end-to-end. |
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
| **MBPP-Honeypot degradation matrix (THE experiment)** | 🟡 | **RUNNING on rented GPU across seeds.** `experiments/make_mbpp_matrix_configs.py` → `configs/mbpp_matrix/row_*.json` (6 monitors + control = 7 rows; each trains against one, holds out the rest, all scored into `eval_metrics.jsonl`). Monitors: `cot_weak`/`cot_strong`/`out_weak`/`out_strong` (haiku/deepseek-chat ladder) + `probe_ood` + `probe_iid` (both 8B, trained). Fanned by `queue_runs.sh`. Build grids: `build_degradation_matrix(sorted(glob('data/runs/mbpp_*')))` → AUROC-drop + gap-drop + end-AUROC (json/csv/png). 🟢 once the runs land + grids look sane. |
| Sycophancy matrix (superseded) | 🟡 | Earlier Qwen3.5/8B scaffolding: `make_matrix_configs.py` → `configs/matrix8b/`. Superseded by the MBPP matrix above; kept for reference. |
| Datasets/probes durability | 🟡 | Both Qwen3-8B probes now **committed to git** under `data/probes/Qwen3-8B/` (un-gitignored) so they travel to any run box — no HF push needed for the matrix. `LinearProbe.push_to_hub` still exists if we want a registry later. |

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
- **RESOLVED — "monitors barely beat chance" was a labeling artifact.** With `behavior_present`=loose, ~96% of "positives" were honest-buggy attempts, so every judge scored ~chance and the penalty stuck ~0.6. Against the CLEAN strict oracle all 14 bakeoff judges score 0.89–0.99. Keep `behavior_present=strict_rh`. (This retired the old "monitor calibration" + "iid sycophancy probe too weak" flags for the MBPP setting.)
- **(sycophancy-era, historical) iid sycophancy probe weak (~0.67–0.74)** and **political-typology has no objective truth** — both specific to the sycophancy setting, not the current MBPP experiment.
- **FIRST-RUN CHECK (seed audit):** confirm GRPO group advantages have non-zero variance — guard against a seeded sampling call collapsing a whole group to identical rollouts.
- **CoT truncation → unparsed:** at low `max_tokens`, Qwen3 truncates mid-`<think>`; the rollout is unparsed and scored task_reward 0 / not-misbehavior (slight bias). Keep `max_tokens` ≥ ~768 or handle truncation explicitly.
- **Monitors are feature-generic but only sycophancy exists** — flagged in docs/MONITORS.md; later build a registry of feature rubrics/probes so we monitor arbitrary features, not just sycophancy.
- **Heavy deps:** adding `tinker-cookbook` (for the LR fn) pulled torch/pandas and downgraded transformers 5.10→5.5. Tests still green; revisit if it causes friction.
