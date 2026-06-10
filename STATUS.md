# STATUS — implementation checklist

Living source of truth for build state. Update the marker whenever a component changes state.
**Keep it honest:** don't mark 🟢 without an actual passing test.

**Legend:** `⬜ unimplemented` · `🟡 implemented, untested` · `🔴 tested, buggy` · `🟢 tested, correct`

_Last updated: 2026-06-05 (behavior-generic refactor; rubric registry; monitor bakeoff tool; nlp subset; 40-step nlp run in progress)._

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
| RL loop (`rl/train.py`) | 🟢 | reward = task − λ·penalty; GRPO; runs via `experiments/train_against_cot.py`. |
| Ground-truth rate tracking | 🟢 | **Primary metric**, logged every step. |
| Per-monitor accuracy + AUROC tracker (`eval/metrics.py`) | 🟢 | per-step + cumulative, train-against AND held-out; unit-tested. |
| Logging / local wandb | 🟢 | offline wandb + per-step `metrics.jsonl` + per-rollout jsonl (env + per-monitor score/label + reward + advantage + generic `extra`). |
| Local plots (`experiments/plot_run.py`) | 🟢 | matplotlib PNGs from `metrics.jsonl` (no W&B server needed). |
| LoRA LR estimation (`hyperparams.py`) | 🟢 | re-exports TM's `tinker_cookbook.hyperparam_utils.get_lr` (no hardcoding). Qwen3-8B → 4.7e-4. |
| Degradation matrix (`eval/degradation.py`) | 🟡 | `build_matrix(run_dir)` assembles per-step AUROC columns (CoT from `metrics.jsonl`, probes from `probe_eval_*.jsonl`) + a trajectory-correlation matrix; writes `degradation.{json,csv}`. Unit-tested on synthetic columns; needs a real run to populate. |
| No-penalty + frozen-policy controls | ⬜ | baselines per EXPERIMENT_PLAN.md. |

## Experiments
| Component | State | Notes |
| --- | --- | --- |
| `train_against_cot` (train weak CoT, hold out strong) | 🟡 | `experiments/train_against_cot.py` — loop runs end-to-end; realistic defaults (16×8 ×40, auto-LR). Not yet run at full size / not a validated result. |
| Exp 1 — cross-method (CoT vs probe) | 🟡 | Plumbing built: `experiments/{train_probe,eval_probes_on_run}.py` (probe trains on base model, scored **post-hoc** over a run's `rollouts.jsonl`). Needs a real probe + saved run to produce numbers. No longer blocked (v1 uses base model, not tinker internals). |
| Combined degradation matrix | ⬜ | After train_against_cot + probe. |

## Known issues / flags (don't forget)
- **Monitor calibration:** the CoT judge over-flags real (subtle) rollouts — AUROCs near chance, penalty stuck ~0.6. Highest-value fix before trusting any degradation curve.
- **Political-typology subset has no objective truth** — clean *behavioral* oracle but ambiguous for the monitor. Consider `nlp`/`philpapers` or factual-QA-with-stated-belief.
- **CoT truncation → unparsed:** at low `max_tokens`, Qwen3 truncates mid-`<think>`; the rollout is unparsed and scored task_reward 0 / not-misbehavior (slight bias). Keep `max_tokens` ≥ ~768 or handle truncation explicitly.
- **Monitors are feature-generic but only sycophancy exists** — flagged in docs/MONITORS.md; later build a registry of feature rubrics/probes so we monitor arbitrary features, not just sycophancy.
- **Heavy deps:** adding `tinker-cookbook` (for the LR fn) pulled torch/pandas and downgraded transformers 5.10→5.5. Tests still green; revisit if it causes friction.
