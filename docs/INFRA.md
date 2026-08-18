# Infrastructure

How the RL substrate, algorithm, and logging fit together. Build state in [`../STATUS.md`](../STATUS.md).

## Layers

```
experiment script
  └─ RL algorithm (GRPO)          # rewards -> group-relative advantages   (rl/, backend-agnostic)
       └─ RLBackend               # sample rollouts + apply gradient step  (backends/)
            ├─ TinkerBackend      # the reference backend
            ├─ TransformersBackend# local HF+PEFT, for white-box work
            └─ VerlBackend (stub) # later escape hatch for off-tinker policies

nemo backend (backend: "nemo") — a hand-off, not an RLBackend:
experiment config -> MD_* env vars -> NeMo-RL's own GRPO loop     (backends/nemo/)
       └─ our Env + Monitors, as a NeMo-RL environment actor  # same reward, same jsonl schema
       └─ Monitors                # score rollouts -> penalty + held-out metrics
       └─ Env                     # task_reward + behavior_present
       └─ Logger                  # local wandb + rollout sampling
```

The **algorithm/backend split** is deliberate: GRPO just turns rewards into advantages and is
indifferent to who runs the forward/backward. Swapping `tinker` → `verl` should not touch the
algorithm.

## Backends

- **`tinker` (implemented path).** Raw SDK, minimal loop: `ServiceClient` →
  `create_lora_training_client(base_model, rank)` → sample via a sampling client → train step. We keep
  our own loop but **delegate the loss layer to tinker-cookbook primitives** (`compute_advantages`,
  `assemble_training_data`, `incorporate_kl_penalty`, `rl.train.train_step`, `hyperparam_utils.get_lr`)
  rather than reinventing them. Connectivity verified (`experiments/check_tinker.py`).
- **`nemo` (implemented).** Local multi-GPU training via **[NeMo-RL](https://github.com/NVIDIA-NeMo/RL)**
  (Megatron policy workers + LoRA, colocated vLLM generation) — the path for policies tinker does not
  host and for anything that needs our own GPUs. See "The nemo backend" below.
- **`verl` (stub, deferred).** The reason it exists: **tinker hosts no small DeepSeek-R1 distills**,
  and we may specifically want one as a natively-verbalizing policy. verl would let us RL an
  off-tinker policy. Mildly painful setup — only build it if we actually want DeepSeek.

### The nemo backend

NeMo-RL is vendored as the git submodule **`third_party/nemo-rl`** (`git clone --recursive`, or
`git submodule update --init --recursive` in an existing clone). It is pinned to the commit the
NeMo-RL container images are built from, so the prebuilt per-worker venvs under `$NEMO_RL_VENV_DIR`
(`/opt/ray_venvs` in the container) are reused instead of rebuilt. Bump the pin with a normal
submodule update when you want a newer NeMo-RL.

Unlike `tinker`/`transformers`, this is **not** an `RLBackend`: NeMo-RL owns its own GRPO loop, its
Megatron workers and its vLLM engine, and it needs its own virtualenv (Python 3.13 + torch 2.11 +
Megatron + vLLM) which cannot be merged with this project's. So the seam moves up one level —
`experiments/run_experiment.py` hands the run over to `backends/nemo/launcher.py`, which starts
`backends/nemo/driver.py` inside NeMo-RL's interpreter with this repo on `PYTHONPATH`. What stays
ours:

| piece | where |
| --- | --- |
| the prompts (env pool + train-disjoint holdout) | `backends/nemo/dataset.py` |
| the reward `task_reward − penalty_coef·mean(train_against)` | `backends/nemo/env_actor.py` |
| every monitor's held-out scores + the oracle | same actor, `role="val"` |
| `metrics.jsonl` / `eval_metrics.jsonl` / `rollouts.jsonl` / `eval_rollouts.jsonl` | `backends/nemo/driver.py` |

so `eval/degradation.py` and `eval/plots.py` read a nemo run exactly as they read a tinker run. Held-out
monitors are scored only by the validation actor and never enter the reward — the invariant is
structural, same as on tinker.

**One config file.** `experiments/configs/nemo/grpo.yaml` is the only NeMo-RL config in the repo: no
per-model or per-GPU-count variants. Everything that varies is an `MD_*` environment variable
resolved by `${oc.decode:${oc.env:…}}`, derived from the experiment config by
`backends/nemo/params.py` (pure + unit-tested in `tests/test_nemo_params.py`) and recorded, fully
resolved, as `data/runs/<run>/nemo_config.yaml`. `n_gpus` defaults to every GPU on the machine;
`nemo_options` is the per-run escape hatch (`--set nemo_options={"train_micro_batch_size":2}`).

**Hyperparameter parity with tinker is deliberate** — tinker is the reference, not transformers:
mean-centred group advantages only (`normalize_rewards: false`, `use_leave_one_out_baseline: false`,
matching cookbook `compute_advantages`), one optimizer step per rollout batch so the PPO ratio is 1
(`force_on_policy_ratio: true`), KL applied to the **reward** not the loss (`use_kl_in_reward: true`,
matching `incorporate_kl_penalty`), Adam(0.9, 0.95, 1e-8) with no weight decay, no gradient clipping
and a constant LR (tinker's `AdamParams` defaults as the cookbook uses them), LoRA alpha fixed at 32,
and temperature 1.0. The Megatron/vLLM block (parallel layout, micro-batch size, memory fraction, MoE
backend, rope-fusion off) comes from the measured 2×B200 Qwen3.5 recipes. Both files argue each
choice inline.

**Environment variables.** `.env` is propagated into the NeMo-RL process (`WANDB_API_KEY` today,
monitor API keys as soon as a config lists monitors); an explicit shell export still wins over `.env`.
The W&B mode is resolved exactly as on the tinker path.

### Policy model note
The policy is the RL'd model, so it must be **tinker-hosted** and should **natively verbalize CoT**
(small 2025-era reasoner that has to think through the task — otherwise there's no load-bearing CoT
to monitor or obfuscate). On tinker that means a small **Qwen3 with thinking on**. Caveat:
`Qwen3-4B-Instruct-2507` is the *non-thinking* variant, so the small verbalizing default is
**`Qwen/Qwen3-8B`** (hybrid thinking). For a true small R1-distill policy we'd need the verl backend.

## RL algorithm

- **GRPO** by default (group-relative advantages: sample `group_size` rollouts per prompt, normalize
  rewards within the group, no value network). Matches tinker's streamlined RLVR recipe.
- **DR-GRPO / DAPO** as easy swaps via `rl_algo` (token-level loss normalization, clip-higher,
  dynamic sampling). Start with vanilla GRPO; revisit if we see length/normalization pathologies.
- Reward per rollout: `task_reward − penalty_coef·monitor_penalty(train_against)`. Per-token **KL-to-base
  penalty** is active (`kl_coef=1e-4`, via cookbook `incorporate_kl_penalty` against a base-model sampler)
  — it anchors the policy and prevents the reward-hack collapse (echoes the Obfuscation Atlas finding
  that high KL + penalty can recover honesty).

## Logging (wandb)

- **wandb auto-syncs iff logged in** — `run_experiment.py` picks `online` when a credential is
  configured locally (netrc / `WANDB_API_KEY`), else `offline` (local, never prompts; `WANDB_MODE`
  overrides). Project `monitor-decorrelation`; all runs of a sweep share a **group** per (experiment,
  model) + tags (env, model, seed, train-against). This is where the metric curves live — especially
  the ground-truth trajectory (see EXPERIMENT_PLAN.md "Ground truth is the crux").
- **Rollout persistence sampling:** `log_fraction` (default `1.0` = keep everything) controls what
  fraction of full rollouts we store verbatim, since logging every transcript may get expensive at
  scale. Aggregate metrics are always logged regardless.

### Metrics to log every eval step
- `behavior_rate` — **primary.** Per env.
- For every monitor (train-against *and* held-out): accuracy / AUROC vs. ground truth.
- Monitor penalty, task reward, total reward, KL.
- Derived: pairwise degradation `D[i][j]` once we have multiple detectors.
