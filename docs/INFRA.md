# Infrastructure

How the RL substrate, algorithm, and logging fit together. Build state in [`../STATUS.md`](../STATUS.md).

## Layers

```
experiment script
  └─ RL algorithm (GRPO)          # rewards -> group-relative advantages   (rl/, backend-agnostic)
       └─ RLBackend               # sample rollouts + apply gradient step  (backends/)
            ├─ TinkerBackend      # the only implemented backend
            └─ VerlBackend (stub) # later escape hatch for off-tinker policies
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
- **`verl` (stub, deferred).** The reason it exists: **tinker hosts no small DeepSeek-R1 distills**,
  and we may specifically want one as a natively-verbalizing policy. verl would let us RL an
  off-tinker policy. Mildly painful setup — only build it if we actually want DeepSeek.

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

### RL-debugging metrics (every train step, W&B `train/…`, also in `metrics.jsonl`)
- `loss/loss`, `loss/loss_per_token` — the importance-sampling objective, `-(exp(train_lp − sample_lp)·A).sum()`
  (tinker's `loss:sum`). Cookbook `train_step` drops the `forward_backward` result, so
  `rl/grpo.optim_metrics` recomputes it exactly from the trainer's returned logprobs. Group-centred
  advantages + ratio ≈ 1 make it ≈ `−Σ A·len`: it tracks the length/advantage correlation, not convergence.
- `loss/ratio_{mean,min,max,std}`, `loss/kl_sample_train_{v1,v2}` — sampler/trainer mismatch (should be
  ≈1 / ≈0 on-policy); `loss/entropy` — sampling entropy (−mean sampled logprob).
- `loss/adv/*`, `loss/frac_zero_adv_groups`, `reward/group_std_mean`, `reward/frac_constant_groups` — how
  much gradient signal the batch carried (a constant-reward group contributes nothing).
- `reward/total_{std,min,max}`, `rollout/output_tokens_{mean,min,max}`, `rollout/truncated_rate`,
  `loss/n_action_tokens`, `loss/learning_rate`.
- `invalid_rate` (train + eval; eval also `truncated_rate`; terminal env also `no_submission_rate`) —
  truncated or unparseable rollouts and, in the terminal env, episodes that never submitted: flat −1
  reward and **never monitored**, so every `monitor/*` metric and `reward/monitor_suspiciousness_mean`
  is over the remaining valid rollouts (`monitor/<m>/n_scored` = their count). `reward/penalty_mean` is
  still averaged over all rollouts (0 for invalid ones) — the reward actually deducted per rollout.
- Not available: grad norm (tinker's `optim_step` returns no metrics for LoRA runs) and post-update KL
  (would cost an extra logprob pass per step).
