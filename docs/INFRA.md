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

- **`tinker` (implemented path).** Raw SDK, minimal usage: `ServiceClient` →
  `create_lora_training_client(base_model, rank)` → sample via a sampling client → `forward_backward`
  + `optim_step`. Connectivity verified (`experiments/check_tinker.py`). No cookbook.
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
- Reward per rollout: `task_reward − penalty_coef·monitor_penalty(train_against)`. Optional KL term
  (`kl_coef`) — recall the Obfuscation Atlas finding that high KL + penalty can recover honesty.

## Logging (local wandb)

- **Local/offline wandb** by default (`LoggingConfig.wandb_mode="offline"`), project
  `monitor-decorrelation`. This is where the metric curves live — especially the ground-truth
  trajectory (see EXPERIMENT_PLAN.md "Ground truth is the crux").
- **Rollout persistence sampling:** `log_fraction` (default `1.0` = keep everything) controls what
  fraction of full rollouts we store verbatim, since logging every transcript may get expensive at
  scale. Aggregate metrics are always logged regardless.

### Metrics to log every eval step
- `behavior_rate` — **primary.** Per env.
- For every monitor (train-against *and* held-out): accuracy / AUROC vs. ground truth.
- Monitor penalty, task reward, total reward, KL.
- Derived: pairwise degradation `D[i][j]` once we have multiple detectors.
