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

### Thinking budget (tinker) — optional, off by default

`thinking_budget: <int>` caps the tokens the policy may spend reasoning; when it is spent, the block
is force-closed and the model has to answer. Unset (the default) changes **nothing** — same single
sampling request, same metric keys, same rollouts.

**Why it needs its own protocol.** `tinker.SamplingParams` has six fields (`max_tokens`, `seed`,
`stop`, `temperature`, `top_k`, `top_p`) — no budget, no logits-processor hook, no forced tokens.
tinker's Anthropic-compatible shim even documents that Anthropic's `thinking.budget_tokens` is
"accepted for compatibility but **not applied**". So the budget is built from sample → inspect →
re-sample-from-a-prefix, which is exactly how **both** supported providers document it:

| family | documented recipe | injected on force-close |
| --- | --- | --- |
| Qwen3 (thinking) | generate with `max_new_tokens=thinking_budget`; if `</think>` is missing, append a fixed sentence and generate the answer ([Qwen3 quickstart §Thinking Budget](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html), arXiv 2505.09388 §4.3) | `"\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"` (24 tokens) |
| Nemotron 3 / 3.5 | NVIDIA's `ThinkingBudgetClient`, in every model card + usage cookbook: call 1 with `max_tokens=reasoning_budget`, then continue the assistant message | `".\n</think>\n\n"` — the leading `"."` closes the truncated sentence |

Both cut at **exactly N generated tokens** — no run-on to a sentence or paragraph boundary. NVIDIA
*also* ships a serving-side `ThinkingBudgetLogitsProcessor` that waits for the next newline past the
budget and hard-stops at `budget + grace`; that one needs a logits hook, so it is unreachable from
tinker — and the client recipe above is equally official, printed in the model cards themselves. The
graceful-ending idea survives anyway: NVIDIA's leading `"."` closes the truncated sentence (their
source comments it "reasoning content is too long, closed with a period"), and Qwen's sentence
explains the interruption to the model.

**How this differs from the nemo backend**, which *does* have a logits hook and therefore took a
different route (`backends/nemo/thinking_budget.py`, a subclass of **vLLM's** generic
`ThinkingTokenBudgetLogitsProcessor`): nemo forces the bare `</think>` token ids with none of the
provider's closing text, its cap is approximate by a token or two (vLLM skips re-parsing while its
countdown runs), the forced token is part of the sampled sequence and **is** trained on, and it has
no allow-list — the tags come from environment variables, so it will budget any policy you point it
at. Neither backend implements NVIDIA's newline grace window.

**Support is an allow-list** (`rl/thinking_budget.py`): Qwen3-8B, Qwen3-30B-A3B, and the four
Nemotron 3/3.5 policies. Everything else tinker hosts is refused **at config load**, with the reason
— Qwen3.5/3.6 (documented only for Alibaba's *hosted* endpoints; different `<think>` ids and a
pre-opening template, which is exactly how SGLang shipped a budget of 200 that produced ~1400
reasoning tokens), the `-Instruct-2507` Qwen3s (no thinking block at all), DeepSeek-V3.1 and Kimi-K2.6
(binary think/no-think only), gpt-oss (harmony *effort*, not a cap), Llama-3.2 (not a reasoner), and
Inkling (continuous `effort` — use `thinking_effort`). The closing text is what makes the forced
close in-distribution, so it is never guessed or ported across families, and the tag ids always come
from the policy's own tokenizer.

**The protocol.** Pass 1 samples `min(max_tokens, budget)` tokens for the whole batch (one request
per prompt, so a GRPO group still shares one prefill) with **no** stop sequence — a rollout that
closes its reasoning early just keeps writing inside the same call, and one that finishes there costs
a single request and zero extra tokens. If the block is still open, the closing text is spliced in
and a continuation request resumes from `prompt + everything so far`. That prefix is byte-identical
to what pass 1 processed (the `ModelInput` is *appended to*, never rebuilt), which is tinker's
documented "sequence extension property" for KV reuse. Because sampling is autoregressive, splitting
one generation in two draws from the same distribution as generating it in one go; `max_tokens` still
bounds the whole completion, forced tokens included, so a budgeted rollout is never longer than an
unbudgeted one. With `max_tokens <= budget` the budget cannot bind and the run is identical to an
unbudgeted one.

**Injected tokens are not trained on.** Each interruption becomes a cookbook `Transition`, with the
injected tokens in the *observation* of the one that follows (`rl/grpo.py`). The cookbook masks
observation deltas out of the loss and merges the transitions back into one datum, so the forced
`</think>` carries no advantage and needs no invented logprob — unlike the nemo/vLLM path, where the
forced token is part of the sampled sequence and *is* trained on.

**What it costs (measured, not modelled).** A budgeted batch logs its bill twice: `…_ideal` (one
request per rollout, as an in-engine budget would do it) beside `…_actual` (every request we really
made), each prefill split into prefix-cache hits and misses read off
`SampleResponse.prompt_cache_hit_tokens`. See `rl/token_accounting.py` and the metric list under
"Logging". The overhead is the continuation's prefill: the prompt part is usually a cache hit (the
cache is block-granular — 64 tokens — so short prompts miss entirely), the truncated reasoning is
not. Decode goes the other way: the injected closer costs *us* no decode step, so actual decode comes
in slightly below ideal.

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

**Thinking budget (optional, off).** `thinking_budget: <int>` caps the tokens the policy may spend
inside `<think>…</think>`; unset (the default) is uncapped and changes nothing about the engine. It
cannot be a plain config value: NeMo-RL builds its `SamplingParams` from a fixed kwargs list, so
vLLM's own `thinking_token_budget` — and therefore its built-in budget logits processor — is
unreachable from YAML. What *is* reachable is `policy.generation.vllm_kwargs`, which NeMo-RL splats
into `vllm.LLM(**kwargs)` and which accepts `logits_processors: [<fqcn>]`. So `params.py` names
`backends/nemo/thinking_budget.py` there and puts the budget in `vllm_cfg.env_vars`; the processor
(a subclass of vLLM's, with the budget and the `<think>`/`</think>` ids taken from the environment
instead of from per-request sampling params) drives the logits of `</think>` once the count is
reached. **No NeMo-RL source is modified.** Two caveats: the forced `</think>` is part of the sampled
sequence and is trained on at whatever logprob the policy gave it, and the cap is approximate by a
token or two (vLLM's state machine skips re-parsing while a countdown runs).

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
  overrides). Project `monitor-correlation-impossible-bench`; all runs of a sweep share a **group** per (experiment,
  model) + tags (env, model, seed, train-against). This is where the metric curves live — especially
  the ground-truth trajectory (see EXPERIMENT_PLAN.md "Ground truth is the crux").
- **Rollout persistence sampling:** `log_fraction` (default `1.0` = keep everything) controls what
  fraction of full rollouts we store verbatim, since logging every transcript may get expensive at
  scale. Aggregate metrics are always logged regardless.

### Token metrics
`tokens/{input,output}_total`, `…_per_rollout`, `tokens/output_max`, `tokens/truncated_rate`,
`tokens/cum_total` — always. **With a thinking budget**, also
`tokens/{prefill,decode}_{ideal,actual}_total` (+ `_per_rollout`),
`tokens/prefill_{ideal,actual}_cache_{hit,miss}_total`, `tokens/prefill_overhead_ratio`,
`tokens/prefill_miss_overhead_ratio` (the one that costs full price — cached prefill is billed at 80%
off), `tokens/prefill_actual_cache_hit_rate`, `tokens/budget_forced_rate` (how often the cap actually
bound), `tokens/sampling_requests_per_rollout` and `tokens/thinking_per_rollout`. Those keys are
**absent** without a budget, so unbudgeted runs' rows are unchanged. Plotted to
`{train,eval}/tokens_budget.png`.

`tokens/thinking_per_rollout` measures the whole reasoning block, so on a forced rollout it lands
*above* the budget by the length of the closing text that precedes `</think>` (Nemotron +1, Qwen +21).
The policy still sampled exactly `budget` tokens; the rest is the documented closer.

Only *sampling* has an ideal-vs-actual split. **Training tokens are identical either way**: the datum
is built from the rollout's final token sequence (prompt + completion, injected tokens included, just
masked), which is exactly the sequence an in-engine budget would have produced — the two-call
protocol is invisible to `forward_backward`.

### Metrics to log every eval step
- `behavior_rate` — **primary.** Per env.
- For every monitor (train-against *and* held-out): accuracy / AUROC vs. ground truth.
- Monitor penalty, task reward, total reward, KL.
- Derived: pairwise degradation `D[i][j]` once we have multiple detectors.
