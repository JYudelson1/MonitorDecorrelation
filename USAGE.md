# Usage — executable entrypoints

The runnable scripts (everything else under `src/` is library code). All run via `uv`; secrets load
from `.env` (`TINKER_API_KEY`, `OPENROUTER_API_KEY`, …).

> **Real runs are GPU (rented CUDA box); local MPS is for plumbing only.** The current experiment is
> **Qwen3-8B** — RL on tinker (remote) + white-box probe forward passes on the base model. Probe scoring
> at 2048-token sequences is GPU-preferred; on a CUDA machine `uv sync --extra gpu` adds the optional
> kernels (flash-attn etc., Linux-gated so a plain `uv sync` on a Mac skips them). Full step-by-step in
> [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md).
>
> *(Legacy caveat for the Qwen3.5-4B env: its hybrid linear-attention OOMs MPS at 4096 tokens —
> single-sequence O(T²), ~27 GiB; `batch_size=1` doesn't help, only shorter sequences do. Not on the
> Qwen3-8B path.)*

| Script | What it does | Outputs |
| --- | --- | --- |
| `experiments/run_experiment.py` | **The RL experiment runner.** GRPO with any mix of monitors (CoT judges + white-box probes), each train-against or held-out, from a validated JSON/YAML config. | `data/runs/<run>/` + wandb |
| `scripts/queue_runs.sh` | **Batch launcher.** Fan a config (or dir) × seeds `0..N-1` into ≤J parallel runs; per-job logs, resume (`-s`), detach (`-D`), shared probe server (`-P`). | one `data/runs/<run>/` per (config×seed) |
| `experiments/probe_server.py` | Shared frozen-model activation server so parallel runs don't each load the base model (`queue_runs.sh -P` starts it). | localhost server |
| `experiments/make_mbpp_matrix_configs.py` | Generate the matrix configs (`--penalty-coef`, `--tag`, `--n-steps`) → `configs/mbpp_matrix[_lowpen]/row_*.json`. | matrix config files |
| `experiments/analyze_coupling.py` | **The headline analysis.** Directed detector-coupling β(A→B) over a set of runs → pooled (7b) + conditioned-on-target (10) heatmaps + CIs. | `data/runs/_coupling/*.png` |
| `experiments/train_probe.py` | Train a white-box linear probe on contrastive deception/hacking data (base model, Atlas §4.3). `--preset {simple_deception,diverse_deception,mbpp}`. | `data/probes/<model>/<datasets>/` |
| `experiments/build_syco_pairs.py` | Build real-CoT sycophancy contrastive pairs (resample policy, label by ground-truth/judge/both) for the on-domain probe. | a `*.jsonl` of pairs |
| `experiments/test_many_monitors.py` | Monitor bakeoff: policy behavior frequency + each candidate monitor's detection AUROC (class-balanced). | `data/runs/<name>/monitor_bakeoff.json` |
| `experiments/eval_probes_on_run.py` | Post-hoc: score a probe over a run's `rollouts.jsonl`, per step (original-model probing). | `data/runs/<run>/probe_eval_<name>.jsonl` |
| `scripts/train_impossiblebench_nemo.sh` | **RL Qwen3.5 on ImpossibleBench with the `nemo` backend** (NeMo-RL, this machine's GPUs). Checks the submodule + venv, builds the venv if missing, forwards `--set …`. | `data/runs/ib_Qwen3.5-4B_nemo_s0/` + wandb |
| `scripts/train_impossiblebench_inkling.sh` | **RL Inkling-Small on the ImpossibleBench impossible subset** (oneoff + conflicting), plotting to W&B. Thin wrapper over the runner: checks credentials up front, forwards `--set …`. | `data/runs/ib_Inkling-Small_s0/` + wandb |
| `scripts/plot_run.py` | Render a run's metrics to PNGs locally (no W&B server). | `data/runs/<run>/*.png` |
| `tests/test_*.py` | Offline unit tests (env, GRPO alignment, metrics, probe logic). | stdout (run via `uv run pytest`) |

## The experiment runner (config-driven)

A run is fully described by a **schema-validated** JSON or YAML config (see `experiments/configs/`):
policy, env, RL hyperparams, and a `monitors` list of mixed kinds (`cot` / `probe`), each
`role: train_against` or `held_out`. A malformed config is rejected at load (pydantic, extra keys
forbidden). So "train against CoT", "train against a probe", "train against strong CoT while holding
out weak CoT + two probes" are all just different configs — no per-experiment scripts.

```bash
uv run python experiments/run_experiment.py --config experiments/configs/mbpp_matrix/row_control.json
uv run python experiments/run_experiment.py --config experiments/configs/train_against_cot.json
# quick one-off overrides of top-level fields:
uv run python experiments/run_experiment.py --config <cfg> --set run_name=quick n_steps=2 batch_size=2
```

Config fields (top level): `run_name experiment description policy backend{tinker,transformers}
env{sycophancy,mbpp_honeypot,impossiblebench} subset env_options n_steps batch_size group_size
eval_every eval_size eval_samples_per_prompt max_tokens thinking_effort penalty_coef penalty_schedule
kl_coef kl_discount_factor lora_rank lr seed n_prompts_pool probe_server_url monitors`.
`monitors: []` is legal and means a pure-task-reward run (no monitor in the gradient, none measured).
`env_options` holds env-specific constructor kwargs (impossiblebench only) and `thinking_effort`
[0,1) conditions the reasoning length of TML-rendered policies (Inkling; ignored elsewhere). Each monitor: `{kind:"cot", name, role, model_id,
use_cot?, behavior?, threshold?}` (output monitor = `use_cot:false`) or `{kind:"probe", name, role,
probe_path, probe_model?, threshold?, batch_size?}`.

For running the full matrix across seeds, use `scripts/queue_runs.sh` (next section) rather than a
hand-rolled loop.

### Queueing many runs (seeds × configs, in parallel)

`scripts/queue_runs.sh` fans a config (or a whole directory of configs) across seeds `0..N-1` and runs
up to `J` at a time (`xargs -P`; BSD/GNU compatible). Each `(config × seed)` becomes its own run:
`run_name`'s `_s<n>` token is rewritten to `_s<seed>`, so every job lands in its own
`data/runs/<run>/` with its rollouts/metrics/plots. Full per-job output → `data/runs/<run>/run.log`;
the terminal shows one ▶/✓/✗ line per job and tails the log of any job that fails.

```bash
# the whole 7-row matrix, seeds 0-2, 4 in parallel:
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4 -d   # -d = dry-run (print plan only)
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4 -s   # -s = resume: skip finished runs
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4 -D   # -D = detach: nohup the batch,
                                                                       #   survives SSH disconnect
# ⭐ recommended for a real batch: -s -P -D (resume-safe · shared probe server · detached), high -j:
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix_lowpen -n 5 -j 12 -s -P -D
scripts/queue_runs.sh -c <cfg> -U http://127.0.0.1:8177 ...   # -U = reuse an already-running probe server
# extra --set overrides after `--` apply to every job (e.g. a fast smoke of the whole matrix):
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 1 -j 2 -- n_steps=3 batch_size=4
```

A run is marked done by a `data/runs/<run>/QUEUE_DONE` sentinel (written on success) — that's what `-s`
checks. Follow a single live job with `tail -f data/runs/<run>/run.log`. With `-D`, the plan still
prints to your terminal, then the batch re-execs under `nohup` into `data/runs/_batch_<ts>.log` and
returns a PID immediately (follow it with `tail -f data/runs/_batch_<ts>.log`).

**Shared probe server (`-P`).** Probes read activations by running the policy's text through a frozen
copy of the base model. Without `-P`, *each* run process loads its own ~16 GB copy, so `-j` is capped by
GPU memory (~4 on an 80 GB card). `-P` starts one `experiments/probe_server.py` (model read from the
first config's `policy`, port `$PROBE_PORT`, default 8177), points every run at it via
`PROBE_SERVER_URL`, and tears it down when the batch ends — so `-j` is then bounded by tinker/API limits
instead of memory. `-U http://host:port` reuses an already-running server. Per-config equivalent:
`{"probe_server_url": "..."}`. Start one standalone with
`uv run python experiments/probe_server.py --model Qwen/Qwen3-8B`.

### Analyzing a run set — directed coupling (the headline)

```bash
uv run python experiments/analyze_coupling.py 'data/runs/mbpp_Qwen3-8B_*lowpen' --bootstrap 3000
uv run python experiments/analyze_coupling.py 'data/runs/mbpp_*' --metric dprime_margin --out data/runs/_coup
```
β(A→B) = per unit detector A's reliability (d′) degrades within a run, how much does B's — the DIRECTED,
asymmetric coupling. Writes `coupling_pooled.png` (all runs, driver→responder) + `coupling_by_target.png`
(conditioned on the training target: *does training against i degrade held-out j?*) + `coupling.json`.
`--metric auroc` (→ d′, resolution-aware clip) or `dprime_margin` (native, non-saturating); `--bootstrap N`
adds per-cell 90% CIs. Reads `eval_metrics.jsonl` (+ dir name for the target). See `docs/DEGRADATION_METRICS.md`.

**W&B syncing is automatic iff you're logged in.** `run_experiment.py` picks `wandb_mode` per run:
`online` if a credential is configured locally (`wandb login` → `~/.netrc`, or `WANDB_API_KEY` set),
else `offline`. The check is purely local (no network, never prompts), so it's disconnect-safe. Set
`WANDB_MODE` explicitly (`online`/`offline`/`disabled`) to override. The chosen mode is printed in each
run's banner. Offline runs lose nothing — upload later with `wandb sync wandb/offline-run-*`.

## ImpossibleBench × Inkling-Small

```bash
# full run (40 steps, 16x8 rollouts/step) — plots to W&B, aborts up front if credentials are missing
scripts/train_impossiblebench_inkling.sh

# smoke test first (2 steps, tiny batch, short completions) — same code path, minutes not hours
scripts/train_impossiblebench_inkling.sh --set run_name=ib_smoke n_steps=2 batch_size=2 group_size=4 \
    eval_size=2 eval_every=1 max_tokens=2048 thinking_effort=0.1
```

Everything after the script name is forwarded to `run_experiment.py`. Config:
`experiments/configs/impossiblebench_inkling_small.json`.

- **The setting.** `subset: "impossible"` = `oneoff` + `conflicting`, the two mutated splits where no
  correct implementation can pass — so `behavior_present` ("passed the tests") *is* the cheating oracle.
  One prompt → one assistant message → one fenced code block; no tool loop, no retry-on-failure.
- **Watch these metrics.** `train/behavior_rate` (= cheat rate, the primary signal),
  `train/reward/task_mean` (fraction of the check passing), `train/unparsed_rate` (answers with no
  codeblock — spikes mean `max_tokens` is cutting the model off mid-thought).
- **Token counts** come free with every run: `tokens/{input,output}_total` (the whole batch at that
  step), `tokens/{input,output}_per_rollout`, `tokens/output_max`, `tokens/truncated_rate` (fraction of
  completions that hit `max_tokens` — read output length next to it, since a censored length flattens
  out), and `tokens/cum_total` (running total, train batches only). They're in `metrics.jsonl` /
  `eval_metrics.jsonl`, in W&B under `train/tokens/*` and `eval/tokens/*`, plotted to
  `data/runs/<run>/{train,eval}/tokens.png`, and echoed on each step line (`tok=…in/…out per rollout`).
- **`thinking_effort`** conditions Inkling's reasoning length (0.1 minimal … 0.9 high, the model
  default). High effort on these tasks costs thousands of CoT tokens per rollout, so it must be paired
  with a large `max_tokens` or the answer never arrives (the rollout is `unparsed`).
- **No LLM judges are configured** (`monitors: []`) — this is the plain-RL baseline for the env. Add
  monitors the same way as the MBPP matrix rows once a decorrelation row is wanted; the CoT judges need
  `OPENROUTER_API_KEY`.
- **Untrusted code executes locally** (subprocess + per-statement alarm + address-space rlimit, not a
  sandbox). See the warning in `envs/impossiblebench.py`.

## The `nemo` backend (local multi-GPU, NeMo-RL)

`backend: "nemo"` trains on **this machine's GPUs** via [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
(Megatron policy workers + LoRA, colocated vLLM generation) instead of on tinker. Everything else is
unchanged: same envs, same monitors, same reward, same `data/runs/<run>/` artifacts, so
`analyze_coupling.py` / `eval/degradation.py` / `scripts/plot_run.py` read a nemo run as usual.

```bash
# one-time (or after a fresh clone without --recursive):
git submodule update --init --recursive

# every config field works the same; n_gpus defaults to every GPU on the machine
scripts/train_impossiblebench_nemo.sh
scripts/train_impossiblebench_nemo.sh --set n_gpus=1
scripts/train_impossiblebench_nemo.sh --set policy=Qwen/Qwen3.5-35B-A3B-Base run_name=ib_35b_nemo

# any existing config can be moved onto it
uv run python experiments/run_experiment.py --config experiments/configs/full_matrix.yaml \
    --set backend=nemo policy=Qwen/Qwen3.5-4B-Base
```

- **One config file.** `experiments/configs/nemo/grpo.yaml` is the only NeMo-RL config in the repo.
  Everything that varies between runs is an `MD_*` env var derived from the experiment config by
  `backends/nemo/params.py`; the fully-resolved result is saved as `data/runs/<run>/nemo_config.yaml`.
- **nemo-only config fields.** `n_gpus` (default: all visible GPUs), `max_prompt_tokens` (NeMo-RL
  budgets ONE sequence length for prompt+completion, so `max_total_sequence_length = max_tokens +
  max_prompt_tokens`), and `nemo_options` — the last-word override of any derived knob, e.g.
  `--set 'nemo_options={"train_micro_batch_size":2,"checkpoint_enabled":false}'`.
- **Hyperparameters follow the tinker backend**, not the transformers one: mean-centred group
  advantages (no std normalisation, no leave-one-out), one optimizer step per rollout batch, KL applied
  to the reward, Adam(0.9, 0.95) with no weight decay or gradient clipping, constant LR, LoRA alpha 32,
  temperature 1.0. The reasoning for each is inline in `experiments/configs/nemo/grpo.yaml`.
- **`.env` is propagated** to the NeMo-RL process (`WANDB_API_KEY`, and monitor API keys once a config
  lists monitors). An explicit shell export still wins over `.env`.
- **Two virtualenvs.** NeMo-RL pins Python 3.13 + torch 2.11 + Megatron + vLLM and lives in
  `third_party/nemo-rl/.venv`; this project keeps its own. Do **not** point both at the same
  `UV_PROJECT_ENVIRONMENT` — each `uv run` would uninstall the other's dependencies. The launcher pins
  NeMo-RL's explicitly for exactly this reason. On a NeMo-RL container the image exports
  `UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv`, which this project's `uv run` will claim; that is fine
  (the backend never uses it) but it does mean the image's prebuilt driver venv is replaced.
- **What is NOT carried over.** `thinking_effort` (Inkling/TML-only; nemo policies use the HF chat
  template), and `lr=None` still resolves through tinker-cookbook's LoRA-LR heuristic on this side
  before being passed down.

## Probes (white-box)

```bash
# the OOD deception probe (probe_ood) on the policy's base model → data/probes/<model>/diverse_deception
uv run python experiments/train_probe.py --model Qwen/Qwen3-8B --preset diverse_deception
# the iid MBPP reward-hack probe (probe_iid), split-safe (disjoint train task_ids, no RL-eval leakage):
uv run python experiments/train_probe.py --model Qwen/Qwen3-8B --preset mbpp \
    --out data/probes/Qwen3-8B/mbpp_honeypot
```
Both Qwen3-8B probes are already trained + committed under `data/probes/Qwen3-8B/` — retrain only if the
model or split changes.

Two probing modes (see `docs/MONITORS.md`): **original-model** (frozen base on the text; tinker
backend, post-hoc or in-reward) and **trained-model** (live policy activations; local
`TransformersBackend`). Probes must use the **same base model** as the policy (off-model is unreliable).

## Probe-investigation scripts (2026-06-15 — see docs/PROBES.md)

```bash
# Build CoT contrastive pairs on-policy, generic over env (measures natural base rate, nudges only the
# rare class as last resort; natural-only by default). --judge-model uses a stronger MASK lie-oracle.
uv run python experiments/build_cot_pairs.py --env mask --policy Qwen/Qwen3-8B --n-prompts 300 --judge-model anthropic/claude-sonnet-4.6

# CoT-vs-noCoT transfer 2x2 on a *_cot_pairs.jsonl. --within-generation / --preserve-thinking read CoT
# (the follow-up default is no-CoT on thinking models — the CoT-stripping bug). Prints mean|Δ(CoT,noCoT)|.
uv run python experiments/run_cot_transfer.py --pairs data/probes/mask_cot_pairs_qwen3-8b.jsonl --model Qwen/Qwen3-8B --within-generation

# Transfer a saved probe to OOD targets (MASK subtle rollouts + MBPP reward-hacking) — the OA cross-domain test.
uv run python experiments/run_probe_transfer.py --probe data/probes/Qwen3-8B/diverse_deception --model Qwen/Qwen3-8B

# One-offs: check_mask_judge.py (haiku vs sonnet label agreement), run_mask_clean_probe.py (relabel+retrain),
# run_mask_blatant_probe.py (train on MASK propositions, eval on subtle rollouts).
```

`train_probe.py` gains `--preset {simple_deception,diverse_deception,mbpp}` (OA recipes); the probe seam
gains `extract_activations(progress=True, within-gen via follow_up=None, preserve_thinking=...)`.

## Run outputs (`data/runs/<run>/`)
- `run_info.json` — full validated config + which model each monitor was (train-against vs held-out) + `env.behavior_name`.
- `config.json` — the effective config (after `--set` overrides); re-run with `--config data/runs/<run>/config.json`.
- `metrics.jsonl` — per **train** step (ground-truth strict/loose/hardcoding rates, reward, train-against monitor score). 
- `eval_metrics.jsonl` — per **eval** (every `eval_every`) on the fixed held-out set: **every** monitor's AUROC/brier/class-split. **This is what the degradation matrix reads.**
- `eval_rollouts.jsonl` — one record per held-out rollout with every monitor's score+label (post-hoc recalibration).
- `rollouts.jsonl` — sampled training rollouts with metadata (env oracle, per-monitor score/label, reward).
- `train/` + `eval/` — each has `ground_truth.png` + `monitors.png` (labels derive from `behavior_name`), auto-plotted on finish.
- `final_checkpoint.txt` — tinker path of the saved final weights (7-day TTL); `run.log` (via `queue_runs.sh`); `QUEUE_DONE` sentinel on success.

> Note: don't nest `&`/`nohup` when backgrounding a run by hand — it orphans the process. Use
> `queue_runs.sh -D` (handles detachment cleanly) or launch as a single foreground command.
