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
| `scripts/queue_runs.sh` | **Batch launcher.** Fan a config (or dir) × seeds `0..N-1` into ≤J parallel runs; per-job logs, resume (`-s`), detach (`-D`). | one `data/runs/<run>/` per (config×seed) |
| `experiments/make_mbpp_matrix_configs.py` | Generate the MBPP-Honeypot matrix configs (6 monitors + control = 7 rows) → `configs/mbpp_matrix/row_*.json`. | matrix config files |
| `experiments/train_probe.py` | Train a white-box linear probe on contrastive deception/hacking data (base model, Atlas §4.3). `--preset {simple_deception,diverse_deception,mbpp}`. | `data/probes/<model>/<datasets>/` |
| `experiments/build_syco_pairs.py` | Build real-CoT sycophancy contrastive pairs (resample policy, label by ground-truth/judge/both) for the on-domain probe. | a `*.jsonl` of pairs |
| `experiments/test_many_monitors.py` | Monitor bakeoff: policy behavior frequency + each candidate monitor's detection AUROC (class-balanced). | `data/runs/<name>/monitor_bakeoff.json` |
| `experiments/eval_probes_on_run.py` | Post-hoc: score a probe over a run's `rollouts.jsonl`, per step (original-model probing). | `data/runs/<run>/probe_eval_<name>.jsonl` |
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
env subset n_steps batch_size group_size eval_every eval_size max_tokens penalty_coef kl_coef
kl_discount_factor lora_rank lr seed monitors`. Each monitor: `{kind:"cot", name, role, model_id,
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

**W&B syncing is automatic iff you're logged in.** `run_experiment.py` picks `wandb_mode` per run:
`online` if a credential is configured locally (`wandb login` → `~/.netrc`, or `WANDB_API_KEY` set),
else `offline`. The check is purely local (no network, never prompts), so it's disconnect-safe. Set
`WANDB_MODE` explicitly (`online`/`offline`/`disabled`) to override. The chosen mode is printed in each
run's banner. Offline runs lose nothing — upload later with `wandb sync wandb/offline-run-*`.

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
