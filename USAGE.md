# Usage — executable entrypoints

The runnable scripts (everything else under `src/` is library code). All run via `uv`; secrets load
from `.env` (`TINKER_API_KEY`, `OPENROUTER_API_KEY`, …).

> **GPU acceleration (rented CUDA box only).** The white-box probe forward passes use Qwen3.5's hybrid
> linear-attention layers, which are slow on macOS/MPS (torch fallback — no CUDA kernels). On a CUDA
> machine, install the optional kernels with `uv sync --extra gpu` (flash-attn + flash-linear-attention
> + causal-conv1d). They're CUDA/Linux-only and gated by `sys_platform == 'linux'`, so a plain
> `uv sync` on a Mac never tries to build them. (There's no "CUDA-available" marker in pyproject; Linux
> is the proxy.)
>
> **Worse than slow — MPS OOMs the 4B probe at 4096-token sequences** (single-sequence O(T²) attention,
> ~27 GiB peak). `batch_size=1` doesn't help; only shorter sequences do. So real probe runs are
> **GPU-only** — full step-by-step in [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md). Local stays on the
> smoke config (`max_tokens=1024`) for plumbing validation.

| Script | What it does | Outputs |
| --- | --- | --- |
| `experiments/run_experiment.py` | **The RL experiment runner.** GRPO with any mix of monitors (CoT judges + white-box probes), each train-against or held-out, from a validated JSON/YAML config. | `data/runs/<run>/` + wandb (offline) |
| `experiments/train_probe.py` | Train a white-box linear probe on contrastive deception/sycophancy data (base model, Atlas §4.3). | `data/probes/<model>/<datasets>/` |
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
uv run python experiments/run_experiment.py --config experiments/configs/train_against_cot.json
uv run python experiments/run_experiment.py --config experiments/configs/full_matrix.json
# quick one-off overrides of top-level fields:
uv run python experiments/run_experiment.py --config <cfg> --set run_name=quick n_steps=2 batch_size=2
```

Config fields (top level): `run_name experiment description policy backend{tinker,transformers}
subset n_steps batch_size group_size max_tokens penalty_coef lora_rank lr seed n_prompts_pool
monitors`. Each monitor: `{kind:"cot", name, role, model_id, behavior?, threshold?}` or
`{kind:"probe", name, role, probe_path, probe_model?, threshold?, batch_size?}`.

## Probes (white-box)

```bash
# train an ood deception probe on the policy's base model (probe lands in data/probes/<model>/<datasets>)
uv run python experiments/train_probe.py --model Qwen/Qwen3.5-4B --datasets doluschat
# on-domain sycophancy probe (needs real-CoT pairs first):
uv run python experiments/build_syco_pairs.py --policy Qwen/Qwen3.5-4B --label-by both \
    --out data/probes/syco_cot_pairs.jsonl
uv run python experiments/train_probe.py --model Qwen/Qwen3.5-4B --datasets sycophancy_cot \
    --follow-up-question "Did you answer based on the user's identity?" --follow-up-answer "No"
```

Two probing modes (see `docs/MONITORS.md`): **original-model** (frozen base on the text; tinker
backend, post-hoc or in-reward) and **trained-model** (live policy activations; local
`TransformersBackend`). Probes must use the **same base model** as the policy (off-model is unreliable).

## Run outputs (`data/runs/<run>/`)
- `run_info.json` — full validated config + which model each monitor was (train-against vs held-out).
- `metrics.jsonl` — one row per step (ground-truth rate, reward, per-monitor accuracy/AUROC). Feeds `plot_run.py`.
- `rollouts.jsonl` — sampled rollouts with metadata (env oracle, per-monitor score/label, reward, advantage).
- `ground_truth.png`, `monitors.png` — auto-plotted on finish (or via `plot_run.py`).

> Note: don't nest `&`/`nohup` when backgrounding a run — it orphans the process. Launch it as a
> single foreground command (the harness/your shell handles backgrounding).
