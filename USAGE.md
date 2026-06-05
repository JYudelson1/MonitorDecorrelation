# Usage — executable entrypoints

The runnable scripts (everything else under `src/` is library code). All run via `uv`; secrets load
from `.env` (`TINKER_API_KEY`, `OPENROUTER_API_KEY`, …).

| Script | What it does | Outputs |
| --- | --- | --- |
| `experiments/train_against_cot.py` | **The RL experiment.** GRPO against a CoT monitor, holding out another; logs ground-truth rate + per-monitor accuracy/AUROC. | `data/runs/<run>/` + wandb (offline) |
| `scripts/plot_run.py` | Render a run's metrics to PNGs locally (no W&B server). | `data/runs/<run>/*.png` |
| `tests/test_*.py` | Offline unit tests (env parsing/oracle, GRPO alignment, metrics). No network. | stdout |

## Common commands

```bash
# 0. one-time: install deps + add keys
uv sync && cp .env.example .env   # then edit .env

# 1. the experiment (defaults: Qwen3-8B, 16x8 rollouts x 40 steps, auto-LR)
uv run python experiments/train_against_cot.py --run-name myrun
#    smaller/cheaper:
uv run python experiments/train_against_cot.py --run-name myrun \
    --batch-size 8 --group-size 8 --n-steps 20 --strong-monitor meta-llama/llama-3.3-70b-instruct

# 2. plot a finished (or in-progress) run
uv run python scripts/plot_run.py --run myrun

## `train_against_cot.py` flags
`--run-name --base-model --n-steps --batch-size --group-size --max-tokens --penalty-coef
--lora-rank --lr (default = TM-estimated LoRA LR) --weak-monitor --strong-monitor`

## Run outputs (`data/runs/<run>/`)
- `run_info.json` — full config + **which model each monitor was** (train-against vs held-out), policy, subset, LR. Decode the run here; `plot_run.py` reads it to label which monitor was trained against.
- `metrics.jsonl` — one row per step (ground-truth rate, reward, per-monitor accuracy/AUROC, …). Feeds `plot_run.py`.
- `rollouts.jsonl` — sampled rollouts with metadata (env oracle, per-monitor score/label, reward, advantage, `extra`). Fraction kept = `LoggingConfig.log_fraction`.
- `ground_truth.png`, `monitors.png` — after `plot_run.py`.
- wandb offline run under `wandb/` (sync later with `wandb sync`, or just use the PNGs).

> Note: don't nest `&`/`nohup` when backgrounding a run — it orphans the process. Launch it as a
> single foreground command (the harness/your shell handles backgrounding).
