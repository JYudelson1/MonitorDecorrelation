#!/usr/bin/env bash
# Train Inkling-Small on hardest-1024 Codeforces with the ImpossibleBench prompt (tasks POSSIBLE),
# logging to Weights & Biases.
#
#   scripts/train_codeforces_ib_inkling.sh                        # the config as-is (40 steps)
#   scripts/train_codeforces_ib_inkling.sh --set n_steps=2 batch_size=2 group_size=4   # smoke test
#   scripts/train_codeforces_ib_inkling.sh --set seed=1 run_name=cfib_Inkling-Small_s1   # another seed
#
# Everything after the script name is forwarded to experiments/run_experiment.py, so `--set key=value`
# overrides any config field. Env:
#   ALLOW_OFFLINE_WANDB=1   don't abort when this machine has no wandb credential (logs locally only)
#
# The env needs the built dataset (data/codeforces_ib/hardest1024.jsonl.gz):
#   uv run python experiments/build_codeforces_ib_data.py --n-hardest 1024
#
# Hyperparameters live in the config (experiments/configs/codeforces_ib_inkling_small.json). The one
# that is NOT auto-derived is the learning rate: tinker-cookbook's LoRA-LR heuristic explicitly refuses
# Inkling ("not yet calibrated"), so the config pins lr=2e-4 by hand — see the config for the rationale.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="experiments/configs/codeforces_ib_inkling_small.json"

# --- fail loudly on a missing credential, rather than 20 minutes into a run ---------------------
# TINKER_API_KEY / WANDB_API_KEY normally come from .env (loaded by run_experiment.py); accept either
# a shell export or that file.
have_key() {  # have_key VAR -> the var is set in the shell, or assigned in .env
    [ -n "${!1:-}" ] && return 0
    [ -f .env ] && grep -qE "^[[:space:]]*(export[[:space:]]+)?$1[[:space:]]*=[[:space:]]*\S" .env
}

if ! have_key TINKER_API_KEY; then
    echo "ERROR: TINKER_API_KEY is not set (shell or .env) — the policy trains on tinker." >&2
    exit 1
fi

# W&B: the runner silently falls back to offline logging when logged out. This run is supposed to plot
# to wandb, so treat "not logged in" as a configuration error unless it is waived explicitly.
if ! have_key WANDB_API_KEY && [ ! -f "${HOME}/.netrc" ] && [ -z "${WANDB_MODE:-}" ]; then
    if [ "${ALLOW_OFFLINE_WANDB:-0}" != "1" ]; then
        echo "ERROR: no wandb credential found (WANDB_API_KEY in the shell or .env, or ~/.netrc)." >&2
        echo "       This run is meant to plot to W&B. Fix with 'wandb login', or re-run with" >&2
        echo "       ALLOW_OFFLINE_WANDB=1 to log locally only (data/runs/<run_name>/)." >&2
        exit 1
    fi
    echo "WARNING: no wandb credential — logging offline only (ALLOW_OFFLINE_WANDB=1)." >&2
fi

if [ ! -f data/codeforces_ib/hardest1024.jsonl.gz ] && ! printf '%s\n' "$@" | grep -q "subset="; then
    echo "ERROR: data/codeforces_ib/hardest1024.jsonl.gz is missing — build it first:" >&2
    echo "       uv run python experiments/build_codeforces_ib_data.py --n-hardest 1024" >&2
    exit 1
fi

echo "==> codeforces_ib × Inkling-Small   config=${CONFIG}"
exec uv run python experiments/run_experiment.py --config "${CONFIG}" "$@"
