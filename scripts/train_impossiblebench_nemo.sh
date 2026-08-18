#!/usr/bin/env bash
# Train Qwen3.5 on the IMPOSSIBLE subset of Impossible-LiveCodeBench with the **nemo** backend
# (NeMo-RL: Megatron + LoRA, colocated vLLM) on this machine's GPUs.
#
#   scripts/train_impossiblebench_nemo.sh                                   # config as-is, all GPUs
#   scripts/train_impossiblebench_nemo.sh --set n_gpus=1                    # pin the GPU count
#   scripts/train_impossiblebench_nemo.sh --set policy=Qwen/Qwen3.5-35B-A3B-Base run_name=ib_35b
#   scripts/train_impossiblebench_nemo.sh --set n_steps=2 batch_size=2 group_size=4 eval_size=2  # smoke
#   scripts/train_impossiblebench_nemo.sh --set thinking_budget=1024                # cap the CoT
#
# Everything after the script name is forwarded to experiments/run_experiment.py, so `--set key=value`
# overrides any config field (dict fields take JSON: --set 'nemo_options={"train_micro_batch_size":2}').
#
# Unlike the tinker script there is no TINKER_API_KEY to check — the policy trains here. What this
# does check is that NeMo-RL is actually checked out, because it is a submodule and a plain
# `git clone` leaves it empty.
#   ALLOW_OFFLINE_WANDB=1   don't abort when this machine has no wandb credential (logs locally only)
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="experiments/configs/impossiblebench_nemo.json"
NEMO_RL_ROOT="${MD_NEMO_RL_ROOT:-third_party/nemo-rl}"

if [ ! -d "${NEMO_RL_ROOT}/nemo_rl" ]; then
    echo "ERROR: NeMo-RL is not checked out at ${NEMO_RL_ROOT}." >&2
    echo "       It is a git submodule of this repo — run:" >&2
    echo "           git submodule update --init --recursive" >&2
    echo "       (or clone this repo with --recursive), or set MD_NEMO_RL_ROOT." >&2
    exit 1
fi
if [ ! -x "${NEMO_RL_ROOT}/.venv/bin/python" ]; then
    echo "NOTE: no NeMo-RL venv at ${NEMO_RL_ROOT}/.venv — building it now (this takes a while)." >&2
    UV_PROJECT_ENVIRONMENT="$(cd "${NEMO_RL_ROOT}" && pwd)/.venv" uv sync --locked --directory "${NEMO_RL_ROOT}"
fi

have_key() {  # have_key VAR -> the var is set in the shell, or assigned in .env
    [ -n "${!1:-}" ] && return 0
    [ -f .env ] && grep -qE "^[[:space:]]*(export[[:space:]]+)?$1[[:space:]]*=[[:space:]]*\S" .env
}

if ! have_key WANDB_API_KEY && [ ! -f "${HOME}/.netrc" ] && [ -z "${WANDB_MODE:-}" ]; then
    if [ "${ALLOW_OFFLINE_WANDB:-0}" != "1" ]; then
        echo "ERROR: no wandb credential found (WANDB_API_KEY in the shell or .env, or ~/.netrc)." >&2
        echo "       Fix with 'wandb login', or re-run with ALLOW_OFFLINE_WANDB=1 to log locally only." >&2
        exit 1
    fi
    echo "WARNING: no wandb credential — logging offline only (ALLOW_OFFLINE_WANDB=1)." >&2
fi

echo "==> impossiblebench x NeMo-RL   config=${CONFIG}   nemo-rl=${NEMO_RL_ROOT}"
exec uv run python experiments/run_experiment.py --config "${CONFIG}" "$@"
