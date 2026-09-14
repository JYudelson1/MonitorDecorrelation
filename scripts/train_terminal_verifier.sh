#!/usr/bin/env bash
# Train Qwen3-8B on the terminal-verifier env (rg_obfuscation / "Output supervision can obfuscate the
# CoT", multi-turn terminal setting) — the no-monitor CONTROL config by default.
#
#   scripts/train_terminal_verifier.sh                                   # the config as-is (40 steps)
#   scripts/train_terminal_verifier.sh --set run_name=tv_smoke n_steps=2 batch_size=4 group_size=4 \
#       eval_size=4 eval_every=1                                          # smoke test
#   scripts/train_terminal_verifier.sh --set seed=1 run_name=tv_Qwen3-8B_control_s1
#
# Everything after the script name is forwarded to experiments/run_experiment.py (`--set key=value`
# overrides any config field). Env: ALLOW_OFFLINE_WANDB=1 to proceed without a wandb credential.
#
# ⚠️ The policy runs arbitrary shell commands in throwaway temp dirs on THIS machine (timeout + minimal
# env, not a sandbox) — see envs/terminal_verifier.py. Fine on a dedicated box.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${TV_CONFIG:-experiments/configs/terminal_verifier_qwen3_8b_control.json}"

have_key() {  # have_key VAR -> the var is set in the shell, or assigned in .env
    [ -n "${!1:-}" ] && return 0
    [ -f .env ] && grep -qE "^[[:space:]]*(export[[:space:]]+)?$1[[:space:]]*=[[:space:]]*\S" .env
}

if ! have_key TINKER_API_KEY; then
    echo "ERROR: TINKER_API_KEY is not set (shell or .env) — the policy trains on tinker." >&2
    exit 1
fi
if ! have_key WANDB_API_KEY && [ ! -f "${HOME}/.netrc" ] && [ -z "${WANDB_MODE:-}" ]; then
    if [ "${ALLOW_OFFLINE_WANDB:-0}" != "1" ]; then
        echo "ERROR: no wandb credential found (WANDB_API_KEY in the shell or .env, or ~/.netrc)." >&2
        echo "       Fix with 'wandb login', or re-run with ALLOW_OFFLINE_WANDB=1 to log locally only." >&2
        exit 1
    fi
    echo "WARNING: no wandb credential — logging offline only (ALLOW_OFFLINE_WANDB=1)." >&2
fi

echo "==> terminal_verifier × Qwen3-8B   config=${CONFIG}"
exec uv run python experiments/run_experiment.py --config "${CONFIG}" "$@"
