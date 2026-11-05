#!/usr/bin/env bash
# serve_small.sh <model> <port> <gpu_mem_frac> — one small Qwen judge on GPU 0, same vLLM binary/CUDA_HOME as
# scripts/setup_vllm.sh, --reasoning-parser qwen3 (required for the reasoning split + thinking_token_budget).
set -euo pipefail
R=/workspace/MonitorDecorrelation
export CUDA_HOME=$R/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$PATH
model=$1 port=$2 frac=$3
extra=()
[[ "$model" == *Qwen3.5* ]] && extra+=(--limit-mm-per-prompt '{"image":0,"video":0}')
cd $R
exec env CUDA_VISIBLE_DEVICES=0 $R/.venv-vllm/bin/vllm serve "$model" --port "$port" \
  --reasoning-parser qwen3 --max-num-seqs 256 --kv-cache-dtype fp8 --max-model-len 32768 \
  --gpu-memory-utilization "$frac" --disable-uvicorn-access-log "${extra[@]}"
