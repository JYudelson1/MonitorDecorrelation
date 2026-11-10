#!/usr/bin/env bash
# serve_mbpp_judges_1gpu.sh — the judge layout of experiments/configs/mbpp_q3_8b_qmonitors on a 1×H200 box:
# GPU 0 of serve_mbpp_judges_2gpu.sh on its own, with the one q35 engine on :8002 directly (no load balancer),
# so the q35_* configs' base_url is unchanged.
#
#   GPU 0  Qwen3.5-35B-A3B-FP8            :8002  (--gpu-memory-utilization 0.5 ≈ 72 GB)
#   GPU 0  Qwen3-1.7B (q17_*)             :8011  (0.15)
#   GPU 0  probe server, Qwen3-8B bf16 ~22 GB, NOT capped — what's left on the GPU is its headroom
#
# Capacity: ~1/3 of a full q35 engine (2-GPU box ≈ 1.35 engines) — about one training row at the per-row
# speed four rows got on a 2-GPU box. Prereq and behaviour as serve_mbpp_judges_2gpu.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
R=$(pwd)
VLLM=$R/.venv-vllm/bin/vllm
[[ -x "$VLLM" ]] || { echo "no $VLLM — run scripts/setup_vllm.sh once first"; exit 1; }
export CUDA_HOME=$R/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
LOG=logs/vllm
mkdir -p "$LOG"

stop() {  # stop() <pgrep -f pattern>
  local pids
  pids=$(pgrep -f "$1" || true)
  [[ -z "$pids" ]] && return 0
  echo "stopping: $1 ($pids)"
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f "$1" >/dev/null || return 0; sleep 2; done
  kill -9 $(pgrep -f "$1") 2>/dev/null || true; sleep 5
}
stop "[v]llm serve Qwen/Qwen3.5-35B-A3B-FP8"
stop "[v]llm serve Qwen/Qwen3-1.7B"

wait_up() {  # wait_up <name> <port>
  for _ in $(seq 1 180); do
    curl -sf -m 3 "localhost:$2/health" >/dev/null && { echo "  $1 up on :$2"; return 0; }
    sleep 5
  done
  echo "✗ $1 not healthy on :$2 after 15 min — see $LOG/$1.log"; tail -n 20 "$LOG/$1.log"; exit 1
}

echo "starting qwen3.5-35b-8002 on GPU 0 (mem 0.5)"
CUDA_VISIBLE_DEVICES=0 setsid -f "$VLLM" serve Qwen/Qwen3.5-35B-A3B-FP8 --port 8002 \
  --reasoning-parser qwen3 --scheduling-policy priority --max-num-seqs 512 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.5 --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --compilation-config '{"max_cudagraph_capture_size": 1024}' --disable-uvicorn-access-log \
  >"$LOG/qwen3.5-35b-8002.log" 2>&1 </dev/null
wait_up qwen3.5-35b-8002 8002
echo "starting qwen3-1.7b-8011 on GPU 0 (mem 0.15)"
CUDA_VISIBLE_DEVICES=0 setsid -f "$VLLM" serve Qwen/Qwen3-1.7B --port 8011 --reasoning-parser qwen3 \
  --scheduling-policy priority --max-num-seqs 512 --kv-cache-dtype fp8 --max-model-len 32768 \
  --gpu-memory-utilization 0.15 --disable-uvicorn-access-log >"$LOG/qwen3-1.7b-8011.log" 2>&1 </dev/null
wait_up qwen3-1.7b-8011 8011
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
