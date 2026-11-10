#!/usr/bin/env bash
# serve_mbpp_judges_4gpu.sh — the judge layout of experiments/configs/mbpp_q3_8b_qmonitors on a 4-GPU box
# (written for 4×H100 NVL, 94 GB each; works on any 4 cards that each fit one FP8 q35 engine):
#
#   GPU 0  Qwen3-1.7B (q17_*)             :8011  (--gpu-memory-utilization 0.3)
#   GPU 0  probe server(s), Qwen3-8B bf16 ~17–20 GB each, NOT capped — the rest of GPU 0 is its headroom
#   GPU 1  Qwen3.5-35B-A3B-FP8  engine    :8003  (0.92)
#   GPU 2  Qwen3.5-35B-A3B-FP8  engine    :8004  (0.92)
#   GPU 3  Qwen3.5-35B-A3B-FP8  engine    :8005  (0.92)
#   :8002  scripts/load_balancer.py over 8003–8005, equal weights (identical engines) — the q35_* base_url
#
# Prereq and behaviour as serve_mbpp_judges_2gpu.sh (setup_vllm.sh once; stops earlier copies; each server
# detached and health-checked before the next; logs in logs/vllm/<name>.log; nothing supervises them).
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
stop "[l]oad_balancer.py --listen-port 8002"
stop "[v]llm serve Qwen/Qwen3.5-35B-A3B-FP8"
stop "[v]llm serve Qwen/Qwen3-1.7B"

wait_up() {  # wait_up <name> <port>
  for _ in $(seq 1 180); do
    curl -sf -m 3 "localhost:$2/health" >/dev/null && { echo "  $1 up on :$2"; return 0; }
    sleep 5
  done
  echo "✗ $1 not healthy on :$2 after 15 min — see $LOG/$1.log"; tail -n 20 "$LOG/$1.log"; exit 1
}

q35() {  # q35 <gpu> <port> <mem frac>
  local name=qwen3.5-35b-$2
  echo "starting $name on GPU $1 (mem $3)"
  CUDA_VISIBLE_DEVICES=$1 setsid -f "$VLLM" serve Qwen/Qwen3.5-35B-A3B-FP8 --port "$2" \
    --reasoning-parser qwen3 --scheduling-policy priority --max-num-seqs 512 --kv-cache-dtype fp8 \
    --gpu-memory-utilization "$3" --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --compilation-config '{"max_cudagraph_capture_size": 1024}' --disable-uvicorn-access-log \
    >"$LOG/$name.log" 2>&1 </dev/null
  wait_up "$name" "$2"
}

q35 1 8003 0.92
q35 2 8004 0.92
q35 3 8005 0.92
echo "starting qwen3-1.7b-8011 on GPU 0 (mem 0.3)"
CUDA_VISIBLE_DEVICES=0 setsid -f "$VLLM" serve Qwen/Qwen3-1.7B --port 8011 --reasoning-parser qwen3 \
  --scheduling-policy priority --max-num-seqs 512 --kv-cache-dtype fp8 --max-model-len 32768 \
  --gpu-memory-utilization 0.3 --disable-uvicorn-access-log >"$LOG/qwen3-1.7b-8011.log" 2>&1 </dev/null
wait_up qwen3-1.7b-8011 8011
echo "starting lb-8002 → 8003 8004 8005"
setsid -f .venv/bin/python scripts/load_balancer.py --listen-port 8002 --backend-ports 8003 8004 8005 --weights 1 1 1 \
  >"$LOG/lb-8002.log" 2>&1 </dev/null
wait_up lb-8002 8002
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
