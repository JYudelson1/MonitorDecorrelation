#!/usr/bin/env bash
# setup_vllm.sh — bring up the local Qwen judge servers that the terminal-verifier
# Qwen-matrix configs expect on :8001 (Qwen3-30B) and :8002 (Qwen3.5-35B).
#
#   2 GPUs:  GPU0 → Qwen3 :8001;  GPU1 → Qwen3.5 :8002
#   4 GPUs:  GPU0 → Qwen3 :8001;  GPU{1,2,3} → Qwen3.5 :{8003,8004,8005};
#            load balancer on :8002 → those three backends
#
# Requires exactly 2 or 4 visible GPUs, each an H200 (or ≥80 GiB so either FP8 MoE
# judge fits on one card with the concurrency settings below). Ctrl-C / exit kills every child.
#
# vLLM: uses `vllm` on PATH if present; otherwise installs into a dedicated
# `.venv-vllm/` (kept out of the project env so it doesn't fight setup_box.sh's torch).
set -euo pipefail
cd "$(dirname "$0")/.."

# Floor for non-H200 cards: FP8 weights are ~30–35 GB, but max-num-seqs 256–512 +
# speculative decoding needs H100-80-class headroom. nvidia-smi reports MiB.
MIN_MEM_MIB=$((80 * 1024))
VLLM_VENV=".venv-vllm"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found; need a GPU box"; exit 1; }
command -v uv >/dev/null || { echo "uv not on PATH; need uv to install/run vllm"; exit 1; }

ensure_vllm() {
  if command -v vllm >/dev/null; then
    VLLM=$(command -v vllm)
    echo "using vllm on PATH: $VLLM"
    return
  fi
  if [[ -x "$VLLM_VENV/bin/vllm" ]]; then
    VLLM="$VLLM_VENV/bin/vllm"
    echo "using vllm from $VLLM"
    return
  fi
  echo "vllm not found; installing into $VLLM_VENV/ (python 3.12, --torch-backend=auto)"
  uv venv --python 3.12 --seed "$VLLM_VENV"
  uv pip install --python "$VLLM_VENV" vllm --torch-backend=auto
  VLLM="$VLLM_VENV/bin/vllm"
  [[ -x "$VLLM" ]] || { echo "vllm install finished but $VLLM is missing"; exit 1; }
  echo "installed vllm → $VLLM"
}
ensure_vllm

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
NGPU=${#GPU_ROWS[@]}
case "$NGPU" in
  2|4) ;;
  *) echo "need exactly 2 or 4 GPUs, found $NGPU"; exit 1 ;;
esac

for row in "${GPU_ROWS[@]}"; do
  # csv: index, name, memory.total (MiB) — name may contain spaces
  idx=$(cut -d',' -f1 <<<"$row" | tr -d ' ')
  name=$(cut -d',' -f2 <<<"$row" | sed 's/^ *//;s/ *$//')
  mem=$(cut -d',' -f3 <<<"$row" | tr -d ' ' | cut -d. -f1)
  if [[ "$name" == *H200* ]]; then
    echo "GPU $idx: $name (${mem} MiB) — H200 OK"
  elif (( mem >= MIN_MEM_MIB )); then
    echo "GPU $idx: $name (${mem} MiB) — not H200, but ≥80 GiB so either model should fit on one card"
  else
    echo "GPU $idx: $name (${mem} MiB) — need H200 (or ≥80 GiB) to serve Qwen3-30B / Qwen3.5-35B FP8 on one GPU" >&2
    exit 1
  fi
done
echo "found $NGPU GPUs, all usable"

PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Qwen3-30B always on GPU 0 → :8001 (clients hardcode this).
CUDA_VISIBLE_DEVICES=0 "$VLLM" serve Qwen/Qwen3-30B-A3B-FP8 \
  --port 8001 \
  --reasoning-parser qwen3 \
  --scheduling-policy priority \
  --max-num-seqs 512 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --speculative-config '{"method": "eagle3", "model": "AngelSlim/Qwen3-a3B_eagle3", "num_speculative_tokens": 2}' \
  &
PIDS+=($!)
echo "Qwen3-30B-A3B-FP8 on GPU 0 → :8001 (pid ${PIDS[-1]})"

serve_q35() {
  local gpu=$1 port=$2
  CUDA_VISIBLE_DEVICES=$gpu "$VLLM" serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --port "$port" \
    --reasoning-parser qwen3 \
    --scheduling-policy priority \
    --max-num-seqs 256 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --compilation-config '{"max_cudagraph_capture_size": 1024}' \
    &
  PIDS+=($!)
  echo "Qwen3.5-35B-A3B-FP8 on GPU $gpu → :$port (pid ${PIDS[-1]})"
}

if [[ "$NGPU" -eq 2 ]]; then
  serve_q35 1 8002
else
  serve_q35 1 8003
  serve_q35 2 8004
  serve_q35 3 8005
  uv run python scripts/load_balancer.py --listen-port 8002 --backend-ports 8003 8004 8005 &
  PIDS+=($!)
  echo "load balancer :8002 → :8003 :8004 :8005 (pid ${PIDS[-1]})"
fi

echo "all servers launched; waiting (Ctrl-C to stop)"
wait
