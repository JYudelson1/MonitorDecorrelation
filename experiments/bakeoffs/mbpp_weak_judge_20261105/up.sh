#!/usr/bin/env bash
# up.sh <model> <port> <frac>: start serve_small.sh, wait for /health, run the thinking admission check.
W=/workspace/weak_bakeoff; cd $W
model=$1 port=$2 frac=$3
setsid nohup ./serve_small.sh "$model" "$port" "$frac" > logs/srv_$port.log 2>&1 < /dev/null &
pid=$!
for i in $(seq 1 180); do
  curl -sf localhost:$port/health >/dev/null && break
  kill -0 $pid 2>/dev/null || { echo "DIED"; tail -40 logs/srv_$port.log; exit 1; }
  sleep 5
done
echo "$model up on :$port after $((i*5))s (pid $pid)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
/workspace/MonitorDecorrelation/.venv-vllm/bin/python verify_thinking.py "$model" http://localhost:$port/v1
