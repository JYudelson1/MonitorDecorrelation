#!/usr/bin/env bash
# down.sh <port>: stop the small-candidate vLLM server listening on <port> (never touches :8002).
port=$1; [[ "$port" == 8002 ]] && { echo "refusing to touch :8002"; exit 1; }
for p in $(pgrep -f -- "--port $port "); do kill $p; done
for i in $(seq 1 30); do pgrep -f -- "--port $port " >/dev/null || break; sleep 2; done
sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
