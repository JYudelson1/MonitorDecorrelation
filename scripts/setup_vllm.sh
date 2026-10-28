#!/usr/bin/env bash
# setup_vllm.sh — bring up the local Qwen judge servers that the terminal-verifier
# Qwen-matrix configs expect on :8001 (Qwen3-30B) and :8002 (Qwen3.5-35B).
#
#   2 GPUs:  GPU0 → Qwen3 :8001;  GPU1 → Qwen3.5 :8002
#   4 GPUs:  GPU0 → Qwen3 :8001;  GPU{1,2,3} → Qwen3.5 :{8003,8004,8005};
#            load balancer on :8002 → those three backends
# (GPU0..3: the GPUs it uses, in order: every GPU nvidia-smi lists, or those VLLM_GPUS names, below.)
#
# Requires exactly 2 or 4 GPUs, each an H200 (or ≥80 GiB so either FP8 MoE
# judge fits on one card with the concurrency settings below). Ctrl-C / exit kills every child.
# Quiet: server output goes to logs/vllm/<name>.log; the terminal shows readiness + any server death.
#
# vLLM: uses `vllm` on PATH if present; otherwise installs into a dedicated
# `.venv-vllm/` (kept out of the project env so it doesn't fight setup_box.sh's torch).
# Then exports CUDA_HOME at a toolkit matching vllm's torch (see ensure_cuda_toolkit).
# Options, all from the environment and off by default (then nothing changes):
# - VLLM_VERSION=X.Y.Z installs exactly that release and refuses any other vllm it finds; unset, the newest is
#   installed and whatever is found is used. Either way the version is printed and the vLLM env's packages are written
#   to logs/vllm/freeze.txt.
# - VLLM_HOST=127.0.0.1: every server, the load balancer too, listens on that address only (vllm serve --host; the
#   balancer's --listen-host). Unset: vllm's default, every interface, and the balancer's own default.
# - VLLM_RESTART=1: a server that dies is restarted, the others keep serving: its process group (with any engine
#   process that outlived it) is killed, its log kept as <name>.log.<n>, and it starts again, at most
#   VLLM_RESTART_MAX times (default 10) each. Unset: the first death stops everything.
# - VLLM_GPUS=4,5,6,7: the GPUs to use, by nvidia-smi's index, in that order (2 or 4 of them): on a machine with more,
#   or beside another server; each server gets its own (Qwen3 the first). Unset, a CUDA_VISIBLE_DEVICES of GPU
#   indices is taken instead (nvidia-smi ignores that variable, so the count would see every GPU), else every GPU.
# The v2 probe rows' batch runs VLLM_VERSION=0.30.0 VLLM_HOST=127.0.0.1 VLLM_RESTART=1 (scripts/rl_batch.py,
# docs/GPU_RUNBOOK.md).
set -euo pipefail
cd "$(dirname "$0")/.."
VLLM_VERSION="${VLLM_VERSION:-}"
[[ -z "$VLLM_VERSION" || "$VLLM_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-z0-9.+-]*)$ ]] \
  || { echo "VLLM_VERSION=$VLLM_VERSION is not a release number like 0.32.0" >&2; exit 1; }
VLLM_HOST="${VLLM_HOST:-}"
VLLM_RESTART="${VLLM_RESTART:-}"
VLLM_RESTART_MAX="${VLLM_RESTART_MAX:-10}"
[[ "$VLLM_RESTART_MAX" =~ ^[0-9]+$ ]] || { echo "VLLM_RESTART_MAX=$VLLM_RESTART_MAX is not a count" >&2; exit 1; }
# each server in a process group of its own, so what is left of a dead one can be killed before it restarts
[[ -z "$VLLM_RESTART" ]] || set -m

# Floor for non-H200 cards: FP8 weights are ~30–35 GB, but max-num-seqs 256–512 +
# speculative decoding needs H100-80-class headroom. nvidia-smi reports MiB.
MIN_MEM_MIB=$((80 * 1024))
VLLM_VENV=".venv-vllm"
# CUDA compiler pieces vLLM's JIT (DeepGEMM, FlashInfer) needs, as cuda-toolkit extras: torch pins
# cuda-toolkit to one release, so these land at exactly that release (see ensure_cuda_toolkit).
CTK_EXTRAS="nvcc,crt,cccl,cudart,nvvm"
# Every uv call on $VLLM_VENV passes --no-config: run from the repo, `uv pip` otherwise applies this
# project's [tool.uv] override-dependencies (transformers>=5.14.0) to vllm's env too, replacing vllm's own
# transformers bound (vllm 0.31 wants <5.18 and got 5.18.0).

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found; need a GPU box"; exit 1; }
command -v uv >/dev/null || { echo "uv not on PATH; need uv to install/run vllm"; exit 1; }

# Triton (vLLM's JIT) compiles C against the venv Python's Python.h. A system Python without its headers (the Nebius
# image's python3.12, 2026-10-28: "Python.h: No such file or directory", then every Qwen3.5 server died at its start)
# can't serve, so the venv is made on uv's own Python (--managed-python), which ships them, and one without is made
# again.
has_headers() {
  "$1" -c 'import os, sys, sysconfig; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' \
    2>/dev/null
}

ensure_vllm() {
  if command -v vllm >/dev/null; then
    VLLM=$(command -v vllm)
    echo "using vllm on PATH: $VLLM"
    return
  fi
  if [[ -x "$VLLM_VENV/bin/vllm" ]] && ! has_headers "$VLLM_VENV/bin/python"; then
    echo "$VLLM_VENV's Python has no Python.h (vLLM's Triton JIT needs it): making it again on uv's own Python"
    rm -rf "$VLLM_VENV"
  fi
  if [[ -x "$VLLM_VENV/bin/vllm" ]]; then
    VLLM="$VLLM_VENV/bin/vllm"
    echo "using vllm from $VLLM"
    uv pip check --no-config --python "$VLLM_VENV" >/dev/null 2>&1 \
      || echo "WARNING: $VLLM_VENV has unsatisfied requirements (uv pip check --no-config --python $VLLM_VENV); consider rm -rf $VLLM_VENV and rerunning" >&2
    return
  fi
  echo "vllm not found; installing vllm${VLLM_VERSION:+==$VLLM_VERSION} into $VLLM_VENV/ (uv's python 3.12, --torch-backend=auto)"
  uv venv --no-config --managed-python --python 3.12 --seed "$VLLM_VENV"
  uv pip install --no-config --python "$VLLM_VENV" "vllm${VLLM_VERSION:+==$VLLM_VERSION}" --torch-backend=auto
  # now that torch has pinned cuda-toolkit, add the matching compiler pieces
  ctk=$("$VLLM_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("cuda-toolkit"))' 2>/dev/null || true)
  [[ -z "$ctk" ]] || uv pip install --no-config --python "$VLLM_VENV" "cuda-toolkit[$CTK_EXTRAS]==$ctk"
  VLLM="$VLLM_VENV/bin/vllm"
  [[ -x "$VLLM" ]] || { echo "vllm install finished but $VLLM is missing"; exit 1; }
  echo "installed vllm → $VLLM"
}
ensure_vllm
# The folder of $VLLM, absolute: first on every server's PATH, so the tools its env installed are found. FlashInfer
# JIT-compiles kernels as an engine starts (sm_103, 2026-10-28) and runs the env's ninja, by name, from its build folder:
# "No such file or directory: 'ninja'" when the servers started as .venv-vllm/bin/vllm without it on PATH.
VLLM_BIN=$(cd "$(dirname "$VLLM")" && pwd)

# The python behind $VLLM: a sibling `python` in its venv, else its shebang.
VLLM_PY="$(dirname "$VLLM")/python"
[[ -x "$VLLM_PY" ]] || VLLM_PY=$(head -1 "$VLLM" | sed 's/^#!//')
"$VLLM_PY" -c 'import vllm' 2>/dev/null || { echo "can't find the python behind $VLLM (tried $VLLM_PY)"; exit 1; }
VLLM_HAVE=$("$VLLM_PY" -c 'import importlib.metadata as m; print(m.version("vllm"))')
echo "vllm $VLLM_HAVE ($VLLM_PY)"
if [[ -n "$VLLM_VERSION" && "$VLLM_HAVE" != "$VLLM_VERSION" ]]; then
  echo "vllm $VLLM_HAVE is not VLLM_VERSION=$VLLM_VERSION: rm -rf $VLLM_VENV (or take $VLLM off PATH) and run again" >&2
  exit 1
fi
mkdir -p logs/vllm
uv pip freeze --no-config --python "$VLLM_PY" > logs/vllm/freeze.txt 2>/dev/null \
  || "$VLLM_PY" -m pip freeze > logs/vllm/freeze.txt 2>/dev/null || true
echo "the vLLM environment's packages: logs/vllm/freeze.txt ($(wc -l < logs/vllm/freeze.txt) lines)"

# vLLM JIT-compiles kernels at runtime (DeepGEMM for the FP8 block-scaled linears, FlashInfer for
# Qwen3.5's GDN prefill) with the nvcc under $CUDA_HOME — NOT the CUDA that torch ships with. Two ways
# this breaks on rented boxes (both seen on vast.ai, 2026-10-05; a third, linking, at add_link_shims):
#   1. the image exports CUDA_HOME=/usr/local/cuda with an older toolkit than vllm's pip torch
#      (system nvcc 12.8 vs torch cu132) → DeepGEMM: "NVCC version must be at least 12.9";
#   2. the pip toolkit next to torch (nvidia/cu13/) is itself mixed: torch pins cuda-toolkit==13.2.1
#      (cudart 13.2) but other vllm deps pull nvidia-cuda-nvcc unpinned (13.4) → CCCL: "CUDA compiler
#      and CUDA toolkit headers are incompatible".
# So: require a toolkit whose nvcc release == its own CUDART_VERSION, matches torch's CUDA major and
# is ≥ 12.9; prefer the pip one (repairing it by pinning the compiler pieces to torch's cuda-toolkit
# version when we own the venv); export it as CUDA_HOME for every server below.
DG_MIN_NVCC="12.9"
ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }
nvcc_release() { "$1/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || true; }
cudart_release() {  # CUDART_VERSION 13020 → 13.2
  local v
  v=$(grep -oP '#define CUDART_VERSION\s+\K[0-9]+' "$1/include/cuda_runtime_api.h" 2>/dev/null) || return 0
  echo "$((v / 1000)).$(((v % 1000) / 10))"
}
# prints: <torch CUDA> <pip toolkit dir|none> <installed cuda-toolkit version|none>
torch_cuda_info() {
  "$VLLM_PY" - <<'PY'
import os, importlib.metadata as md, torch
cuda = torch.version.cuda or ""
tk = "none"
try:
    import nvidia
    for base in nvidia.__path__:
        cand = os.path.join(base, "cu" + cuda.split(".")[0])
        if cuda and os.path.isdir(os.path.join(cand, "include")):
            tk = cand
            break
except ImportError:
    pass
try:
    ctk = md.version("cuda-toolkit")
except md.PackageNotFoundError:
    ctk = "none"
print(cuda or "none", tk, ctk)
PY
}
# "ok", or why the toolkit at $1 can't serve torch's CUDA major $2
toolkit_problem() {
  local dir=$1 major=$2 rel hdr
  rel=$(nvcc_release "$dir"); hdr=$(cudart_release "$dir")
  if [[ -z "$rel" ]]; then echo "no nvcc"
  elif [[ "${rel%%.*}" != "$major" ]]; then echo "nvcc $rel, torch needs CUDA $major.x"
  elif ! ver_ge "$rel" "$DG_MIN_NVCC"; then echo "nvcc $rel < $DG_MIN_NVCC (DeepGEMM floor)"
  elif [[ "$rel" != "$hdr" ]]; then echo "nvcc $rel but headers are CUDA ${hdr:-missing}"
  else echo ok
  fi
}

# A pip toolkit ships runtime libs only, under lib/: no lib64/, no unversioned libX.so dev symlinks, no
# libcuda stub. FlashInfer's JIT links with `-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs -lcudart -lcuda`
# (+ -lnvrtc etc. per op; "ld: cannot find -lcudart" otherwise), so add what a real toolkit has: lib64,
# libX.so → libX.so.N for every lib, and stubs/libcuda.so → the driver's real libcuda.so.1 (fine to link
# against).
add_link_shims() {
  local tk=$1 so libcuda
  [[ -e "$tk/lib64" ]] || ln -s lib "$tk/lib64"
  for so in "$tk"/lib/lib*.so.[0-9]*; do
    [[ -e "$so" ]] || continue
    so=$(basename "$so")
    [[ -e "$tk/lib/${so%%.so.*}.so" ]] || ln -s "$so" "$tk/lib/${so%%.so.*}.so"
  done
  if [[ ! -e "$tk/lib/stubs/libcuda.so" ]]; then
    libcuda=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF; exit}')
    [[ -z "$libcuda" ]] || { mkdir -p "$tk/lib/stubs"; ln -sf "$libcuda" "$tk/lib/stubs/libcuda.so"; }
  fi
}

ensure_cuda_toolkit() {
  local torch_cuda pip_tk ctk major driver_cuda cand why seen=" "
  command -v c++ >/dev/null || { echo "no host C++ compiler (c++); vLLM's JIT kernels need one (apt-get install -y g++)" >&2; exit 1; }
  read -r torch_cuda pip_tk ctk < <(torch_cuda_info)
  [[ -n "$torch_cuda" && "$torch_cuda" != none ]] || { echo "vllm's torch has no CUDA build (or failed to import) in $VLLM_PY"; exit 1; }
  major=${torch_cuda%%.*}

  # The driver must cover torch's CUDA major (within a major, minor-version compatibility covers it:
  # e.g. a CUDA 13.0 driver runs cu132 torch and the cubins nvcc 13.2 builds).
  driver_cuda=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)
  if [[ -n "$driver_cuda" ]] && (( ${driver_cuda%%.*} < major )); then
    echo "vllm's torch is built for CUDA $torch_cuda but this driver only supports CUDA $driver_cuda." >&2
    echo "Reinstall vllm for CUDA ${driver_cuda%%.*} (rm -rf $VLLM_VENV, then a cu${driver_cuda%%.*}x vllm wheel)." >&2
    exit 1
  fi

  # Our own venv + a pip toolkit that isn't self-consistent → pin nvcc/crt/cccl/nvvm to the
  # cuda-toolkit release torch already pinned (its extras pin each piece to that exact release).
  if [[ "$VLLM" == "$VLLM_VENV/bin/vllm" && "$ctk" != none ]] && (( major >= 13 )); then
    why=$(toolkit_problem "$pip_tk" "$major")
    if [[ "$why" != ok ]]; then
      echo "pip CUDA toolkit in $VLLM_VENV is unusable ($why); pinning its compiler to cuda-toolkit $ctk"
      uv pip install --no-config --python "$VLLM_PY" "cuda-toolkit[$CTK_EXTRAS]==$ctk" --reinstall-package cuda-toolkit
      read -r torch_cuda pip_tk ctk < <(torch_cuda_info)
    fi
  fi

  for cand in "$pip_tk" "${CUDA_HOME:-}" /usr/local/cuda; do
    [[ -n "$cand" && "$cand" != none && "$seen" != *" $cand "* ]] || continue
    seen+="$cand "
    why=$(toolkit_problem "$cand" "$major")
    if [[ "$why" == ok ]]; then
      [[ "$cand" != "$pip_tk" ]] || add_link_shims "$cand"
      export CUDA_HOME="$cand"
      export PATH="$CUDA_HOME/bin:$PATH"
      echo "CUDA_HOME=$CUDA_HOME (nvcc $(nvcc_release "$cand"); torch CUDA $torch_cuda; driver CUDA ${driver_cuda:-?})"
      return
    fi
    echo "skipping toolkit $cand ($why)"
  done
  echo "no usable CUDA toolkit (nvcc == its headers, CUDA $major.x, ≥ $DG_MIN_NVCC); vLLM's JIT kernels would fail." >&2
  echo "Fix: uv pip install --no-config --python $VLLM_PY 'cuda-toolkit[$CTK_EXTRAS]==<torch's cuda-toolkit version>', or point CUDA_HOME at one." >&2
  exit 1
}
ensure_cuda_toolkit

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
gpu_index() { cut -d',' -f1 <<<"$1" | tr -d ' '; }
VLLM_GPUS="${VLLM_GPUS:-}"
if [[ -z "$VLLM_GPUS" && "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  VLLM_GPUS=$CUDA_VISIBLE_DEVICES
fi
if [[ -n "$VLLM_GPUS" ]]; then
  [[ "$VLLM_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "VLLM_GPUS=$VLLM_GPUS is not a comma list of GPU indices" >&2; exit 1; }
  IFS=',' read -ra WANT <<<"$VLLM_GPUS"
  PICKED=()
  for want in "${WANT[@]}"; do
    row=""
    for r in "${GPU_ROWS[@]}"; do
      if [[ "$(gpu_index "$r")" == "$want" ]]; then row=$r; fi
    done
    if [[ -z "$row" ]]; then echo "VLLM_GPUS=$VLLM_GPUS: nvidia-smi lists no GPU $want" >&2; exit 1; fi
    for p in "${PICKED[@]+"${PICKED[@]}"}"; do
      if [[ "$(gpu_index "$p")" == "$want" ]]; then echo "VLLM_GPUS=$VLLM_GPUS names GPU $want twice" >&2; exit 1; fi
    done
    PICKED+=("$row")
  done
  GPU_ROWS=("${PICKED[@]}")
  echo "the GPUs of VLLM_GPUS=$VLLM_GPUS, in that order"
fi
NGPU=${#GPU_ROWS[@]}
case "$NGPU" in
  2|4) ;;
  *) echo "need exactly 2 or 4 GPUs, found $NGPU (VLLM_GPUS picks them on a machine with more)"; exit 1 ;;
esac

GPU_IDS=()  # nvidia-smi's index of GPU0..GPU3 above: each server's own CUDA_VISIBLE_DEVICES
for row in "${GPU_ROWS[@]}"; do
  # csv: index, name, memory.total (MiB) — name may contain spaces
  idx=$(gpu_index "$row")
  GPU_IDS+=("$idx")
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

command -v curl >/dev/null || { echo "curl not found; needed to health-check the servers"; exit 1; }

# Servers are quiet: each one's full output (startup, per-10s throughput stats, errors) goes to
# $LOG_DIR/<name>.log, overwritten per launch; the terminal only shows launches, readiness and deaths.
# Per-request access logs are off (--disable-uvicorn-access-log) — one line per judge call otherwise.
LOG_DIR="logs/vllm"
mkdir -p "$LOG_DIR"

# Refuse to start if a port is taken (e.g. an earlier setup_vllm.sh still running): the new server would
# die on bind, and worse, the old one would answer the readiness check below.
if [[ "$NGPU" -eq 2 ]]; then PORTS=(8001 8002); else PORTS=(8001 8002 8003 8004 8005); fi
for port in "${PORTS[@]}"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "port $port is already in use — another setup_vllm.sh still running? (ss -ltnp | grep :$port)" >&2
    exit 1
  fi
done
PIDS=()
declare -A NAME_OF PORT_OF LOG_OF READY LAUNCH_OF RESTARTS
cleanup() {
  local pid
  for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
    if [[ -n "$VLLM_RESTART" ]]; then kill -- "-$pid" 2>/dev/null || true; else kill "$pid" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
}
if [[ -n "$VLLM_RESTART" ]]; then
  # stop for good on a signal: carrying on would take every server it just stopped for dead, and start them again
  trap cleanup EXIT
  trap 'cleanup; trap - EXIT; exit 143' TERM
  trap 'cleanup; trap - EXIT; exit 130' INT
else
  trap cleanup EXIT INT TERM
fi

# launch <name> <port> <cmd...>: run cmd in the background with all output → $LOG_DIR/<name>.log. $RELAUNCH, set by the
# serve_* function calling it, is how to start that server again (VLLM_RESTART).
RELAUNCH=""
launch() {
  local name=$1 port=$2 pid; shift 2
  "$@" >"$LOG_DIR/$name.log" 2>&1 &
  pid=$!
  PIDS+=("$pid"); NAME_OF[$pid]=$name; PORT_OF[$pid]=$port; LOG_OF[$pid]=$LOG_DIR/$name.log
  LAUNCH_OF[$pid]=$RELAUNCH
  echo "$name → :$port (pid $pid, log $LOG_DIR/$name.log)"
}

HOST_ARGS=()
[[ -z "$VLLM_HOST" ]] || HOST_ARGS=(--host "$VLLM_HOST")

# Qwen3-30B always on GPU0 (the first GPU) → :8001 (clients hardcode this).
serve_q3() {
  RELAUNCH=serve_q3
  launch qwen3-30b-8001 8001 env PATH="$VLLM_BIN:$PATH" CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$VLLM" serve Qwen/Qwen3-30B-A3B-FP8 \
    --port 8001 \
    --reasoning-parser qwen3 \
    --scheduling-policy priority \
    --max-num-seqs 512 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.92 \
    --speculative-config '{"method": "eagle3", "model": "AngelSlim/Qwen3-a3B_eagle3", "num_speculative_tokens": 2}' \
    --disable-uvicorn-access-log ${HOST_ARGS[@]+"${HOST_ARGS[@]}"}
}

serve_q35() {
  local gpu=$1 port=$2
  RELAUNCH="serve_q35 $gpu $port"
  launch "qwen3.5-35b-$port" "$port" env PATH="$VLLM_BIN:$PATH" CUDA_VISIBLE_DEVICES="$gpu" "$VLLM" serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --port "$port" \
    --reasoning-parser qwen3 \
    --scheduling-policy priority \
    --max-num-seqs 256 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --compilation-config '{"max_cudagraph_capture_size": 1024}' \
    --disable-uvicorn-access-log ${HOST_ARGS[@]+"${HOST_ARGS[@]}"}
}

# TCP passthrough, so :8002/health answers once a backend does
serve_lb() {
  RELAUNCH=serve_lb
  launch lb-8002 8002 uv run python scripts/load_balancer.py --listen-port 8002 --backend-ports 8003 8004 8005 \
    ${VLLM_HOST:+--listen-host "$VLLM_HOST"}
}

# The engines' weights, downloaded before any engine starts: one `hf download` per repo, as scripts/nebius_job.py fetches
# the probes' reader. On Nebius VMs (2026-10-28) the engines' own downloads, unauthenticated and at once, stalled for
# good: no bytes coming in, every file .incomplete at or near its size, the engines waiting. A repo that still fails
# after 3 tries is left to its engine.
JUDGE_REPOS=(Qwen/Qwen3-30B-A3B-FP8 AngelSlim/Qwen3-a3B_eagle3 Qwen/Qwen3.5-35B-A3B-FP8)
predownload() {
  local hf repo try
  hf="$(dirname "$VLLM")/hf"
  if [[ ! -x "$hf" ]]; then
    echo "no hf CLI beside $VLLM: each engine downloads its own weights"
    return 0
  fi
  for repo in "${JUDGE_REPOS[@]}"; do
    for try in 1 2 3; do
      if HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 timeout 1200 "$hf" download "$repo" >/dev/null; then
        echo "weights: $repo"
        break
      fi
      echo "hf download $repo failed (try $try of 3)"
    done
  done
}
predownload

serve_q3
if [[ "$NGPU" -eq 2 ]]; then
  serve_q35 "${GPU_IDS[1]}" 8002
else
  serve_q35 "${GPU_IDS[1]}" 8003
  serve_q35 "${GPU_IDS[2]}" 8004
  serve_q35 "${GPU_IDS[3]}" 8005
  serve_lb
fi

# restart_dead <pid> <n>: VLLM_RESTART's <n>th restart of the server <pid>, which died: kill what is left of its process
# group, keep its log as <log>.<n>, and start it again the way it was started.
restart_dead() {
  local pid=$1 n=$2 name=${NAME_OF[$1]} relaunch=${LAUNCH_OF[$1]} log=${LOG_OF[$1]} p keep=()
  kill -KILL -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 -- "-$pid" 2>/dev/null || break; sleep 1; done
  mv -f "$log" "$log.$n"
  for p in "${PIDS[@]}"; do [[ "$p" == "$pid" ]] || keep+=("$p"); done
  PIDS=(${keep[@]+"${keep[@]}"})
  unset "NAME_OF[$pid]" "PORT_OF[$pid]" "LOG_OF[$pid]" "LAUNCH_OF[$pid]" "READY[$pid]"
  echo "restarting $name ($n of at most $VLLM_RESTART_MAX; the log of the one that died: $log.$n)"
  $relaunch
}

if [[ -n "$VLLM_RESTART" ]]; then
  echo "starting (a first start JIT-compiles kernels: a few minutes); a server that dies is started again (VLLM_RESTART)"
else
  echo "starting (a first start JIT-compiles kernels: a few minutes); Ctrl-C stops everything"
fi
# Watch: announce each server once /health answers; if any process dies, show its log tail and stop
# everything (a dead judge would break the run anyway), or with VLLM_RESTART start that one again.
while :; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rc=0; wait "$pid" || rc=$?
      echo "!! ${NAME_OF[$pid]} exited (code $rc) — last lines of ${LOG_OF[$pid]}:" >&2
      tail -n 40 "${LOG_OF[$pid]}" >&2
      [[ -n "$VLLM_RESTART" ]] || exit 1
      name=${NAME_OF[$pid]}
      n=$(( ${RESTARTS[$name]:-0} + 1 ))
      RESTARTS[$name]=$n
      if (( n > VLLM_RESTART_MAX )); then
        echo "!! $name died $n times (VLLM_RESTART_MAX=$VLLM_RESTART_MAX): stopping everything" >&2
        exit 1
      fi
      restart_dead "$pid" "$n"
      continue
    fi
    if [[ -z "${READY[$pid]:-}" ]] && curl -sf -o /dev/null --max-time 2 "http://localhost:${PORT_OF[$pid]}/health"; then
      READY[$pid]=1
      echo "✓ ${NAME_OF[$pid]} ready on :${PORT_OF[$pid]}"
      (( ${#READY[@]} == ${#PIDS[@]} )) && echo "all servers ready (logs: $LOG_DIR/*.log)"
    fi
  done
  sleep 5
done
