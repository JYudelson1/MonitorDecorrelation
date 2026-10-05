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
# Then exports CUDA_HOME at a toolkit matching vllm's torch (see ensure_cuda_toolkit).
set -euo pipefail
cd "$(dirname "$0")/.."

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

ensure_vllm() {
  if command -v vllm >/dev/null; then
    VLLM=$(command -v vllm)
    echo "using vllm on PATH: $VLLM"
    return
  fi
  if [[ -x "$VLLM_VENV/bin/vllm" ]]; then
    VLLM="$VLLM_VENV/bin/vllm"
    echo "using vllm from $VLLM"
    uv pip check --no-config --python "$VLLM_VENV" >/dev/null 2>&1 \
      || echo "WARNING: $VLLM_VENV has unsatisfied requirements (uv pip check --no-config --python $VLLM_VENV); consider rm -rf $VLLM_VENV and rerunning" >&2
    return
  fi
  echo "vllm not found; installing into $VLLM_VENV/ (python 3.12, --torch-backend=auto)"
  uv venv --no-config --python 3.12 --seed "$VLLM_VENV"
  uv pip install --no-config --python "$VLLM_VENV" vllm --torch-backend=auto
  # now that torch has pinned cuda-toolkit, add the matching compiler pieces
  ctk=$("$VLLM_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("cuda-toolkit"))' 2>/dev/null || true)
  [[ -z "$ctk" ]] || uv pip install --no-config --python "$VLLM_VENV" "cuda-toolkit[$CTK_EXTRAS]==$ctk"
  VLLM="$VLLM_VENV/bin/vllm"
  [[ -x "$VLLM" ]] || { echo "vllm install finished but $VLLM is missing"; exit 1; }
  echo "installed vllm → $VLLM"
}
ensure_vllm

# The python behind $VLLM: a sibling `python` in its venv, else its shebang.
VLLM_PY="$(dirname "$VLLM")/python"
[[ -x "$VLLM_PY" ]] || VLLM_PY=$(head -1 "$VLLM" | sed 's/^#!//')
"$VLLM_PY" -c 'import vllm' 2>/dev/null || { echo "can't find the python behind $VLLM (tried $VLLM_PY)"; exit 1; }

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
