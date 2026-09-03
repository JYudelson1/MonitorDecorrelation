#!/usr/bin/env bash
# setup_box.sh — point this uv project's torch at the newest PyTorch CUDA index the box's
# driver actually supports, then sync and verify. Idempotent; rerun on every new box.
#
#   ./setup_box.sh            # torch only
#   ./setup_box.sh --extra gpu  # also flash-attn etc (args are passed through to `uv sync`)
#
# Why: `uv sync` pulls torch from PyPI, whose default wheel targets the newest CUDA (cu130 as of
# torch 2.11+). Rented boxes often run older drivers; torch then silently falls back to CPU and
# everything is 50x slower. The driver's max CUDA (nvidia-smi header) is the binding constraint —
# not nvcc, not what's in /usr/local/cuda.
set -euo pipefail
cd "$(dirname "$0")"

# ---- 1. driver's max supported CUDA, e.g. "12.8" -----------------------------------------------
DRV=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1) \
  || { echo "no nvidia-smi / no GPU visible; nothing to do"; exit 1; }
DRV_NUM=$(( 10#${DRV%.*} * 100 + 10#${DRV#*.} ))     # 12.8 -> 1208, 13.0 -> 1300
echo "driver supports CUDA <= $DRV"

# ---- 2. which python are we resolving for (wheel tag like cp311) -------------------------------
PYBIN=$(uv python find)
PY=$("$PYBIN" -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')

# ---- 3. newest index <= driver that has a linux x86_64 wheel for our python ---------------------
# Candidate list is hardcoded newest-first; add to the front when PyTorch adds a new cuXXX index.
TAG=""
for cand in cu132 cu130 cu128 cu126 cu124 cu121 cu118; do
  n=${cand#cu}; cand_num=$(( 10#${n:0:2} * 100 + 10#${n:2} ))
  (( cand_num <= DRV_NUM )) || continue
  listing=$(curl -fsSL "https://download.pytorch.org/whl/$cand/torch/" || true)
  newest=$(grep -oP "torch-\K[0-9]+\.[0-9]+\.[0-9]+(?=%2B$cand-$PY-$PY-manylinux[^\"]*x86_64\.whl)" \
             <<<"$listing" | sort -V | tail -1)
  if [[ -n "$newest" ]]; then
    TAG=$cand; echo "using $TAG (newest torch there for $PY: $newest)"; break
  fi
  echo "  $cand: no $PY linux wheel, skipping"
done
[[ -n "$TAG" ]] || { echo "no compatible PyTorch index found for driver $DRV"; exit 1; }
URL="https://download.pytorch.org/whl/$TAG"

# ---- 4. write it into pyproject.toml ------------------------------------------------------------
if grep -q 'download\.pytorch\.org/whl/cu' pyproject.toml; then
  # already configured for some cuXXX — just retarget the index (name + url)
  sed -i -E "s#(download\.pytorch\.org/whl/)cu[0-9]+#\1$TAG#g; s#pytorch-cu[0-9]+#pytorch-$TAG#g" pyproject.toml
  echo "retargeted existing pytorch index -> $TAG"
else
  # first time: adds torch as a direct dep, the [[tool.uv.index]] block, and the tool.uv.sources pin
  uv add torch --index "pytorch-$TAG=$URL"
  echo "added torch + pytorch-$TAG index to pyproject"
fi
# belt & braces: the index must be `explicit = true` or uv may pull *other* deps from it too
if ! grep -A3 "url = \"$URL\"" pyproject.toml | grep -q 'explicit = true'; then
  sed -i -E "s#^(url = \"$URL\")\$#\1\nexplicit = true#" pyproject.toml
  echo "marked index explicit"
fi

# ---- 5. sync + verify ---------------------------------------------------------------------------
uv sync "$@"
uv run python - <<'EOF'
import torch as t
assert t.cuda.is_available(), f"torch {t.__version__} (built for CUDA {t.version.cuda}) still can't see the GPU"
print(f"OK: torch {t.__version__} / CUDA {t.version.cuda} / {t.cuda.get_device_name(0)}")
EOF