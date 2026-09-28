#!/usr/bin/env bash
# Bootstrap the CUDA-enabled PyTorch wheel for the pixi bench-gpu env.
# macOS is intentionally skipped - conda-forge pytorch (CPU) is the default there.
set -euo pipefail

case "$(uname)" in
    Darwin*)
        echo "skipping CUDA bootstrap on macOS"
        exit 0
        ;;
esac

INDEX="${TORCHFITS_TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"

# Avoid writing into ~/.local on CANFAR (/arc/home) — concurrent sessions
# corrupt shared user-site packages mid-uninstall.
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMPDIR:-/tmp}/torchfits-pip-cache}"
mkdir -p "${PIP_CACHE_DIR}"

# Match the wheel lane the repo tracks (scripts/torch_lanes.json, rendered into
# constraints-wheel.txt). The index otherwise installs the latest torch minor,
# which fails the extension ABI check. Read from constraints-wheel.txt like
# scripts/cibw_before_build.sh does, so advancing the lane cannot leave this
# script installing the previous minor.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="$(grep -E '^torch>=' "${ROOT_DIR}/constraints-wheel.txt" | head -1)"
if [[ -z "${PIN}" ]]; then
  echo "error: no torch pin in ${ROOT_DIR}/constraints-wheel.txt" >&2
  exit 1
fi
TORCH_SPEC="${TORCHFITS_TORCH_SPEC:-${PIN}}"

python -m pip install \
    --no-cache-dir \
    --force-reinstall \
    --no-user \
    "${TORCH_SPEC}" \
    --index-url "${INDEX}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
