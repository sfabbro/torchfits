#!/usr/bin/env bash
# Build the published wheel matrix with cibuildwheel (same [tool.cibuildwheel]
# config CI uses).
#
# Linux x86_64 / aarch64: run on that arch (Docker + manylinux_2_28). GHA
# splits the two Linux jobs; locally this builds the host arch only.
# macOS arm64: run on a Mac (no Docker). GHA macos-15 also produces these.
# CUDA: there is no separate torchfits CUDA wheel — the CPU-linked wheel
# loads CUDA torch of the same minor. Verify on CANFAR with
#   bash scripts/verify_wheel_cuda_canfar.sh
#
# Usage:
#   bash scripts/cibuildwheel.sh              # host arch, cp310-cp314
#   CIBW_ARCHS=aarch64 bash scripts/cibuildwheel.sh
#   CIBW_BUILD=cp314-* bash scripts/cibuildwheel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v cibuildwheel >/dev/null 2>&1; then
  python3 -m pip install --user 'cibuildwheel>=4.1.1,<5'
  export PATH="${HOME}/.local/bin:${PATH}"
fi

cibuildwheel "$@"
