#!/usr/bin/env bash
# Install the wheel-lane torch + build frontend into the cibuildwheel build
# venv. Isolated builds would resolve torch>=2.10 to a newer minor and fail
# the ABI check. Always pull CPU torch — the extension has no .cu kernels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="$(grep -E '^torch>=' "${ROOT}/constraints-wheel.txt" | head -1)"
test -n "${PIN}"

python -m pip install -U pip
python -m pip install \
  "scikit-build-core>=1.0.3" \
  "nanobind>=2.13.0" \
  "numpy>=1.20.0" \
  "cmake>=3.21" \
  "ninja>=1.10" \
  "${PIN}" \
  --extra-index-url https://download.pytorch.org/whl/cpu
