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

python -m pip install \
    --no-cache-dir \
    --force-reinstall \
    --no-deps \
    "torch==2.10.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

# nvidia-cuda-runtime-cu12 + nvidia-cudnn-cu12 etc. are NOT pinned here on purpose: their
# exact versions on the cu128 index drift across torch releases (e.g. 12.8.90 -> 12.8.96),
# and conda-forge's cuda-cudart-dev=12.8.* in $CONDA_PREFIX/lib already supplies a valid
# libcudart.so.12.8.90 with `cudaGetDriverEntryPointByVersion`. Pinning an exact cu12 version
# not present on the index surfaces "No matching distribution found" errors.
# If the cu128 wheel metadata declares new deps that conda cudart-dev doesn't cover,
# pip will short-fetch them via its normal dependency resolver (we use --no-deps here only
# to avoid pulling the cpu conda pytorch back on top).
