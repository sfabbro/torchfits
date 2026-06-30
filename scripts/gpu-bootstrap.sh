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
    "torch>=2,<3"

# CUDA runtime libraries (nvidia-* packages) are dynamically resolved and installed
# automatically by pip without hardcoded version pins.
