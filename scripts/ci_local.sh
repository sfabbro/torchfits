#!/usr/bin/env bash
# Mirror GitHub Comprehensive CI + Documentation build locally before pushing.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FAST="${CI_LOCAL_FAST:-0}"

echo "=== ci_local: lint ==="
pixi run ruff check .
pixi run ruff format --check .
python3 scripts/check_duplicate_cpp.py
# Resolves the [cpu]/[cuda] extra pins against download.pytorch.org (needs network).
pixi run check-torch-pins
# Verifies the committed lane pins match scripts/torch_lanes.json (no network).
pixi run check-lane

echo "=== ci_local: docs contract ==="
PYTHONPATH=src pixi run pytest tests/test_docs_integrity.py tests/test_package_isolation.py -q
pixi run docs-build

if [[ "${FAST}" == "1" ]]; then
  echo "=== ci_local: fast mode (skip release-gate) ==="
  exit 0
fi

echo "=== ci_local: release gate ==="
pixi run release-gate

echo "=== ci_local: OK ==="
