#!/usr/bin/env bash
# Mirror GitHub Comprehensive CI + Documentation build locally before pushing.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FAST="${CI_LOCAL_FAST:-0}"

echo "=== ci_local: lint ==="
pixi run ruff check .
pixi run ruff format --check .
pixi run python scripts/check_duplicate_cpp.py
# Resolves the [cpu]/[cuda] extra pins against download.pytorch.org (needs network).
pixi run check-torch-pins
# Verifies the committed lane pins match scripts/torch_lanes.json (no network).
pixi run check-lane

echo "=== ci_local: docs contract ==="
# One task, not a hand-copied file list: this used to spell out the pytest
# command and the docs build separately, which is a third copy of the same
# contract (the other two are pixi.toml's `docs-contract` and the CI job of the
# same name) and nothing held them together. PYTHONPATH=src went with it --
# the editable install already resolves both `torchfits` and the compiled
# `_C` out of the source tree, so it was doing nothing.
pixi run docs-contract

if [[ "${FAST}" == "1" ]]; then
  echo "=== ci_local: fast mode (skip release-gate) ==="
  exit 0
fi

echo "=== ci_local: release gate ==="
pixi run release-gate

echo "=== ci_local: OK ==="
