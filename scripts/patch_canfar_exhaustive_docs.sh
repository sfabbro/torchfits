#!/usr/bin/env bash
# Patch docs/benchmarks.md from a CANFAR exhaustive run (CSV must exist locally).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: $0 <exhaustive_cuda_run_id>}"
OUT_DIR="${ROOT_DIR}/benchmarks_results/${RUN_ID}"
CSV="${OUT_DIR}/results.csv"
DEFICITS="${OUT_DIR}/torchfits_deficits.csv"

if [[ ! -f "${CSV}" ]]; then
  cat >&2 <<EOF
missing ${CSV}

Fetch from VOSpace (after a CANFAR session completes):

  bash scripts/fetch_canfar_bench_vos.sh ${RUN_ID}

Or if vcp is unavailable, fall back to log import:
  pixi run "${PYTHON_BINARY:-python3}" scripts/import_canfar_bench_artifacts.py \\
    benchmarks_results/canfar_${RUN_ID}/canfar_logs.txt ${RUN_ID}

Expected layout:
  benchmarks_results/${RUN_ID}/results.csv
  benchmarks_results/${RUN_ID}/torchfits_deficits.csv
EOF
  exit 1
fi

"${PYTHON_BINARY:-python3}" scripts/patch_bench_docs.py \
  --csv "${CSV}" \
  --deficits "${DEFICITS}" \
  --run-id "${RUN_ID}"

pixi run docs-build
pixi run pytest tests/test_docs_integrity.py -q

echo "Patched docs/benchmarks.md from ${RUN_ID}"
