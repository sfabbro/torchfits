#!/usr/bin/env bash
# Exhaustive lab-profile bench-all (CPU mmap + CUDA GPU transports when available),
# then patch docs/benchmarks.md from the resulting CSV.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-exhaustive_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT_DIR}/benchmarks_results"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
OUT_DIR="${LOG_DIR}/${RUN_ID}"

mkdir -p "$LOG_DIR"

echo "=== torchfits exhaustive benchmark run: ${RUN_ID} ===" | tee "$LOG_FILE"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

pixi run -e bench-gpu gpu-bootstrap >>"$LOG_FILE" 2>&1
# Rebuild only when the native extension is missing or its build id disagrees
# with the torch-free library (avoids flaky LTO rebuilds mid-run). Checking only
# _C would let a half-rebuilt tree through: the three native artifacts carry one
# build id, and a mismatch surfaces as an ImportError on the first metadata call
# rather than as a build failure.
if ! pixi run -e bench-gpu python -c "import torchfits._C, torchfits._core as c; assert c.core_library_build_id() == c.__build_id__ == torchfits._C.__core_build_id__" >>"$LOG_FILE" 2>&1; then
  bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt >>"$LOG_FILE" 2>&1
  pixi run -e bench-gpu bench-gpu-install >>"$LOG_FILE" 2>&1
fi
pixi run -e bench-gpu gpu-env-check >>"$LOG_FILE" 2>&1

set +e
pixi run -e bench-gpu python benchmarks/bench_all.py \
  --profile lab \
  --scope all \
  --mmap-matrix \
  --run-id "$RUN_ID" \
  --keep-temp >>"$LOG_FILE" 2>&1
BENCH_RC=$?
set -e

echo "bench-all exit code: ${BENCH_RC}" | tee -a "$LOG_FILE"
echo "Finished bench-all: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

CSV="${OUT_DIR}/results.csv"
DEFICITS="${OUT_DIR}/torchfits_deficits.csv"

if [[ ! -f "$CSV" ]]; then
  # Non-zero whatever the bench reported: this is this script's own failure, and
  # '${BENCH_RC:-1}' used to hand back the bench's 0 and look like success.
  echo "ERROR: missing ${CSV}" | tee -a "$LOG_FILE"
  exit 1
fi

"${PYTHON_BINARY:-python3}" scripts/patch_bench_docs.py \
  --csv "$CSV" \
  --deficits "$DEFICITS" \
  --run-id "$RUN_ID" >>"$LOG_FILE" 2>&1

echo "Patched docs/benchmarks.md from ${RUN_ID}" | tee -a "$LOG_FILE"
echo "Artifacts: ${OUT_DIR}/" | tee -a "$LOG_FILE"
echo "Log: ${LOG_FILE}" | tee -a "$LOG_FILE"

exit "$BENCH_RC"
