#!/usr/bin/env bash
# Exhaustive lab-profile bench-all (CPU mmap + CUDA GPU transports when available),
# then patch docs/benchmarks.md from the resulting CSV.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-exhaustive_0.5.0b3_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT_DIR}/benchmarks_results"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
OUT_DIR="${LOG_DIR}/${RUN_ID}"

mkdir -p "$LOG_DIR"

echo "=== torchfits exhaustive benchmark run: ${RUN_ID} ===" | tee "$LOG_FILE"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

# Ensure CFITSIO vendored and GPU extension built against CUDA torch.
bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt >>"$LOG_FILE" 2>&1
pixi run -e bench-gpu bench-gpu-install >>"$LOG_FILE" 2>&1
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
  echo "ERROR: missing ${CSV}" | tee -a "$LOG_FILE"
  exit "${BENCH_RC:-1}"
fi

python scripts/patch_bench_docs.py \
  --csv "$CSV" \
  --deficits "$DEFICITS" \
  --run-id "$RUN_ID" >>"$LOG_FILE" 2>&1

echo "Patched docs/benchmarks.md from ${RUN_ID}" | tee -a "$LOG_FILE"
echo "Artifacts: ${OUT_DIR}/" | tee -a "$LOG_FILE"
echo "Log: ${LOG_FILE}" | tee -a "$LOG_FILE"

exit "$BENCH_RC"
