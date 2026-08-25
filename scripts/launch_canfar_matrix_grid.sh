#!/usr/bin/env bash
# Launch the full deep-review benchmark grid on CANFAR headless:
# python 3.10-3.14 x torch lanes 2.10-2.13 x {CPU, CUDA}, plus one cu130
# spot leg on 2.13/py3.13. Each leg gets a unique run-id; logs/results are
# collected per-session in VOSpace and fetched back by each poller.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONS=(3.10 3.11 3.12 3.13 3.14)
TORCHES=${TORCHES:-"2.13"}
read -r -a TORCHES <<< "$TORCHES"

launch_leg() {
  local py="$1" torch="$2" cuda="$3" cu="$4" tag="$5"
  local run_id
  run_id="exhaustive_matrix_${tag}_$(date -u +%Y%m%d_%H%M%S)"
  echo "== launching py=${py} torch=${torch} cuda=${cuda} cu=${cu} run_id=${run_id}"
  TORCHFITS_BENCH_MODE=matrix \
  TORCHFITS_BENCH_RUN_ID="${run_id}" \
  TORCHFITS_BENCH_PYTHON="${py}" \
  TORCHFITS_BENCH_TORCH="${torch}" \
  TORCHFITS_BENCH_CUDA="${cuda}" \
  TORCHFITS_BENCH_CU_FLAVOR="${cu}" \
  bash scripts/launch_canfar_gpu_bench.sh || echo "[FAIL] ${run_id}"
}

for torch in ${TORCHES}; do
  for py in "${PYTHONS[@]}"; do
    for cuda in 0 1; do
      if [[ "${cuda}" == "1" ]]; then
        if [[ "${torch}" == "2.13" ]]; then
          cu="cu129"
        elif [[ "${torch}" == "2.12" ]]; then
          cu="cu126"
        elif [[ "${torch}" == "2.11" ]]; then
          cu="cu128"
        else
          cu="cu128"
        fi
      else
        cu="cpu"
      fi
      tag="py${py//./}t${torch//./}${cu}"
      launch_leg "${py}" "${torch}" "${cuda}" "${cu}" "${tag}"
      sleep 2
    done
  done
done

launch_leg "3.13" "2.13" "1" "cu130" "py313t213cu130"
echo "grid launch complete"
