#!/usr/bin/env bash
# Runs inside astroai/base on CANFAR (headless GPU session).
# Logs + benchmark CSVs land under ${TMP_SCRATCH_DIR}/torchfits-gpu-bench/<run-id>/.
set -euo pipefail

: "${TORCHFITS_BENCH_RUN_ID:=exhaustive_cuda_$(date -u +%Y%m%d_%H%M%S)}"
: "${TORCHFITS_BENCH_MODE:=exhaustive}"

SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
RUN_DIR="${SCRATCH}/torchfits-gpu-bench/${TORCHFITS_BENCH_RUN_ID}"
mkdir -p "$RUN_DIR"

# ponytail: notebook image ships pixi pointed at /usr/local/share (not writable); use scratch
export PIXI_HOME="${PIXI_HOME:-${SCRATCH}/torchfits-pixi-home}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${SCRATCH}/torchfits-pixi-cache}"
mkdir -p "${PIXI_HOME}" "${PIXI_CACHE_DIR}"

if [[ -z "${TORCHFITS_BENCH_LOG_REDIRECTED:-}" ]]; then
  export TORCHFITS_BENCH_LOG_REDIRECTED=1
  exec > >(tee -a "${RUN_DIR}/stdout.log") 2> >(tee -a "${RUN_DIR}/stderr.log" >&2)
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== torchfits CANFAR GPU bench ==="
echo "run_id=${TORCHFITS_BENCH_RUN_ID} mode=${TORCHFITS_BENCH_MODE}"
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "scratch=${SCRATCH} run_dir=${RUN_DIR}"
echo "TMP_SRC_DIR=${TMP_SRC_DIR:-unset}"

# Prefer requested TORCH_NUM_THREADS; otherwise nproc (cgroup-aware). Floor at 4
# when the container only sees 1 CPU (under-provisioned headless) so ATen parallel
# paths still run for large-N table benches against fitsio.
NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
if [[ "${NPROC}" -lt 4 ]]; then
  NPROC=4
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NPROC}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NPROC}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${NPROC}}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-${NPROC}}"
echo "threads: OMP=${OMP_NUM_THREADS} TORCH=${TORCH_NUM_THREADS} nproc=$(nproc 2>/dev/null || echo unknown)"

if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L || true
fi

bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt

# Keep pip/user-site off the shared /arc home — concurrent sessions and
# leftover ~/.local packages have corrupted setuptools uninstalls there.
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

pixi install

case "${TORCHFITS_BENCH_MODE}" in
  exhaustive-cpu)
    # CPU scorecard: no CUDA torch swap, no gpu-env-check.
    pixi run -e bench-all bench-install
    pixi run -e bench-all python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-8}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
    echo "exhaustive-cpu threads: TORCH=${TORCH_NUM_THREADS}"
    pixi run -e bench-all python benchmarks/bench_all.py \
      --profile lab \
      --suite release \
      --mmap-matrix \
      --no-gpu \
      --run-id "${TORCHFITS_BENCH_RUN_ID}" \
      --keep-temp
    ;;
  *)
    pixi run -e bench-gpu gpu-bootstrap
    pixi run -e bench-gpu bench-gpu-install
    pixi run -e bench-gpu gpu-env-check
    pixi run -e bench-gpu python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"

    case "${TORCHFITS_BENCH_MODE}" in
      smoke)
        mkdir -p benchmarks_results
        pixi run -e bench-gpu pytest tests/test_scale_on_device.py -q
        pixi run -e bench-gpu python benchmarks/bench_gpu_transports.py \
          --run-id "${TORCHFITS_BENCH_RUN_ID}" \
          --output "benchmarks_results/${TORCHFITS_BENCH_RUN_ID}/gpu_transports.csv"
        ;;
      release-gate)
        pixi run -e bench-gpu release-gate
        ;;
      exhaustive)
        pixi run -e bench-gpu python benchmarks/bench_all.py \
          --profile lab \
          --suite release \
          --mmap-matrix \
          --run-id "${TORCHFITS_BENCH_RUN_ID}" \
          --keep-temp
        ;;
      *)
        echo "unknown TORCHFITS_BENCH_MODE=${TORCHFITS_BENCH_MODE}" >&2
        exit 1
        ;;
    esac
    ;;
esac

BENCH_OUT="benchmarks_results/${TORCHFITS_BENCH_RUN_ID}"
PERSISTED=0
ARC_URI=""
VOS_URI=""

if [[ -d "${BENCH_OUT}" ]]; then
  cp -a "${BENCH_OUT}" "${RUN_DIR}/benchmarks_results"

  # Persistent on-cluster archive under /arc (survives session delete; scratch does not).
  if [[ -z "${TORCHFITS_ARC_DEST:-}" ]]; then
    if [[ -n "${USER:-}" && -d "/arc/home/${USER}" ]]; then
      TORCHFITS_ARC_DEST="/arc/home/${USER}/torchfits-gpu-bench/${TORCHFITS_BENCH_RUN_ID}"
    elif [[ -n "${HOME:-}" && -d "${HOME}" && "${HOME}" == /arc/* ]]; then
      TORCHFITS_ARC_DEST="${HOME}/torchfits-gpu-bench/${TORCHFITS_BENCH_RUN_ID}"
    fi
  fi
  if [[ -n "${TORCHFITS_ARC_DEST:-}" ]]; then
    echo "archiving bench CSVs to ${TORCHFITS_ARC_DEST}"
    mkdir -p "$(dirname "${TORCHFITS_ARC_DEST}")"
    rm -rf "${TORCHFITS_ARC_DEST}"
    cp -a "${BENCH_OUT}" "${TORCHFITS_ARC_DEST}"
    cp -a "${RUN_DIR}/manifest.txt" "${TORCHFITS_ARC_DEST}/scratch_manifest.txt" 2>/dev/null || true
    # Keep session logs next to CSVs so a dead poller still has evidence on /arc.
    cp -a "${RUN_DIR}/stdout.log" "${TORCHFITS_ARC_DEST}/" 2>/dev/null || true
    cp -a "${RUN_DIR}/stderr.log" "${TORCHFITS_ARC_DEST}/" 2>/dev/null || true
    ARC_URI="${TORCHFITS_ARC_DEST}"
    PERSISTED=1
    echo "TORCHFITS_ARC_URI=${ARC_URI}"
  else
    echo "WARN: no /arc home found; skipping ARC archive" >&2
  fi

  if [[ -n "${TORCHFITS_VOS_DEST:-}" ]]; then
    if bash scripts/publish_canfar_bench_vos.sh "${BENCH_OUT}" "${TORCHFITS_VOS_DEST}"; then
      VOS_URI="${TORCHFITS_VOS_DEST}"
      PERSISTED=1
    else
      echo "WARN: VOS publish failed for ${TORCHFITS_VOS_DEST}" >&2
    fi
  fi
fi

{
  echo "TORCHFITS_BENCH_RUN_DIR=${RUN_DIR}"
  echo "TORCHFITS_BENCH_RUN_ID=${TORCHFITS_BENCH_RUN_ID}"
  echo "TORCHFITS_BENCH_MODE=${TORCHFITS_BENCH_MODE}"
  echo "TORCHFITS_BENCH_GIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ -n "${ARC_URI}" ]]; then
    echo "TORCHFITS_ARC_URI=${ARC_URI}"
  fi
  if [[ -n "${VOS_URI}" ]]; then
    echo "TORCHFITS_VOS_URI=${VOS_URI}"
  fi
} > "${RUN_DIR}/manifest.txt"
# Rewrite into ARC copy now that manifest is complete.
if [[ -n "${ARC_URI}" && -d "${ARC_URI}" ]]; then
  cp -a "${RUN_DIR}/manifest.txt" "${ARC_URI}/manifest.txt"
fi

if [[ "${PERSISTED}" -ne 1 ]]; then
  echo "ERROR: bench finished but neither /arc nor vos: archive succeeded" >&2
  echo "  ARC=${TORCHFITS_ARC_DEST:-unset} VOS=${TORCHFITS_VOS_DEST:-unset}" >&2
  exit 1
fi

echo "=== done; scratch=${RUN_DIR} arc=${ARC_URI:-none} vos=${VOS_URI:-none} ==="
