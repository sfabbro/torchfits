#!/usr/bin/env bash
# Runs inside astroai/base on CANFAR (headless session): full-grid matrix bench
# leg, pixi-first. Unlike the old uv-venv + wheel-bundle approach, each leg
# builds its own pixi environment for the requested CPython tier
# (bench-py310..bench-py314 / bench-gpu-py310..bench-gpu-py314) and applies the
# requested torch lane with scripts/release_lane.py --apply (the lane pin
# drives the conda pytorch version AND the C++ ABI the extension is built
# against). CUDA legs swap the conda CPU torch for the pip cu-wheel of the
# same lane via the gpu-bootstrap task.
# Logs + benchmark CSVs land under ${TMP_SCRATCH_DIR}/torchfits-gpu-bench/<run-id>/.
set -euo pipefail

: "${TORCHFITS_BENCH_RUN_ID:=exhaustive_matrix_$(date -u +%Y%m%d_%H%M%S)}"
: "${TORCHFITS_BENCH_PYTHON:=3.13}"
: "${TORCHFITS_BENCH_TORCH:=2.13}"
: "${TORCHFITS_BENCH_CUDA:=0}"
# cu flavor defaults per lane (2.10-2.12 ship cu128; 2.13 defaults cu129,
# cu130 is the spot leg); overridable per leg.
: "${TORCHFITS_BENCH_CU_FLAVOR:=}"

SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
RUN_DIR="${SCRATCH}/torchfits-gpu-bench/${TORCHFITS_BENCH_RUN_ID}"
mkdir -p "$RUN_DIR"

export PIXI_HOME="${PIXI_HOME:-${SCRATCH}/torchfits-pixi-home}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${SCRATCH}/torchfits-pixi-cache}"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${PIXI_HOME}" "${PIXI_CACHE_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}"

if [[ -z "${TORCHFITS_BENCH_LOG_REDIRECTED:-}" ]]; then
  export TORCHFITS_BENCH_LOG_REDIRECTED=1
  exec > >(tee -a "${RUN_DIR}/stdout.log") 2> >(tee -a "${RUN_DIR}/stderr.log" >&2)
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY="${TORCHFITS_BENCH_PYTHON}"
LANE="${TORCHFITS_BENCH_TORCH}"
PY_TAG="py${PY//./}"
if [[ -z "${TORCHFITS_BENCH_CU_FLAVOR}" ]]; then
  # PyTorch wheel flavors: 2.10/2.11 ship cu128; 2.12+ ship cu129 only
  # (no cu128 wheels exist for 2.12/2.13).
  if [[ "${LANE}" == "2.13" ]]; then
    TORCHFITS_BENCH_CU_FLAVOR="cu129"
  elif [[ "${LANE}" == "2.12" ]]; then
    TORCHFITS_BENCH_CU_FLAVOR="cu126"
  else
    TORCHFITS_BENCH_CU_FLAVOR="cu128"
  fi
fi
if [[ "${TORCHFITS_BENCH_CUDA}" == "1" ]]; then
  ENV_NAME="bench-gpu-${PY_TAG}"
else
  ENV_NAME="bench-${PY_TAG}"
fi

echo "=== torchfits CANFAR matrix bench (pixi-first) ==="
echo "run_id=${TORCHFITS_BENCH_RUN_ID} mode=matrix py=${PY} torch=${LANE} cuda=${TORCHFITS_BENCH_CUDA} cu=${TORCHFITS_BENCH_CU_FLAVOR} env=${ENV_NAME}"
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "scratch=${SCRATCH} run_dir=${RUN_DIR}"

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

# --- real-astronomy fixtures are fetched inside every session (idempotent) ---
echo "fetching CFHT MegaCam / MegaPipe fixtures..."
bash scripts/fetch_cfht_megacam_sample.sh
bash scripts/fetch_cfht_megapipe_sample.sh
MEGACAM_COUNT="$(find benchmarks_data/cfht_megacam -name '*.fits*' 2>/dev/null | wc -l)"
if [[ "${MEGACAM_COUNT}" -lt 1 ]]; then
  echo "ERROR: megacam fixtures missing after fetch" >&2
  exit 1
fi
echo "megacam files=${MEGACAM_COUNT}"

# --- torch lane: render the repo for the requested minor ---
# pixi-first: run the (stdlib-only) renderer with the pixi default env, never
# with a bare python3 (no home/user-site writes). Lane 2.13 is the committed
# state, so only non-2.13 lanes need the render. pixi auto-installs the
# default env from the committed lock on first use.
if [[ "${LANE}" != "2.13" ]]; then
  echo "applying torch lane ${LANE} (release_lane.py --apply)"
  pixi run -e default python scripts/release_lane.py --lane "${LANE}" --apply
fi

# --- pixi env for the python tier (pixi re-locks when pixi.toml changed) ---
echo "pixi install -e ${ENV_NAME}"
pixi install -e "${ENV_NAME}"

if [[ "${TORCHFITS_BENCH_CUDA}" == "1" ]]; then
  # Swap the conda CPU torch for the lane-matched cu wheel, build the C++
  # extension against it, and verify CUDA reachability.
  NEXT_MINOR=$(( ${LANE#*.} + 1 ))
  TORCHFITS_TORCH_SPEC="torch>=${LANE},<${LANE%%.*}.${NEXT_MINOR}" \
  TORCHFITS_TORCH_INDEX="https://download.pytorch.org/whl/${TORCHFITS_BENCH_CU_FLAVOR}" \
    pixi run -e "${ENV_NAME}" gpu-bootstrap
  pixi run -e "${ENV_NAME}" bench-gpu-install
  pixi run -e "${ENV_NAME}" gpu-env-check
else
  pixi run -e "${ENV_NAME}" bench-install
fi

# --- verify leg: lane pin, ABI match + roundtrip (+ CUDA on GPU legs) ---
pixi run -e "${ENV_NAME}" python - "${LANE}" <<'PY'
import subprocess, sys
import torch, torchfits, pathlib, tempfile
lane = sys.argv[1] if len(sys.argv) > 1 else "2.13"
print("lane", lane, flush=True)
subprocess.run([sys.executable, "scripts/release_lane.py", "--lane", lane, "--check"], check=True)
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "cuda_rt", torch.version.cuda, flush=True)
print("torchfits", torchfits.__version__, "python", sys.version.split()[0], flush=True)
f = str(pathlib.Path(tempfile.mkdtemp()) / "t.fits")
data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
torchfits.write(f, data)
assert torch.equal(torchfits.read(f), data), "roundtrip mismatch"
print("roundtrip OK", flush=True)
if torch.cuda.is_available():
    d = torch.zeros(4, device="cuda")
    assert torch.equal(d, torch.zeros(4, device="cuda"))
    print("cuda tensor OK", flush=True)
PY
CUDA_OK="$(pixi run -e "${ENV_NAME}" python -c 'import torch; print(1 if torch.cuda.is_available() else 0)')"
if [[ "${TORCHFITS_BENCH_CUDA}" == "1" && "${CUDA_OK}" != "1" ]]; then
  echo "ERROR: CUDA requested but torch.cuda.is_available() is False" >&2
  exit 1
fi
echo "[ OK ] matrix lane=${LANE} py=${PY} cu=${TORCHFITS_BENCH_CU_FLAVOR} cuda_available=${CUDA_OK}"

# --- exhaustive run ---
GPU_FLAGS=()
if [[ "${TORCHFITS_BENCH_CUDA}" != "1" ]]; then
  GPU_FLAGS=(--no-gpu)
fi
mkdir -p "benchmarks_results/${TORCHFITS_BENCH_RUN_ID}"
pixi run -e "${ENV_NAME}" python benchmarks/bench_all.py \
  --profile lab \
  --suite release \
  --mmap-matrix \
  "${GPU_FLAGS[@]}" \
  --run-id "${TORCHFITS_BENCH_RUN_ID}" \
  --keep-temp

# --- real-astronomy megacam cutouts (fixtures fetched above) ---
pixi run -e "${ENV_NAME}" python benchmarks/bench_megacam_cutouts.py \
  --run-id "${TORCHFITS_BENCH_RUN_ID}" \
  --profile lab \
  --max-files "${TORCHFITS_BENCH_MEGACAM_FILES:-10}"

pixi list -e "${ENV_NAME}" > "benchmarks_results/${TORCHFITS_BENCH_RUN_ID}/environment.txt"

BENCH_OUT="benchmarks_results/${TORCHFITS_BENCH_RUN_ID}"
PERSISTED=0
ARC_URI=""
VOS_URI=""

if [[ -d "${BENCH_OUT}" ]]; then
  cp -a "${BENCH_OUT}" "${RUN_DIR}/benchmarks_results"

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
  echo "TORCHFITS_BENCH_MODE=matrix"
  echo "TORCHFITS_BENCH_PYTHON=${PY}"
  echo "TORCHFITS_BENCH_TORCH=${LANE}"
  echo "TORCHFITS_BENCH_CUDA=${TORCHFITS_BENCH_CUDA}"
  echo "TORCHFITS_BENCH_CU_FLAVOR=${TORCHFITS_BENCH_CU_FLAVOR}"
  echo "TORCHFITS_BENCH_ENV=${ENV_NAME}"
  echo "TORCHFITS_BENCH_GIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ -n "${ARC_URI}" ]]; then
    echo "TORCHFITS_ARC_URI=${ARC_URI}"
  fi
  if [[ -n "${VOS_URI}" ]]; then
    echo "TORCHFITS_VOS_URI=${VOS_URI}"
  fi
} > "${RUN_DIR}/manifest.txt"
if [[ -n "${ARC_URI}" && -d "${ARC_URI}" ]]; then
  cp -a "${RUN_DIR}/manifest.txt" "${ARC_URI}/manifest.txt"
fi

if [[ "${PERSISTED}" -ne 1 ]]; then
  echo "ERROR: bench finished but neither /arc nor vos: archive succeeded" >&2
  echo "  ARC=${TORCHFITS_ARC_DEST:-unset} VOS=${TORCHFITS_VOS_DEST:-unset}" >&2
  exit 1
fi

echo "=== done; scratch=${RUN_DIR} arc=${ARC_URI:-none} vos=${VOS_URI:-none} ==="
