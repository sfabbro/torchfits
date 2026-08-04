#!/usr/bin/env bash
# Runs inside astroai/base on CANFAR (headless session): full-grid matrix bench
# leg. Unlike the pixi-based modes, this builds a plain venv from prebuilt
# torchfits wheels (uploaded to VOSpace once), so it can target any
# python (3.10-3.14) x torch lane (2.10-2.13) x device (CPU / CUDA flavor).
# Logs + benchmark CSVs land under ${TMP_SCRATCH_DIR}/torchfits-gpu-bench/<run-id>/.
set -euo pipefail

: "${TORCHFITS_BENCH_RUN_ID:=exhaustive_matrix_$(date -u +%Y%m%d_%H%M%S)}"
: "${TORCHFITS_BENCH_PYTHON:=3.13}"
: "${TORCHFITS_BENCH_TORCH:=2.13}"
: "${TORCHFITS_BENCH_CUDA:=0}"
: "${TORCHFITS_BENCH_CU_FLAVOR:=cu129}"
: "${TORCHFITS_BENCH_WHEELS:=vos:sfabbro/torchfits-gpu-bench/wheels}"

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

echo "=== torchfits CANFAR matrix bench ==="
echo "run_id=${TORCHFITS_BENCH_RUN_ID} mode=matrix py=${TORCHFITS_BENCH_PYTHON} torch=${TORCHFITS_BENCH_TORCH} cuda=${TORCHFITS_BENCH_CUDA} cu=${TORCHFITS_BENCH_CU_FLAVOR}"
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "scratch=${SCRATCH} run_dir=${RUN_DIR} wheels=${TORCHFITS_BENCH_WHEELS}"

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

# --- python via uv (image python is a single version; uv fetches the rest) ---
PY="${TORCHFITS_BENCH_PYTHON}"
PY_TAG="cp${PY//./}"
VENV="${SCRATCH}/matrix-venv-py${PY}-torch${TORCHFITS_BENCH_TORCH}${TORCHFITS_BENCH_CUDA}"
rm -rf "${VENV}"
if command -v uv >/dev/null 2>&1; then
  UV="uv"
else
  python3 -m pip install -q --user uv 2>/dev/null || pip install -q uv
  UV="${HOME}/.local/bin/uv"
fi
"${UV}" python install "${PY}" 2>/dev/null || true
"${UV}" venv --python "${PY}" "${VENV}"
VENV_PY="${VENV}/bin/python"
"${UV}" pip install --python "${VENV_PY}" -q --upgrade pip

# --- torch from the lane + flavor index ---
LANE="${TORCHFITS_BENCH_TORCH}"
MAJOR="${LANE%%.*}"
MINOR="${LANE##*.}"
NEXT_MINOR=$((MINOR + 1))
TORCH_SPEC="torch>=${LANE},<${MAJOR}.${NEXT_MINOR}"
if [[ "${TORCHFITS_BENCH_CUDA}" == "1" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/${TORCHFITS_BENCH_CU_FLAVOR}"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi
echo "installing ${TORCH_SPEC} from ${TORCH_INDEX}"
"${UV}" pip install --python "${VENV_PY}" -q --no-cache-dir "${TORCH_SPEC}" --index-url "${TORCH_INDEX}"

# --- torchfits wheel (release 1.0.0 for lane 2.13, dev build otherwise) ---
WHEELS_DIR="${SCRATCH}/torchfits-wheels"
mkdir -p "${WHEELS_DIR}"
if [[ -z "$(ls "${WHEELS_DIR}"/*.whl 2>/dev/null || true)" ]]; then
  echo "fetching wheel bundle from ${TORCHFITS_BENCH_WHEELS}"
  # vos CLI may live outside the image PATH; bootstrap a private prefix if needed.
  # Never write into ${HOME}/.local/torchfits-vos/bin (a symlink may point back
  # into lib/python/bin and clobber the real console script).
  if ! command -v vcp >/dev/null; then
    _vos_root="${HOME}/.local/torchfits-vos"
    mkdir -p "${_vos_root}/lib/python"
    export PATH="${_vos_root}/lib/python/bin:${PATH}"
    if [[ ! -x "${_vos_root}/lib/python/bin/vcp" ]]; then
      PYTHONNOUSERSITE=1 python3 -m pip install -q --upgrade --target "${_vos_root}/lib/python" vos
    fi
    export PYTHONPATH="${_vos_root}/lib/python${PYTHONPATH:+:${PYTHONPATH}}"
    if [[ ! -x "${_vos_root}/lib/python/bin/vcp" ]]; then
      echo "ERROR: vos console script not found under private prefix" >&2
      exit 1
    fi
  fi
  # Trailing slash on the VOS source: copy the container's contents (not a
  # nested subdirectory) into WHEELS_DIR.
  vcp "${TORCHFITS_BENCH_WHEELS}/" "${WHEELS_DIR}/" || vcp "${TORCHFITS_BENCH_WHEELS}"/* "${WHEELS_DIR}/"
fi
ls "${WHEELS_DIR}" | head -25 || true
if [[ "${LANE}" == "2.13" ]]; then
  TF_WHEEL="$(ls "${WHEELS_DIR}"/torchfits-1.0.0-${PY_TAG}-${PY_TAG}-linux_x86_64.whl 2>/dev/null | head -1 || true)"
else
  TF_WHEEL="$(ls "${WHEELS_DIR}"/torchfits-1.0.0.dev0+torch${LANE//./}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl 2>/dev/null | head -1 || true)"
fi
if [[ -z "${TF_WHEEL}" ]]; then
  echo "ERROR: no torchfits wheel for py=${PY} torch=${LANE} in ${WHEELS_DIR}" >&2
  ls "${WHEELS_DIR}" >&2 || true
  exit 1
fi
echo "wheel: ${TF_WHEEL}"
"${UV}" pip install --python "${VENV_PY}" -q --no-cache-dir "${TF_WHEEL}"

# --- bench dependencies, pinned per python (latest floors do not resolve on 3.10/3.11) ---
case "${PY}" in
  3.10)
    BENCH_DEPS=(
      "numpy>=1.26,<2.3" "astropy>=5.0,<7.0" "matplotlib>=3.8,<3.11"
      "pandas>=2.2,<3" "pyarrow" "fitsio" "psutil" "pytest"
    )
    ;;
  3.11)
    BENCH_DEPS=(
      "numpy>=2.3,<2.5" "astropy" "matplotlib" "pandas"
      "pyarrow" "fitsio" "psutil" "pytest"
    )
    ;;
  *)
    BENCH_DEPS=("numpy" "astropy" "matplotlib" "pandas" "pyarrow" "fitsio" "psutil" "pytest")
    ;;
esac
"${UV}" pip install --python "${VENV_PY}" -q "${BENCH_DEPS[@]}"

# --- CUDA runtime search path (pip torch wheel keeps libs under site-packages) ---
if [[ "${TORCHFITS_BENCH_CUDA}" == "1" ]]; then
  SP="${VENV}/lib/python${PY}/site-packages"
  export LD_LIBRARY_PATH="${SP}/torch/lib:${SP}/nvidia/cuda_runtime/lib:${SP}/nvidia/nccl/lib:${SP}/nvidia/cudnn/lib:${SP}/nvidia/cublas/lib:${SP}/nvidia/cusparse/lib:${SP}/nvidia/cusolver/lib:${LD_LIBRARY_PATH:-}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

# --- verify leg: ABI match + roundtrip (+ CUDA reachability on GPU legs) ---
"${VENV_PY}" - <<'PY'
import sys, torch, torchfits, pathlib, tempfile
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
CUDA_OK=$("${VENV_PY}" -c "import torch; print(1 if torch.cuda.is_available() else 0)")
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
"${VENV_PY}" benchmarks/bench_all.py \
  --profile lab \
  --suite release \
  --mmap-matrix \
  "${GPU_FLAGS[@]}" \
  --run-id "${TORCHFITS_BENCH_RUN_ID}" \
  --keep-temp

# --- real-astronomy megacam cutouts (fixtures fetched above) ---
"${VENV_PY}" benchmarks/bench_megacam_cutouts.py \
  --run-id "${TORCHFITS_BENCH_RUN_ID}" \
  --profile lab \
  --max-files "${TORCHFITS_BENCH_MEGACAM_FILES:-10}"

"${UV}" pip freeze --python "${VENV_PY}" > "benchmarks_results/${TORCHFITS_BENCH_RUN_ID}/requirements.txt"

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
  echo "TORCHFITS_BENCH_PYTHON=${TORCHFITS_BENCH_PYTHON}"
  echo "TORCHFITS_BENCH_TORCH=${TORCHFITS_BENCH_TORCH}"
  echo "TORCHFITS_BENCH_CUDA=${TORCHFITS_BENCH_CUDA}"
  echo "TORCHFITS_BENCH_CU_FLAVOR=${TORCHFITS_BENCH_CU_FLAVOR}"
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
