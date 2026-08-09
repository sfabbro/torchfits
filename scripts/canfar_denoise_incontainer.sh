#!/usr/bin/env bash
# Runs inside astroai/base on CANFAR (headless GPU session): trains the
# Noise2Noise dark->blank CR denoiser on public CFHT MegaCam calibration
# frames and evaluates transfer on the real science frames, all on GPU.
#
# Data is fetched in-container from CADC (public direct data service) via
# the idempotent fetch scripts — no manual data staging needed. Products
# (cleaned metrics, summaries, probe + injection results) land under
# ${TMP_SCRATCH_DIR}/torchfits-denoise/<run-id>/ and are archived to /arc
# and published to VOSpace when those destinations are configured.
set -euo pipefail

: "${TORCHFITS_DENOISE_RUN_ID:=denoise_$(date -u +%Y%m%d_%H%M%S)}"
: "${TORCHFITS_DENOISE_MODE:=full}"
# lr 1e-3 diverges after ~epoch 4 (l1 rises, CR removal drops); 4 epochs is
# the stable maximum with the fixed LR — see docs/denoise-pipeline.md.
: "${TORCHFITS_DENOISE_EPOCHS:=4}"
: "${TORCHFITS_DENOISE_PAIRS:=4}"
: "${TORCHFITS_DENOISE_PATCHES:=64}"
: "${TORCHFITS_DENOISE_EVAL_HDUS:=8}"

SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
RUN_DIR="${SCRATCH}/torchfits-denoise/${TORCHFITS_DENOISE_RUN_ID}"
mkdir -p "$RUN_DIR"

# ponytail: notebook image ships pixi pointed at /usr/local/share (not writable); use scratch
export PIXI_HOME="${PIXI_HOME:-${SCRATCH}/torchfits-pixi-home}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${SCRATCH}/torchfits-pixi-cache}"
mkdir -p "${PIXI_HOME}" "${PIXI_CACHE_DIR}"

if [[ -z "${TORCHFITS_DENOISE_LOG_REDIRECTED:-}" ]]; then
  export TORCHFITS_DENOISE_LOG_REDIRECTED=1
  exec > >(tee -a "${RUN_DIR}/stdout.log") 2> >(tee -a "${RUN_DIR}/stderr.log" >&2)
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== torchfits CANFAR denoise job ==="
echo "run_id=${TORCHFITS_DENOISE_RUN_ID} mode=${TORCHFITS_DENOISE_MODE}"
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "scratch=${SCRATCH} run_dir=${RUN_DIR}"

if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L || true
fi

bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

pixi install

pixi run -e bench-gpu gpu-bootstrap
pixi run -e bench-gpu bench-gpu-install
pixi run -e bench-gpu python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"

# Public data, fetched in-container from CADC (idempotent).
bash scripts/fetch_cfht_megacam_sample.sh
bash scripts/fetch_cfht_calib_frames.sh

DENOISE_ARGS=(
  --mode both
  --epochs "${TORCHFITS_DENOISE_EPOCHS}"
  --n-pairs "${TORCHFITS_DENOISE_PAIRS}"
  --n-patches "${TORCHFITS_DENOISE_PATCHES}"
  --full-eval-files 1
  --full-hdus 40
  --eval-hdus "${TORCHFITS_DENOISE_EVAL_HDUS}"
  --inject-stars
  --device auto
  --out-dir "${RUN_DIR}/products"
)
if [[ "${TORCHFITS_DENOISE_MODE}" == "smoke" ]]; then
  DENOISE_ARGS+=(--epochs 1 --n-pairs 1 --n-patches 8 --eval-hdus 2)
fi
echo "example args: ${DENOISE_ARGS[*]}"
pixi run -e bench-gpu python examples/example_megacam_cr_denoise.py "${DENOISE_ARGS[@]}"

cp -a "${RUN_DIR}/products" benchmarks_results 2>/dev/null || true
cp -a benchmarks_results "${RUN_DIR}/" 2>/dev/null || true

# Persistent on-cluster archive under /arc (survives session delete; scratch does not).
PERSISTED=0
ARC_URI=""
VOS_URI=""
if [[ -z "${TORCHFITS_ARC_DEST:-}" ]]; then
  if [[ -n "${USER:-}" && -d "/arc/home/${USER}" ]]; then
    TORCHFITS_ARC_DEST="/arc/home/${USER}/torchfits-denoise/${TORCHFITS_DENOISE_RUN_ID}"
  elif [[ -n "${HOME:-}" && -d "${HOME}" && "${HOME}" == /arc/* ]]; then
    TORCHFITS_ARC_DEST="${HOME}/torchfits-denoise/${TORCHFITS_DENOISE_RUN_ID}"
  fi
fi
if [[ -n "${TORCHFITS_ARC_DEST:-}" ]]; then
  echo "archiving products to ${TORCHFITS_ARC_DEST}"
  mkdir -p "$(dirname "${TORCHFITS_ARC_DEST}")"
  rm -rf "${TORCHFITS_ARC_DEST}"
  cp -a "${RUN_DIR}/products" "${TORCHFITS_ARC_DEST}/"
  cp -a "${RUN_DIR}/stdout.log" "${TORCHFITS_ARC_DEST}/" 2>/dev/null || true
  cp -a "${RUN_DIR}/stderr.log" "${TORCHFITS_ARC_DEST}/" 2>/dev/null || true
  ARC_URI="${TORCHFITS_ARC_DEST}"
  PERSISTED=1
  echo "TORCHFITS_ARC_URI=${ARC_URI}"
else
  echo "WARN: no /arc home found; skipping ARC archive" >&2
fi
if [[ -n "${TORCHFITS_VOS_DEST:-}" ]]; then
  if command -v vcp >/dev/null && vcp -r "${RUN_DIR}/products" "${TORCHFITS_VOS_DEST}"; then
    VOS_URI="${TORCHFITS_VOS_DEST}"
    PERSISTED=1
    echo "TORCHFITS_VOS_URI=${VOS_URI}"
  else
    echo "WARN: VOS publish failed for ${TORCHFITS_VOS_DEST}" >&2
  fi
fi

if [[ "${PERSISTED}" -ne 1 ]]; then
  echo "ERROR: denoise finished but neither /arc nor vos: archive succeeded" >&2
  exit 1
fi

echo "=== done; scratch=${RUN_DIR} arc=${ARC_URI:-none} vos=${VOS_URI:-none} ==="
