#!/usr/bin/env bash
# Launch a headless CANFAR GPU session that trains the Noise2Noise dark->blank
# CR denoiser on public CFHT MegaCam calibration frames and evaluates transfer
# on the real science frames (see canfar_denoise_incontainer.sh). Data is
# fetched in-container from CADC; products are archived to /arc and VOSpace.
#
# Usage:
#   bash scripts/launch_canfar_denoise.sh
#   TORCHFITS_DENOISE_MODE=smoke bash scripts/launch_canfar_denoise.sh
set -euo pipefail

ROOT_DIR="${TORCHFITS_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

RUN_ID="${TORCHFITS_DENOISE_RUN_ID:-denoise_$(date -u +%Y%m%d_%H%M%S)}"
MODE="${TORCHFITS_DENOISE_MODE:-full}"
IMAGE="${TORCHFITS_CANFAR_IMAGE:-astroai/base:latest}"
GPU="${TORCHFITS_CANFAR_GPU:-1}"
CPU="${TORCHFITS_CANFAR_CPU:-8}"
MEMORY="${TORCHFITS_CANFAR_MEMORY:-32}"
SAFE_TAG="$(printf '%s' "${RUN_ID}" | tr '_.' '--' | tr -cd '[:alnum:]-')"
NAME="${TORCHFITS_CANFAR_NAME:-torchfits-denoise-${SAFE_TAG}}"
REPO_URL="${TORCHFITS_GIT_URL:-https://github.com/astroai/torchfits.git}"
LOCAL_OUT="${ROOT_DIR}/benchmarks_results/canfar_${RUN_ID}"
POLL_SECS="${TORCHFITS_CANFAR_POLL_SECS:-30}"
MAX_WAIT_SECS="${TORCHFITS_CANFAR_MAX_WAIT_SECS:-21600}"
CLONE_DIR="/scratch/torchfits"
VOS_DEST="${TORCHFITS_VOS_DEST:-vos:sfabbro/torchfits-denoise/${RUN_ID}}"

mkdir -p "$LOCAL_OUT"
if ! command -v canfar >/dev/null; then
  echo "canfar CLI not found (install from canfar-portal or CANFAR image venv)" >&2
  exit 1
fi

LAUNCH_LOG="${LOCAL_OUT}/launcher.log"
echo "=== CANFAR denoise launcher ===" | tee -a "${LAUNCH_LOG}"
echo "image=${IMAGE} gpu=${GPU} cpu=${CPU} memory=${MEMORY}G mode=${MODE} run_id=${RUN_ID}" | tee -a "${LAUNCH_LOG}"
echo "vos_dest=${VOS_DEST}" | tee -a "${LAUNCH_LOG}"

REMOTE_PLAIN="git clone --depth 1 --branch ${TORCHFITS_GIT_REF:-main} ${REPO_URL} ${CLONE_DIR}; cd ${CLONE_DIR}; bash scripts/canfar_denoise_incontainer.sh"
REMOTE_CMD="$(printf '%s' "${REMOTE_PLAIN}" | tr ' ' '\t')"

CREATE_LOG="${LOCAL_OUT}/create.log"
CREATE_ARGS=(
  create headless "${IMAGE}"
  --name "${NAME}"
  --cpu "${CPU}"
  --memory "${MEMORY}"
  --env "TORCHFITS_DENOISE_RUN_ID=${RUN_ID}"
  --env "TORCHFITS_DENOISE_MODE=${MODE}"
  --env "TORCHFITS_DENOISE_EPOCHS=${TORCHFITS_DENOISE_EPOCHS:-4}"
  --env "TORCHFITS_DENOISE_PAIRS=${TORCHFITS_DENOISE_PAIRS:-4}"
  --env "TORCHFITS_DENOISE_PATCHES=${TORCHFITS_DENOISE_PATCHES:-64}"
  --env "TORCHFITS_DENOISE_EVAL_HDUS=${TORCHFITS_DENOISE_EVAL_HDUS:-8}"
  --env "TORCHFITS_DENOISE_LOG_REDIRECTED=1"
  --env "PIXI_HOME=/scratch/torchfits-pixi-home"
  --env "PIXI_CACHE_DIR=/scratch/torchfits-pixi-cache"
  --env "TORCHFITS_VOS_DEST=${VOS_DEST}"
  --env "TORCH_NUM_THREADS=${CPU}"
  --env "OMP_NUM_THREADS=${CPU}"
)
if [[ -n "${GPU}" ]]; then
  CREATE_ARGS+=(--gpu "${GPU}")
fi
set +o pipefail
canfar "${CREATE_ARGS[@]}" -- bash -c "${REMOTE_CMD}" 2>&1 | tee "${CREATE_LOG}"
CREATE_RC=${PIPESTATUS[0]}
set -o pipefail
if [[ "${CREATE_RC}" -ne 0 ]]; then
  echo "canfar create failed (rc=${CREATE_RC}); see ${CREATE_LOG}" >&2
  exit 1
fi
echo "session created; logs under ${LOCAL_OUT}" | tee -a "${LAUNCH_LOG}"
echo "tracking: canfar logs <session_id> (see ${CREATE_LOG})" | tee -a "${LAUNCH_LOG}"
echo "results: bash scripts/fetch_canfar_bench_vos.sh ${RUN_ID} (from ${VOS_DEST})" | tee -a "${LAUNCH_LOG}"
