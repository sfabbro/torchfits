#!/usr/bin/env bash
# OPTIONAL CUDA verification tier — never part of preflight/ci-local/
# release-gate. Verifies the published torchfits wheels against lane-pinned
# CUDA-flavored PyTorch on real GPUs via headless CANFAR sessions, one
# session per torch lane, in parallel (scripts/verify_wheel_cuda_canfar_incontainer.sh).
#
# Soft-fail by default: per-lane [FAIL] lines are printed and results land
# under benchmarks_results/canfar_cuda_verify_<run-id>/ but the script exits 0
# (use TORCHFITS_CUDA_VERIFY_STRICT=1 to fail hard).
#
# Usage:
#   bash scripts/verify_wheel_cuda_canfar.sh [--lanes 2.12,2.13] [--jobs 2] [--ref main]
#
# Env overrides:
#   TORCHFITS_LANES, TORCHFITS_JOBS, TORCHFITS_GIT_REF (default main),
#   TORCHFITS_WHEEL_URL (default: PyPI torchfits==<lane version>),
#   TORCHFITS_CANFAR_IMAGE, TORCHFITS_CANFAR_CPU, TORCHFITS_CANFAR_MEMORY,
#   TORCHFITS_CUDA_VERIFY_STRICT, TORCHFITS_CANFAR_MAX_WAIT_SECS.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LANES="${TORCHFITS_LANES:-}"
if [[ -z "${LANES}" ]]; then
  LANES="$(python3 -c 'import json; print(",".join(json.load(open("scripts/torch_lanes.json"))))')"
fi
JOBS="${TORCHFITS_JOBS:-2}"
GIT_REF="${TORCHFITS_GIT_REF:-main}"
IMAGE="${TORCHFITS_CANFAR_IMAGE:-astroai/base:latest}"
CPU="${TORCHFITS_CANFAR_CPU:-4}"
MEMORY="${TORCHFITS_CANFAR_MEMORY:-16}"
MAX_WAIT_SECS="${TORCHFITS_CANFAR_MAX_WAIT_SECS:-5400}"
STRICT="${TORCHFITS_CUDA_VERIFY_STRICT:-0}"
WHEEL_URL="${TORCHFITS_WHEEL_URL:-}"
POLL_SECS="${TORCHFITS_CANFAR_POLL_SECS:-30}"
REPO_URL="${TORCHFITS_GIT_URL:-https://github.com/astroai/torchfits.git}"
CLONE_DIR="/tmp/torchfits"
RUN_ID="cuda_verify_$(date -u +%Y%m%d_%H%M%S)"
LOCAL_OUT="${ROOT_DIR}/benchmarks_results/canfar_${RUN_ID}"
mkdir -p "$LOCAL_OUT"

if ! command -v canfar >/dev/null; then
  echo "canfar CLI not found (install from canfar-portal or CANFAR image venv)" >&2
  exit 1
fi

verify_lane() {
  local lane="$1"
  local safe run_id name session_id status rc=0
  safe="$(printf '%s' "cv-${lane}-${RUN_ID}" | tr '_.' '--' | tr -cd '[:alnum:]-')"
  run_id="${RUN_ID}-${lane}"
  name="torchfits-${safe}"
  local dir="$LOCAL_OUT/${lane}"
  mkdir -p "$dir"

  echo "=== [${lane}] launching GPU session ${name} ==="
  local create_log="$dir/create.log"
  local remote
  remote="git clone --depth 1 --branch ${GIT_REF} ${REPO_URL} ${CLONE_DIR}; cd ${CLONE_DIR}; bash scripts/verify_wheel_cuda_canfar_incontainer.sh"
  local tabs
  tabs="$(printf '%s' "${remote}" | tr ' ' '\t')"
  if ! canfar create headless "${IMAGE}" \
    --name "${name}" \
    --cpu "${CPU}" \
    --memory "${MEMORY}" \
    --gpu 1 \
    --env "TORCHFITS_CUDA_VERIFY_LANE=${lane}" \
    --env "TORCHFITS_WHEEL_URL=${WHEEL_URL}" \
    --env "PIXI_HOME=/scratch/torchfits-pixi-home" \
    --env "PIXI_CACHE_DIR=/scratch/torchfits-pixi-cache" \
    -- bash -c "${tabs}" > "$create_log" 2>&1; then
    echo "[FAIL] lane=${lane}: session create failed (see ${create_log})"
    return 1
  fi

  session_id="$(
    python3 - "$create_log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"ID:\s*([^)]+)\)", text)
print(m.group(1).strip() if m else "")
PY
  )"
  if [[ -z "${session_id}" ]]; then
    echo "[FAIL] lane=${lane}: could not parse session ID (see ${create_log})"
    return 1
  fi
  echo "$session_id" > "$dir/session_id.txt"

  local status="" start
  start="$(date +%s)"
  while true; do
    status="$(canfar info "${session_id}" 2>/dev/null | python3 -c '
import re, sys
text = sys.stdin.read()
m = re.search(r"^\s*Status\s+(\S+)", text, re.MULTILINE)
print(m.group(1) if m else "")
' || true)"
    case "${status}" in
      Succeeded|Completed|Failed|Error|Terminating) break ;;
    esac
    if (( $(date +%s) - start > MAX_WAIT_SECS )); then
      echo "[FAIL] lane=${lane}: timeout after ${MAX_WAIT_SECS}s (status=${status:-unknown})"
      return 1
    fi
    sleep "${POLL_SECS}"
  done

  canfar logs "${session_id}" > "$dir/canfar_logs.txt" 2>&1 || true

  if [[ "${status}" == "Succeeded" || "${status}" == "Completed" ]]; then
    if rg -q '^\[ OK \]' "$dir/canfar_logs.txt"; then
      rg '^\[ OK \]' "$dir/canfar_logs.txt"
    else
      echo "[FAIL] lane=${lane}: session ${status} but no [ OK ] marker in logs"
      rc=1
    fi
  else
    echo "[FAIL] lane=${lane}: session status ${status}"
    rc=1
  fi
  return "$rc"
}

export -f verify_lane
export ROOT_DIR LANES RUN_ID LOCAL_OUT IMAGE CPU MEMORY GIT_REF REPO_URL CLONE_DIR
export WHEEL_URL MAX_WAIT_SECS POLL_SECS

echo "=== CANFAR CUDA wheel verification (soft-fail, strict=${STRICT}) ==="
echo "lanes=${LANES} jobs=${JOBS} ref=${GIT_REF} image=${IMAGE} results=${LOCAL_OUT}"
echo "${LANES}" | tr ',' '\n' | xargs -P "${JOBS}" -I{} bash -c 'verify_lane "$1"' _ {}

set +e
FAILED="$(grep -h '^\[FAIL\]' "${LOCAL_OUT}"/*/canfar_logs.txt 2>/dev/null | wc -l)"
set -e
echo "=== summary: $(grep -rh '^\[ OK \]' "${LOCAL_OUT}" 2>/dev/null | wc -l) ok, ${FAILED} failed ==="
if [[ "${FAILED}" -gt 0 ]]; then
  grep -rh '^\[FAIL\]' "${LOCAL_OUT}" 2>/dev/null || true
fi
if [[ "${STRICT}" == "1" && "${FAILED}" -gt 0 ]]; then
  exit 1
fi
echo "soft-fail: exiting 0 (set TORCHFITS_CUDA_VERIFY_STRICT=1 to fail hard)"
