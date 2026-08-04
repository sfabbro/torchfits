#!/usr/bin/env bash
# Launch a headless session on CANFAR staging, clone torchfits, run pixi bench/tests
# (mode exhaustive / exhaustive-cpu / matrix), and capture platform logs locally under
# benchmarks_results/canfar_<run-id>/. Matrix legs target any python x torch lane x device
# from the prebuilt wheel bundle (see canfar_matrix_bench_incontainer.sh).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GIT_REF="${TORCHFITS_GIT_REF:-main}"
MODE="${TORCHFITS_BENCH_MODE:-exhaustive}"
case "${MODE}" in
  exhaustive-cpu)
    DEFAULT_RUN_ID="exhaustive_cpu_$(date -u +%Y%m%d_%H%M%S)"
    # Omit --gpu (do not pass --gpu 0 — Skaha rejects zero). CPU-only scorecard
    # is still enforced in-container via --no-gpu.
    DEFAULT_GPU=""
    ;;
  matrix)
    DEFAULT_RUN_ID="exhaustive_matrix_$(date -u +%Y%m%d_%H%M%S)"
    if [[ "${TORCHFITS_BENCH_CUDA:-0}" == "1" ]]; then
      DEFAULT_GPU=1
    else
      DEFAULT_GPU=""
    fi
    ;;
  *)
    DEFAULT_RUN_ID="exhaustive_cuda_$(date -u +%Y%m%d_%H%M%S)"
    DEFAULT_GPU=1
    ;;
esac
RUN_ID="${TORCHFITS_BENCH_RUN_ID:-${DEFAULT_RUN_ID}}"
IMAGE="${TORCHFITS_CANFAR_IMAGE:-astroai/base:latest}"
# Empty TORCHFITS_CANFAR_GPU means "omit --gpu" (CPU session).
if [[ -n "${TORCHFITS_CANFAR_GPU+x}" ]]; then
  GPU="${TORCHFITS_CANFAR_GPU}"
else
  GPU="${DEFAULT_GPU}"
fi
CPU="${TORCHFITS_CANFAR_CPU:-8}"
MEMORY="${TORCHFITS_CANFAR_MEMORY:-32}"
# ponytail: CANFAR session names allow [A-Za-z0-9-] only (no dots/underscores)
SAFE_TAG="$(printf '%s' "${RUN_ID}" | tr '_.' '--' | tr -cd '[:alnum:]-')"
NAME="${TORCHFITS_CANFAR_NAME:-torchfits-gpu-${SAFE_TAG}}"
REPO_URL="${TORCHFITS_GIT_URL:-https://github.com/astroai/torchfits.git}"
LOCAL_OUT="${ROOT_DIR}/benchmarks_results/canfar_${RUN_ID}"
POLL_SECS="${TORCHFITS_CANFAR_POLL_SECS:-30}"
MAX_WAIT_SECS="${TORCHFITS_CANFAR_MAX_WAIT_SECS:-14400}"
CLONE_DIR="/scratch/torchfits"
VOS_DEST="${TORCHFITS_VOS_DEST:-vos:sfabbro/torchfits-gpu-bench/${RUN_ID}}"

mkdir -p "$LOCAL_OUT"

if ! command -v canfar >/dev/null; then
  echo "canfar CLI not found (install from canfar-portal or CANFAR image venv)" >&2
  exit 1
fi

LAUNCH_LOG="${LOCAL_OUT}/launcher.log"
if [[ -z "${TORCHFITS_CANFAR_EXISTING_SESSION:-}" ]]; then
  : > "${LAUNCH_LOG}"
fi

echo "=== CANFAR GPU bench launcher ===" | tee -a "${LAUNCH_LOG}"
echo "server: $(canfar auth show 2>&1 | rg 'Server' || true)" | tee -a "${LAUNCH_LOG}"
echo "image=${IMAGE} gpu=${GPU:-omit} cpu=${CPU} memory=${MEMORY}G ref=${GIT_REF} mode=${MODE} run_id=${RUN_ID}" | tee -a "${LAUNCH_LOG}"
echo "vos_dest=${VOS_DEST}" | tee -a "${LAUNCH_LOG}"

if [[ -n "${TORCHFITS_CANFAR_EXISTING_SESSION:-}" ]]; then
  SESSION_ID="${TORCHFITS_CANFAR_EXISTING_SESSION}"
  echo "${SESSION_ID}" > "${LOCAL_OUT}/session_id.txt"
  echo "poller resume session_id=${SESSION_ID}" | tee -a "${LAUNCH_LOG}"
else
  # ponytail: skaha splits cmd args on spaces; tabs keep bash -c script as one token (no $ or &)
  # Optional: TORCHFITS_VOS_BUNDLE=vos:.../tree.bundle clones an uploaded git bundle
  # (lets CANFAR soak an unpushed local commit without a GitHub push).
  if [[ "${MODE}" == "matrix" ]]; then
    INCONTAINER="scripts/canfar_matrix_bench_incontainer.sh"
  else
    INCONTAINER="scripts/canfar_gpu_bench_incontainer.sh"
  fi
  if [[ -n "${TORCHFITS_VOS_BUNDLE:-}" ]]; then
    REMOTE_PLAIN="vcp ${TORCHFITS_VOS_BUNDLE} /scratch/torchfits.bundle; git clone /scratch/torchfits.bundle ${CLONE_DIR}; cd ${CLONE_DIR}; bash ${INCONTAINER}"
  else
    REMOTE_PLAIN="git clone --depth 1 --branch ${GIT_REF} ${REPO_URL} ${CLONE_DIR}; cd ${CLONE_DIR}; bash ${INCONTAINER}"
  fi
  REMOTE_CMD="$(printf '%s' "${REMOTE_PLAIN}" | tr ' ' '\t')"

  CREATE_LOG="${LOCAL_OUT}/create.log"
  CREATE_ARGS=(
    create headless "${IMAGE}"
    --name "${NAME}"
    --cpu "${CPU}"
    --memory "${MEMORY}"
    --env "TORCHFITS_BENCH_RUN_ID=${RUN_ID}"
    --env "TORCHFITS_BENCH_MODE=${MODE}"
    --env "TORCHFITS_GIT_REF=${GIT_REF}"
    --env "TORCHFITS_BENCH_LOG_REDIRECTED=1"
    --env "PIXI_HOME=/scratch/torchfits-pixi-home"
    --env "PIXI_CACHE_DIR=/scratch/torchfits-pixi-cache"
    --env "TORCHFITS_VOS_DEST=${VOS_DEST}"
    --env "TORCHFITS_BENCH_PYTHON=${TORCHFITS_BENCH_PYTHON:-3.13}"
    --env "TORCHFITS_BENCH_TORCH=${TORCHFITS_BENCH_TORCH:-2.13}"
    --env "TORCHFITS_BENCH_CUDA=${TORCHFITS_BENCH_CUDA:-0}"
    --env "TORCHFITS_BENCH_CU_FLAVOR=${TORCHFITS_BENCH_CU_FLAVOR:-cu129}"
    --env "TORCHFITS_BENCH_WHEELS=${TORCHFITS_BENCH_WHEELS:-vos:sfabbro/torchfits-gpu-bench/wheels}"
    --env "TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-${CPU}}"
    --env "OMP_NUM_THREADS=${OMP_NUM_THREADS:-${CPU}}"
  )
  if [[ -n "${GPU}" ]]; then
    CREATE_ARGS+=(--gpu "${GPU}")
  fi
  set +o pipefail
  canfar "${CREATE_ARGS[@]}" -- bash -c "${REMOTE_CMD}" 2>&1 | tee "${CREATE_LOG}"
  CREATE_RC=${PIPESTATUS[0]}
  set -o pipefail
  if [[ "${CREATE_RC}" -ne 0 ]]; then
    if rg -q 'No authentication provided for unknown or private image' "${CREATE_LOG}" 2>/dev/null; then
      cat >&2 <<EOF
canfar create failed: private registry image (${IMAGE}).

${IMAGE} is not in \`canfar image ls\`; x509 alone only pulls contributed/public
images. For astroai/base configure Harbor once:

  canfar config set registry.url https://images.canfar.net
  canfar config set registry.username <harbor-user>
  canfar config set registry.secret <harbor-token>

Or use a listed image, e.g.:
  TORCHFITS_CANFAR_IMAGE=astroai/notebook:latest pixi run bench-canfar-gpu
EOF
    fi
    echo "canfar create failed (rc=${CREATE_RC}); see ${CREATE_LOG}" >&2
    exit 1
  fi

  SESSION_ID="$(
    python3 - "${CREATE_LOG}" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
m = re.search(r"ID:\s*([^)]+)\)", text)
print(m.group(1).strip() if m else "")
PY
  )"
  if [[ -z "${SESSION_ID}" ]]; then
    echo "could not parse session ID from ${CREATE_LOG}" >&2
    exit 1
  fi
  echo "${SESSION_ID}" > "${LOCAL_OUT}/session_id.txt"
  echo "session_id=${SESSION_ID}" | tee -a "${LAUNCH_LOG}"
fi
echo "${RUN_ID}" > "${LOCAL_OUT}/run_id.txt"
echo "${VOS_DEST}" > "${LOCAL_OUT}/vos_dest.txt"

# Detach poll+fetch by default so a dead parent shell cannot drop durable results.
# Set TORCHFITS_CANFAR_FOREGROUND=1 to wait in-process.
if [[ "${TORCHFITS_CANFAR_FOREGROUND:-0}" != "1" && -z "${TORCHFITS_CANFAR_POLLER:-}" ]]; then
  POLLER_LOG="${LOCAL_OUT}/poller.log"
  # Double-fork out of the launcher process group (Cursor/agent shells
  # often reap the whole group when the parent command exits; nohup alone
  # is not enough without leaving the session).
  CANFAR_BIN="$(command -v canfar)"
  CANFAR_DIR="$(cd "$(dirname "${CANFAR_BIN}")" && pwd)"
  POLLER_PATH="${PATH}:${CANFAR_DIR}"
  {
    echo '#!/usr/bin/env bash'
    echo "export PATH=$(printf %q "${POLLER_PATH}")"
    echo 'export TORCHFITS_CANFAR_POLLER=1'
    echo 'export TORCHFITS_CANFAR_FOREGROUND=1'
    echo "export TORCHFITS_GIT_REF=$(printf %q "${GIT_REF}")"
    echo "export TORCHFITS_BENCH_MODE=$(printf %q "${MODE}")"
    echo "export TORCHFITS_BENCH_RUN_ID=$(printf %q "${RUN_ID}")"
    echo "export TORCHFITS_CANFAR_IMAGE=$(printf %q "${IMAGE}")"
    echo "export TORCHFITS_CANFAR_GPU=$(printf %q "${GPU}")"
    echo "export TORCHFITS_CANFAR_CPU=$(printf %q "${CPU}")"
    echo "export TORCHFITS_CANFAR_MEMORY=$(printf %q "${MEMORY}")"
    echo "export TORCHFITS_CANFAR_NAME=$(printf %q "${NAME}")"
    echo "export TORCHFITS_VOS_DEST=$(printf %q "${VOS_DEST}")"
    echo "export TORCHFITS_BENCH_PYTHON=$(printf %q "${TORCHFITS_BENCH_PYTHON:-3.13}")"
    echo "export TORCHFITS_BENCH_TORCH=$(printf %q "${TORCHFITS_BENCH_TORCH:-2.13}")"
    echo "export TORCHFITS_BENCH_CUDA=$(printf %q "${TORCHFITS_BENCH_CUDA:-0}")"
    echo "export TORCHFITS_BENCH_CU_FLAVOR=$(printf %q "${TORCHFITS_BENCH_CU_FLAVOR:-cu129}")"
    echo "export TORCHFITS_BENCH_WHEELS=$(printf %q "${TORCHFITS_BENCH_WHEELS:-vos:sfabbro/torchfits-gpu-bench/wheels}")"
    echo "export TORCHFITS_CANFAR_POLL_SECS=$(printf %q "${POLL_SECS}")"
    echo "export TORCHFITS_CANFAR_MAX_WAIT_SECS=$(printf %q "${MAX_WAIT_SECS}")"
    echo "export TORCHFITS_CANFAR_EXISTING_SESSION=$(printf %q "${SESSION_ID}")"
    echo "exec bash $(printf %q "${ROOT_DIR}/scripts/launch_canfar_gpu_bench.sh")"
  } > "${LOCAL_OUT}/poller_daemon.sh"
  chmod +x "${LOCAL_OUT}/poller_daemon.sh"
  python3 - "${LOCAL_OUT}/poller_daemon.sh" "${LOCAL_OUT}/poller.pid" "${POLLER_LOG}" <<'PY'
import os, sys

daemon, pid_path, log_path = sys.argv[1:4]
# First fork.
if os.fork() > 0:
    raise SystemExit(0)
os.setsid()
# Second fork — orphan under launchd/init.
if os.fork() > 0:
    raise SystemExit(0)
os.chdir("/")
os.umask(0)
with open(log_path, "a", encoding="utf-8") as log:
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
devnull = open(os.devnull, "r")
os.dup2(devnull.fileno(), 0)
pid = os.spawnvpe(os.P_NOWAIT, "/bin/bash", ["bash", daemon], os.environ)
with open(pid_path, "w", encoding="utf-8") as fh:
    fh.write(str(pid))
os.waitpid(pid, 0)
raise SystemExit(0)
PY
  # Give the daemon a moment to write the pid file.
  sleep 0.2
  echo "detached poller pid=$(cat "${LOCAL_OUT}/poller.pid") log=${POLLER_LOG}" | tee -a "${LAUNCH_LOG}"
  echo "session_id=${SESSION_ID} run_id=${RUN_ID} vos=${VOS_DEST}"
  exit 0
fi

terminal_status() {
  local info_status ps_status
  info_status="$(
    canfar info "${SESSION_ID}" 2>/dev/null | python3 -c '
import re, sys

text = sys.stdin.read()
m = re.search(r"^\s*Status\s+(\S+)", text, re.MULTILINE)
print(m.group(1) if m else "")
' || true
  )"
  if [[ -n "${info_status}" ]]; then
    echo "${info_status}"
    return
  fi

  # ponytail: completed headless sessions may 404 on info; ps --all keeps history
  ps_status="$(
    canfar ps --all --json 2>/dev/null | SESSION_ID="${SESSION_ID}" python3 -c '
import json, os, sys

sid = os.environ["SESSION_ID"]
try:
    rows = json.load(sys.stdin)
except json.JSONDecodeError:
    raise SystemExit(0)
for row in rows:
    if row.get("id") == sid:
        print(row.get("status", ""))
        break
' 2>/dev/null || true
  )"
  if [[ -n "${ps_status}" ]]; then
    echo "${ps_status}"
  fi
}

STATUS=""
POLL_START="$(date +%s)"
while true; do
  STATUS="$(terminal_status || true)"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${STATUS:-unknown}" | tee -a "${LOCAL_OUT}/launcher.log"
  case "${STATUS}" in
    Succeeded|Completed|Failed|Error|Terminating)
      break
      ;;
  esac
  if (( $(date +%s) - POLL_START > MAX_WAIT_SECS )); then
    echo "timeout after ${MAX_WAIT_SECS}s waiting for terminal status" | tee -a "${LOCAL_OUT}/launcher.log"
    STATUS="Failed"
    break
  fi
  sleep "${POLL_SECS}"
done

canfar logs "${SESSION_ID}" > "${LOCAL_OUT}/canfar_logs.txt" 2>&1 || true
canfar events "${SESSION_ID}" > "${LOCAL_OUT}/canfar_events.txt" 2>&1 || true

if [[ "${STATUS}" == "Succeeded" || "${STATUS}" == "Completed" ]]; then
  if command -v vcp >/dev/null && bash scripts/fetch_canfar_bench_vos.sh "${RUN_ID}"; then
    echo "fetched benchmarks_results/${RUN_ID} from ${VOS_DEST}" | tee -a "${LOCAL_OUT}/launcher.log"
  elif python3 scripts/import_canfar_bench_artifacts.py "${LOCAL_OUT}/canfar_logs.txt" "${RUN_ID}" --dest "${ROOT_DIR}/benchmarks_results" 2>/dev/null; then
    echo "imported benchmarks_results/${RUN_ID} from session logs (vcp fallback)" | tee -a "${LOCAL_OUT}/launcher.log"
  else
    echo "fetch results: bash scripts/fetch_canfar_bench_vos.sh ${RUN_ID}" | tee -a "${LOCAL_OUT}/launcher.log"
    echo "  (vos: ${VOS_DEST})" | tee -a "${LOCAL_OUT}/launcher.log"
  fi
fi

echo "finished status=${STATUS}" | tee -a "${LOCAL_OUT}/launcher.log"
echo "local artifacts: ${LOCAL_OUT}" | tee -a "${LOCAL_OUT}/launcher.log"
echo "vos artifacts: ${VOS_DEST}" | tee -a "${LOCAL_OUT}/launcher.log"

case "${STATUS}" in
  Succeeded|Completed) exit 0 ;;
  *) exit 1 ;;
esac
