#!/usr/bin/env bash
# Download torchfits CANFAR bench artifacts from VOSpace to benchmarks_results/<run-id>/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: $0 <run-id>}"
VOS_BASE="${TORCHFITS_VOS_BASE:-vos:sfabbro/torchfits-gpu-bench}"
VOS_URI="${VOS_BASE}/${RUN_ID}"
LOCAL_DIR="${ROOT_DIR}/benchmarks_results/${RUN_ID}"

if command -v vcp >/dev/null; then
  VCP=(vcp)
else
  # vos is a pixi pypi-dependency; the script is not on PATH outside the env.
  VCP=(pixi run vcp)
fi

mkdir -p "${LOCAL_DIR}"
"${VCP[@]}" "${VOS_URI}/" "${LOCAL_DIR}/"
# ponytail: pre-fix uploads nested <run-id>/ inside dest; flatten for patch scripts
nested="${LOCAL_DIR}/${RUN_ID}"
if [[ -d "${nested}" ]]; then
  shopt -s nullglob dotglob
  mv "${nested}"/* "${LOCAL_DIR}/" 2>/dev/null || true
  rmdir "${nested}" 2>/dev/null || true
  shopt -u nullglob dotglob
fi
echo "fetched ${VOS_URI} -> ${LOCAL_DIR}"
