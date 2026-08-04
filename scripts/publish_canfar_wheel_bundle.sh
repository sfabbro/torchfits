#!/usr/bin/env bash
# Publish the prebuilt torchfits wheel matrix (all python x torch lane combos)
# to VOSpace so every CANFAR matrix session installs identical binaries
# without rebuilding. Idempotent: skips uploads for files that already exist.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SRC="${TORCHFITS_WHEEL_SRC:-dist-local}"
DEST="${TORCHFITS_WHEELS_DEST:-vos:sfabbro/torchfits-gpu-bench/wheels}"

if ! command -v vcp >/dev/null; then
  echo "vcp not found (vos 3.x CLI; e.g. /opt/astroai/venv/cadc/bin/vcp)" >&2
  exit 1
fi
if [[ ! -d "${SRC}" ]]; then
  echo "no ${SRC}/ directory (run the wheel matrix build first)" >&2
  exit 1
fi

mkdir -p "benchmarks_results/wheels-manifest"
MANIFEST="benchmarks_results/wheels-manifest/manifest.txt"
{
  echo "torchfits wheel bundle for CANFAR matrix legs"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "torch_lanes: $(python3 scripts/release_lane.py --print-pins | sed 's/.*torch=//')"
  echo "---"
} > "${MANIFEST}"
for w in "${SRC}"/*.whl; do
  sha256sum "${w}" >> "${MANIFEST}"
done
cp -a "${MANIFEST}" "${SRC}/manifest.txt"

# vos 3.x splits the CLI: use vmkdir/vls binaries (vcp mkdir/ls do not exist).
if command -v vmkdir >/dev/null; then
  vmkdir -p "${DEST}"
elif ! vcp mkdir "${DEST}" 2>/dev/null && ! vcp mkdir -p "${DEST}" 2>/dev/null; then
  echo "could not create ${DEST}" >&2
  exit 1
fi
UPLOADED=0
for w in "${SRC}"/*.whl "${SRC}/manifest.txt"; do
  NAME="$(basename "${w}")"
  if command -v vls >/dev/null && vls "${DEST}/${NAME}" >/dev/null 2>&1; then
    echo "exists: ${NAME}"
    continue
  fi
  if ! command -v vls >/dev/null && vcp ls "${DEST}/${NAME}" >/dev/null 2>&1; then
    echo "exists: ${NAME}"
    continue
  fi
  echo "uploading ${NAME}"
  vcp "${w}" "${DEST}/${NAME}"
  UPLOADED=$((UPLOADED + 1))
done

echo "bundle at ${DEST} (uploaded=${UPLOADED})"
vls -l "${DEST}" 2>/dev/null || vcp ls -l "${DEST}" || true
