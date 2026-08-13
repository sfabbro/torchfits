#!/usr/bin/env bash
# Runs inside astroai/base on CANFAR (headless GPU session): install the
# published (CPU-linked) torchfits wheel for one torch lane and verify it
# against a CUDA-flavored lane torch on a real GPU. There is no separate
# CUDA torchfits wheel. Driver-vs-flavor fit is probed, not assumed: try
# each CUDA flavor the lane ships (scripts/torch_lanes.json); the first one
# where torch.cuda.is_available() succeeds wins.
#
# Prints a one-line "[ OK ] lane=<lane> cu=<flavor> driver=<ver>" or
# "[FAIL] lane=<lane> ..." result marker that the launcher greps for.
set -euo pipefail

: "${TORCHFITS_CUDA_VERIFY_LANE:?set by the launcher}"
WHEEL_URL="${TORCHFITS_WHEEL_URL:-}"

SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
WORK="${SCRATCH}/torchfits-cuda-verify/${TORCHFITS_CUDA_VERIFY_LANE}"
mkdir -p "$WORK"
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
export PYTHONNOUSERSITE=1

DRIVER="unknown"
if command -v nvidia-smi >/dev/null; then
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  nvidia-smi -L || true
fi
echo "driver=${DRIVER:-unknown}"

SPEC="$(pixi run python - "${TORCHFITS_CUDA_VERIFY_LANE}" <<'PY'
import json, pathlib, sys
lane = sys.argv[1]
major, minor = (int(x) for x in lane.split("."))
lanes = json.loads(pathlib.Path("scripts/torch_lanes.json").read_text())
print(">={},<{}.{}".format(lane, major, minor + 1))
PY
)"
VERSION="$(pixi run python - "${TORCHFITS_CUDA_VERIFY_LANE}" <<'PY'
import json, pathlib, re, sys
lane = sys.argv[1]
lanes = json.loads(pathlib.Path("scripts/torch_lanes.json").read_text())
if lane in lanes:
    version = lanes[lane]["torchfits_version"]
else:
    current = next(iter(lanes.values()))
    version = f"{current['torchfits_version']}.dev0+torch{lane.replace('.', '')}"
print(version)
PY
)"
CU_FLAVORS="$(pixi run python - "${TORCHFITS_CUDA_VERIFY_LANE}" <<'PY'
import json, pathlib, sys
lane = sys.argv[1]
lanes = json.loads(pathlib.Path("scripts/torch_lanes.json").read_text())
if lane in lanes:
    flavors = lanes[lane]["cu"]
else:
    flavors = next(iter(lanes.values()))["cu"]
print(" ".join(flavors))
PY
)"

PYBIN="$(pixi run python -c 'import sys; print(sys.executable)')"
VENV="$WORK/venv"
rm -rf "$VENV"
"$PYBIN" -m venv "$VENV"

echo "=== lane ${TORCHFITS_CUDA_VERIFY_LANE} (torchfits ${VERSION}) probing CUDA flavors: ${CU_FLAVORS} ==="

PICKED=""
for cu in ${CU_FLAVORS}; do
  echo "--- trying ${cu} ---"
  "$VENV/bin/pip" install -q --upgrade pip
  if "$VENV/bin/pip" install -q "torch${SPEC}" --index-url "https://download.pytorch.org/whl/${cu}"; then
    if "$VENV/bin/python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
      PICKED="${cu}"
      break
    fi
  fi
  "$VENV/bin/pip" uninstall -q -y torch 2>/dev/null || true
done

if [[ -z "${PICKED}" ]]; then
  echo "[FAIL] lane=${TORCHFITS_CUDA_VERIFY_LANE} driver=${DRIVER:-unknown}: no CUDA flavor in (${CU_FLAVORS}) gave torch.cuda.is_available()"
  exit 1
fi

echo "=== installing torchfits wheel ==="
if [[ -n "${WHEEL_URL}" ]]; then
  "$VENV/bin/pip" install -q "${WHEEL_URL}"
else
  "$VENV/bin/pip" install -q "torchfits==${VERSION}"
fi
"$VENV/bin/pip" install -q pytest

echo "=== import + cuda + roundtrip + release smoke ==="
if ! TORCHFITS_EXPECT_VERSION="${VERSION}" "$VENV/bin/python" - <<'PY'
import os
import tempfile
from pathlib import Path

import torch
import torchfits

assert torchfits.__version__ == os.environ["TORCHFITS_EXPECT_VERSION"], torchfits.__version__
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))
img = torch.arange(16, dtype=torch.float32).reshape(4, 4)
with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
    path = Path(fh.name)
try:
    torchfits.write(str(path), img, overwrite=True)
    out = torchfits.read(str(path))
    assert torch.allclose(out.cpu(), img)
finally:
    path.unlink(missing_ok=True)
print("roundtrip ok")
PY
then
  echo "[FAIL] lane=${TORCHFITS_CUDA_VERIFY_LANE} cu=${PICKED} driver=${DRIVER:-unknown}: import/roundtrip failed"
  exit 1
fi

if ! "$VENV/bin/python" -m pytest -q -c /dev/null --noconftest \
    /tmp/torchfits/tests/test_release_smoke.py; then
  echo "[FAIL] lane=${TORCHFITS_CUDA_VERIFY_LANE} cu=${PICKED} driver=${DRIVER:-unknown}: release smoke failed"
  exit 1
fi

echo "[ OK ] lane=${TORCHFITS_CUDA_VERIFY_LANE} cu=${PICKED} driver=${DRIVER:-unknown}"
