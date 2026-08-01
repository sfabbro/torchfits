#!/usr/bin/env bash
# Build a local wheel and smoke-import it in a fresh venv (outside the src tree).
# Full multi-Python matrix is covered by cibuildwheel on tagged releases.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR="${TMPDIR:-/tmp}/torchfits-clean-install-$$"
DIST="$WORKDIR/dist"
mkdir -p "$DIST"
trap 'rm -rf "$WORKDIR"' EXIT

cd "$ROOT"
echo "=== building wheel into $DIST ==="
pixi run python -m pip wheel . -w "$DIST" --no-deps --no-build-isolation
WHEEL="$(ls -1 "$DIST"/torchfits-*.whl | head -1)"
test -n "$WHEEL"
echo "wheel: $WHEEL"
SHASUM="$(shasum -a 256 "$WHEEL" | awk '{print $1}')"
echo "sha256: $SHASUM"

PIXI_PY="$(pixi run python -c 'import sys; print(sys.executable)')"
TAG="$(basename "$WHEEL")"
# e.g. torchfits-1.0.0rc1-cp313-cp313-macosx_14_0_arm64.whl
PY_TAG="$(echo "$TAG" | sed -n 's/.*-\(cp[0-9][0-9][0-9]\)-.*/\1/p')"

smoke() {
  local py="$1"
  local venv="$WORKDIR/venv-$(basename "$py")"
  echo "=== clean install smoke: $py ==="
  "$py" -m venv "$venv"
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
  python -m pip install -q --upgrade pip
  # Documented CPU-only one-liner (docs/install.md): local wheel + ABI-lane
  # torch pin, torch pulled from the PyTorch CPU index (numpy/pyarrow come
  # in as torchfits dependencies).
  python -m pip install -q "$WHEEL" "torch>=2.10,<2.11" --extra-index-url https://download.pytorch.org/whl/cpu
  python - <<'PY'
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchfits

assert torchfits.__version__
img = torch.arange(16, dtype=torch.float32).reshape(4, 4)
with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
    path = Path(fh.name)
try:
    torchfits.write(str(path), img, overwrite=True)
    out = torchfits.read(str(path))
    assert torch.allclose(out.cpu(), img)
    table = {"RA": np.array([1.0, 2.0]), "ID": np.array([1, 2], dtype=np.int64)}
    torchfits.write(str(path), table, overwrite=True)
    arrow = torchfits.table.read(str(path), hdu=1)
    assert arrow.num_rows == 2
finally:
    path.unlink(missing_ok=True)
print("ok", torchfits.__version__)
PY
  deactivate
}

smoke "$PIXI_PY"
echo "=== clean_install_smoke done (abi=$PY_TAG) ==="
echo "NOTE: multi-Python wheels are built by .github/workflows/build_wheels.yml on tag."
