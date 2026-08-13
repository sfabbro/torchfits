#!/usr/bin/env bash
# Build a torchfits wheel locally for every (torch lane, CPython) cell and
# drop the wheels into dist-local/. Docker-free smoke (linux_x86_64 / macos
# tags). Published manylinux / macosx wheels come from
# `bash scripts/cibuildwheel.sh` / `.github/workflows/build_wheels.yml`.
#
# Each cell builds in a throwaway copy of the tree whose lane pins are
# rendered onto the requested torch lane by scripts/release_lane.py, against
# the lane-pinned torch from the PyTorch CPU index. The committed repo state
# is never touched.
#
# Usage:
#   bash scripts/build_wheels_local.sh [--lanes 2.12,2.13] [--pythons 3.11,3.13] [--jobs 2]
#
# Flags:
#   --lanes      comma-separated torch lanes (default: all in scripts/torch_lanes.json)
#   --pythons    comma-separated CPython versions (default: 3.10,3.11,3.12,3.13,3.14)
#   --jobs       parallel build cells (default: 2)
#   --prerelease prerelease suffix (e.g. rc5) to render onto the lane version,
#                mirroring `release_lane.py --prerelease` (matches the rc tag
#                state the CI wheel build runs on)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="dist-local"
JOBS=2
LANES=""
PYTHONS="3.10,3.11,3.12,3.13,3.14"
PRERELEASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lanes) LANES="$2"; shift 2 ;;
    --pythons) PYTHONS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --prerelease) PRERELEASE="$2"; shift 2 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) OUT_DIR="$1"; shift ;;
  esac
done

cd "$ROOT"
if [[ -z "$LANES" ]]; then
  LANES="$(pixi run python -c 'import json; print(",".join(json.load(open("scripts/torch_lanes.json"))))')"
fi
mkdir -p "$OUT_DIR"

if [[ ! -d extern/cfitsio ]]; then
  echo "== vendoring cfitsio (needed by every build) =="
  bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt
fi

PIXI_PY="$(pixi run python -c 'import sys; print(sys.executable)')"
PIXI_PREFIX="$(pixi run python -c 'import sys; print(sys.prefix)')"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/logs"

next_minor() {
  pixi run python -c "major, minor = map(int, '$1'.split('.')); print(f'<{major}.{minor + 1}')"
}

build_cell() {
  local lane="$1" py="$2"
  local copy="$WORK/$lane-py$py"
  local log="$WORK/logs/$lane-py$py.log"
  mkdir -p "$copy"
  if {
    echo "== $lane / py$py: copying tree and rendering lane pins =="
    tar -C "$ROOT" -cf - \
      --exclude=.git --exclude=.pixi --exclude=dist-local --exclude=dist \
      --exclude=build --exclude='*.egg-info' --exclude=__pycache__ \
      --exclude='.venv' --exclude='venv*' --exclude=docs \
      . | tar -C "$copy" -xf -
    "$PIXI_PY" "$copy/scripts/release_lane.py" --lane "$lane" \
      $(if [[ -n "$PRERELEASE" ]]; then printf '%s' "--prerelease $PRERELEASE"; fi) \
      --apply
    "$PIXI_PY" "$copy/scripts/release_lane.py" --check
    echo "== $lane / py$py: creating venv (uv pulls CPython $py) =="
    uv venv --python "$py" --no-project "$copy/.venv" --quiet
    uv pip install --python "$copy/.venv/bin/python" --quiet pip
    echo "== $lane / py$py: installing torch $lane + build deps =="
    uv pip install --python "$copy/.venv/bin/python" --quiet \
      "torch>=$lane,$(next_minor "$lane")" \
      --extra-index-url https://download.pytorch.org/whl/cpu
    uv pip install --python "$copy/.venv/bin/python" --quiet \
      "numpy" "nanobind" "scikit-build-core"
    echo "== $lane / py$py: building wheel =="
    (cd "$copy" && CMAKE_PREFIX_PATH="$PIXI_PREFIX" MACOSX_DEPLOYMENT_TARGET=11.0 \
      ./.venv/bin/python -m pip wheel . \
      --no-deps --no-build-isolation -w "$ROOT/$OUT_DIR")
  } >"$log" 2>&1; then
    echo "[ OK ] $lane / py$py"
  else
    echo "[FAIL] $lane / py$py (see $log)"
    tail -5 "$log"
    return 1
  fi
}

export -f build_cell next_minor
export ROOT OUT_DIR WORK PIXI_PY PIXI_PREFIX PRERELEASE

cells=()
for lane in $(echo "$LANES" | tr ',' ' '); do
  for py in $(echo "$PYTHONS" | tr ',' ' '); do
    cells+=("$lane $py")
  done
done

echo "== building ${#cells[@]} cells in $OUT_DIR (jobs=$JOBS) =="
printf '%s\n' "${cells[@]}" | xargs -P "$JOBS" -n2 bash -c 'build_cell "$1" "$2"' _

echo "== wheels =="
ls -lh "$OUT_DIR"/torchfits-*.whl
