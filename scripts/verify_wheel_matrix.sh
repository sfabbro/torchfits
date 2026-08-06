#!/usr/bin/env bash
# Verify locally built wheels (scripts/build_wheels_local.sh output) in fresh
# venvs, mirroring the cibuildwheel test command: install the wheel plus the
# lane-pinned torch from the PyTorch CPU index, then run import + CLI + the
# release smoke test. Exits non-zero if any wheel fails.
#
# Usage:
#   bash scripts/verify_wheel_matrix.sh [--lanes 2.12,2.13] [--jobs 2] [wheel-dir]
#   default wheel-dir: dist-local
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANES=""
JOBS=2
WHEEL_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lanes) LANES="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) WHEEL_DIR="$1"; shift ;;
  esac
done
WHEEL_DIR="${WHEEL_DIR:-dist-local}"

cd "$ROOT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/logs"

next_minor() {
  pixi run python -c "major, minor = map(int, '$1'.split('.')); print(f'<{major}.{minor + 1}')"
}

verify_cell() {
  local wheel="$1"
  local log="$WORK/logs/$(basename "$wheel").log"
  local lane
  # torchfits-1.0.0-cp313-cp313-linux_x86_64.whl -> lane from version, py from tag
  lane="$(basename "$wheel" | sed -n 's/^torchfits-\([0-9][0-9.]*[a-z0-9+]*\)-cp.*/\1/p' |
    pixi run python -c '
import json, re, sys
v = sys.stdin.read().strip()
lanes = json.load(open("scripts/torch_lanes.json"))
lane = next((l for l, c in lanes.items() if c["torchfits_version"] == v), "")
if not lane:
    # Prerelease wheels (e.g. 1.0.0rc5) carry a PEP 440 suffix on the lane
    # version; strip it before matching torch_lanes.json.
    base = re.sub(r"(rc|a|b|\.dev0\+\S*)\d*$", "", v) if v else ""
    lane = next((l for l, c in lanes.items() if c["torchfits_version"] == base), "")
if not lane:
    m = re.fullmatch(r".*\.dev0\+torch(\d)(\d+)", v)
    if m:
        lane = f"{m.group(1)}.{m.group(2)}"
print(lane)')"
  py="$(basename "$wheel" | sed -n 's/^torchfits-[^-]*-cp\([0-9][0-9][0-9]\)-\?.*/\1/p' | sed 's/^3/3./')"
  if [[ -z "$lane" || -z "$py" ]]; then
    echo "[FAIL] $(basename "$wheel"): cannot map to lane/python" >&2
    return 1
  fi
  if [[ -n "$LANES" ]] && ! echo "$LANES" | tr ',' '\n' | grep -qx "$lane"; then
    echo "[SKIP] $(basename "$wheel"): lane $lane not requested"
    return 0
  fi
  local venv="$WORK/venv-$(basename "$wheel")"
  if {
    echo "== $(basename "$wheel"): fresh venv py$py =="
    uv venv --python "$py" --no-project "$venv" --quiet
    uv pip install --python "$venv/bin/python" --quiet \
      "$ROOT/$WHEEL_DIR/$(basename "$wheel")" \
      "torch>=$lane,$(next_minor "$lane")" \
      --extra-index-url https://download.pytorch.org/whl/cpu
    uv pip install --python "$venv/bin/python" --quiet pytest
    TORCH_LIB="$("$venv/bin/python" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
    export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "== $(basename "$wheel"): import + CLI + release smoke =="
    "$venv/bin/python" -c 'import torchfits; print("import ok:", torchfits.__version__)'
    "$venv/bin/torchfits" --help >/dev/null
    "$venv/bin/python" -m pytest -c /dev/null --noconftest \
      "$ROOT/tests/test_release_smoke.py" -q
  } >"$log" 2>&1; then
    echo "[ OK ] $(basename "$wheel")"
  else
    echo "[FAIL] $(basename "$wheel") (see $log)"
    tail -20 "$log"
    return 1
  fi
}

export -f verify_cell next_minor
export ROOT WORK LANES WHEEL_DIR

shopt -s nullglob
wheels=("$WHEEL_DIR"/torchfits-*.whl)
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "no wheels found in $WHEEL_DIR" >&2
  exit 2
fi

echo "== verifying ${#wheels[@]} wheels (jobs=$JOBS) =="
printf '%s\n' "${wheels[@]}" | xargs -P "$JOBS" -n1 bash -c 'verify_cell "$1"' _
