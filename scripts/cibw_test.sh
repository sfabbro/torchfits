#!/usr/bin/env bash
# cibuildwheel test-command: lane-pinned CPU torch + release smoke.
# {project} is passed as $1 (the mounted source tree).
set -euo pipefail

PROJECT="${1:?cibuildwheel test-command must pass project path}"
PIN="$(grep -E '^torch>=' "${PROJECT}/constraints-wheel.txt" | head -1)"
test -n "${PIN}"

python -m pip install "${PIN}" --extra-index-url https://download.pytorch.org/whl/cpu

TORCH_LIB="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
if [ "$(uname)" = "Darwin" ]; then
  export DYLD_FALLBACK_LIBRARY_PATH="${TORCH_LIB}${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python -m pytest -c /dev/null --noconftest "${PROJECT}/tests/test_release_smoke.py" -q
