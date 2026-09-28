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

python -c 'import torchfits._C as C; assert C.HAS_BZIP2, "wheel must link libbz2 (HAS_BZIP2)"'
# The three native artifacts ship together and share one build id. If the wheel
# lost libtorchfits_core, or shipped a _core from a different build than the
# library it dlopens, the failure is a metadata ImportError (or a cross-version
# call into C) that no functional test above would reach.
python -c 'import torchfits._core as c; assert c.TORCH_FREE, "the metadata core must be torch-free"; assert c.core_library_build_id() == c.__build_id__, "build id mismatch"; import torchfits._C as C; assert C.__core_build_id__ == c.__build_id__, "extension and library are from different builds"'
python -m pytest -c /dev/null --noconftest \
  "${PROJECT}/tests/test_release_smoke.py" \
  "${PROJECT}/tests/test_bz2.py" -q
