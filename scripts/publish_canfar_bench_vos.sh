#!/usr/bin/env bash
# Upload torchfits CANFAR bench artifacts to VOSpace via vcp.
# vos/CADC CLI is preinstalled in astroai/base images — never pip-install
# into $HOME here (concurrent sessions corrupt shared user-site packages).
set -euo pipefail

bench_out="${1:?usage: $0 <benchmarks_results/run-id-dir>}"
vos_dest="${2:?usage: $0 <bench-out> <vos:user/path/run-id>}"

if ! command -v vcp >/dev/null; then
  echo "vcp not on PATH (astroai/base preinstalls vos); refusing to pip-install into \$HOME" >&2
  exit 1
fi

if command -v vmkdir >/dev/null; then
  vmkdir -p "${vos_dest}" 2>/dev/null || true
fi

# ponytail: copy directory *contents* (not the directory name) into vos_dest
vcp "${bench_out}/." "${vos_dest}/"
echo "TORCHFITS_VOS_URI=${vos_dest}"
