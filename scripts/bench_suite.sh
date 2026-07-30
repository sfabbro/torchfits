#!/usr/bin/env bash
# Run a named modular benchmark suite (see benchmarks/suites.py).
# Usage:
#   bash scripts/bench_suite.sh hcompress
#   bash scripts/bench_suite.sh fitstable_predicate --no-mmap
# Extra args after the suite name are forwarded to bench_all.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <suite> [extra bench_all args...]" >&2
  pixi run -e "${TORCHFITS_BENCH_ENV:-bench-all}" python -c \
    "from benchmarks.suites import list_suite_names; print('suites:', ', '.join(list_suite_names()))"
  exit 2
fi

SUITE="$1"
shift
# Pixi forwards a leading `--` when callers use `pixi run bench-suite NAME -- --flag`.
if [[ "${1:-}" == "--" ]]; then
  shift
fi
ENV_NAME="${TORCHFITS_BENCH_ENV:-bench-all}"
RUN_ID="${TORCHFITS_BENCH_RUN_ID:-suite_${SUITE}_$(date -u +%Y%m%d_%H%M%S)}"

# Disable SharedReadMeta stale-stat revalidation during one-shot benches —
# 1s polling adds jitter that invents CFITSIO-parity flips vs fitsio.
export TORCHFITS_SHARED_META_VALIDATE="${TORCHFITS_SHARED_META_VALIDATE:-0}"

echo "==> suite=${SUITE} run_id=${RUN_ID}"
pixi run -e "$ENV_NAME" python benchmarks/bench_all.py \
  --suite "$SUITE" \
  --run-id "$RUN_ID" \
  "$@"
