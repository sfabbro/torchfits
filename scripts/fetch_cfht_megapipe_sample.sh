#!/usr/bin/env bash
# Idempotent fetch of the public CFHTLS-Deep D1 MegaPipe g/r/i mosaics + SExtractor
# catalog from the CADC Direct Data Service (~5.3 GB total; skips existing files).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/benchmarks_data/cfht_megapipe"
BASE=https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHTSG

mkdir -p "$DEST"

FILES=(
  D1.IQ.G.fits
  D1.IQ.R.fits
  D1.IQ.I.fits
  D1.IQ.G.cat
)

count=0
for name in "${FILES[@]}"; do
  out="${DEST}/${name}"
  if [[ -s "$out" ]]; then
    echo "skip (exists): ${name}"
    count=$((count + 1))
    continue
  fi
  url="${BASE}/${name}"
  echo "fetch: ${url}"
  if curl -fL -C - --retry 2 --retry-all-errors --connect-timeout 30 -o "${out}.partial" "$url"; then
    mv "${out}.partial" "$out"
    count=$((count + 1))
  else
    rm -f "${out}.partial"
    echo "warn: failed ${name}" >&2
  fi
done

echo "ready: ${count} file(s) under ${DEST}"
