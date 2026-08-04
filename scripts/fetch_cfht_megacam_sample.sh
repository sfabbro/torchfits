#!/usr/bin/env bash
# Idempotent fetch of public CFHT MegaCam FITS from CADC Direct Data Service.
# Files land in benchmarks_data/cfht_megacam/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/benchmarks_data/cfht_megacam"
BASE="https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT"

mkdir -p "$DEST"

# Documented public CFHT MegaCam / MegaPrime examples (CADC archive pattern).
# Replace or extend this list as needed; existing files are skipped.
FILES=(
  2583975o.fits.fz
  2583975p.fits.fz
  1722795p.fits.fz
  2366432o.fits.fz
  2366432p.fits.fz
  2366188o.fits.fz
  2366188p.fits.fz
  2376828o.fits.fz
  2376828p.fits.fz
  2480747o.fits.fz
)

count=0
for name in "${FILES[@]}"; do
  if (( count >= 10 )); then
    break
  fi
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
