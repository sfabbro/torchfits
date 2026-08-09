#!/usr/bin/env bash
# Idempotent fetch of public CFHT MegaCam calibration frames (darks + biases)
# from the CADC Direct Data Service. Lands in benchmarks_data/cfht_megacam/calib/
# (gitignored). Used by examples/example_megacam_cr_denoise.py to train the
# Noise2Noise dark->blank denoiser on real detector-noise twins (zero field ->
# any two darks are perfect N2N pairs; any two biases likewise).
#
# Discovery (reproducible, CADC TAP argus endpoint):
#   The CAOM2 Observation column is `type` (uppercase values 'DARK', 'BIAS',
#   'OBJECT', ...), NOT obs_type, and the service lives at
#   https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync (the old
#   cadc-ccda.hawaii.edu host is retired). Example:
#
#   curl -sL --data-urlencode "REQUEST=doQuery" --data-urlencode "LANG=ADQL" \
#     --data-urlencode "QUERY=SELECT o.observationID, p.time_exposure, a.uri
#       FROM caom2.Observation o
#       JOIN caom2.Plane p ON o.obsID=p.obsID
#       JOIN caom2.Artifact a ON a.planeID=p.planeID
#       WHERE o.collection='CFHT' AND o.instrument_name LIKE '%Mega%'
#         AND o.type='DARK' AND p.time_exposure=250
#         AND o.observationID BETWEEN '2360000' AND '2599999'
#         AND a.productType='calibration' ORDER BY o.observationID" \
#     'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync'
#
#   Verified counts for MegaCam (2026-08-08): DARK 6336, BIAS 18000.
#   The 250 s darks are the closest public exposure time to the 200 s science
#   frames of the sample; darks/biases are picked from the same 2019-2020
#   era (obsID 2360xxx-2584xxx) so the readout electronics match the science.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/benchmarks_data/cfht_megacam/calib"
BASE="https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT"

# 250 s darks, same era as the sample science frames (2366188/2366432/2376828
# of 2019, 2480747/2583975 of 2020). 'd' suffix per CADC artifact pattern.
DARKS=(
  2366052d 2366301d 2366784d 2367719d   # 2019, same nights as 2366188/2366432
  2538866d 2541247d 2541405d 2570498d   # 2020
  2574754d 2576409d 2581632d 2584041d   # 2020, near 2583975
)

# 0 s biases, same era. 'b' suffix per CADC artifact pattern.
BIASES=(
  2360150b 2360151b 2360152b 2360153b   # 2019
  2360154b 2360155b 2360156b 2586437b   # 2019 + 2020
)

fetch_one() {
  local name="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "skip (exists): ${name}.fits.fz"
    return 0
  fi
  local url="${BASE}/${name}.fits.fz"
  echo "fetch: ${url}"
  if curl -fL -C - --retry 2 --retry-all-errors --connect-timeout 30 -o "${out}.partial" "$url"; then
    mv "${out}.partial" "$out"
  else
    rm -f "${out}.partial"
    echo "warn: failed ${name}" >&2
  fi
}

ok=0
for kind in darks biases; do
  mkdir -p "${DEST}/${kind}"
  for name in $(if [[ "$kind" == darks ]]; then printf '%s\n' "${DARKS[@]}"; else printf '%s\n' "${BIASES[@]}"; fi); do
    fetch_one "$name" "${DEST}/${kind}/${name}.fits.fz" && ok=$((ok + 1))
  done
done

echo "ready: ${ok} calibration frame(s) under ${DEST}"
