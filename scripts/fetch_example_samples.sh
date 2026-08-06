#!/usr/bin/env bash
# Idempotent fetch of examples/_sample_data.py SAMPLES into the torchfits sample
# cache via curl. Skips manga_logcube (~200MB) unless --with-manga is given.
set -euo pipefail

# Keep shell downloads aligned with examples/_sample_data.py and
# torchfits.cache.sample_cache_root().
CACHE_DIR="${TORCHFITS_SAMPLE_CACHE:-${TORCHFITS_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/torchfits}/samples}"
mkdir -p "$CACHE_DIR"

WITH_MANGA=0
[[ "${1:-}" == "--with-manga" ]] && WITH_MANGA=1

# Mirrors examples/_sample_data.py SAMPLES; dest keeps the URL's suffix.
SAMPLES=(
  "horsehead:http://data.astropy.org/tutorials/FITS-images/HorseHead.fits"
  "chandra_events:http://data.astropy.org/tutorials/FITS-tables/chandra_events.fits"
  "sdss_spectrum:https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/0751/spec-0751-52251-0160.fits"
  "m13_blue_0001:http://data.astropy.org/tutorials/FITS-images/M13_blue_0001.fits"
  "m13_blue_0002:http://data.astropy.org/tutorials/FITS-images/M13_blue_0002.fits"
  "m13_blue_0003:http://data.astropy.org/tutorials/FITS-images/M13_blue_0003.fits"
  "m13_blue_0004:http://data.astropy.org/tutorials/FITS-images/M13_blue_0004.fits"
  "m13_blue_0005:http://data.astropy.org/tutorials/FITS-images/M13_blue_0005.fits"
  "fits_header_mef:http://data.astropy.org/tutorials/FITS-Header/input_file.fits"
  "sdss_lupton_g:http://data.astropy.org/visualization/reprojected_sdss_g.fits.bz2"
  "sdss_lupton_r:http://data.astropy.org/visualization/reprojected_sdss_r.fits.bz2"
  "sdss_lupton_i:http://data.astropy.org/visualization/reprojected_sdss_i.fits.bz2"
  "spitzer_example:http://data.astropy.org/photometry/spitzer_example_image.fits"
  "radio_cube_c14:http://data.astropy.org/tutorials/FITS-cubes/reduced_TAN_C14.fits"
  "galaxy_zoo1_table2:https://galaxy-zoo-1.s3.amazonaws.com/GalaxyZoo1_DR_table2.fits"
)
if [[ $WITH_MANGA -eq 1 ]]; then
  SAMPLES+=("manga_logcube:https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/7443/stack/manga-7443-12703-LOGCUBE.fits.gz")
fi

for entry in "${SAMPLES[@]}"; do
  name="${entry%%:*}"
  url="${entry#*:}"
  base="$(basename "$url")"
  ext="${base#*.}"
  dest="${CACHE_DIR}/${name}.${ext}"
  if [[ -s "$dest" ]]; then
    echo "skip (cached): ${name}"
    continue
  fi
  echo "fetch: ${url}"
  if curl -fL --retry 2 --connect-timeout 30 -o "${dest}.partial" "$url"; then
    mv "${dest}.partial" "$dest"
  else
    rm -f "${dest}.partial"
    echo "warn: failed ${name}" >&2
  fi
done

echo "ready: samples under ${CACHE_DIR}"
