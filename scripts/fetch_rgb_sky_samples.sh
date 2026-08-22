#!/usr/bin/env bash
# Fetch public RGB sky cutouts for examples/example_rgb_sky.py.
# Anonymous HTTP. HSC-SSP is not fetched (PDR3 needs an account).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_DIR="${TORCHFITS_SAMPLE_CACHE:-${TORCHFITS_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/torchfits}/samples}/rgb_sky"
mkdir -p "$CACHE_DIR"

fetch() {
  local dest="$1"
  local url="$2"
  if [[ -s "$dest" ]]; then
    echo "skip (cached): $(basename "$dest")"
    return 0
  fi
  echo "fetch: $url"
  if curl -fL --retry 2 --connect-timeout 30 --max-time 180 -o "${dest}.partial" "$url"; then
    mv "${dest}.partial" "$dest"
  else
    rm -f "${dest}.partial"
    echo "warn: failed $(basename "$dest")" >&2
  fi
}

# IC 3418 (VCC 1217): RA 12h29m43.9s Dec +11°24′17″
fetch "$CACHE_DIR/ic3418_grz.fits" \
  "https://www.legacysurvey.org/viewer/fits-cutout?ra=187.4329&dec=11.4047&layer=ls-dr10&pixscale=0.262&bands=grz&size=900"

# NGC 4438 / 4435 (Arp 120): RA 12h27m46s Dec +13°00′31″
fetch "$CACHE_DIR/ngc4438_grz.fits" \
  "https://www.legacysurvey.org/viewer/fits-cutout?ra=186.9417&dec=13.0086&layer=ls-dr10&pixscale=0.262&bands=grz&size=1200"

# JWST NIRCam HiPS at SMACS 0723 (Webb's First Deep Field).
HIPS="https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
SRA="110.834"
SDEC="-73.454"
for filt in F115W F150W F200W; do
  fetch "$CACHE_DIR/jwst_smacs_${filt}.fits" \
    "${HIPS}?hips=CDS/P/JWST/${filt}&width=800&height=800&fov=0.04&projection=TAN&coordsys=icrs&ra=${SRA}&dec=${SDEC}&format=fits"
done

# HST WFC3 UVIS OPAL Jupiter (program 16266, visit 09). FLT is ~40 MB;
# keep an 800px downsample of the full SCI so the disk and limb stay in frame.
JUP_STAMPS=(
  "jupiter_f395n.fits:ieaq09b9q"
  "jupiter_f502n.fits:ieaq09b8q"
  "jupiter_f631n.fits:ieaq09b7q"
)
need_jup=0
for spec in "${JUP_STAMPS[@]}"; do
  dest="$CACHE_DIR/${spec%%:*}"
  [[ -s "$dest" ]] || need_jup=1
done
if [[ "$need_jup" -eq 1 ]]; then
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/torchfits-jup.XXXXXX")"
  trap 'rm -rf "$tmp"' EXIT
  for spec in "${JUP_STAMPS[@]}"; do
    dest="$CACHE_DIR/${spec%%:*}"
    root="${spec##*:}"
    if [[ -s "$dest" ]]; then
      continue
    fi
    flt="$tmp/${root}_flt.fits"
    fetch "$flt" \
      "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/${root}_flt.fits"
  done
  (
    export TORCHFITS_JUP_TMP="$tmp" TORCHFITS_JUP_CACHE="$CACHE_DIR"
    cd "$ROOT" && pixi run python -c '
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchfits

tmp = Path(os.environ["TORCHFITS_JUP_TMP"])
cache = Path(os.environ["TORCHFITS_JUP_CACHE"])
size = 800
pairs = (
    ("ieaq09b9q", "jupiter_f395n.fits"),
    ("ieaq09b8q", "jupiter_f502n.fits"),
    ("ieaq09b7q", "jupiter_f631n.fits"),
)
for root, name in pairs:
    dest = cache / name
    if dest.is_file() and dest.stat().st_size > 0:
        continue
    src = tmp / f"{root}_flt.fits"
    if not src.is_file() or src.stat().st_size <= 0:
        print(f"warn: missing {src.name}", flush=True)
        continue
    tensor = torchfits.read_tensor(str(src), hdu=1).float()
    tensor = torch.nan_to_num(tensor, nan=0.0)
    scaled = F.interpolate(
        tensor.unsqueeze(0).unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    torchfits.write(str(dest), scaled, overwrite=True)
    print(f"downsample {src.name} -> {name} {size}x{size}", flush=True)
')
  rm -rf "$tmp"
  trap - EXIT
fi

(
  export TORCHFITS_RGB_CACHE="$CACHE_DIR"
  cd "$ROOT" && pixi run python -c '
import os
from pathlib import Path

import torch
import torchfits

cache = Path(os.environ["TORCHFITS_RGB_CACHE"])
for name in ("jwst_smacs_F115W.fits", "jwst_smacs_F150W.fits", "jwst_smacs_F200W.fits"):
    path = cache / name
    if not path.is_file() or path.stat().st_size <= 0:
        continue
    tensor = torchfits.read_tensor(str(path), hdu=0).float()
    if not torch.isfinite(tensor).any():
        path.unlink()
        print(f"warn: dropped empty HiPS {name}", flush=True)
')

echo "ready: cutouts under ${CACHE_DIR}"
