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

# JWST NIRCam HiPS at Stephan's Quintet (F115W has no coverage here).
HIPS="https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
QRA="338.99754"
QDEC="33.96049"
for filt in F150W F200W F444W; do
  fetch "$CACHE_DIR/jwst_quintet_${filt}.fits" \
    "${HIPS}?hips=CDS/P/JWST/${filt}&width=800&height=800&fov=0.06&projection=TAN&coordsys=icrs&ra=${QRA}&dec=${QDEC}&format=fits"
done

# HST WFC3 UVIS OPAL Jupiter (program 16266, visit 09). FLT is ~40 MB;
# keep a 720px flux-centroid crop of SCI.
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
  TORCHFITS_JUP_TMP="$tmp" TORCHFITS_JUP_CACHE="$CACHE_DIR" \
    (cd "$ROOT" && pixi run python -c '
import os
from pathlib import Path

import torch
import torchfits

tmp = Path(os.environ["TORCHFITS_JUP_TMP"])
cache = Path(os.environ["TORCHFITS_JUP_CACHE"])
size = 720
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
    weight = torch.clamp(torch.nan_to_num(tensor), min=0.0)
    height, width = int(tensor.shape[0]), int(tensor.shape[1])
    yy = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    xx = torch.arange(width, dtype=torch.float32).unsqueeze(0)
    mass = float(weight.sum())
    if mass <= 0.0:
        print(f"warn: empty SCI in {src.name}", flush=True)
        continue
    cy = int(round(float((weight * yy).sum() / mass)))
    cx = int(round(float((weight * xx).sum() / mass)))
    y0 = max(0, min(height - size, cy - size // 2))
    x0 = max(0, min(width - size, cx - size // 2))
    torchfits.write(str(dest), tensor[y0 : y0 + size, x0 : x0 + size], overwrite=True)
    print(f"crop {src.name} -> {name} {size}x{size} @ ({y0},{x0})", flush=True)
')
  rm -rf "$tmp"
  trap - EXIT
fi

TORCHFITS_RGB_CACHE="$CACHE_DIR" \
  (cd "$ROOT" && pixi run python -c '
import os
from pathlib import Path

import torch
import torchfits

cache = Path(os.environ["TORCHFITS_RGB_CACHE"])
for name in ("jwst_quintet_F150W.fits", "jwst_quintet_F200W.fits", "jwst_quintet_F444W.fits"):
    path = cache / name
    if not path.is_file() or path.stat().st_size <= 0:
        continue
    tensor = torchfits.read_tensor(str(path), hdu=0).float()
    if not torch.isfinite(tensor).any():
        path.unlink()
        print(f"warn: dropped empty HiPS {name}", flush=True)
')

echo "ready: cutouts under ${CACHE_DIR}"
