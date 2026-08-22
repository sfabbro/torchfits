"""Auto RGB collage: Virgo dwarf, merger, JWST SMACS 0723, HST Jupiter.

Same ``rgb()`` defaults on every panel. Prefers cached public FITS from
``scripts/fetch_rgb_sky_samples.sh``:

- Legacy Survey ``grz`` (nanomaggies → ``calibrated=True``)
- CDS JWST HiPS F115W/F150W/F200W at SMACS 0723
- HST WFC3 UVIS OPAL F395N/F502N/F631N Jupiter (full-frame SCI downsample)

Missing files fall back to synthetic stamps so the example still runs
under ``TORCHFITS_EXAMPLE_FAST=1``. Gallery PNGs are written only from
real cutouts. HSC-SSP is not fetched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import CACHE_DIR  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import torchfits  # noqa: E402
from torchfits.transforms import lupton_rgb, rgb  # noqa: E402
from torchfits.transforms.rgb import write_rgb_image  # noqa: E402

FETCH_CMD = "bash scripts/fetch_rgb_sky_samples.sh"
SKY_DIR = CACHE_DIR / "rgb_sky"
TILE = 360
GALLERY_DIR = ROOT / "docs" / "assets" / "gallery"
_READ_ERR = (OSError, RuntimeError, ValueError, IndexError)


def _fast_mode() -> bool:
    return os.environ.get("TORCHFITS_EXAMPLE_FAST", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
    )


def _resize(image: torch.Tensor, size: int) -> torch.Tensor:
    """Resize (H, W, 3) to size×size; area-filter when shrinking."""
    if int(image.shape[0]) == size and int(image.shape[1]) == size:
        return image
    work = image.permute(2, 0, 1).unsqueeze(0).float()
    shrinking = min(int(image.shape[0]), int(image.shape[1])) > size
    out = F.interpolate(
        work,
        size=(size, size),
        mode="area" if shrinking else "nearest",
    )
    return out.squeeze(0).permute(1, 2, 0)


def _read_2d(path: Path) -> torch.Tensor | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    for hdu in (0, 1):
        try:
            tensor = torchfits.read_tensor(str(path), hdu=hdu).float()
        except _READ_ERR:
            continue
        if tensor.ndim == 2 and tensor.numel() > 0 and torch.isfinite(tensor).any():
            return tensor
    return None


def _read_grz(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        tensor = torchfits.read_tensor(str(path), hdu=0).float()
    except _READ_ERR:
        return None
    if tensor.ndim == 3 and int(tensor.shape[0]) in (3, 4):
        return tensor[0], tensor[1], tensor[2]
    if tensor.ndim == 3 and int(tensor.shape[-1]) in (3, 4):
        return tensor[..., 0], tensor[..., 1], tensor[..., 2]
    if tensor.ndim != 2:
        return None
    try:
        r_band = torchfits.read_tensor(str(path), hdu=1).float()
        z_band = torchfits.read_tensor(str(path), hdu=2).float()
    except _READ_ERR:
        return None
    return tensor, r_band, z_band


def _read_band_set(paths: tuple[Path, ...]) -> tuple[torch.Tensor, ...] | None:
    bands: list[torch.Tensor] = []
    for path in paths:
        band = _read_2d(path)
        if band is None:
            return None
        bands.append(band)
    shape = tuple(bands[0].shape)
    if any(tuple(band.shape) != shape for band in bands):
        return None
    return tuple(bands)


def _noise(n: int, seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return scale * torch.randn(n, n, generator=gen)


def _mesh(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, n),
        torch.linspace(-1.0, 1.0, n),
        indexing="ij",
    )
    return yy, xx


def _synthetic_dwarf(n: int = TILE) -> tuple[torch.Tensor, ...]:
    yy, xx = _mesh(n)
    sky = 0.08 + _noise(n, 0, 0.015)
    rr = torch.sqrt((xx + 0.15) ** 2 + (yy - 0.05) ** 2)
    body = 1.6 * torch.exp(-rr / 0.08)
    tail = 0.12 * torch.exp(-((xx - 0.35) ** 2) / 0.18 - (yy - 0.2 * xx) ** 2 / 0.012)
    g = sky + 0.7 * body + 0.5 * tail
    r = sky + 0.9 * body + 0.45 * tail
    i = sky + 1.1 * body + 0.25 * tail
    z = sky + 1.0 * body + 0.25 * tail
    return g, r, i, z


def _synthetic_merger(n: int = TILE) -> tuple[torch.Tensor, ...]:
    yy, xx = _mesh(n)
    sky = 0.08 + _noise(n, 1, 0.015)
    a = torch.exp(-((xx + 0.22) ** 2 + (yy + 0.08) ** 2) / 0.012)
    b = 0.6 * torch.exp(-((xx - 0.28) ** 2 + (yy - 0.18) ** 2) / 0.01)
    arc = 0.25 * torch.exp(-((torch.sqrt(xx * xx + yy * yy) - 0.55) ** 2) / 0.008)
    g = sky + 4.0 * a + 3.0 * b + arc
    r = sky + 5.0 * a + 3.5 * b + 0.9 * arc
    i = sky + 6.0 * a + 4.0 * b + 0.8 * arc
    z = sky + 5.5 * a + 3.8 * b + 0.7 * arc
    return g, r, i, z


def _synthetic_jwst(n: int = TILE) -> tuple[torch.Tensor, ...]:
    yy, xx = _mesh(n)
    sky = 0.05 + _noise(n, 2, 0.01)
    cores = torch.zeros(n, n)
    for cx, cy, amp in ((-0.3, -0.2, 8.0), (0.15, 0.1, 6.0), (0.4, -0.15, 5.0)):
        cores = cores + amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.006)
    tidal = 0.4 * torch.exp(-((yy - 0.4 * xx) ** 2) / 0.02)
    f150 = sky + 0.6 * cores + tidal
    f200 = sky + 0.9 * cores + 0.7 * tidal
    f444 = sky + 1.1 * cores + 0.5 * tidal
    return f150, f200, f444


def _synthetic_jupiter(n: int = TILE) -> tuple[torch.Tensor, ...]:
    yy, xx = _mesh(n)
    disk = (xx * xx + yy * yy) < 0.72
    bands = 0.5 + 0.35 * torch.sin(yy * 14.0)
    r = torch.where(disk, 40.0 * (bands + 0.4), 1.0)
    g = torch.where(disk, 38.0 * (bands + 0.25), 1.0)
    b = torch.where(disk, 32.0 * (0.7 - 0.2 * bands), 1.0)
    return b, g, r


def _assert_visible(image: torch.Tensor, label: str) -> None:
    luma = (image.clamp(0, 1) * 255).mean(dim=-1)
    if float(luma.mean()) < 8.0 or float(torch.quantile(luma, 0.90)) < 20.0:
        raise RuntimeError(f"{label} looks near-black — refuse to write gallery asset")


_GUTTER = 12


def _grid(tiles: list[torch.Tensor], columns: int) -> torch.Tensor:
    """Assemble tiles on a near-black mount with an even outer frame."""
    rows = (len(tiles) + columns - 1) // columns
    height = rows * TILE + (rows + 1) * _GUTTER
    width = columns * TILE + (columns + 1) * _GUTTER
    canvas = torch.full((height, width, 3), 0.02)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        y = _GUTTER + row * (TILE + _GUTTER)
        x = _GUTTER + col * (TILE + _GUTTER)
        canvas[y : y + TILE, x : x + TILE] = tile
    return canvas


def _collage(panels: list[torch.Tensor]) -> torch.Tensor:
    tiles = [_resize(p.float(), TILE) for p in panels]
    return _grid(tiles, columns=2)


def _strip(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _grid([_resize(left.float(), TILE), _resize(right.float(), TILE)], columns=2)


def main() -> int:
    dwarf_path = SKY_DIR / "ic3418_grz.fits"
    merger_path = SKY_DIR / "ngc4438_grz.fits"
    jwst_paths = (
        SKY_DIR / "jwst_smacs_F115W.fits",
        SKY_DIR / "jwst_smacs_F150W.fits",
        SKY_DIR / "jwst_smacs_F200W.fits",
    )
    jup_paths = (
        SKY_DIR / "jupiter_f395n.fits",
        SKY_DIR / "jupiter_f502n.fits",
        SKY_DIR / "jupiter_f631n.fits",
    )
    if not _fast_mode():
        print(f"cutouts: {SKY_DIR} (fetch via: {FETCH_CMD})")

    dwarf_grz = _read_grz(dwarf_path)
    merger_grz = _read_grz(merger_path)
    jwst_bands = _read_band_set(jwst_paths)
    jup_bands = _read_band_set(jup_paths)
    all_real = all(
        item is not None for item in (dwarf_grz, merger_grz, jwst_bands, jup_bands)
    )

    if dwarf_grz is None:
        print("dwarf: synthetic (no ic3418_grz.fits)")
        dg, dr, di, dz = _synthetic_dwarf()
        dwarf = rgb(dg, dr, di, dz)
        lupton = lupton_rgb(r=di, g=dr, b=dg, Q=8.0, stretch=0.5)
    else:
        print("dwarf: Legacy Survey grz (calibrated)")
        dg, dr, dz = dwarf_grz
        dwarf = rgb(dg, dr, dz, calibrated=True)
        lupton = lupton_rgb(r=dz, g=dr, b=dg, Q=8.0, stretch=0.5)

    if merger_grz is None:
        print("merger: synthetic (no ngc4438_grz.fits)")
        merger = rgb(*_synthetic_merger())
    else:
        print("merger: Legacy Survey grz (calibrated)")
        mg, mr, mz = merger_grz
        merger = rgb(mg, mr, mz, calibrated=True)

    if jwst_bands is None:
        print("jwst: synthetic (no SMACS F115W/F150W/F200W)")
        jwst = rgb(*_synthetic_jwst())
    else:
        print("jwst: CDS HiPS SMACS 0723 F115W/F150W/F200W")
        jwst = rgb(*jwst_bands)

    if jup_bands is None:
        print("jupiter: synthetic (no OPAL F395N/F502N/F631N)")
        jupiter = rgb(*_synthetic_jupiter())
    else:
        print("jupiter: HST OPAL WFC3 F395N/F502N/F631N (full disk)")
        jupiter = rgb(*jup_bands)

    collage = _collage([dwarf, merger, jwst, jupiter])
    strip = _strip(dwarf, lupton)
    _assert_visible(collage, "rgb_sky_collage")
    _assert_visible(strip, "rgb_vs_lupton_dwarf")

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    collage_path = out_dir / "rgb_sky_collage.png"
    strip_path = out_dir / "rgb_vs_lupton_dwarf.png"
    write_rgb_image(str(collage_path), collage)
    write_rgb_image(str(strip_path), strip)
    if not _fast_mode() and all_real:
        GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        write_rgb_image(str(GALLERY_DIR / "rgb_sky_collage.png"), collage)
        write_rgb_image(str(GALLERY_DIR / "rgb_vs_lupton_dwarf.png"), strip)
    elif not _fast_mode():
        print(f"skip gallery: missing real cutouts; run {FETCH_CMD}")
    print(f"wrote {collage_path}")
    print(f"wrote {strip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
