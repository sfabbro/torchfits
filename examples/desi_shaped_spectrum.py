"""Tiny DESI-shaped multi-arm spectrum Dataset demo (synthetic MEF).

Mirrors the DESI spectra HDU naming (B/R/Z FLUX+IVAR) without downloading
DR1. Real layout notes:
https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_REDUX/SPECPROD/healpix/SURVEY/PROGRAM/PIXGROUP/PIXNUM/spectra-SURVEY-PROGRAM-PIXNUM.html
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from torchfits.data import FitsSpectrumDataset


def _write_desi_shaped(path: Path) -> None:
    hdus = [fits.PrimaryHDU()]
    for name, nwave in (
        ("B_FLUX", 20),
        ("B_IVAR", 20),
        ("R_FLUX", 24),
        ("R_IVAR", 24),
        ("Z_FLUX", 18),
        ("Z_IVAR", 18),
    ):
        data = np.arange(4 * nwave, dtype=np.float32).reshape(4, nwave)
        hdus.append(fits.ImageHDU(data, name=name))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="torchfits_desi_") as tmp:
        _run(Path(tmp) / "desi_shaped.fits")


def _run(path: Path) -> None:
    _write_desi_shaped(path)
    ds = FitsSpectrumDataset(
        [str(path)],
        hdu=["B_FLUX", "R_FLUX", "Z_FLUX"],
        ivar_hdu=["B_IVAR", "R_IVAR", "Z_IVAR"],
        row=2,
        layout="dict",
    )
    arms = ds[0]
    for name, arm in arms.items():
        print(
            f"{name}: flux={tuple(arm['flux'].shape)} ivar={tuple(arm['ivar'].shape)}"
        )
    stitched = FitsSpectrumDataset(
        [str(path)],
        hdu=["B_FLUX", "R_FLUX", "Z_FLUX"],
        ivar_hdu=["B_IVAR", "R_IVAR", "Z_IVAR"],
        row=2,
        layout="concat",
    )[0]
    print(f"concat flux={tuple(stitched['flux'].shape)}")


if __name__ == "__main__":
    main()
