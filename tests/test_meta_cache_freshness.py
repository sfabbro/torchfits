"""HDU metadata lookups: honest errors + cache freshness on replacement (r4c-08/09/13)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

import torchfits
from torchfits._io_engine import hdu_api


def test_named_hdu_on_missing_file_raises_io_error(tmp_path):
    """A missing file must not be reported as "HDU not found" (r4c-08).

    The EXTNAME scan used to swallow the file-level open failure, probe up to
    1024 phantom HDUs, and end with a misleading ValueError.
    """
    missing = str(tmp_path / "no_such.fits")
    with pytest.raises((OSError, RuntimeError)):
        torchfits.read_header(missing, hdu="EVENTS")


def test_named_hdu_on_existing_file_still_resolves(tmp_path):
    path = str(tmp_path / "named.fits")
    img = fits.ImageHDU(np.zeros((2, 2), dtype=np.float32), name="SCI")
    fits.HDUList([fits.PrimaryHDU(), img]).writeto(path, overwrite=True)
    hdr = torchfits.read_header(path, hdu="SCI")
    assert hdr["NAXIS1"] == 2


def test_missing_name_on_existing_file_still_raises_value_error(tmp_path):
    path = str(tmp_path / "named.fits")
    img = fits.ImageHDU(np.zeros((2, 2), dtype=np.float32), name="SCI")
    fits.HDUList([fits.PrimaryHDU(), img]).writeto(path, overwrite=True)
    with pytest.raises(ValueError, match="not found"):
        torchfits.read_header(path, hdu="NOPE")


def test_meta_and_data_caches_rotate_on_file_replacement(tmp_path):
    """Replacing a file must invalidate every path-keyed cache (r4c-13)."""
    path = str(tmp_path / "rot.fits")

    def replace_with(primary_data: np.ndarray) -> None:
        tmp = str(tmp_path / "rot_next.fits")
        fits.HDUList([fits.PrimaryHDU(primary_data)]).writeto(tmp, overwrite=True)
        os.replace(tmp, path)

    replace_with(np.full((2, 2), 1.0, dtype=np.float32))
    assert torchfits.read(path).shape == (2, 2)
    assert torchfits.read_header(path)["NAXIS1"] == 2

    replace_with(np.full((3, 3), 2.0, dtype=np.float32))
    out = torchfits.read(path)
    assert out.shape == (3, 3), "stale data cache after file replacement"
    assert float(out[0, 0]) == 2.0
    assert torchfits.read_header(path)["NAXIS1"] == 3, (
        "stale header cache after file replacement"
    )

    # A payload-extending replacement must also rotate the autodetect answer.
    tmp = str(tmp_path / "rot_mef.fits")
    img = fits.ImageHDU(np.zeros((4, 4), dtype=np.float32))
    fits.HDUList([fits.PrimaryHDU(), img]).writeto(tmp, overwrite=True)
    os.replace(tmp, path)
    assert hdu_api.autodetect_hdu(path) == 1, (
        "stale autodetect cache after file replacement"
    )
    assert torchfits.read(path, hdu="auto").shape == (4, 4)


def test_autodetect_negative_result_stays_fresh(tmp_path):
    """The cached no-payload answer must rotate when a payload appears (r4c-09)."""
    path = str(tmp_path / "empty.fits")
    fits.HDUList([fits.PrimaryHDU()]).writeto(path, overwrite=True)
    assert hdu_api.autodetect_hdu(path) == 0
    assert hdu_api.autodetect_hdu(path) == 0  # repeat: must stay 0

    tmp = str(tmp_path / "empty_next.fits")
    img = fits.ImageHDU(np.zeros((4, 4), dtype=np.float32))
    fits.HDUList([fits.PrimaryHDU(), img]).writeto(tmp, overwrite=True)
    os.replace(tmp, path)
    assert hdu_api.autodetect_hdu(path) == 1
