from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

import torchfits


astropy = pytest.importorskip("astropy")
fitsio = pytest.importorskip("fitsio")
healpy = pytest.importorskip("healpy")


ASTROPY_ROOT = Path(astropy.__path__[0])
FITSIO_ROOT = Path(fitsio.__file__).resolve().parent
HEALPY_ROOT = Path(healpy.__file__).resolve().parent

SCALE_FIXTURE = ASTROPY_ROOT / "io" / "fits" / "tests" / "data" / "scale.fits"
VLA_FIXTURE = (
    ASTROPY_ROOT / "io" / "fits" / "tests" / "data" / "variable_length_table.fits"
)
SIP_WCS_FIXTURE = ASTROPY_ROOT / "nddata" / "tests" / "data" / "sip-wcs.fits"
GZIP_FIXTURE = FITSIO_ROOT / "test_images" / "test_gzip_compressed_image.fits.fz"
HEALPY_WEIGHT_FIXTURE = HEALPY_ROOT / "data" / "weight_ring_n00016.fits"


def _nested_numeric_max_abs(ref: list[object], got: list[object]) -> float:
    max_abs = 0.0
    for ref_row, got_row in zip(ref, got, strict=True):
        ref_arr = np.asarray(ref_row, dtype=np.float64)
        got_arr = np.asarray(got_row, dtype=np.float64)
        if ref_arr.size == 0 and got_arr.size == 0:
            continue
        max_abs = max(max_abs, float(np.max(np.abs(ref_arr - got_arr))))
    return max_abs


def test_local_scaled_image_fixture_matches_astropy_physical_values() -> None:
    image, header = torchfits.read(SCALE_FIXTURE.as_posix(), return_header=True)

    with fits.open(SCALE_FIXTURE) as hdul:
        expected_header = hdul[0].header.copy()
        expected = hdul[0].data

    np.testing.assert_allclose(image.cpu().numpy(), expected, atol=1e-4, rtol=0.0)
    assert float(header["BSCALE"]) == float(expected_header["BSCALE"])
    assert float(header["BZERO"]) == float(expected_header["BZERO"])


def test_local_variable_length_table_fixture_matches_astropy() -> None:
    table = torchfits.table.read(VLA_FIXTURE.as_posix(), hdu=1)

    with fits.open(VLA_FIXTURE) as hdul:
        expected = hdul[1].data
        expected_var = [list(row) for row in expected["var"].tolist()]
        expected_xyz = [list(row) for row in expected["xyz"].tolist()]

    assert table.column_names == ["var", "xyz"]
    got_var = table["var"].to_pylist()
    got_xyz = table["xyz"].to_pylist()
    assert _nested_numeric_max_abs(expected_var, got_var) == 0.0
    assert _nested_numeric_max_abs(expected_xyz, got_xyz) == 0.0


def test_local_gzip_compressed_image_matches_fitsio() -> None:
    image = torchfits.read(GZIP_FIXTURE.as_posix(), hdu=1)
    expected = fitsio.read(GZIP_FIXTURE.as_posix(), ext=1)
    np.testing.assert_allclose(image.cpu().numpy(), expected, atol=0.0, rtol=0.0)


def test_local_healpy_weight_table_matches_astropy_rows() -> None:
    table = torchfits.table.read(HEALPY_WEIGHT_FIXTURE.as_posix(), hdu=1)

    with fits.open(HEALPY_WEIGHT_FIXTURE) as hdul:
        expected = hdul[1].data

    assert table.column_names == list(expected.names)
    assert table.num_rows == len(expected)
    for name in expected.names:
        np.testing.assert_allclose(
            np.asarray(table[name].to_pylist(), dtype=np.float64),
            np.asarray(expected[name], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )


def test_local_sip_wcs_fixture_matches_astropy_world_coordinates() -> None:
    awcs = AstropyWCS(fits.getheader(SIP_WCS_FIXTURE))
    twcs = torchfits.get_wcs(SIP_WCS_FIXTURE.as_posix(), hdu=0)
    xy = np.array([[0.0, 0.0], [20.0, 10.0], [80.0, 40.0]], dtype=np.float64)

    ra_ast, dec_ast = awcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(xy[:, 0]), torch.from_numpy(xy[:, 1]), origin=0
    )

    np.testing.assert_allclose(ra_t.cpu().numpy(), ra_ast, atol=5e-6, rtol=0.0)
    np.testing.assert_allclose(dec_t.cpu().numpy(), dec_ast, atol=5e-6, rtol=0.0)
