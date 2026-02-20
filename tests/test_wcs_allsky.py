import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from torchfits.wcs.core import WCS


ALLSKY = ("AIT", "MOL", "HPX")


def _header(proj: str) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 360
    header["NAXIS2"] = 180
    header["CRPIX1"] = 180.0
    header["CRPIX2"] = 90.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CD1_1"] = -1.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0
    header["CTYPE1"] = f"RA---{proj}"
    header["CTYPE2"] = f"DEC--{proj}"
    return header


def _grid() -> tuple[np.ndarray, np.ndarray]:
    # Interior domain where all three all-sky projections are stable in both libs.
    x = np.linspace(150.0, 210.0, 9, dtype=np.float64)
    y = np.linspace(65.0, 115.0, 9, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


@pytest.mark.parametrize("proj", ALLSKY)
def test_allsky_forward_matches_astropy(proj: str) -> None:
    header = _header(proj)
    awcs = AstropyWCS(header)
    twcs = WCS(dict(header))

    x, y = _grid()

    ra_ast, dec_ast = awcs.all_pix2world(x, y, 0)
    ra_t, dec_t = twcs.pixel_to_world(torch.from_numpy(x), torch.from_numpy(y), origin=0)

    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite overlap for {proj}"

    np.testing.assert_allclose(_ra_delta(ra_t_np[valid], ra_ast[valid]), 0.0, atol=1e-8)
    np.testing.assert_allclose(dec_t_np[valid] - dec_ast[valid], 0.0, atol=1e-8)


@pytest.mark.parametrize("proj", ALLSKY)
def test_allsky_inverse_matches_astropy_interior(proj: str) -> None:
    header = _header(proj)
    awcs = AstropyWCS(header)
    twcs = WCS(dict(header))

    x, y = _grid()
    ra_ast, dec_ast = awcs.all_pix2world(x, y, 0)

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]

    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, 0)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid),
        torch.from_numpy(dec_valid),
        origin=0,
    )

    x_t_np = x_t.cpu().numpy()
    y_t_np = y_t.cpu().numpy()
    finite = np.isfinite(x_ast) & np.isfinite(y_ast) & np.isfinite(x_t_np) & np.isfinite(y_t_np)
    assert np.any(finite), f"No finite inverse overlap for {proj}"

    np.testing.assert_allclose(x_t_np[finite], x_ast[finite], atol=1e-6)
    np.testing.assert_allclose(y_t_np[finite], y_ast[finite], atol=1e-6)
