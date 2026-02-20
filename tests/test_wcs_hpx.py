import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from torchfits.wcs.core import WCS


def _header() -> fits.Header:
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
    header["CTYPE1"] = "RA---HPX"
    header["CTYPE2"] = "DEC--HPX"
    return header


def _ra_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def test_hpx_forward_matches_astropy_across_equatorial_and_polar_samples() -> None:
    header = _header()
    awcs = AstropyWCS(header)
    twcs = WCS(dict(header))

    x = np.array([150.0, 165.0, 180.0, 195.0, 210.0], dtype=np.float64)
    y = np.array([65.0, 80.0, 90.0, 100.0, 115.0], dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    px = xx.ravel()
    py = yy.ravel()

    ra_ast, dec_ast = awcs.all_pix2world(px, py, 0)
    ra_t, dec_t = twcs.pixel_to_world(torch.from_numpy(px), torch.from_numpy(py), origin=0)

    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid)

    np.testing.assert_allclose(_ra_delta(ra_t_np[valid], ra_ast[valid]), 0.0, atol=1e-8)
    np.testing.assert_allclose(dec_t_np[valid] - dec_ast[valid], 0.0, atol=1e-8)


def test_hpx_inverse_matches_astropy() -> None:
    header = _header()
    awcs = AstropyWCS(header)
    twcs = WCS(dict(header))

    x = np.linspace(150.0, 210.0, 7, dtype=np.float64)
    y = np.linspace(70.0, 110.0, 7, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    px = xx.ravel()
    py = yy.ravel()

    ra, dec = awcs.all_pix2world(px, py, 0)
    valid = np.isfinite(ra) & np.isfinite(dec)

    px_ast, py_ast = awcs.all_world2pix(ra[valid], dec[valid], 0)
    px_t, py_t = twcs.world_to_pixel(
        torch.from_numpy(ra[valid]),
        torch.from_numpy(dec[valid]),
        origin=0,
    )

    np.testing.assert_allclose(px_t.cpu().numpy(), px_ast, atol=1e-6)
    np.testing.assert_allclose(py_t.cpu().numpy(), py_ast, atol=1e-6)
