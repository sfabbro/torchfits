import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from torchfits.wcs.core import WCS as TorchWCS


PROJECTIONS = ("TAN", "SIN", "ARC", "AIT", "MOL", "HPX", "CEA", "MER")


def _make_header(projection: str) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 256
    header["NAXIS2"] = 256
    header["CTYPE1"] = f"RA---{projection}"
    header["CTYPE2"] = f"DEC--{projection}"
    header["CRPIX1"] = 128.0
    header["CRPIX2"] = 128.0
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 0.0
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"

    if projection in {"AIT", "MOL", "HPX"}:
        header["CD1_1"] = -1.0
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 1.0
    elif projection in {"CEA", "MER"}:
        header["CD1_1"] = -0.5
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.5
        if projection == "CEA":
            header["PV2_1"] = 1.0
    else:
        header["CD1_1"] = -0.01
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.01

    return header


def _sample_pixels(header: fits.Header, n_side: int = 7) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(96.0, header["NAXIS1"] - 96.0, n_side, dtype=np.float64)
    y = np.linspace(96.0, header["NAXIS2"] - 96.0, n_side, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_wrap_delta(ra_a: np.ndarray, ra_b: np.ndarray) -> np.ndarray:
    return ((ra_a - ra_b + 180.0) % 360.0) - 180.0


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_forward_matches_astropy(projection: str, origin: int) -> None:
    header = _make_header(projection)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x, y = _sample_pixels(header)

    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x),
        torch.from_numpy(y),
        origin=origin,
    )

    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for projection={projection}"

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]

    np.testing.assert_allclose(dra, 0.0, atol=5e-9)
    np.testing.assert_allclose(ddec, 0.0, atol=5e-9)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_inverse_matches_astropy(projection: str, origin: int) -> None:
    header = _make_header(projection)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for projection={projection}"

    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]

    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid),
        torch.from_numpy(dec_valid),
        origin=origin,
    )

    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-6)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-6)

    np.testing.assert_allclose(x_t.cpu().numpy(), x[valid], atol=1e-6)
    np.testing.assert_allclose(y_t.cpu().numpy(), y[valid], atol=1e-6)
