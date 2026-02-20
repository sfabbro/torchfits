import numpy as np
import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

import torchfits


def _wave_1d_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 1,
        "CTYPE1": "WAVE",
        "CRPIX1": 1.0,
        "CRVAL1": 4000.0,
        "CDELT1": 10.0,
    }


def _cube_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 3,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRPIX3": 1.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CRVAL3": 5000.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
        "CDELT3": 2.0,
    }


@pytest.mark.parametrize("origin", [0, 1])
def test_wcs_spectral_1d_matches_astropy(origin: int) -> None:
    header = _wave_1d_header()

    twcs = torchfits.WCS(header)
    awcs = AstropyWCS(header)

    pixels = np.array([0.0, 1.0, 10.0, 25.0], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels[:, None]), origin=origin)
    world_a = awcs.all_pix2world(pixels[:, None], origin)

    np.testing.assert_allclose(world_t.cpu().numpy(), world_a, atol=1e-12)

    pixels_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=origin)
    pixels_a = awcs.all_world2pix(world_a, origin)

    np.testing.assert_allclose(pixels_t.cpu().numpy(), pixels_a, atol=1e-9)
    np.testing.assert_allclose(pixels_t.cpu().numpy(), pixels[:, None], atol=1e-9)


@pytest.mark.parametrize("origin", [0, 1])
def test_wcs_spatial_plus_spectral_3d_matches_astropy(origin: int) -> None:
    header = _cube_header()

    twcs = torchfits.WCS(header)
    awcs = AstropyWCS(header)

    # Keep points around CRPIX for robust inversion across projections.
    x = np.linspace(20.0, 44.0, 7, dtype=np.float64)
    y = np.linspace(20.0, 44.0, 7, dtype=np.float64)
    z = np.array([0.0, 3.0, 11.0], dtype=np.float64)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    pixels = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=origin)
    world_a = awcs.all_pix2world(pixels, origin)

    np.testing.assert_allclose(world_t.cpu().numpy(), world_a, atol=1e-8)

    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=origin)
    pix_a = awcs.all_world2pix(world_a, origin)

    np.testing.assert_allclose(pix_t.cpu().numpy(), pix_a, atol=1e-6)
    np.testing.assert_allclose(pix_t.cpu().numpy(), pixels, atol=1e-6)
