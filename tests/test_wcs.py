import pytest
import torch

from torchfits.wcs import WCS

pytestmark = pytest.mark.skip(reason="WCS not implemented yet")


def test_wcs_2d_creation():
    header = {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 1024,
        "CRPIX2": 1024,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.1,
        "CDELT2": 0.1,
        "NAXIS1": 2048,
        "NAXIS2": 2048,
    }
    wcs = WCS(header)
    assert wcs.naxis == 2


def test_wcs_2d_pixel_to_world():
    header = {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 1,
        "CRPIX2": 1,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.1,
        "CDELT2": 0.1,
        "NAXIS1": 2,
        "NAXIS2": 2,
    }
    wcs = WCS(header)
    pixels = torch.tensor([[1.0, 1.0]])
    world = wcs.pixel_to_world(pixels)
    assert torch.allclose(world, torch.tensor([[180.0, 0.0]]), atol=1e-6)


def test_wcs_2d_world_to_pixel():
    header = {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 1,
        "CRPIX2": 1,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.1,
        "CDELT2": 0.1,
        "NAXIS1": 2,
        "NAXIS2": 2,
    }
    wcs = WCS(header)
    world = torch.tensor([[180.0, 0.0]])
    pixels = wcs.world_to_pixel(world)
    assert torch.allclose(pixels, torch.tensor([[1.0, 1.0]]), atol=1e-6)
