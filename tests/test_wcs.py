import os
import tempfile

import numpy as np
import torch
from astropy.io import fits

import torchfits
from torchfits.wcs import WCS


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
    # torchfits uses Python-style 0-based pixel coordinates.
    pixels = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    world = wcs.pixel_to_world(pixels)
    assert torch.allclose(
        world, torch.tensor([[180.0, 0.0]], dtype=torch.float64), atol=1e-6
    )


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
    world = torch.tensor([[180.0, 0.0]], dtype=torch.float32)
    pixels = wcs.world_to_pixel(world)
    assert torch.allclose(
        pixels, torch.tensor([[0.0, 0.0]], dtype=torch.float64), atol=1e-6
    )


def test_get_wcs_from_file_auto_hdu():
    image = np.random.normal(size=(32, 32)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        hdu = fits.PrimaryHDU(image)
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
        hdu.header["CRPIX1"] = 1.0
        hdu.header["CRPIX2"] = 1.0
        hdu.header["CRVAL1"] = 180.0
        hdu.header["CRVAL2"] = 0.0
        hdu.header["CDELT1"] = -0.1
        hdu.header["CDELT2"] = 0.1
        hdu.writeto(f.name, overwrite=True)
        path = f.name

    try:
        wcs = torchfits.get_wcs(path, hdu="auto")
        assert isinstance(wcs, WCS)
        world = wcs.pixel_to_world(torch.tensor([[0.0, 0.0]], dtype=torch.float64))
        assert torch.allclose(
            world, torch.tensor([[180.0, 0.0]], dtype=torch.float64), atol=1e-6
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_get_wcs_auto_hdu_skips_empty_primary():
    image = np.random.normal(size=(16, 16)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        primary = fits.PrimaryHDU()
        ext = fits.ImageHDU(image)
        ext.header["EXTNAME"] = "SCI"
        ext.header["CTYPE1"] = "RA---TAN"
        ext.header["CTYPE2"] = "DEC--TAN"
        ext.header["CRPIX1"] = 1.0
        ext.header["CRPIX2"] = 1.0
        ext.header["CRVAL1"] = 10.0
        ext.header["CRVAL2"] = -5.0
        ext.header["CDELT1"] = -0.01
        ext.header["CDELT2"] = 0.01
        fits.HDUList([primary, ext]).writeto(f.name, overwrite=True)
        path = f.name

    try:
        wcs = torchfits.get_wcs(path, hdu="auto")
        world = wcs.pixel_to_world(torch.tensor([[0.0, 0.0]], dtype=torch.float64))
        assert torch.allclose(
            world, torch.tensor([[10.0, -5.0]], dtype=torch.float64), atol=1e-6
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)
