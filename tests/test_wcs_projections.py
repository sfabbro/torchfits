import torch
import torchfits
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
import pytest


def create_header(projection="TAN"):
    header = fits.Header()
    header["SIMPLE"] = "T"
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 100
    header["NAXIS2"] = 100
    header["CTYPE1"] = f"RA---{projection}"
    header["CTYPE2"] = f"DEC--{projection}"
    header["CRPIX1"] = 50.0
    header["CRPIX2"] = 50.0
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 45.0
    header["CDELT1"] = -0.01
    header["CDELT2"] = 0.01
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    return header


def run_projection_test(projection):
    print(f"\nTesting {projection} projection...")
    header = create_header(projection)

    # Create Astropy WCS
    awcs = AstropyWCS(header)

    # Create TorchFits WCS
    # We need to convert astropy header to dict for torchfits
    # Pass raw values, wcs.py handles formatting
    header_dict = dict(header)
    twcs = torchfits.WCS(header_dict)

    # Generate test pixels
    # Center, corners, and some random points
    pixels = torch.tensor(
        [
            [
                50.0,
                50.0,
            ],  # CRPIX (0-based in numpy/torch? No, FITS is 1-based, let's check convention)
            # Astropy WCS uses 0-based by default if initialized from header?
            # Actually Astropy WCS follows FITS 1-based convention but pixel_to_world takes 0-based or 1-based depending on origin argument.
            # Let's assume 0-based for input pixels to match typical python usage.
            [0.0, 0.0],
            [99.0, 99.0],
            [25.0, 75.0],
        ],
        dtype=torch.float64,
    )

    # Astropy calculation (0-based)
    # all_pix2world with origin=0 for 0-based indexing
    expected_world = awcs.all_pix2world(pixels.numpy(), 0)

    # TorchFits
    # Now uses 0-based indexing internally
    actual_world = twcs.pixel_to_world(pixels)

    print("TorchFits CRPIX:", twcs.crpix)
    print("TorchFits CRVAL:", twcs.crval)
    print("TorchFits CDELT:", twcs.cdelt)

    print("Pixels:", pixels)
    print("Expected World:", expected_world)
    print("Actual World:", actual_world)

    np.testing.assert_allclose(actual_world.numpy(), expected_world, atol=1e-5)
    print("Forward transform passed!")

    # Inverse transform
    actual_pixels = twcs.world_to_pixel(actual_world)
    print("Actual Pixels:", actual_pixels)

    np.testing.assert_allclose(actual_pixels.numpy(), pixels.numpy(), atol=1e-5)


@pytest.mark.parametrize("projection", ["TAN", "SIN"])
def test_wcs_projections(projection):
    run_projection_test(projection)
