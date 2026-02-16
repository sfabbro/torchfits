
import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# MOCK c++ extension to avoid build dependency for this pure-python test
sys.modules["torchfits.cpp"] = MagicMock()

# Add source to path to pick up new module
sys.path.insert(0, os.path.abspath("src"))

from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
from torchfits.wcs.core import WCS as TorchWCS

# Assume run from project root, path to benchmark file
BENCH_FILE = "bench_20k_float32.fits"

def test_wcs_tan_projection():
    """
    Validate TorchWCS TAN projection against Astropy implementation.
    """
    try:
        hdul = fits.open(BENCH_FILE)
        header = hdul[0].header
    except FileNotFoundError:
        # Create a dummy header if file not found
        header = fits.Header()
        header['NAXIS'] = 2
        header['NAXIS1'] = 2048
        header['NAXIS2'] = 2048
    
    # Ensure WCS is present (inject if missing)
    if 'CTYPE1' not in header or 'TAN' not in header['CTYPE1']:
        print("Injecting synthetic TAN WCS into header for testing...")
        header['CTYPE1'] = 'RA---TAN'
        header['CTYPE2'] = 'DEC--TAN'
        header['CRPIX1'] = header['NAXIS1'] / 2.0
        header['CRPIX2'] = header['NAXIS2'] / 2.0
        header['CRVAL1'] = 180.0
        header['CRVAL2'] = 45.0
        header['CD1_1'] = -0.00027777 # -1 arcsec/pix
        header['CD1_2'] = 0.0
        header['CD2_1'] = 0.0
        header['CD2_2'] = 0.00027777 # 1 arcsec/pix

    # Create WCS objects
    wcs_astropy = AstropyWCS(header)
    wcs_torch = TorchWCS(header)
    
    # 1. Test Pixel -> World
    # Generate random pixel coordinates
    H, W = header['NAXIS2'], header['NAXIS1']
    N_samples = 100
    
    # Random pixels within image bounds
    x_pix = np.random.uniform(0, W, N_samples)
    y_pix = np.random.uniform(0, H, N_samples)
    
    # Astropy Ground Truth
    ra_astro, dec_astro = wcs_astropy.all_pix2world(x_pix, y_pix, 0) # 0-based
    
    # Torch Prediction
    x_tensor = torch.tensor(x_pix, dtype=torch.float64)
    y_tensor = torch.tensor(y_pix, dtype=torch.float64)
    
    ra_torch, dec_torch = wcs_torch.pixel_to_world(x_tensor, y_tensor)
    
    # Validate
    # High precision check (float64)
    np.testing.assert_allclose(ra_torch.numpy(), ra_astro, rtol=1e-7, atol=1e-5, err_msg="RA mismatch")
    np.testing.assert_allclose(dec_torch.numpy(), dec_astro, rtol=1e-7, atol=1e-5, err_msg="Dec mismatch")
    
    print("\nPixel -> World: PASSED")
    
    # 2. Test World -> Pixel
    # Use the sky coordinates we just generated
    ra_in = ra_astro
    dec_in = dec_astro
    
    # Astropy Ground Truth (Back to pixels)
    x_astro_back, y_astro_back = wcs_astropy.all_world2pix(ra_in, dec_in, 0)
    
    # Torch Prediction
    ra_tensor = torch.tensor(ra_in, dtype=torch.float64)
    dec_tensor = torch.tensor(dec_in, dtype=torch.float64)
    
    x_torch_back, y_torch_back = wcs_torch.world_to_pixel(ra_tensor, dec_tensor)
    
    # Validate
    np.testing.assert_allclose(x_torch_back.numpy(), x_astro_back, rtol=1e-5, atol=1e-3, err_msg="X pixel mismatch")
    np.testing.assert_allclose(y_torch_back.numpy(), y_astro_back, rtol=1e-5, atol=1e-3, err_msg="Y pixel mismatch")
    
    print("World -> Pixel: PASSED")

if __name__ == "__main__":
    test_wcs_tan_projection()
