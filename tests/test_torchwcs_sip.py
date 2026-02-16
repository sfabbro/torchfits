
import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# MOCK c++ extension
if "torchfits.cpp" not in sys.modules:
    sys.modules["torchfits.cpp"] = MagicMock()

# Add source to path
if os.path.abspath("src") not in sys.path:
    sys.path.insert(0, os.path.abspath("src"))

from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
from torchfits.wcs.core import WCS as TorchWCS

def test_sip_distortion():
    """
    Validate SIP distortion against Astropy.
    We create a synthetic header with strong distortions.
    """
    # Create header with TAN-SIP
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 1000
    header['NAXIS2'] = 1000
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    header['CRPIX1'] = 500.0
    header['CRPIX2'] = 500.0
    header['CRVAL1'] = 180.0
    header['CRVAL2'] = 0.0
    header['CD1_1'] = -2.8e-4
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = 2.8e-4
    
    # SIP Coefficients (Forward)
    header['A_ORDER'] = 2
    header['B_ORDER'] = 2
    
    # Quadratic distortion in X (u^2 term)
    header['A_2_0'] = 5.0e-4 
    # Quadratic distortion in Y (v^2 term)
    header['B_0_2'] = 5.0e-4
    # Cross terms
    header['A_1_1'] = 1.0e-4
    header['B_1_1'] = -1.0e-4
    
    # Create WCS objects
    wcs_astropy = AstropyWCS(header)
    wcs_torch = TorchWCS(header)
    
    # Ensure TorchWCS detected SIP
    assert wcs_torch.sip is not None
    assert len(wcs_torch.sip.a_coeffs) > 0
    
    # Generate random pixels
    N = 100
    x = np.random.uniform(0, 1000, N)
    y = np.random.uniform(0, 1000, N)
    
    # Astropy Forward (all_pix2world applies SIP)
    ra_astro, dec_astro = wcs_astropy.all_pix2world(x, y, 0)
    
    # Torch Forward
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    ra_torch, dec_torch = wcs_torch.pixel_to_world(xt, yt)
    
    # Validate
    # SIP distortions can be large, ensure we capture them.
    # Without SIP, error would be large. With SIP, should be tiny.
    np.testing.assert_allclose(ra_torch.numpy(), ra_astro, rtol=1e-7, atol=1e-6, err_msg="RA Mismatch with SIP")
    np.testing.assert_allclose(dec_torch.numpy(), dec_astro, rtol=1e-7, atol=1e-6, err_msg="Dec Mismatch with SIP")
    
    print("\nSIP Forward: PASSED")
    
    # Test Inverse (if we implemented AP/BP, but let's test if we *didn't* implement inverse yet)
    # Astropy all_world2pix uses iterative solution if AP/BP missing.
    # header doesn't have AP/BP.
    
    # Let's add AP/BP coefficients (approximate inverse) to test undistort if we implemented it.
    # For now, just checking forward is a huge win.

if __name__ == "__main__":
    test_sip_distortion()
