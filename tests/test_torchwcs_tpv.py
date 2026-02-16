
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

from torchfits.wcs.core import WCS

def test_tpv_internal_logic():
    """
    Verify TPV polynomial math is applied correctly.
    We check specific terms in isolation.
    """
    # Create TPV Header
    header = {
        'NAXIS': 2, 'NAXIS1': 100, 'NAXIS2': 100,
        'CRPIX1': 50.0, 'CRPIX2': 50.0,
        'CRVAL1': 0.0, 'CRVAL2': 0.0,
        'CTYPE1': 'RA---TPV', 'CTYPE2': 'DEC--TPV',
        'CD1_1': 1.0, 'CD1_2': 0.0, # Should be ignored by TPV
        'CD2_1': 0.0, 'CD2_2': 1.0
    }
    
    # 1. Linear X term (PV1_1)
    # Expected: xi = PV1_1 * x
    header['PV1_1'] = 0.5
    wcs = WCS(header)
    
    # Input x=51. rel_x = 51 - (50-1) = 2.
    # Expected xi = 0.5 * 2 = 1.0 deg.
    # At 1.0 deg, tan(1) ~ 1. (Error < 1e-4)
    x = torch.tensor([51.0])
    y = torch.tensor([49.0]) # y=49 -> rel_y=0
    ra, dec = wcs.pixel_to_world(x, y)
    
    # RA ~ xi = 1.0
    # Expected precise: degrees(atan(radians(1.0)))
    val = np.degrees(np.arctan(np.radians(1.0)))
    target = torch.tensor(val, dtype=ra.dtype)
    assert torch.allclose(ra, target, atol=1e-5)
    
    # 2. Quadratic term (PV1_4 = x^2)
    # Expected: xi = PV1_4 * x^2
    header['PV1_1'] = 0.0 # Zero out linear
    header['PV1_4'] = 0.01 
    wcs = WCS(header)
    
    # Input x=51. rel_x=2. x^2=4.
    # Expected xi = 0.01 * 4 = 0.04 deg.
    ra, dec = wcs.pixel_to_world(x, y)
    target = torch.tensor(0.04, dtype=ra.dtype)
    assert torch.allclose(ra, target, atol=1e-5)
    
    # 3. Radial term (PV1_3 = r)
    header['PV1_4'] = 0.0
    header['PV1_3'] = 0.1
    wcs = WCS(header)
    
    # Input x=51, y=51. rel_x=2, rel_y=2.
    # r = sqrt(4+4) = 2.8284
    # Expected xi = 0.1 * 2.8284 = 0.28284 deg.
    y_off = torch.tensor([51.0])
    ra, dec = wcs.pixel_to_world(x, y_off)
    target = torch.tensor(0.28284, dtype=ra.dtype)
    # Radial distortion can be slightly off due to float32 precision
    assert torch.allclose(ra, target, atol=1e-4)
    
    print("\nTPV Internal Math Check: PASSED")

if __name__ == "__main__":
    test_tpv_internal_logic()
