import torch
import torchfits
import numpy as np

def test_wcs_spectral():
    # Create a 1D WCS header
    header = {
        'NAXIS': '1',
        'CTYPE1': 'WAVE',
        'CRPIX1': '1.0',
        'CRVAL1': '4000.0',
        'CDELT1': '10.0'
        # 'CUNIT1': 'Angstrom' # Removed to avoid unit conversion to meters
    }
    
    wcs = torchfits.WCS(header)
    assert wcs.naxis == 1
    
    # Test pixel_to_world
    # Pixel 0 (1-based index 1) -> 4000.0
    pixels = torch.tensor([[0.0], [1.0], [10.0]], dtype=torch.float64)
    world = wcs.pixel_to_world(pixels)

    
    assert torch.allclose(world[0], torch.tensor([4000.0], dtype=torch.float64))
    assert torch.allclose(world[1], torch.tensor([4010.0], dtype=torch.float64))
    assert torch.allclose(world[2], torch.tensor([4100.0], dtype=torch.float64))
    
    # Test world_to_pixel
    pixels_back = wcs.world_to_pixel(world)
    assert torch.allclose(pixels, pixels_back)

    
def test_wcs_3d():
    print("Testing 3D WCS (RA, DEC, WAVE)...")
    
    header = {
        'NAXIS': '3',
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'CTYPE3': 'WAVE',
        'CRPIX1': '10.0',
        'CRPIX2': '10.0',
        'CRPIX3': '1.0',
        'CRVAL1': '180.0',
        'CRVAL2': '0.0',
        'CRVAL3': '5000.0',
        'CDELT1': '-0.01',
        'CDELT2': '0.01',
        'CDELT3': '2.0',
    }
    
    wcs = torchfits.WCS(header)
    assert wcs.naxis == 3
    
    # Test pixel (9, 9, 0) -> (180, 0, 5000)
    # Note: 0-based index 9 is 1-based 10 (CRPIX)
    pixels = torch.tensor([[9.0, 9.0, 0.0]], dtype=torch.float64)
    world = wcs.pixel_to_world(pixels)
    
    print(f"Pixels: {pixels}")
    print(f"World: {world}")
    
    assert torch.allclose(world[0, 0], torch.tensor(180.0, dtype=torch.float64))
    assert torch.allclose(world[0, 1], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(world[0, 2], torch.tensor(5000.0, dtype=torch.float64))
    
    # Test world_to_pixel
    pixels_back = wcs.world_to_pixel(world)
    assert torch.allclose(pixels, pixels_back, atol=1e-5)
    
    assert torch.allclose(pixels, pixels_back, atol=1e-5)

