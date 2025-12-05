import torchfits
from astropy.io import fits
import os
import torch

def test_get_header():
    filename = "test_get_header.fits"
    
    # Create file with multiple HDUs
    hdu0 = fits.PrimaryHDU(data=torch.zeros(10, 10).numpy())
    hdu0.header['HDU0'] = 'Value0'
    
    hdu1 = fits.ImageHDU(data=torch.zeros(10, 10).numpy(), name='IMAGE1')
    hdu1.header['HDU1'] = 'Value1'
    
    hdul = fits.HDUList([hdu0, hdu1])
    hdul.writeto(filename, overwrite=True)
    
    try:
        # Test get_header by index
        h0 = torchfits.get_header(filename, 0)
        assert h0['HDU0'] == 'Value0'
        
        h1 = torchfits.get_header(filename, 1)
        assert h1['HDU1'] == 'Value1'
        
        # Test get_header by name
        h_named = torchfits.get_header(filename, 'IMAGE1')
        assert h_named['HDU1'] == 'Value1'
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)
