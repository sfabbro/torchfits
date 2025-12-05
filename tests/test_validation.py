import torchfits
from astropy.io import fits
import os
import torch


def create_valid_fits(filename):
    hdu = fits.PrimaryHDU(data=torch.zeros(10, 10).numpy())
    hdu.writeto(filename, overwrite=True)


def test_validation():
    filename = "test_valid.fits"
    create_valid_fits(filename)

    try:
        hdul = torchfits.HDUList.fromfile(filename)
        is_valid = hdul.validate()
        assert is_valid

    finally:
        if os.path.exists(filename):
            os.remove(filename)
