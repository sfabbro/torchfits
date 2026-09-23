import torch
from astropy.io import fits

import torchfits


def test_read_header(tmp_path):
    filename = str(tmp_path / "test_read_header.fits")

    # Create file with multiple HDUs
    hdu0 = fits.PrimaryHDU(data=torch.zeros(10, 10).numpy())
    hdu0.header["HDU0"] = "Value0"

    hdu1 = fits.ImageHDU(data=torch.zeros(10, 10).numpy(), name="IMAGE1")
    hdu1.header["HDU1"] = "Value1"

    hdul = fits.HDUList([hdu0, hdu1])
    hdul.writeto(filename, overwrite=True)

    # Test read_header by index
    h0 = torchfits.read_header(filename, 0)
    assert h0["HDU0"] == "Value0"

    h1 = torchfits.read_header(filename, 1)
    assert h1["HDU1"] == "Value1"

    # Test read_header by name
    h_named = torchfits.read_header(filename, "IMAGE1")
    assert h_named["HDU1"] == "Value1"
