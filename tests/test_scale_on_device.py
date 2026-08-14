"""Scale-on-device reads through the live production path."""

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits


def test_read_int8_fits_matches_fitsio(tmp_path):
    fitsio = pytest.importorskip("fitsio")
    data = np.array([-128, 0, 127, 42], dtype=np.int8).reshape(2, 2)
    path = tmp_path / "int8.fits"
    fits.PrimaryHDU(data).writeto(path, overwrite=True)

    expected = fitsio.read(str(path))
    if torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = None

    if dev is not None:
        got = torchfits.read(str(path), device=dev, scale_on_device=True)
        assert got.dtype == torch.int8
        assert got.device.type == dev
        assert got.cpu().numpy().tolist() == expected.tolist()
    else:
        got = torchfits.read(str(path), scale_on_device=True)
        assert got.dtype == torch.int8
        assert got.numpy().tolist() == expected.tolist()


def test_read_uint16_fits_native_dtype_accelerator(tmp_path):
    if torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    else:
        pytest.skip("No GPU/MPS accelerator available")
    data = np.array([[0, 32768, 65535]], dtype=np.uint16)
    path = tmp_path / "uint16.fits"
    fits.PrimaryHDU(data).writeto(path, overwrite=True)

    got = torchfits.read(str(path), device=dev, scale_on_device=True)
    assert got.dtype == torch.uint16
    assert got.device.type == dev
    assert got.cpu().numpy().tolist() == data.tolist()
