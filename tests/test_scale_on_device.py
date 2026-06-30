"""Tests for narrow-dtype scale-on-device helpers."""

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits
from torchfits._io_engine.read_dispatch import (
    _apply_scale_on_device,
    _apply_unsigned_offset,
)


def test_apply_unsigned_offset_uint16_cpu():
    raw = torch.tensor([-1, 0, 1], dtype=torch.int16)
    out = _apply_unsigned_offset(raw, torch.uint16, 32768)
    assert out.dtype == torch.uint16
    assert out.tolist() == [32767, 32768, 32769]


def test_apply_unsigned_offset_uint16_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    raw = torch.tensor([-1, 0, 1], dtype=torch.int16)
    out = _apply_unsigned_offset(raw, torch.uint16, 32768, device="cuda")
    assert out.device.type == "cuda"
    assert out.dtype == torch.uint16
    assert out.cpu().tolist() == [32767, 32768, 32769]


def test_apply_scale_on_device_signed_byte():
    raw = torch.tensor([0, 128, 255], dtype=torch.uint8)
    out = _apply_scale_on_device(
        raw,
        scaled=True,
        bscale=1.0,
        bzero=-128.0,
        device="cpu",
    )
    assert out.dtype == torch.int8
    assert out.tolist() == [-128, 0, 127]


def test_apply_scale_on_device_signed_byte_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    raw = torch.tensor([0, 128, 255], dtype=torch.uint8)
    out = _apply_scale_on_device(
        raw,
        scaled=True,
        bscale=1.0,
        bzero=-128.0,
        device="cuda",
    )
    assert out.device.type == "cuda"
    assert out.dtype == torch.int8
    assert out.cpu().tolist() == [-128, 0, 127]


def test_read_int8_fits_matches_fitsio(tmp_path):
    fitsio = pytest.importorskip("fitsio")
    data = np.array([-128, 0, 127, 42], dtype=np.int8).reshape(2, 2)
    path = tmp_path / "int8.fits"
    fits.PrimaryHDU(data).writeto(path, overwrite=True)

    expected = fitsio.read(str(path))
    if torch.cuda.is_available():
        got = torchfits.read(str(path), device="cuda", scale_on_device=True)
        assert got.dtype == torch.int8
        assert got.device.type == "cuda"
        assert got.cpu().numpy().tolist() == expected.tolist()
    else:
        got = torchfits.read(str(path), scale_on_device=True)
        assert got.dtype == torch.int8
        assert got.numpy().tolist() == expected.tolist()


def test_read_uint16_fits_native_dtype_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    data = np.array([[0, 32768, 65535]], dtype=np.uint16)
    path = tmp_path / "uint16.fits"
    fits.PrimaryHDU(data).writeto(path, overwrite=True)

    got = torchfits.read(str(path), device="cuda", scale_on_device=True)
    assert got.dtype == torch.uint16
    assert got.device.type == "cuda"
    assert got.cpu().numpy().tolist() == data.tolist()
