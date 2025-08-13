import os
import sys

import pytest
import torch

import torchfits as tf
from torchfits.openmp_guard import detect_duplicate_openmp

# Opt-in model: disable by default to avoid unstable segfaults until
# environment OpenMP duplication is resolved. Export
# TORCHFITS_ENABLE_COMPRESSION_TESTS=1 to run these.
if not os.environ.get("TORCHFITS_ENABLE_COMPRESSION_TESTS"):
    pytest.skip(
        "Compression tests disabled by default (set TORCHFITS_ENABLE_COMPRESSION_TESTS=1 to enable).",
        allow_module_level=True,
    )

_omp_state = detect_duplicate_openmp()

pytestmark = pytest.mark.skipif(
    not hasattr(tf, "fits_reader_cpp")
    or not hasattr(tf.fits_reader_cpp, "CompressionConfig"),
    reason="Compression API not available",
)


# Generate a simple test image
def _make_image():
    return torch.arange(0, 64, dtype=torch.int16).view(8, 8)


@pytest.mark.parametrize("compression_type", ["None", "GZIP", "RICE"])
def test_compressed_write_read_roundtrip(tmp_path, compression_type):
    img = _make_image()
    out = tmp_path / f"comp_{compression_type.lower()}.fits"

    from torchfits import fits_reader_cpp  # low-level API

    cfg = tf.fits_reader_cpp.CompressionConfig()
    cfg.type = getattr(tf.fits_reader_cpp.CompressionType, compression_type)
    if compression_type != "None" and hasattr(cfg, "quantize_level"):
        cfg.quantize_level = 0

    fits_reader_cpp.write_tensor_to_fits_advanced(
        str(out), img, {"OBJECT": "COMPTEST"}, cfg, True, False
    )

    data, header = tf.read(str(out), hdu=0)
    assert torch.equal(data.to(img.dtype), img)
    assert header.get("OBJECT") == "COMPTEST"


@pytest.mark.parametrize("compression_type", ["HCOMPRESS"])
def test_compressed_hcompress_roundtrip(tmp_path, compression_type):
    img = _make_image().to(torch.float32)
    out = tmp_path / f"comp_{compression_type.lower()}.fits"

    from torchfits import fits_reader_cpp

    cfg = tf.fits_reader_cpp.CompressionConfig()
    cfg.type = getattr(tf.fits_reader_cpp.CompressionType, compression_type)
    if hasattr(cfg, "quantize_level"):
        cfg.quantize_level = 0

    fits_reader_cpp.write_tensor_to_fits_advanced(
        str(out), img, {"OBJECT": "HCOMP"}, cfg, True, False
    )

    data, header = tf.read(str(out), hdu=0)
    assert torch.allclose(data.to(img.dtype), img, atol=1e-3, rtol=1e-3)
    assert header.get("OBJECT") == "HCOMP"
