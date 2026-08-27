"""
Test compressed FITS image handling.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

import torchfits


class TestCompression:
    """Test compressed FITS image functionality."""

    def create_compressed_fits(self, shape=(1000, 1000), compression="RICE_1"):
        """Create a compressed FITS file."""
        data = np.random.normal(100, 10, shape).astype(np.float32)

        # Add some structure to make compression more realistic
        y, x = np.ogrid[: shape[0], : shape[1]]
        data += 50 * np.sin(2 * np.pi * x / 100) * np.cos(2 * np.pi * y / 100)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            # Create compressed HDU
            hdu = fits.CompImageHDU(data, compression_type=compression)
            hdu.writeto(f.name, overwrite=True)
            return f.name, data

    def test_rice_compression(self):
        """Test RICE compressed images."""
        filepath, expected_data = self.create_compressed_fits(compression="RICE_1")

        try:
            # Compressed images are typically in HDU 1
            result, header = torchfits.read(filepath, hdu=1, return_header=True)

            assert isinstance(result, torch.Tensor)
            assert result.shape == expected_data.shape

            # RICE is lossless for integer data, but we're using float
            # Allow for some compression artifacts
            # Relaxed atol to 0.5 as quantization can introduce larger errors
            np.testing.assert_allclose(
                result.numpy(), expected_data, rtol=1e-3, atol=0.5
            )

        finally:
            os.unlink(filepath)

    def test_gzip_compression(self):
        """Test GZIP compressed images."""
        filepath, expected_data = self.create_compressed_fits(compression="GZIP_1")

        try:
            result, header = torchfits.read(filepath, hdu=1, return_header=True)

            assert isinstance(result, torch.Tensor)
            assert result.shape == expected_data.shape

            # GZIP should be lossless, but astropy quantizes floats by default
            np.testing.assert_allclose(
                result.numpy(), expected_data, rtol=1e-5, atol=0.5
            )

        finally:
            os.unlink(filepath)

    def test_tile_aware_reading(self):
        """Test tile-aware reading for compressed images."""
        # Create larger image to test tile optimization
        filepath, expected_data = self.create_compressed_fits((2000, 2000))

        try:
            # Read full image
            full_result, _ = torchfits.read(filepath, hdu=1, return_header=True)

            # Read subset (should use tile-aware optimization)
            # FITS uses 1-based inclusive indexing. Python [500:1500] is 0-based 500 to 1499.
            # So FITS range is 501 to 1500.
            subset_result, _ = torchfits.read(
                filepath + "[1][501:1500,501:1500]", return_header=True
            )

            assert subset_result.shape == (1000, 1000)

            # Verify subset matches full image
            expected_subset = full_result[500:1500, 500:1500]
            torch.testing.assert_close(
                subset_result, expected_subset, rtol=1e-3, atol=1e-1
            )

        finally:
            os.unlink(filepath)

    def test_compression_detection(self):
        """Test automatic compression detection."""
        filepath, expected_data = self.create_compressed_fits()

        try:
            # Should automatically detect and handle compression
            result, header = torchfits.read(filepath, hdu=1, return_header=True)

            assert isinstance(result, torch.Tensor)
            # Header should contain compression info
            # assert isinstance(header, dict) # Header is now a custom object
            assert "ZIMAGE" in header or "ZCMPTYPE" in header

        finally:
            os.unlink(filepath)

    def test_multiple_compression_types(self):
        """Test different compression algorithms."""
        compression_types = ["RICE_1", "GZIP_1"]

        for comp_type in compression_types:
            try:
                filepath, expected_data = self.create_compressed_fits(
                    shape=(500, 500), compression=comp_type
                )

                result, header = torchfits.read(filepath, hdu=1, return_header=True)

                assert isinstance(result, torch.Tensor)
                assert result.shape == expected_data.shape

            except Exception as e:
                # Some compression types might not be available
                if "not supported" in str(e).lower():
                    pytest.skip(f"Compression type {comp_type} not supported")
                else:
                    raise
            finally:
                if "filepath" in locals() and os.path.exists(filepath):
                    os.unlink(filepath)


class TestCompressionPerformance:
    """Test performance aspects of compressed image reading."""

    def test_large_integer_rice_roundtrip_exact(self):
        """Large integer RICE reads should stay exact (parallel path safety net)."""
        shape = (3072, 3072)
        # Integer ramps are lossless under RICE and exercise row-wise tiles.
        data = (
            np.arange(shape[0] * shape[1], dtype=np.int64).reshape(shape) % 32767
        ).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            try:
                torchfits.write(
                    f.name, torch.from_numpy(data), overwrite=True, compress="RICE_1"
                )
                result = torchfits.read(f.name, hdu=1, mmap=False)
                np.testing.assert_array_equal(result.numpy(), data)
            finally:
                os.unlink(f.name)

    def test_large_compressed_image(self):
        """Test reading large compressed images."""
        import time

        # Create large compressed image
        shape = (4000, 4000)
        data = np.random.normal(100, 10, shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.CompImageHDU(data, compression_type="RICE_1")
            hdu.writeto(f.name, overwrite=True)

            try:
                start_time = time.time()
                result, _ = torchfits.read(f.name, hdu=1, return_header=True)
                read_time = time.time() - start_time

                assert result.shape == shape
                # Should complete in reasonable time
                assert read_time < 60.0  # 60 seconds max

            finally:
                os.unlink(f.name)

    def test_compressed_subset_performance(self):
        """Test performance of reading subsets from compressed images."""
        import time

        # Create large compressed image
        shape = (3000, 3000)
        data = np.random.normal(100, 10, shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.CompImageHDU(data, compression_type="RICE_1")
            hdu.writeto(f.name, overwrite=True)

            try:
                # Read small subset - should be fast due to tile optimization
                start_time = time.time()
                # FITS 1-based inclusive: 1001 to 1500 (length 500)
                subset, _ = torchfits.read(
                    f.name + "[1][1001:1500,1001:1500]", return_header=True
                )
                subset_time = time.time() - start_time

                assert subset.shape == (500, 500)
                # Subset should be much faster than full image
                assert subset_time < 10.0  # 10 seconds max for subset

            finally:
                os.unlink(f.name)


def test_compress_empty_primary_mef_roundtrip():
    """Compressing a MEF with an empty primary must not shift output HDU indices."""
    from astropy.io import fits

    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    b = np.arange(48, dtype=np.float32).reshape(6, 8)
    src = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    src.close()
    dst = src.name.replace(".fits", "_c.fits")
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(a, name="A"), fits.ImageHDU(b, name="B")]
    ).writeto(src.name, overwrite=True)
    try:
        with torchfits.open(src.name) as hdul:
            torchfits.write(dst, hdul, overwrite=True, compress=True)
        with torchfits.open(dst) as out:
            assert len(out) == 3
            assert np.allclose(out[1].to_tensor("cpu").numpy(), a)
            assert np.allclose(out[2].to_tensor("cpu").numpy(), b)
    finally:
        for p in (src.name, dst):
            if os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
