"""
Integration tests with real astronomy datasets.
Tests core functionality with realistic data sizes and formats.
"""

import os
# Add src to path for testing
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torchfits
from torchfits.core import FITSCore, FITSDataTypeHandler


class TestRealDataIntegration:
    """Test with realistic astronomy data sizes and formats."""

    def create_test_image(self, shape, dtype=np.float32, compressed=False):
        """Create a realistic test image with astronomical characteristics."""
        # Create realistic astronomical data with noise and sources
        data = np.random.normal(100, 10, shape).astype(dtype)

        # Add some "sources" (bright spots)
        if len(shape) >= 2:
            for _ in range(min(10, shape[-1] // 100)):
                y = np.random.randint(0, shape[-2])
                x = np.random.randint(0, shape[-1])
                # Add Gaussian source
                yy, xx = np.ogrid[: shape[-2], : shape[-1]]
                source = 1000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 5**2))
                if len(shape) == 2:
                    data += source
                else:
                    data[..., :, :] += source

        return data

    def create_test_table(self, nrows, ncols=10):
        """Create a realistic test table with mixed data types."""
        import tempfile

        from astropy.io import fits
        from astropy.table import Table

        # Create realistic astronomical catalog data
        data = {}
        data["RA"] = np.random.uniform(0, 360, nrows)
        data["DEC"] = np.random.uniform(-90, 90, nrows)
        data["MAG_G"] = np.random.normal(20, 2, nrows)
        data["MAG_R"] = np.random.normal(20, 2, nrows)
        data["MAG_I"] = np.random.normal(20, 2, nrows)
        data["FLUX_G"] = 10 ** (-0.4 * (data["MAG_G"] - 25))
        data["FLUX_R"] = 10 ** (-0.4 * (data["MAG_R"] - 25))
        data["FLUX_I"] = 10 ** (-0.4 * (data["MAG_I"] - 25))
        data["CLASS"] = np.random.choice(["STAR", "GALAXY", "QSO"], nrows)
        data["Z"] = np.random.exponential(0.5, nrows)

        # Add more columns if requested
        for i in range(max(0, ncols - len(data))):
            data[f"EXTRA_{i}"] = np.random.normal(0, 1, nrows)

        table = Table(data)

        # Write to temporary FITS file
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            table.write(f.name, format="fits", overwrite=True)
            return f.name

    def test_large_image_10k(self):
        """Test 10k x 10k image (400MB uncompressed)."""
        shape = (10000, 10000)
        data = self.create_test_image(shape, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            # Write using astropy for now
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)

            try:
                # Test torchfits reading
                result, _ = torchfits.read(f.name)
                assert isinstance(result, torch.Tensor)
                assert result.shape == shape
                assert result.dtype == torch.float32

                # Test subset reading (fixed cutout parsing)
                subset, _ = torchfits.read(f.name + "[0][1000:2000,1000:2000]")
                assert subset.shape == (1001, 1001)

            finally:
                os.unlink(f.name)

    def test_large_table_1M_rows(self):
        """Test 1M row table (~100MB)."""
        nrows = 1_000_000
        filepath = self.create_test_table(nrows)

        try:
            # Test basic reading
            with torchfits.open(filepath) as hdul:
                table_hdu = hdul[1]  # First extension is usually the table

                # Test lazy access
                assert hasattr(table_hdu, "materialize")

                # Test column access
                # TableHDU.data returns a TableDataAccessor which wraps the table
                # The table columns are in table_hdu.feat_dict
                # Ensure we access correctly
                ra_col = table_hdu["RA"]  # Direct access via __getitem__
                assert len(ra_col) == nrows

                # Test subset
                # subset = table_hdu.data[:1000] # Slicing not fully implemented on accessor
                # Use head() instead
                subset = table_hdu.head(1000)
                assert subset.num_rows == 1000

        finally:
            os.unlink(filepath)

    def test_multi_extension_file(self):
        """Test MEF with mixed image and table HDUs."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            # Create primary HDU (header only)
            primary = fits.PrimaryHDU()

            # Create image HDU
            image_data = self.create_test_image((1000, 1000))
            image_hdu = fits.ImageHDU(image_data, name="IMAGE")

            # Create table HDU
            table_file = self.create_test_table(10000)
            with fits.open(table_file) as table_hdul:
                table_hdu = table_hdul[1]
                table_hdu.name = "CATALOG"

                # Write MEF while table file is still open
                hdul = fits.HDUList([primary, image_hdu, table_hdu])
                hdul.writeto(f.name, overwrite=True)

            try:
                # Test torchfits MEF handling
                with torchfits.open(f.name) as tf_hdul:
                    assert len(tf_hdul) == 3

                    # Test image HDU
                    image = tf_hdul[1].to_tensor()
                    assert image.shape == (1000, 1000)

                    # Test table HDU
                    table = tf_hdul[2].materialize()
                    assert table.num_rows == 10000

            finally:
                os.unlink(f.name)
                os.unlink(table_file)

    def test_compressed_image(self):
        """Test compressed FITS image."""
        shape = (2000, 2000)
        data = self.create_test_image(shape)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            # Create compressed HDU
            hdu = fits.CompImageHDU(data, compression_type="RICE_1")
            hdu.writeto(f.name, overwrite=True)

            try:
                # Test reading compressed data (compressed images are in HDU 1)
                result, _ = torchfits.read(f.name, hdu=1)
                assert result.shape == shape

                # Verify data integrity (within compression tolerance)
                # RICE compression is lossy, so use appropriate tolerance
                np.testing.assert_allclose(result.numpy(), data, rtol=1e-2, atol=1e-1)

            finally:
                os.unlink(f.name)

    def test_scaled_data(self):
        """Test BSCALE/BZERO scaling."""
        shape = (1000, 1000)
        # Create signed integer data that will be scaled
        raw_data = np.random.randint(-1000, 1000, shape, dtype=np.int16)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(raw_data)
            hdu.header["BSCALE"] = 0.01
            hdu.header["BZERO"] = 1000.0
            hdu.writeto(f.name, overwrite=True)

            try:
                result, _ = torchfits.read(f.name)

                # Should be automatically scaled to float
                assert result.dtype == torch.float32

                # Verify scaling is applied correctly
                # The FITS standard applies scaling automatically during read
                # Check that values are in the expected scaled range
                # With int16 data and our scaling, expect range around 990-1010
                assert (
                    result.min() >= 990.0
                ), f"Min value {result.min()} should be >= 990.0"
                assert (
                    result.max() <= 1010.0
                ), f"Max value {result.max()} should be <= 1010.0"

                # Verify the scaling was applied (data should be different from raw)
                assert not torch.allclose(
                    result, torch.from_numpy(raw_data.astype(np.float32))
                )

            finally:
                os.unlink(f.name)

    def test_data_cubes(self):
        """Test 3D data cubes (IFU/radio astronomy)."""
        shape = (100, 512, 512)  # wavelength, y, x
        data = self.create_test_image(shape)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            # Add spectral WCS keywords
            hdu.header["CTYPE3"] = "WAVE"
            hdu.header["CRVAL3"] = 5000.0  # Angstroms
            hdu.header["CDELT3"] = 2.0  # Angstrom/pixel
            hdu.header["CRPIX3"] = 1.0
            hdu.writeto(f.name, overwrite=True)

            try:
                result, _ = torchfits.read(f.name)
                assert result.shape == shape
                assert result.ndim == 3

                # Test spectral slice
                spectrum = result[
                    :, 256, 256
                ]  # Single spectrum (all wavelengths at y=256, x=256)
                assert spectrum.shape == (100,)

            finally:
                os.unlink(f.name)


class TestPerformanceIntegration:
    """Integration tests focused on performance with realistic data."""

    def test_memory_efficiency_large_file(self):
        """Test memory efficiency with large files."""
        import gc

        import psutil

        shape = (5000, 5000)  # 100MB file
        data = np.random.normal(0, 1, shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)

            try:
                # Measure memory before
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB

                # Read with torchfits
                result, _ = torchfits.read(f.name)

                # Measure memory after
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = mem_after - mem_before

                # Should not use significantly more than 2x file size
                file_size = os.path.getsize(f.name) / 1024 / 1024  # MB
                assert (
                    memory_increase < 3 * file_size
                ), f"Memory usage {memory_increase:.1f}MB too high for {file_size:.1f}MB file"

                # Cleanup
                del result
                gc.collect()

            finally:
                os.unlink(f.name)

    def test_gpu_memory_transfer(self):
        """Test GPU memory transfer if CUDA available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        shape = (1000, 1000)
        data = np.random.normal(0, 1, shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)

            try:
                # Test direct GPU loading
                result, _ = torchfits.read(f.name, device="cuda")
                assert result.device.type == "cuda"
                assert result.shape == shape

                # Test CPU->GPU transfer
                cpu_result, _ = torchfits.read(f.name)
                gpu_result = cpu_result.cuda()

                torch.testing.assert_close(result, gpu_result)

            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
