"""
Test the main torchfits API functions.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torchfits


class TestMainAPI:
    """Test main API functions."""

    def create_test_fits(self, shape=(100, 100), dtype=np.float32):
        """Create a test FITS file."""
        data = np.random.normal(0, 1, shape).astype(dtype)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)
            return f.name, data

    def test_read_basic(self):
        """Test basic read functionality."""
        filepath, expected_data = self.create_test_fits()

        try:
            result, header = torchfits.read(filepath, return_header=True)
            assert isinstance(result, torch.Tensor)
            assert result.shape == expected_data.shape
            np.testing.assert_allclose(result.numpy(), expected_data, rtol=1e-5)
        finally:
            os.unlink(filepath)

    def test_read_device(self):
        """Test reading to different devices."""
        filepath, expected_data = self.create_test_fits()

        try:
            # CPU
            result_cpu, _ = torchfits.read(filepath, device="cpu", return_header=True)
            assert result_cpu.device.type == "cpu"

            # GPU if available
            if torch.cuda.is_available():
                result_gpu, _ = torchfits.read(
                    filepath, device="cuda", return_header=True
                )
                assert result_gpu.device.type == "cuda"
                torch.testing.assert_close(result_cpu, result_gpu.cpu())
        finally:
            os.unlink(filepath)

    def test_read_precision(self):
        """Test mixed precision reading."""
        filepath, expected_data = self.create_test_fits()

        try:
            # FP16
            result_fp16, _ = torchfits.read(filepath, fp16=True, return_header=True)
            assert result_fp16.dtype == torch.float16

            # BF16
            result_bf16, _ = torchfits.read(filepath, bf16=True, return_header=True)
            assert result_bf16.dtype == torch.bfloat16
        finally:
            os.unlink(filepath)

    def test_write_basic(self):
        """Test basic write functionality."""
        data = torch.randn(50, 50)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            try:
                torchfits.write(f.name, data, overwrite=True)

                # Read back and verify
                result, _ = torchfits.read(f.name, return_header=True)
                torch.testing.assert_close(result, data)
            finally:
                os.unlink(f.name)

    def test_open_context_manager(self):
        """Test open() as context manager."""
        filepath, expected_data = self.create_test_fits()

        try:
            with torchfits.open(filepath) as hdul:
                assert len(hdul) >= 1
                result = hdul[0].to_tensor()
                assert isinstance(result, torch.Tensor)
        finally:
            os.unlink(filepath)

    def test_batch_operations(self):
        """Test batch reading functions."""
        files = []
        expected_shapes = []

        try:
            # Create multiple test files
            for i in range(3):
                shape = (50 + i * 10, 50 + i * 10)
                filepath, _ = self.create_test_fits(shape)
                files.append(filepath)
                expected_shapes.append(shape)

            # Test batch reading
            results = torchfits.read_batch(files)
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.shape == expected_shapes[i]

            # Test batch info
            info = torchfits.get_batch_info(files)
            assert info["num_files"] == 3
            assert info["valid_files"] == 3

        finally:
            for f in files:
                if os.path.exists(f):
                    os.unlink(f)

    def test_cache_functions(self):
        """Test cache management functions."""
        # Test cache stats
        stats = torchfits.get_cache_performance()
        assert isinstance(stats, dict)

        # Test cache clearing
        torchfits.clear_file_cache()  # Should not raise

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Non-existent file
        with pytest.raises((FileNotFoundError, RuntimeError)):
            torchfits.read("nonexistent.fits")

        # Invalid HDU
        filepath, _ = self.create_test_fits()
        try:
            with pytest.raises((ValueError, RuntimeError)):
                torchfits.read(filepath, hdu=999)
        finally:
            os.unlink(filepath)

        # Invalid device
        filepath, _ = self.create_test_fits()
        try:
            with pytest.raises(ValueError):
                torchfits.read(filepath, device="invalid")
        finally:
            os.unlink(filepath)


class TestTableAPI:
    """Test table-specific API functions."""

    def create_test_table(self, nrows=1000):
        """Create a test FITS table."""
        from astropy.table import Table

        data = {
            "ID": np.arange(nrows),
            "RA": np.random.uniform(0, 360, nrows),
            "DEC": np.random.uniform(-90, 90, nrows),
            "MAG": np.random.normal(20, 2, nrows),
            "FLAG": np.random.choice([0, 1], nrows),
        }

        table = Table(data)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            table.write(f.name, format="fits", overwrite=True)
            return f.name

    def test_large_table_reading(self):
        """Test large table reading function."""
        filepath = self.create_test_table(10000)

        try:
            # Test streaming read
            result = torchfits.read_large_table(
                filepath, max_memory_mb=10, streaming=True
            )
            assert isinstance(result, dict)
            assert "RA" in result
            assert len(result["RA"]) == 10000

            # Test non-streaming read
            result2 = torchfits.read_large_table(filepath, streaming=False)
            assert isinstance(result2, dict)

        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
