"""
Performance tests for torchfits.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torchfits


class TestPerformance:
    """Performance benchmarks and regression tests."""

    def create_test_data(self, shape, dtype=np.float32):
        """Create test data with realistic astronomical characteristics."""
        data = np.random.normal(100, 10, shape).astype(dtype)

        # Add some structure
        if len(shape) >= 2:
            y, x = np.ogrid[: shape[-2], : shape[-1]]
            data += 20 * np.sin(2 * np.pi * x / 200) * np.cos(2 * np.pi * y / 200)

        return data

    def create_fits_file(self, data):
        """Create FITS file from data."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)
            return f.name

    def test_zero_copy_performance(self):
        """Test zero-copy tensor creation performance."""
        shape = (2000, 2000)
        data = self.create_test_data(shape)
        filepath = self.create_fits_file(data)

        try:
            # Measure read time
            start_time = time.time()
            result, _ = torchfits.read(filepath)
            read_time = time.time() - start_time

            # Should be fast due to zero-copy optimization
            assert read_time < 5.0  # 5 seconds max for 2k x 2k image
            assert result.shape == shape

        finally:
            os.unlink(filepath)

    def test_memory_efficiency(self):
        """Test memory efficiency of reading operations."""
        shape = (3000, 3000)  # ~36MB file
        data = self.create_test_data(shape)
        filepath = self.create_fits_file(data)

        try:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            result, _ = torchfits.read(filepath)

            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = mem_after - mem_before

            # Should not use significantly more than 2x file size
            file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
            assert memory_increase < 3 * file_size

            del result

        finally:
            os.unlink(filepath)

    def test_batch_reading_performance(self):
        """Test performance of batch reading operations."""
        files = []

        try:
            # Create multiple test files
            for i in range(5):
                shape = (500, 500)
                data = self.create_test_data(shape)
                filepath = self.create_fits_file(data)
                files.append(filepath)

            # Test batch reading performance
            start_time = time.time()
            results = torchfits.read_batch(files)  # Removed max_workers
            batch_time = time.time() - start_time

            assert len(results) == 5
            # Batch reading should be reasonably fast
            assert batch_time < 15.0  # 15 seconds max for 5 files

        finally:
            for f in files:
                if os.path.exists(f):
                    os.unlink(f)

    def test_cache_performance(self):
        """Test file caching performance improvement."""
        shape = (1000, 1000)
        data = self.create_test_data(shape)
        filepath = self.create_fits_file(data)

        try:
            # Clear cache first
            torchfits.clear_file_cache()

            # First read (cache miss)
            start_time = time.time()
            result1, _ = torchfits.read(filepath)
            first_read_time = time.time() - start_time

            # Second read (cache hit)
            start_time = time.time()
            result2, _ = torchfits.read(filepath)
            second_read_time = time.time() - start_time

            # Cache hit should be faster (though not always guaranteed due to OS caching)
            # At minimum, should not be significantly slower
            assert second_read_time <= first_read_time * 2

            # Results should be identical
            torch.testing.assert_close(result1, result2)

        finally:
            os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_large_file_performance(self):
        """Test performance with large files."""
        shape = (5000, 5000)  # ~100MB file
        data = self.create_test_data(shape, dtype=np.float32)
        filepath = self.create_fits_file(data)

        try:
            start_time = time.time()
            result, _ = torchfits.read(filepath)
            read_time = time.time() - start_time

            assert result.shape == shape
            # Should complete in reasonable time
            assert read_time < 30.0  # 30 seconds max for 100MB file

        finally:
            os.unlink(filepath)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_transfer_performance(self):
        """Test GPU transfer performance."""
        shape = (2000, 2000)
        data = self.create_test_data(shape)
        filepath = self.create_fits_file(data)

        try:
            # Test direct GPU loading
            start_time = time.time()
            result_gpu, _ = torchfits.read(filepath, device="cuda")
            gpu_read_time = time.time() - start_time

            assert result_gpu.device.type == "cuda"
            assert result_gpu.shape == shape

            # Should complete in reasonable time
            assert gpu_read_time < 10.0  # 10 seconds max

        finally:
            os.unlink(filepath)

    def test_subset_reading_performance(self):
        """Test performance of subset reading."""
        shape = (4000, 4000)
        data = self.create_test_data(shape)
        filepath = self.create_fits_file(data)

        try:
            # Read small subset
            start_time = time.time()
            subset, _ = torchfits.read(filepath + "[0][1000:2000,1000:2000]")
            subset_time = time.time() - start_time

            # FITS slicing is 1-based inclusive, so 1000:2000 is 1001 elements
            assert subset.shape == (1001, 1001)
            # Subset should be much faster than full image
            assert subset_time < 5.0  # 5 seconds max for 1k x 1k subset

        finally:
            os.unlink(filepath)


class TestPerformanceRegression:
    """Regression tests to ensure performance doesn't degrade."""

    def test_read_performance_baseline(self):
        """Baseline performance test for read operations."""
        shape = (1000, 1000)
        data = np.random.normal(0, 1, shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)

            try:
                # Measure multiple reads for consistency
                times = []
                for _ in range(3):
                    start_time = time.time()
                    result, _ = torchfits.read(f.name)
                    read_time = time.time() - start_time
                    times.append(read_time)

                avg_time = sum(times) / len(times)

                # Performance baseline: should read 1M pixels in < 2 seconds
                assert avg_time < 2.0
                assert result.shape == shape

            finally:
                os.unlink(f.name)

    def test_memory_usage_baseline(self):
        """Baseline memory usage test."""
        shape = (2000, 2000)  # ~16MB
        data = np.random.normal(0, 1, shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)

            try:
                import gc

                gc.collect()

                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB

                result, _ = torchfits.read(f.name)

                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = mem_after - mem_before

                # Should not use more than 3x the data size
                data_size = result.numel() * 4 / 1024 / 1024  # MB (float32)
                assert memory_increase < 3 * data_size

                del result
                gc.collect()

            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
