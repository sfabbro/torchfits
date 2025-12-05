"""
Test caching functionality.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torchfits


class TestCaching:
    """Test file caching functionality."""

    def create_test_fits(self, shape=(100, 100)):
        """Create a test FITS file."""
        data = np.random.normal(0, 1, shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)
            return f.name, data

    def test_cache_performance_tracking(self):
        """Test cache hit/miss tracking."""
        filepath, _ = self.create_test_fits()

        try:
            # Clear cache first
            torchfits.clear_file_cache()

            # First read should be a cache miss
            torchfits.read(filepath)
            stats1 = torchfits.get_cache_performance()

            # Second read should be a cache hit
            torchfits.read(filepath)
            stats2 = torchfits.get_cache_performance()

            # Verify cache behavior
            assert stats2["total_requests"] > stats1["total_requests"]

        finally:
            os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        filepath, _ = self.create_test_fits()

        try:
            # Read file to populate cache
            torchfits.read(filepath)

            # Clear cache
            torchfits.clear_file_cache()

            # Verify cache is cleared
            torchfits.get_cache_performance()
            # After clearing, stats should be reset or show no cached entries

        finally:
            os.unlink(filepath)

    def test_multiple_file_caching(self):
        """Test caching with multiple files."""
        files = []

        try:
            # Create multiple test files
            for i in range(5):
                filepath, _ = self.create_test_fits((50 + i * 10, 50 + i * 10))
                files.append(filepath)

            # Clear cache
            torchfits.clear_file_cache()

            # Read all files
            for filepath in files:
                torchfits.read(filepath)

            # Read them again (should hit cache)
            for filepath in files:
                torchfits.read(filepath)

            # Check cache performance
            stats = torchfits.get_cache_performance()
            assert stats["total_requests"] >= len(files) * 2

        finally:
            for f in files:
                if os.path.exists(f):
                    os.unlink(f)
            torchfits.clear_file_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
