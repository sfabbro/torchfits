"""
Test caching functionality.
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

    def test_cached_numpy_reads_survive_repeated_cache_clears(self):
        """Regression: cached numpy reads should remain stable across cache clears."""
        cpp = pytest.importorskip("torchfits.cpp")
        if not hasattr(cpp, "read_full_numpy_cached"):
            pytest.skip("read_full_numpy_cached unavailable in this build")

        file_a, _ = self.create_test_fits((257,))
        file_b, _ = self.create_test_fits((33, 17))

        try:
            for i in range(300):
                torchfits.clear_file_cache()
                if i % 2 == 0:
                    arr = cpp.read_full_numpy_cached(file_a, 0, True)
                    assert arr.shape == (257,)
                else:
                    arr = cpp.read_full_numpy_cached(file_b, 0, True)
                    assert arr.shape == (33, 17)
        finally:
            for path in (file_a, file_b):
                if os.path.exists(path):
                    os.unlink(path)
            torchfits.clear_file_cache()

    def test_cached_multibyte_read_matches_nocache_reference(self):
        """Regression: cached mmap raw path must match the no-cache path exactly."""
        cpp = pytest.importorskip("torchfits.cpp")
        if not hasattr(cpp, "read_full_cached") or not hasattr(
            cpp, "read_full_nocache"
        ):
            pytest.skip("cached/nocache read methods unavailable in this build")

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            data = np.arange(128 * 64, dtype=np.int32).reshape(128, 64) - 1000
            fits.PrimaryHDU(data).writeto(f.name, overwrite=True)
            path = f.name

        try:
            torchfits.clear_file_cache()
            cached = cpp.read_full_cached(path, 0, True).numpy()
            reference = cpp.read_full_nocache(path, 0, True).numpy()
            np.testing.assert_array_equal(cached, reference)
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchfits.clear_file_cache()

    def test_cold_nommap_heuristic_with_cache_enabled_int16(self):
        """Large int16 images should prefer non-mmap even when cache is enabled."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            data = np.arange(1024 * 1024, dtype=np.int16).reshape(1024, 1024)
            fits.PrimaryHDU(data).writeto(f.name, overwrite=True)
            path = f.name

        try:
            torchfits.clear_file_cache()
            assert torchfits._should_use_cold_nommap(
                path, 0, cache_capacity=10, mmap=True
            )
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchfits.clear_file_cache()

    def test_cold_nommap_heuristic_float64_and_small_guard(self):
        """64-bit and sub-1MiB payloads should keep mmap enabled by default."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f_large:
            from astropy.io import fits

            large = np.random.randn(1024, 1024).astype(np.float64)  # ~8 MiB payload
            fits.PrimaryHDU(large).writeto(f_large.name, overwrite=True)
            large_path = f_large.name
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f_small:
            from astropy.io import fits

            small = np.random.randn(128, 128).astype(np.float64)  # <1 MiB payload
            fits.PrimaryHDU(small).writeto(f_small.name, overwrite=True)
            small_path = f_small.name

        try:
            torchfits.clear_file_cache()
            assert not torchfits._should_use_cold_nommap(
                large_path, 0, cache_capacity=10, mmap=True
            )
            assert not torchfits._should_use_cold_nommap(
                small_path, 0, cache_capacity=10, mmap=True
            )
        finally:
            for path in (large_path, small_path):
                if os.path.exists(path):
                    os.unlink(path)
            torchfits.clear_file_cache()

    def test_read_mmap_auto_defaults_to_false_for_compressed(self, monkeypatch):
        """`mmap='auto' should default to non-mmap for compressed image HDUs."""
        cpp = pytest.importorskip("torchfits.cpp")
        if not hasattr(cpp, "read_full_cached"):
            pytest.skip("read_full_cached unavailable in this build")

        observed = []

        monkeypatch.setattr(
            torchfits,
            "_get_image_meta",
            lambda path, hdu: (-32, 2, (64, 64), 1.0, 0.0, True),
        )
        monkeypatch.setattr(
            torchfits,
            "_should_use_cold_nommap",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "cold nommap heuristic should not run for compressed auto mode"
                )
            ),
        )
        monkeypatch.setattr(
            cpp,
            "read_full_cached",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )
        monkeypatch.setattr(
            cpp,
            "read_full",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )

        out = torchfits.read(
            "synthetic_compressed.fits",
            hdu=1,
            mmap="auto",
            cache_capacity=10,
            handle_cache_capacity=16,
        )

        assert isinstance(out, torch.Tensor)
        assert observed == [False]

    def test_read_mmap_true_is_explicit_override(self, monkeypatch):
        """`mmap=True` must stay enabled even for compressed-image metadata."""
        cpp = pytest.importorskip("torchfits.cpp")
        if not hasattr(cpp, "read_full_cached"):
            pytest.skip("read_full_cached unavailable in this build")

        observed = []

        monkeypatch.setattr(
            torchfits,
            "_get_image_meta",
            lambda path, hdu: (-32, 2, (64, 64), 1.0, 0.0, True),
        )
        monkeypatch.setattr(
            torchfits,
            "_should_use_cold_nommap",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "cold nommap heuristic should not run for explicit mmap=True"
                )
            ),
        )
        monkeypatch.setattr(
            cpp,
            "read_full_cached",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )
        monkeypatch.setattr(
            cpp,
            "read_full",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )

        out = torchfits.read(
            "synthetic_compressed.fits",
            hdu=1,
            mmap=True,
            cache_capacity=10,
            handle_cache_capacity=16,
        )

        assert isinstance(out, torch.Tensor)
        assert observed == [True]

    def test_read_rejects_invalid_mmap_mode(self):
        """Only bool or 'auto' mmap mode should be accepted."""
        with pytest.raises(ValueError, match="mmap must be bool or 'auto'"):
            torchfits.read("dummy.fits", mmap="sometimes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
