"""
Test caching functionality.
"""

import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import torch

import torchfits
from torchfits.cache import CacheConfig


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

    def test_get_cache_stats(self):
        """Test get_cache_stats returns expected dictionary structure."""
        from torchfits.cache import get_cache_stats, clear_cache

        # Clear cache to start with a known state
        clear_cache()

        stats = get_cache_stats()

        # Verify it's a dictionary
        assert isinstance(stats, dict)

        # Check for expected keys (dead never-updated counters removed:
        # evictions/memory_usage_mb/disk_usage_gb were always 0 and are gone).
        expected_keys = {
            "hits",
            "misses",
            "io_hits",
            "io_misses",
            "io_total_requests",
            "cpp_cache_size",
            "config",
            "hit_rate",
        }
        assert expected_keys.issubset(stats.keys())

        # Verify types of specific fields
        assert isinstance(stats["hits"], int)
        assert isinstance(stats["misses"], int)
        assert isinstance(stats["hit_rate"], float)
        assert isinstance(stats["config"], dict)

        # Check config keys
        expected_config_keys = {
            "max_files",
            "max_memory_mb",
            "disk_cache_gb",
            "prefetch_enabled",
        }
        assert expected_config_keys.issubset(stats["config"].keys())

        # Basic hit_rate calculation check (should be 0.0 when hits=0, misses=0)
        assert stats["hit_rate"] == 0.0

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        filepath, _ = self.create_test_fits()

        try:
            # Read file to populate cache
            torchfits.read(filepath)
            torchfits.get_cache_performance()

            # Clear cache
            torchfits.clear_file_cache()

            # Verify cache is cleared. This test previously made no assertion
            # at all (deep-review unit 10, TE-003): the comment said "stats
            # should be reset" and nothing checked it, so a clear_file_cache
            # that did nothing would have passed.
            stats = torchfits.get_cache_performance()
            assert stats["total_requests"] == 0, stats
            assert stats["hits"] == 0, stats
            assert stats["misses"] == 0, stats

            # And a read after the clear is served cold, not from the cache.
            torchfits.read(filepath)
            assert torchfits.get_cache_performance()["misses"] >= 1

        finally:
            os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_concurrent_python_lru_access_preserves_outputs(self):
        """Concurrent metadata/LRU access must not corrupt read results."""
        filepath, expected = self.create_test_fits((64, 48))

        def read_once(_: int) -> np.ndarray:
            torchfits.read_header(filepath, hdu=0)
            return np.asarray(torchfits.read(filepath, hdu=0, mmap="auto").numpy())

        try:
            torchfits.clear_file_cache()
            with ThreadPoolExecutor(max_workers=8) as pool:
                outputs = list(pool.map(read_once, range(32)))
            for output in outputs:
                np.testing.assert_array_equal(output, expected)
        finally:
            os.unlink(filepath)
            torchfits.clear_file_cache()

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
        import torchfits.io

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            data = np.arange(1024 * 1024, dtype=np.int16).reshape(1024, 1024)
            fits.PrimaryHDU(data).writeto(f.name, overwrite=True)
            path = f.name

        try:
            torchfits.clear_file_cache()
            assert torchfits.io._should_use_cold_nommap(
                path, 0, cache_capacity=10, mmap=True
            )
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchfits.clear_file_cache()

    def test_cold_nommap_heuristic_float64_and_small_guard(self):
        """64-bit and sub-1MiB payloads should keep mmap enabled by default."""
        import torchfits.io

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
            assert not torchfits.io._should_use_cold_nommap(
                large_path, 0, cache_capacity=10, mmap=True
            )
            assert not torchfits.io._should_use_cold_nommap(
                small_path, 0, cache_capacity=10, mmap=True
            )
        finally:
            for path in (large_path, small_path):
                if os.path.exists(path):
                    os.unlink(path)
            torchfits.clear_file_cache()

    def test_read_mmap_auto_defaults_to_false_for_compressed(self, monkeypatch):
        """`mmap='auto'` should disable mmap for compressed HDUs."""
        import torchfits.io
        import torchfits._C as cpp

        if not hasattr(cpp, "read_full_cached"):
            pytest.skip("read_full_cached unavailable in this build")

        filepath, _ = self.create_test_fits()
        observed = []

        monkeypatch.setattr(
            torchfits.io,
            "_get_image_meta",
            lambda path, hdu: (-32, 2, (64, 64), 1.0, 0.0, True),
        )
        monkeypatch.setattr(
            torchfits.io,
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

        try:
            out = torchfits.read(
                filepath,
                hdu=0,
                mmap="auto",
                cache_capacity=10,
                handle_cache_capacity=16,
            )

            assert isinstance(out, torch.Tensor)
            assert observed == [False]
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_read_mmap_true_is_explicit_override(self, monkeypatch):
        """`mmap=True` must stay enabled for compressed metadata reads."""
        import torchfits.io
        import torchfits._C as cpp

        if not hasattr(cpp, "read_full_cached"):
            pytest.skip("read_full_cached unavailable in this build")

        filepath, _ = self.create_test_fits()
        observed = []

        monkeypatch.setattr(
            torchfits.io,
            "_get_image_meta",
            lambda path, hdu: (-32, 2, (64, 64), 1.0, 0.0, True),
        )
        monkeypatch.setattr(
            torchfits.io,
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

        try:
            out = torchfits.read(
                filepath,
                hdu=0,
                mmap=True,
                cache_capacity=10,
                handle_cache_capacity=16,
            )

            assert isinstance(out, torch.Tensor)
            assert observed == [True]
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_read_mmap_auto_uses_direct_mmap(self, monkeypatch):
        """`mmap='auto'` on a plain uncompressed HDU should use direct mmap."""
        import torchfits.io
        import torchfits._C as cpp

        filepath, _ = self.create_test_fits()
        observed = []
        monkeypatch.setattr(
            cpp,
            "read_full",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((4, 4), dtype=torch.float32),
            )[1],
        )
        if hasattr(cpp, "read_full_cached"):
            monkeypatch.setattr(
                cpp,
                "read_full_cached",
                lambda path, hdu, use_mmap: (
                    observed.append(bool(use_mmap)),
                    torch.zeros((4, 4), dtype=torch.float32),
                )[1],
            )

        try:
            out = torchfits.read(
                filepath,
                hdu=0,
                mmap="auto",
                cache_capacity=10,
                handle_cache_capacity=16,
            )

            assert isinstance(out, torch.Tensor)
            assert observed == [True]
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
            torchfits.clear_file_cache()

    def test_read_rejects_invalid_mmap_mode(self):
        """Only bool or 'auto' mmap mode should be accepted."""
        with pytest.raises(ValueError, match="mmap must be bool or 'auto'"):
            torchfits.read("dummy.fits", mmap="sometimes")

    def test_read_hdu_auto_detects_compressed_image_extension(self):
        """`hdu='auto'` should resolve to the first payload HDU (fitsio-like behavior)."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            image = np.random.normal(size=(64, 64)).astype(np.float32)
            primary = fits.PrimaryHDU()
            compressed = fits.CompImageHDU(image, compression_type="RICE_1")
            fits.HDUList([primary, compressed]).writeto(f.name, overwrite=True)
            path = f.name

        try:
            empty_primary = torchfits.read(path, hdu=0, mmap="auto")
            auto_tensor = torchfits.read(path, hdu="auto", mmap="auto")
            none_tensor = torchfits.read(path, hdu=None, mmap="auto")

            assert isinstance(empty_primary, torch.Tensor)
            assert empty_primary.numel() == 0
            assert isinstance(auto_tensor, torch.Tensor)
            assert isinstance(none_tensor, torch.Tensor)
            assert tuple(auto_tensor.shape) == image.shape
            assert tuple(none_tensor.shape) == image.shape
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchfits.clear_file_cache()

    def test_read_header_auto_matches_detected_hdu(self):
        """`read_header(..., hdu='auto')` should return the detected payload header."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            image = np.random.normal(size=(32, 48)).astype(np.float32)
            primary = fits.PrimaryHDU()
            compressed = fits.CompImageHDU(image, compression_type="RICE_1")
            fits.HDUList([primary, compressed]).writeto(f.name, overwrite=True)
            path = f.name

        try:
            header = torchfits.read_header(path, hdu="auto")
            assert int(header.get("NAXIS", 0)) == 2
            assert int(header.get("NAXIS1", 0)) > 0
            assert int(header.get("NAXIS2", 0)) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchfits.clear_file_cache()


class TestCacheManager:
    """Test CacheManager functionality."""

    def test_get_cache_manager_singleton(self):
        from torchfits.cache import get_cache_manager

        manager1 = get_cache_manager()
        manager2 = get_cache_manager()

        assert manager1 is manager2


class TestCacheConfig:
    """Test CacheConfig functionality."""

    def test_default_initialization(self):
        config = CacheConfig()
        assert config.max_files == 100
        assert config.max_memory_mb == 1024
        assert config.disk_cache_gb == 10
        assert config.prefetch_enabled is True

    def test_custom_initialization(self):
        config = CacheConfig(
            max_files=50, max_memory_mb=512, disk_cache_gb=5, prefetch_enabled=False
        )
        assert config.max_files == 50
        assert config.max_memory_mb == 512
        assert config.disk_cache_gb == 5
        assert config.prefetch_enabled is False

    @patch("torchfits.cache.os.sysconf")
    def test_for_environment_no_psutil(self, mock_sysconf):
        mock_sysconf.side_effect = ValueError("Simulated sysconf error")
        config = CacheConfig.for_environment()
        assert config.max_files == 100
        assert config.max_memory_mb == 1024
        assert config.disk_cache_gb == 5
        assert config.prefetch_enabled is False

    @patch("torchfits.cache.os.sysconf")
    @patch.object(CacheConfig, "_is_hpc_environment", return_value=True)
    def test_for_environment_hpc(self, mock_hpc, mock_sysconf):
        def sysconf_mock(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return (100 * (1024**3)) // 4096
            raise ValueError()

        mock_sysconf.side_effect = sysconf_mock

        config = CacheConfig.for_environment()
        assert config.max_files == 1000
        assert config.max_memory_mb == int(100 * 1024 * 0.3)
        assert config.disk_cache_gb == 50
        assert config.prefetch_enabled is True

    @patch("torchfits.cache.os.sysconf")
    @patch.object(CacheConfig, "_is_hpc_environment", return_value=False)
    @patch.object(CacheConfig, "_is_cloud_environment", return_value=True)
    def test_for_environment_cloud(self, mock_cloud, mock_hpc, mock_sysconf):
        def sysconf_mock(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return (16 * (1024**3)) // 4096
            raise ValueError()

        mock_sysconf.side_effect = sysconf_mock

        config = CacheConfig.for_environment()
        assert config.max_files == 500
        assert config.max_memory_mb == int(16 * 1024 * 0.2)
        assert config.disk_cache_gb == 20
        assert config.prefetch_enabled is True

    @patch("torchfits.cache.os.sysconf")
    @patch.object(CacheConfig, "_is_hpc_environment", return_value=False)
    @patch.object(CacheConfig, "_is_cloud_environment", return_value=False)
    @patch.object(CacheConfig, "_is_gpu_environment", return_value=True)
    def test_for_environment_gpu(self, mock_gpu, mock_cloud, mock_hpc, mock_sysconf):
        def sysconf_mock(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return (32 * (1024**3)) // 4096
            raise ValueError()

        mock_sysconf.side_effect = sysconf_mock

        config = CacheConfig.for_environment()
        assert config.max_files == 200
        assert config.max_memory_mb == int(32 * 1024 * 0.4)
        assert config.disk_cache_gb == 30
        assert config.prefetch_enabled is True

    @patch("torchfits.cache.os.sysconf")
    @patch.object(CacheConfig, "_is_hpc_environment", return_value=False)
    @patch.object(CacheConfig, "_is_cloud_environment", return_value=False)
    @patch.object(CacheConfig, "_is_gpu_environment", return_value=False)
    def test_for_environment_default(
        self, mock_gpu, mock_cloud, mock_hpc, mock_sysconf
    ):
        def sysconf_mock(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return (8 * (1024**3)) // 4096
            raise ValueError()

        mock_sysconf.side_effect = sysconf_mock

        config = CacheConfig.for_environment()
        assert config.max_files == 100
        assert config.max_memory_mb == min(2048, int(8 * 1024 * 0.1))
        assert config.disk_cache_gb == 5
        assert config.prefetch_enabled is False

    @patch.dict(os.environ, {"SLURM_JOB_ID": "12345"})
    def test_is_hpc_environment_true(self):
        assert CacheConfig._is_hpc_environment() is True

    @patch.dict(os.environ, clear=True)
    def test_is_hpc_environment_false(self):
        assert CacheConfig._is_hpc_environment() is False

    @patch.dict(os.environ, {"AWS_EXECUTION_ENV": "AWS_Lambda"})
    def test_is_cloud_environment_true(self):
        assert CacheConfig._is_cloud_environment() is True

    @patch.dict(os.environ, clear=True)
    def test_is_cloud_environment_false(self):
        assert CacheConfig._is_cloud_environment() is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    def test_is_gpu_environment_true(self, mock_count, mock_available):
        assert CacheConfig._is_gpu_environment() is True

    @patch("torch.cuda.is_available", return_value=False)
    def test_is_gpu_environment_false_not_available(self, mock_available):
        assert CacheConfig._is_gpu_environment() is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=0)
    def test_is_gpu_environment_false_no_devices(self, mock_count, mock_available):
        assert CacheConfig._is_gpu_environment() is False


class TestCacheOptimization:
    """Test CacheOptimization functionality."""

    def test_optimize_for_dataset_small(self):
        # Create a test configuration
        config = CacheConfig(disk_cache_gb=10)
        manager = torchfits.cache.CacheManager(config)

        # Mock get_cache_manager to return our specific manager
        with patch("torchfits.cache.get_cache_manager", return_value=manager):
            # Test a small dataset that fits easily in cache
            # 100 files, 10 MB each = ~1 GB total (< 10 GB limit)
            file_paths = ["file_{}.fits".format(i) for i in range(100)]
            torchfits.cache.optimize_for_dataset(file_paths, avg_file_size_mb=10.0)

            # Verification
            # Should enable aggressive caching for the entire dataset
            assert manager.config.max_files == 100
            assert manager.config.prefetch_enabled is True

    def test_optimize_for_dataset_large(self):
        # Create a test configuration
        config = CacheConfig(disk_cache_gb=10)
        manager = torchfits.cache.CacheManager(config)

        # Mock get_cache_manager to return our specific manager
        with patch("torchfits.cache.get_cache_manager", return_value=manager):
            # Test a large dataset that exceeds cache limit
            # 200 files, 100 MB each = ~19.5 GB total (> 10 GB limit)
            file_paths = ["file_{}.fits".format(i) for i in range(200)]
            torchfits.cache.optimize_for_dataset(file_paths, avg_file_size_mb=100.0)

            # Verification
            # Should restrict max files to optimal_files
            # optimal_files = int(10 * 1024 / 100) = 102
            # min(102, 1000) = 102
            assert manager.config.max_files == 102
            # prefetch_enabled should not be forced to True in this branch
            # (assuming default was False, or at least it doesn't change it)

    def test_optimize_for_dataset_huge_many_files(self):
        # Create a test configuration
        config = CacheConfig(disk_cache_gb=100)
        manager = torchfits.cache.CacheManager(config)

        # Mock get_cache_manager to return our specific manager
        with patch("torchfits.cache.get_cache_manager", return_value=manager):
            # Test a very large number of files that exceeds cache limit
            # optimal_files = int(100 * 1024 / 10) = 10240
            # max_files is capped at 1000
            file_paths = ["file_{}.fits".format(i) for i in range(20000)]
            torchfits.cache.optimize_for_dataset(file_paths, avg_file_size_mb=10.0)

            # Verification
            assert manager.config.max_files == 1000


class TestCacheManagerFunctions:
    """Test cache manager module-level functions."""

    def test_clear_cache(self):
        from torchfits.cache import get_cache_manager, clear_cache, get_cache_stats

        # The manager no longer keeps fake never-updated counters; hits/misses
        # come straight from the I/O engine. Clearing must reset them.
        manager = get_cache_manager()
        clear_cache()
        stats_before = get_cache_stats()
        assert stats_before["hits"] == 0
        assert stats_before["misses"] == 0
        assert not hasattr(manager, "_stats")


def test_cached_reads_are_isolated_from_caller_mutation(tmp_path):
    """In-place mutation of a returned result must not poison the cache.

    Regression: cached read results were handed out (and stored) by
    reference, so ``result['flux'].mul_(2)`` silently corrupted every later
    read served from the default on-by-default cache.
    """
    import torch

    import torchfits

    table_path = tmp_path / "alias.fits"
    torchfits.write(
        table_path.as_posix(),
        {"flux": torch.arange(10, dtype=torch.float32)},
        overwrite=True,
    )
    image_path = tmp_path / "img.fits"
    torchfits.write_tensor(image_path.as_posix(), torch.ones(4, 4), overwrite=True)

    expected = torch.arange(10, dtype=torch.float32)

    d1 = torchfits.read(table_path.as_posix(), hdu=1)
    d1["flux"].mul_(100)
    assert torch.equal(torchfits.read(table_path.as_posix(), hdu=1)["flux"], expected)

    d3 = torchfits.read(table_path.as_posix(), hdu=1)
    d3["flux"].mul_(-1)
    assert torch.equal(torchfits.read(table_path.as_posix(), hdu=1)["flux"], expected)

    i1 = torchfits.read(image_path.as_posix())
    i1.add_(5)
    assert torch.equal(torchfits.read(image_path.as_posix()), torch.ones(4, 4))


class TestDeprecatedNoOpContract:
    """Option-A cache no-ops must warn only on explicit user calls.

    Regression (r1a-01): ``get_cache_manager`` / ``configure_for_environment``
    / ``optimize_for_dataset`` invoked the deprecated ``configure_cpp_cache``
    no-op internally, so every process's first ``torchfits`` I/O call emitted a
    ``DeprecationWarning`` — raised as an error under the common
    ``-W error::DeprecationWarning`` / pytest ``filterwarnings = error``
    configurations.
    """

    def test_configure_for_environment_emits_no_deprecation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torchfits.cache.configure_for_environment()
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep == []

    def test_get_cache_manager_creation_emits_no_deprecation(self, monkeypatch):
        monkeypatch.setattr(torchfits.cache, "_cache_manager", None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager = torchfits.cache.get_cache_manager()
        assert isinstance(manager, torchfits.cache.CacheManager)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep == []

    def test_optimize_for_dataset_emits_no_deprecation(self, monkeypatch):
        manager = torchfits.cache.CacheManager(CacheConfig(disk_cache_gb=10))
        monkeypatch.setattr(torchfits.cache, "get_cache_manager", lambda: manager)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torchfits.cache.optimize_for_dataset(["a.fits", "b.fits"], 1.0)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep == []

    def test_explicit_deprecated_calls_still_warn(self):
        """The deprecation contract for explicit user calls is preserved."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torchfits.cache.get_cache_manager().configure_cpp_cache()
            torchfits.cache.configure_cache(1, 2, 3)
        messages = [
            str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert any("configure_cpp_cache" in m for m in messages)
        assert any("configure_cache" in m for m in messages)

    def test_configure_cache_is_documented_noop(self):
        """``configure_cache`` is a documented no-op (changelog 1.1.2): it must
        not replace the global cache manager with phantom-knob config (which
        also raced ``get_cache_manager``'s lock-free read side)."""
        before = torchfits.cache.get_cache_manager()
        stats_before = torchfits.cache.get_cache_stats()["config"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torchfits.cache.configure_cache(7, 8, 9)
        assert torchfits.cache.get_cache_manager() is before
        assert torchfits.cache.get_cache_stats()["config"] == stats_before

    def test_cache_stats_surface_engine_errors(self, monkeypatch):
        """Stats aggregation must not silently swallow engine failures."""
        import torchfits._io_engine.caches as caches

        def boom() -> dict:
            raise RuntimeError("engine broken")

        monkeypatch.setattr(caches, "get_cache_performance", boom)
        with pytest.raises(RuntimeError, match="engine broken"):
            torchfits.cache.get_cache_stats()
        with pytest.raises(RuntimeError, match="engine broken"):
            torchfits.cache.stats()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Deep-review unit 10, TE-002: the selective contract of clear_file_cache
# ---------------------------------------------------------------------------
#
# `clear_file_cache` takes six keyword-only flags -- the entire reason it
# accepts arguments at all -- and `docs/api-core-io.md` documents each one.
# Nothing pinned them. Verified by breaking the implementation: making
# `clear_python_caches` ignore all five Python-side keywords, and making
# `cpp=False` still clear the native cache, each left the whole cache-related
# test selection green (122 passed, 1 skipped).
#
# The caches are module-level OrderedDicts, so the flag -> cache mapping is
# asserted directly. That is deliberately implementation-coupled: the contract
# under test *is* which store each flag clears.


def _fill_all_caches() -> None:
    from torchfits._io_engine import caches

    caches.file_cache[("data",)] = None
    caches.image_meta_cache[("meta", 0)] = None
    caches.header_cards_cache[("meta", 0)] = None
    caches.cold_nommap_cache[("meta", 0)] = None
    caches.auto_mmap_cache[("meta", 0)] = None
    caches.hdu_type_cache[("hdu", 0)] = None
    caches.auto_hdu_cache[("auto",)] = None
    caches.cache_stats["hits"] += 7


_ALL_CACHES = (
    "file_cache",
    "image_meta_cache",
    "header_cards_cache",
    "cold_nommap_cache",
    "auto_mmap_cache",
    "hdu_type_cache",
    "auto_hdu_cache",
)


@pytest.mark.parametrize(
    ("flags", "preserved"),
    [
        ({"data": False}, ("file_cache",)),
        (
            {"meta": False},
            (
                "image_meta_cache",
                "header_cards_cache",
                "cold_nommap_cache",
                "auto_mmap_cache",
            ),
        ),
        ({"hdu_types": False}, ("hdu_type_cache", "auto_hdu_cache")),
    ],
)
def test_clear_file_cache_preserves_only_the_named_caches(flags, preserved):
    """Each keyword must gate exactly the caches it is documented to gate."""
    from torchfits._io_engine import caches

    torchfits.clear_file_cache()
    _fill_all_caches()
    torchfits.clear_file_cache(cpp=False, **flags)

    for name in _ALL_CACHES:
        size = len(getattr(caches, name))
        if name in preserved:
            assert size == 1, f"{flags}: {name} must be preserved, has {size} entries"
        else:
            assert size == 0, f"{flags}: {name} must be cleared, has {size} entries"
    # `stats` was not named, so it is reset.
    assert caches.cache_stats["hits"] == 0, f"{flags}: stats must reset"


def test_clear_file_cache_stats_false_keeps_counters():
    """``stats=False`` is the one flag that preserves rather than clears."""
    from torchfits._io_engine import caches

    torchfits.clear_file_cache()
    _fill_all_caches()
    torchfits.clear_file_cache(cpp=False, stats=False)

    assert caches.cache_stats["hits"] == 7, "stats=False must keep the counters"
    for name in _ALL_CACHES:
        assert len(getattr(caches, name)) == 0, f"{name} must still be cleared"


def test_clear_file_cache_cpp_flag_is_honoured():
    """``cpp=False`` must not touch the native cache, ``cpp=True`` must."""
    from unittest import mock

    fake_cpp = mock.MagicMock()

    torchfits.clear_file_cache(cpp=False, cpp_module=fake_cpp)
    fake_cpp.clear_shared_read_meta_cache.assert_not_called()

    torchfits.clear_file_cache(cpp=True, cpp_module=fake_cpp)
    fake_cpp.clear_shared_read_meta_cache.assert_called_once()


# --------------------------------------------------------------------------
# TS-014: `cache_subsystem_policy` is documented in `docs/api-core-io.md` and
# had no test. TS-002 pinned `clear_file_cache`'s *selective* keyword contract;
# this is the same contract one layer up -- the policy table that decides which
# keywords a named subsystem clears -- and it was equally unpinned. A regression
# that made `fits_header_metadata` clear `data` too, or clear nothing, would
# have been silent.
# --------------------------------------------------------------------------

# The measured table: subsystem name -> the flags it actually enables.
_POLICY_FLAGS = {
    "fits_image_data": {"data"},
    "fits_table_data": {"data"},
    "fits_header_metadata": {"meta", "hdu_types"},
    "fits_header_hdu_metadata": {"meta", "hdu_types"},
}


def _enabled(flags: dict) -> set:
    return {name for name, on in flags.items() if on}


@pytest.mark.parametrize("name, expected", sorted(_POLICY_FLAGS.items()))
def test_each_subsystem_enables_exactly_its_own_flags(name, expected):
    """The selective part: clearing `fits_image_data` must not touch metadata."""
    from torchfits.io import cache_subsystem_policy

    flags = cache_subsystem_policy(name)
    assert _enabled(flags) == expected, flags
    # `handles` is never enabled: the handle cache was removed, and
    # `clear_cache_subsystem` deliberately does not forward it.
    assert flags["handles"] is False


def test_all_subsystem_enables_every_flag():
    from torchfits.io import cache_subsystem_policy

    flags = cache_subsystem_policy("all")
    assert all(flags.values()), flags
    assert set(flags) == {"data", "handles", "meta", "hdu_types", "stats", "cpp"}


def test_policy_returns_a_copy_not_the_shared_table():
    """Mutating the result must not corrupt the policy for the next caller."""
    from torchfits.io import cache_subsystem_policy

    first = cache_subsystem_policy("fits_image_data")
    first["data"] = False
    first["injected"] = True

    second = cache_subsystem_policy("fits_image_data")
    assert second["data"] is True
    assert "injected" not in second


def test_unknown_subsystem_lists_the_valid_names():
    """The error is the only place the four non-"all" names are discoverable."""
    from torchfits.io import cache_subsystem_policy

    with pytest.raises(KeyError) as err:
        cache_subsystem_policy("bogus")
    message = str(err.value)
    assert "bogus" in message
    for name in ["all", *_POLICY_FLAGS]:
        assert name in message, f"{name} missing from the error message"


@pytest.mark.parametrize("flag_name", ["data", "meta", "hdu_types", "handles"])
def test_clear_file_cache_flag_names_are_not_subsystem_names(flag_name):
    """A namespace trap worth pinning: the *keyword* names are not *subsystems*.

    `clear_file_cache(data=...)` takes these words as keywords, so
    `cache_subsystem_policy("data")` is the natural-looking call -- and it
    raises. The valid names are the `fits_*` ones.
    """
    from torchfits.io import cache_subsystem_policy

    with pytest.raises(KeyError):
        cache_subsystem_policy(flag_name)


def test_clear_cache_subsystem_forwards_the_policy_flags():
    """The policy is only useful if `clear_cache_subsystem` actually applies it."""
    from unittest import mock

    from torchfits._io_engine import caches

    with mock.patch.object(caches, "clear_file_cache") as clear:
        caches.clear_cache_subsystem("fits_header_metadata")
    kwargs = clear.call_args.kwargs
    assert kwargs["meta"] is True
    assert kwargs["hdu_types"] is True
    assert kwargs["data"] is False
    assert "handles" not in kwargs, "the removed handle cache must not be forwarded"
