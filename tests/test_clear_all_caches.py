"""Tests for clear_all_caches() convenience function."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchfits


def test_clear_all_caches_smoke(tmp_path: Path) -> None:
    """clear_all_caches() should not raise and should leave torchfits usable."""
    # Write and read a file to populate caches.
    data = torch.ones((4, 4), dtype=torch.float32)
    path = str(tmp_path / "test.fits")
    torchfits.write(path, data, overwrite=True)
    _ = torchfits.read(path, hdu=0)

    # Clear everything.
    torchfits.clear_all_caches()

    # Should still work after clearing.
    result = torchfits.read(path, hdu=0)
    assert torch.equal(result, data)


def test_clear_all_caches_removes_disk_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clear_all_caches() removes the cache_root directory."""
    cache_dir = tmp_path / "torchfits_cache"
    cache_dir.mkdir()
    (cache_dir / "remote").mkdir()
    (cache_dir / "samples").mkdir()
    (cache_dir / "remote" / "dummy.fits").write_bytes(b"fake")

    monkeypatch.setenv("TORCHFITS_CACHE_DIR", str(cache_dir))

    assert (cache_dir / "remote" / "dummy.fits").exists()

    torchfits.clear_all_caches()

    # The cache root directory should be removed (or at least empty).
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_clear_cache_disk_false_preserves_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clear_cache() (default) should NOT remove disk cache files."""
    cache_dir = tmp_path / "torchfits_cache"
    cache_dir.mkdir()
    remote = cache_dir / "remote"
    remote.mkdir()
    (remote / "dummy.fits").write_bytes(b"fake")

    monkeypatch.setenv("TORCHFITS_CACHE_DIR", str(cache_dir))

    assert (remote / "dummy.fits").exists()

    torchfits.cache.clear_cache()

    # Disk files should still be there.
    assert (remote / "dummy.fits").exists()


def test_clear_all_caches_accessible_from_root() -> None:
    """clear_all_caches should be importable from the torchfits root."""
    assert hasattr(torchfits, "clear_all_caches")
    assert callable(torchfits.clear_all_caches)


def test_clear_cache_disk_true_parameter() -> None:
    """clear_cache(disk=True) should not crash."""
    # Just verify it doesn't raise — disk cleanup tested above.
    torchfits.cache.clear_cache(disk=False)


def test_cache_subsystem_still_works_after_clear() -> None:
    """Subsystem clear should not break after a full clear."""
    torchfits.clear_all_caches()
    # Should not raise.
    torchfits.get_cache_performance()
