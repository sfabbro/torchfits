"""Tests for FitsStagedCutoutIterableDataset and ephemeral scratch staging."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch
from astropy.io import fits
from torch.utils.data import DataLoader

from torchfits.data import FitsStagedCutoutIterableDataset
from torchfits.data.remote import ephemeral_scratch_dir


@pytest.fixture(autouse=True)
def _clean_prefetch_state():
    import torchfits.data.remote as remote

    with remote._prefetch_lock:
        threads = list(remote._prefetch_threads.values())
    for t in threads:
        t.join(timeout=1.0)
    with remote._prefetch_lock:
        remote._prefetch_threads.clear()
        remote._prefetch_errors.clear()
    yield
    with remote._prefetch_lock:
        threads = list(remote._prefetch_threads.values())
    for t in threads:
        t.join(timeout=1.0)
    with remote._prefetch_lock:
        remote._prefetch_threads.clear()
        remote._prefetch_errors.clear()


@pytest.fixture
def mosaic_fits_files(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"mosaic_{i}.fits"
        data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100) + (i * 10000)
        fits.PrimaryHDU(data).writeto(str(p), overwrite=True)
        paths.append(str(p))
    return paths


def test_staged_cutout_local_files(mosaic_fits_files):
    """Test extracting cutouts from local mosaic files."""
    ds = FitsStagedCutoutIterableDataset(
        mosaic_fits_files,
        cutouts_per_file=5,
        cutout_size=32,
        device="cpu",
    )
    cutouts = list(ds)
    assert len(cutouts) == 3 * 5
    for c in cutouts:
        assert isinstance(c, torch.Tensor)
        assert c.shape == (1, 32, 32)


def test_staged_cutout_dataloader_collate(mosaic_fits_files):
    """Test DataLoader batching with FitsStagedCutoutIterableDataset."""
    ds = FitsStagedCutoutIterableDataset(
        mosaic_fits_files,
        cutouts_per_file=4,
        cutout_size=(16, 16),
        device="cpu",
    )
    loader = DataLoader(ds, batch_size=4)
    batches = list(loader)
    assert len(batches) == 3
    for b in batches:
        assert b.shape == (4, 1, 16, 16)


def test_staged_cutout_custom_generator(mosaic_fits_files):
    """Test custom cutout coordinate generator."""

    def fixed_coords(h, w, ch, cw):
        return 0, 0, ch, cw

    ds = FitsStagedCutoutIterableDataset(
        mosaic_fits_files[:1],
        cutouts_per_file=3,
        cutout_size=10,
        cutout_generator=fixed_coords,
        add_channel_dim=False,
    )
    stamps = list(ds)
    assert len(stamps) == 3
    for s in stamps:
        assert s.shape == (10, 10)
        assert s[0, 0] == 0.0


def test_staged_cutout_remote_mock_and_cleanup(tmp_path):
    """Test simulated remote URLs with ephemeral staging and automatic post-sampling cleanup."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    real_mosaic = tmp_path / "real_mosaic.fits"
    fits.PrimaryHDU(np.zeros((50, 50), dtype=np.float32)).writeto(
        str(real_mosaic), overwrite=True
    )

    def _mock_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(str(real_mosaic), str(dest))
        return dest

    with mock.patch("torchfits.data.remote._download", side_effect=_mock_download):
        remote_urls = [
            "https://archive.example.org/mosaics/tile1.fits",
            "https://archive.example.org/mosaics/tile2.fits",
        ]
        ds = FitsStagedCutoutIterableDataset(
            remote_urls,
            cutouts_per_file=2,
            cutout_size=16,
            staging_dir=scratch,
            cleanup=True,
        )

        stamps = list(ds)
        assert len(stamps) == 4

        # Verify that temporary downloaded files were cleaned up
        remaining = list(scratch.glob("*.fits"))
        assert len(remaining) == 0


def test_ephemeral_scratch_dir(monkeypatch, tmp_path):
    slurm_dir = tmp_path / "slurm_scratch"
    monkeypatch.setenv("SLURM_TMPDIR", str(slurm_dir))
    scratch = ephemeral_scratch_dir()
    assert str(slurm_dir) in str(scratch)
    assert scratch.is_dir()
