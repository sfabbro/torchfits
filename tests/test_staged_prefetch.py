"""Tests for FitsStagedCutoutIterableDataset and ephemeral scratch staging."""

from __future__ import annotations

from pathlib import Path
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

        tmp = dest.with_suffix(dest.suffix + ".partial")
        shutil.copy(str(real_mosaic), str(tmp))
        tmp.replace(dest)
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


def test_staged_cutout_multi_hdu_and_companions(tmp_path):
    """Test multi-HDU channel stacking and companion ivar/mask payloads."""
    from torchfits.transforms import InterquantileScale

    path = tmp_path / "multihdu_mosaic.fits"
    sci0 = np.ones((60, 60), dtype=np.float32) * 10.0
    sci1 = np.ones((60, 60), dtype=np.float32) * 20.0
    ivar0 = np.ones((60, 60), dtype=np.float32) * 0.1
    ivar1 = np.ones((60, 60), dtype=np.float32) * 0.2
    mask0 = np.ones((60, 60), dtype=np.int16)
    mask1 = np.ones((60, 60), dtype=np.int16)

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(sci0),
            fits.ImageHDU(sci1, name="SCI1"),
            fits.ImageHDU(ivar0, name="IVAR0"),
            fits.ImageHDU(ivar1, name="IVAR1"),
            fits.ImageHDU(mask0, name="MASK0"),
            fits.ImageHDU(mask1, name="MASK1"),
        ]
    )
    hdul.writeto(str(path), overwrite=True)

    # 1. Multi-HDU flux stacking without companions
    ds_flux = FitsStagedCutoutIterableDataset(
        str(path),
        hdu=[0, 1],
        cutouts_per_file=2,
        cutout_size=16,
    )
    samples_flux = list(ds_flux)
    assert len(samples_flux) == 2
    assert samples_flux[0].shape == (2, 16, 16)
    assert (samples_flux[0][0] == 10.0).all()
    assert (samples_flux[0][1] == 20.0).all()

    # 2. Multi-HDU flux with companion ivar and mask + InterquantileScale
    scaler = InterquantileScale(zero_preserving=True)
    ds_comp = FitsStagedCutoutIterableDataset(
        str(path),
        hdu=[0, 1],
        ivar_hdu=[2, 3],
        mask_hdu=[4, 5],
        cutouts_per_file=2,
        cutout_size=16,
        transform=scaler,
    )
    samples_comp = list(ds_comp)
    assert len(samples_comp) == 2
    payload = samples_comp[0]
    assert isinstance(payload, dict)
    assert "flux" in payload and "ivar" in payload and "mask" in payload
    assert payload["flux"].shape == (2, 16, 16)
    assert payload["ivar"].shape == (2, 16, 16)
    assert payload["mask"].shape == (2, 16, 16)
    # Color ratio 20.0 / 10.0 == 2.0 must be preserved after scaling
    ratio = payload["flux"][1] / payload["flux"][0]
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0))


def test_make_loader_staged_remote_downloads_once(tmp_path, monkeypatch):
    """make_loader must not fetch staged remotes a second time.

    ``FitsStagedCutoutIterableDataset`` stages its own copies under
    ``staging_dir`` and never reads the shared remote cache, so make_loader's
    full-file prefetch would download each remote twice (once into the remote
    cache, once into staging) and leave a dead copy behind.
    """
    import shutil

    from torchfits.data import make_loader

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(tmp_path / "remote_cache"))

    real_mosaic = tmp_path / "real_mosaic.fits"
    fits.PrimaryHDU(np.zeros((50, 50), dtype=np.float32)).writeto(
        str(real_mosaic), overwrite=True
    )

    calls: list[tuple[str, Path]] = []

    def _mock_download(url, dest):
        calls.append((url, Path(dest)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        shutil.copy(str(real_mosaic), str(tmp))
        tmp.replace(dest)
        return dest

    with mock.patch("torchfits.data.remote._download", side_effect=_mock_download):
        ds = FitsStagedCutoutIterableDataset(
            ["https://archive.example.org/mosaics/only-tile.fits"],
            cutouts_per_file=2,
            cutout_size=16,
            staging_dir=scratch,
            cleanup=True,
        )
        loader = make_loader(ds, batch_size=2, num_workers=0)
        n = sum(batch.shape[0] for batch in loader)

    assert n == 2
    # Deterministic count proof: exactly one download per remote URL, and it
    # lands in the dataset's own staging directory.
    assert len(calls) == 1, calls
    assert scratch in calls[0][1].parents
    assert not (tmp_path / "remote_cache").exists()


def test_concurrent_staged_iterators_do_not_yank_each_others_files(
    tmp_path, monkeypatch
):
    """Two iterators sharing a staging dir must not delete files out from
    under each other's resolve->open window (r3b-08 regression).

    Iterator A resolves its staged copy (existence-checked path), then a
    concurrent iterator B runs the same file to completion and ``cleanup=True``
    deletes it before A's ``open_subset_reader`` opens it -> A dies mid-epoch.
    The window is pinned deterministically by wedging A's first open until B
    has finished. Fixed by per-iterator private staging dirs
    (``FitsStagedCutoutIterableDataset._stage_root``).
    """
    import shutil
    import threading

    import torchfits.io

    real_mosaic = tmp_path / "real_mosaic.fits"
    fits.PrimaryHDU(np.zeros((50, 50), dtype=np.float32)).writeto(
        str(real_mosaic), overwrite=True
    )

    def _mock_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        shutil.copy(str(real_mosaic), str(tmp))
        tmp.replace(dest)
        return dest

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    url = "https://archive.example.org/mosaics/shared-tile.fits"
    mk = dict(
        cutouts_per_file=2,
        cutout_size=16,
        staging_dir=scratch,
        cleanup=True,
    )
    ds_a = FitsStagedCutoutIterableDataset([url], **mk)
    ds_b = FitsStagedCutoutIterableDataset([url], **mk)

    real_open = torchfits.io.open_subset_reader
    a_opening = threading.Event()
    b_finished = threading.Event()
    state = {"wedged": False}

    def wedged_open(path, hdu=0, device="cpu"):
        if not state["wedged"]:
            state["wedged"] = True
            a_opening.set()
            assert b_finished.wait(timeout=10)
        return real_open(path, hdu=hdu, device=device)

    monkeypatch.setattr(torchfits.io, "open_subset_reader", wedged_open)

    errors: list[BaseException] = []
    got_a: list = []

    def run_a():
        try:
            got_a.extend(list(ds_a))
        except BaseException as exc:  # noqa: BLE001 - recorded for assertion
            errors.append(exc)

    with mock.patch("torchfits.data.remote._download", side_effect=_mock_download):
        thread_a = threading.Thread(target=run_a)
        thread_a.start()
        assert a_opening.wait(timeout=10)
        # B stages the same shared file, reads it, and cleanup=True deletes it
        # while A sits in its resolve->open window.
        got_b = list(ds_b)
        b_finished.set()
        thread_a.join(timeout=30)

    assert not thread_a.is_alive()
    assert len(got_b) == 2
    assert errors == [], f"concurrent cleanup yanked A's staged file: {errors}"
    assert len(got_a) == 2
