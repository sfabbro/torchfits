"""Remote prefetch cache + Dataset peer taxonomy smoke checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


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
def image_fits(tmp_path):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.arange(16, dtype=np.float32).reshape(4, 4)).writeto(
        str(path), overwrite=True
    )
    return path


@pytest.fixture
def spectrum_1d_fits(tmp_path):
    path = tmp_path / "spec1d.fits"
    fits.PrimaryHDU(np.linspace(0, 1, 32, dtype=np.float32)).writeto(
        str(path), overwrite=True
    )
    return path


@pytest.fixture
def desi_shaped_fits(tmp_path):
    """Tiny DESI-like MEF: B/R arms with unequal nwave + IVAR companions."""
    path = tmp_path / "spectra-fake.fits"
    hdus = [fits.PrimaryHDU()]
    for name, nwave in (
        ("B_FLUX", 10),
        ("B_IVAR", 10),
        ("R_FLUX", 12),
        ("R_IVAR", 12),
    ):
        data = np.arange(3 * nwave, dtype=np.float32).reshape(3, nwave)
        hdus.append(fits.ImageHDU(data, name=name))
    fits.HDUList(hdus).writeto(str(path), overwrite=True)
    return path


def test_cache_root_env_override(tmp_path, monkeypatch):
    from torchfits.cache import cache_root, remote_cache_root, sample_cache_root

    root = tmp_path / "tf-cache"
    monkeypatch.setenv("TORCHFITS_CACHE_DIR", str(root))
    monkeypatch.delenv("TORCHFITS_REMOTE_CACHE", raising=False)
    monkeypatch.delenv("TORCHFITS_SAMPLE_CACHE", raising=False)
    assert cache_root() == root
    assert remote_cache_root() == root / "remote"
    assert sample_cache_root() == root / "samples"


def test_cache_root_xdg(tmp_path, monkeypatch):
    from torchfits.cache import cache_root

    monkeypatch.delenv("TORCHFITS_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "xdg" / "torchfits"


def test_resolve_local_path_waits_for_inflight_prefetch(tmp_path, monkeypatch):
    """resolve_local_path must not race a concurrent prefetch for the same URL.

    Regression: prefetch_urls() starts a background download for the
    make_loader/Dataset "read ahead" window; resolve_local_path() used to
    ignore that in-flight download and start a second, concurrent download
    to the same temp file whenever the caller reached that file before the
    prefetch finished (duplicate work, and a real corruption risk since both
    downloads write the same ".partial" path).
    """
    import threading
    from unittest import mock

    import torchfits.data.remote as remote

    calls: list[str] = []
    started = threading.Event()
    release = threading.Event()
    # Set when resolve_local_path observes the live prefetch thread and is
    # about to join it. The download stays blocked until that happens, so
    # ordering does not depend on sleep.
    observed_inflight = threading.Event()

    def _slow_download(url, dest):
        calls.append(url)
        started.set()
        if not release.wait(timeout=2):
            raise AssertionError("in-flight prefetch was not released")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("data")
        return dest

    orig_is_alive = threading.Thread.is_alive

    def _is_alive(self) -> bool:
        alive = orig_is_alive(self)
        if alive and self.name == "torchfits-prefetch":
            observed_inflight.set()
        return alive

    url = "https://example.test/warm-cache.fits"
    outputs: list[str] = []
    monkeypatch.setattr(threading.Thread, "is_alive", _is_alive)
    with mock.patch.object(remote, "_download", side_effect=_slow_download):
        remote.prefetch_urls([url], cache_dir=tmp_path)
        assert started.wait(timeout=1), "prefetch download did not start"
        resolver = threading.Thread(
            target=lambda: outputs.append(
                remote.resolve_local_path(url, cache_dir=tmp_path)
            )
        )
        resolver.start()
        try:
            assert observed_inflight.wait(timeout=1), (
                "resolve_local_path did not observe the in-flight prefetch"
            )
        finally:
            release.set()
            resolver.join(timeout=2)

    assert not resolver.is_alive()
    assert len(calls) == 1, f"expected exactly one download, got {len(calls)}"
    assert len(outputs) == 1
    assert Path(outputs[0]).read_text() == "data"


def test_concurrent_resolve_local_path_downloads_once(tmp_path, monkeypatch):
    import threading
    from unittest import mock

    import torchfits.data.remote as remote

    calls: list[str] = []
    started = threading.Event()
    release = threading.Event()
    # Second resolve has entered _download_once while the first download is
    # still blocked, so release is not a stand-in for scheduler delay.
    second_entered = threading.Event()

    def _slow_download(url, dest):
        calls.append(url)
        started.set()
        if not release.wait(timeout=2):
            raise AssertionError("in-flight download was not released")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("data")
        return dest

    real_download_once = remote._download_once
    entrants = 0
    entrants_lock = threading.Lock()

    def _download_once(cache_key, url, dest):
        nonlocal entrants
        with entrants_lock:
            entrants += 1
            if entrants >= 2:
                second_entered.set()
        return real_download_once(cache_key, url, dest)

    url = "https://example.test/cold-cache.fits"
    outputs: list[str] = []
    with (
        mock.patch.object(remote, "_download_once", side_effect=_download_once),
        mock.patch.object(remote, "_download", side_effect=_slow_download),
    ):
        first = threading.Thread(
            target=lambda: outputs.append(
                remote.resolve_local_path(url, cache_dir=tmp_path)
            )
        )
        second = threading.Thread(
            target=lambda: outputs.append(
                remote.resolve_local_path(url, cache_dir=tmp_path)
            )
        )
        first.start()
        assert started.wait(timeout=1), "first download did not start"
        second.start()
        try:
            assert second_entered.wait(timeout=1), (
                "second resolve did not reach the in-flight download"
            )
        finally:
            release.set()
            first.join(timeout=2)
            second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(calls) == 1
    assert len(outputs) == 2
    assert Path(outputs[0]).read_text() == "data"


def test_remote_cache_path_stable(tmp_path, monkeypatch):
    from torchfits.data.remote import (
        cache_path_for_url,
        is_http_url,
        resolve_local_path,
    )

    monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(tmp_path))
    url = "https://example.edu/data/sample.fits"
    assert is_http_url(url)
    a = cache_path_for_url(url)
    b = cache_path_for_url(url)
    assert a == b
    assert a.parent == tmp_path
    assert Path(resolve_local_path(url, download=False)) == a


def test_fits_tensor_dataset_local(image_fits):
    from torchfits.data import FitsCubeDataset, FitsTensorDataset

    ds = FitsTensorDataset([str(image_fits)], labels=[1], add_channel_dim=True)
    image, label = ds[0]
    assert image.ndim == 3
    assert int(label) == 1

    cube = FitsCubeDataset([str(image_fits)], labels=[0])
    t, _ = cube[0]
    assert t.ndim >= 2


def test_fits_spectrum_1d(spectrum_1d_fits):
    from torchfits.data import FitsSpectrumDataset

    spec = FitsSpectrumDataset([str(spectrum_1d_fits)])
    payload = spec[0]
    assert payload["flux"].ndim == 1
    assert payload["flux"].shape[0] == 32


def test_fits_image_dataset_peer(image_fits):
    from torchfits.data import FitsImageDataset

    ds = FitsImageDataset([str(image_fits)], labels=[0])
    assert "FitsImageDataset" in repr(ds)
    image, _ = ds[0]
    assert image.ndim == 3


def test_desi_shaped_spectrum_layouts(desi_shaped_fits):
    from torchfits.data import FitsSpectrumDataset

    path = str(desi_shaped_fits)
    arms = FitsSpectrumDataset(
        [path],
        hdu=["B_FLUX", "R_FLUX"],
        ivar_hdu=["B_IVAR", "R_IVAR"],
        row=1,
        layout="dict",
    )[0]
    assert set(arms) == {"B_FLUX", "R_FLUX"}
    assert arms["B_FLUX"]["flux"].shape == (10,)
    assert arms["R_FLUX"]["ivar"].shape == (12,)

    concat = FitsSpectrumDataset(
        [path],
        hdu=["B_FLUX", "R_FLUX"],
        ivar_hdu=["B_IVAR", "R_IVAR"],
        row=0,
        layout="concat",
    )[0]
    assert concat["flux"].shape == (22,)
    assert concat["ivar"].shape == (22,)

    with pytest.raises(ValueError, match="equal nwave"):
        _ = FitsSpectrumDataset(
            [path],
            hdu=["B_FLUX", "R_FLUX"],
            row=0,
            layout="stack",
        )[0]


def test_multi_hdu_flux_ivar_companions(tmp_path):
    from torchfits.data import FitsImageDataset

    path = tmp_path / "bands.fits"
    hdus = [fits.PrimaryHDU()]
    for name in ("G", "R", "G_IVAR", "R_IVAR"):
        hdus.append(fits.ImageHDU(np.ones((4, 4), dtype=np.float32), name=name))
    fits.HDUList(hdus).writeto(str(path), overwrite=True)

    payload, _ = FitsImageDataset(
        [str(path)],
        hdu=["G", "R"],
        ivar_hdu=["G_IVAR", "R_IVAR"],
        labels=[0],
        add_channel_dim=False,
    )[0]
    assert isinstance(payload, dict)
    assert payload["flux"].shape == (2, 4, 4)
    assert payload["ivar"].shape == (2, 4, 4)


def test_fits_cube_iterable_dataset(tmp_path):
    from torchfits.data import FitsCubeIterableDataset

    path = tmp_path / "cube.fits"
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    fits.PrimaryHDU(data).writeto(str(path), overwrite=True)

    # 1. Full cube iterable
    ds_full = FitsCubeIterableDataset([str(path)])
    items_full = list(ds_full)
    assert len(items_full) == 1
    assert items_full[0].shape == (2, 3, 4)

    # 2. Sliced cube iterable along leading axis
    ds_sliced = FitsCubeIterableDataset([str(path)], slice_index=1)
    items_sliced = list(ds_sliced)
    assert len(items_sliced) == 1
    assert items_sliced[0].shape == (3, 4)
    assert np.allclose(items_sliced[0].numpy(), data[1])


def test_fits_spectrum_iterable_dataset(desi_shaped_fits):
    from torchfits.data import FitsSpectrumIterableDataset

    path = str(desi_shaped_fits)
    ds = FitsSpectrumIterableDataset(
        [path, path],
        hdu=["B_FLUX", "R_FLUX"],
        ivar_hdu=["B_IVAR", "R_IVAR"],
        row=0,
        layout="dict",
        shuffle_buffer_size=10,
    )
    items = list(ds)
    assert len(items) == 2
    assert set(items[0].keys()) == {"B_FLUX", "R_FLUX"}


def test_rank_world_size_sharding(tmp_path):
    from torchfits.data import FitsTensorIterableDataset
    from torchfits.data.datasets import _resolve_rank_and_world_size

    paths = [str(tmp_path / f"img_{i}.fits") for i in range(10)]
    for p in paths:
        fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32)).writeto(p, overwrite=True)

    # Sharding across rank 0 of 2 vs rank 1 of 2
    ds_rank0 = FitsTensorIterableDataset(paths, rank=0, world_size=2)
    ds_rank1 = FitsTensorIterableDataset(paths, rank=1, world_size=2)

    items0 = list(ds_rank0)
    items1 = list(ds_rank1)
    assert len(items0) == 5
    assert len(items1) == 5

    # Check env var resolution
    r, w = _resolve_rank_and_world_size(None, None)
    assert r >= 0 and w >= 1


def test_buffered_shuffle():
    from torchfits.data.datasets import _buffered_shuffle

    items = list(range(100))
    shuffled = list(_buffered_shuffle(iter(items), buffer_size=20, seed=42))
    assert len(shuffled) == 100
    assert set(shuffled) == set(items)
    assert shuffled != items  # Permuted


# ---------------------------------------------------------------------------
# R3a review: companion arity + staged cutout safety
# ---------------------------------------------------------------------------


@pytest.fixture
def three_band_fits(tmp_path):
    path = tmp_path / "three.fits"
    hdus = [fits.PrimaryHDU()]
    for name in ("G", "R", "Z"):
        hdus.append(fits.ImageHDU(np.ones((4, 4), dtype=np.float32), name=name))
    fits.HDUList(hdus).writeto(str(path), overwrite=True)
    return path


class TestCompanionArity:
    def test_map_rejects_mismatched_ivar_arity(self, three_band_fits):
        from torchfits.data import FitsTensorDataset

        with pytest.raises(ValueError, match="ivar_hdu must match hdu arity"):
            FitsTensorDataset([str(three_band_fits)], hdu=["G", "R"], ivar_hdu=["Z"])

    def test_iterable_rejects_mismatched_ivar_arity(self, three_band_fits):
        from torchfits.data import FitsTensorIterableDataset

        with pytest.raises(ValueError, match="ivar_hdu must match hdu arity"):
            FitsTensorIterableDataset(
                [str(three_band_fits)], hdu=["G", "R"], ivar_hdu=["Z"]
            )

    def test_iterable_rejects_mismatched_mask_arity(self, three_band_fits):
        from torchfits.data import FitsTensorIterableDataset

        with pytest.raises(ValueError, match="mask_hdu must match hdu arity"):
            FitsTensorIterableDataset(
                [str(three_band_fits)], hdu=["G", "R"], mask_hdu=["Z"]
            )


class TestStagedCutoutSafety:
    def test_cutout_size_must_be_positive(self, three_band_fits):
        from torchfits.data import FitsStagedCutoutIterableDataset

        with pytest.raises(ValueError, match="cutout_size"):
            FitsStagedCutoutIterableDataset(
                [str(three_band_fits)], cutout_size=0, hdu="G"
            )

    def test_duplicate_urls_download_once_and_survive_cleanup(self, tmp_path):
        """A staged copy must outlive every occurrence of its URL.

        ``cleanup=True`` used to unlink the staged file after the first
        occurrence's cutouts: a second occurrence re-downloaded it, and a
        concurrent iterator could crash in its resolve->open window on the
        unlinked path.
        """
        import shutil
        from unittest import mock

        import torchfits.data.remote as remote
        from torchfits.data import FitsStagedCutoutIterableDataset

        mosaic = tmp_path / "mosaic.fits"
        fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(
            str(mosaic), overwrite=True
        )

        calls: list[str] = []

        def _serve(url, dest):
            calls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".partial")
            shutil.copy(mosaic, tmp)
            tmp.replace(dest)
            return dest

        url = "https://archive.example.org/mosaics/dup.fits"
        with mock.patch.object(remote, "_download", side_effect=_serve):
            ds = FitsStagedCutoutIterableDataset(
                [url, url],
                cutouts_per_file=1,
                cutout_size=4,
                hdu=0,
                staging_dir=tmp_path / "scratch",
                cleanup=True,
            )
            items = list(ds)

        assert len(items) == 2
        assert all(t.shape == (1, 4, 4) for t in items)
        assert calls == [url]
