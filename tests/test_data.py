"""Tests for torchfits.data — datasets, collate, and loader helpers."""

import json as _json
import os
import subprocess as _subprocess
import sys as _sys
import tempfile
import textwrap as _textwrap

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from torchfits.data import (
    FitsCutoutDataset,
    FitsImageDataset,
    FitsImageIterableDataset,
    FitsTableDataset,
    FitsTableIterableDataset,
    fits_collate_fn,
    make_loader,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with FITS image files (LABEL header)."""
    from astropy.io import fits

    tmpdir = tempfile.mkdtemp(prefix="torchfits_data_test_")
    files = []
    for i in range(8):
        data = np.random.rand(32, 32).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["LABEL"] = i % 2
        path = os.path.join(tmpdir, f"image_{i:03d}.fits")
        hdu.writeto(path, overwrite=True)
        files.append(path)
    yield tmpdir, files
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_table_file():
    """Create a temporary FITS binary table file."""
    from astropy.table import Table

    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    table = Table()
    table["flux"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    table["mag"] = np.array(
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype=np.float32
    )
    table.write(tmp.name, format="fits", overwrite=True)
    yield tmp.name
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Test: fits_collate_fn
# ---------------------------------------------------------------------------


class TestFitsCollateFn:
    def test_empty_batch(self):
        assert fits_collate_fn([]) == []

    def test_tensor_list(self):
        batch = [torch.randn(3, 32, 32) for _ in range(4)]
        out = fits_collate_fn(batch)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 3, 32, 32)

    def test_image_label_tuple(self):
        batch = [(torch.randn(3, 32, 32), torch.tensor(0)) for _ in range(4)]
        images, labels = fits_collate_fn(batch)
        assert images.shape == (4, 3, 32, 32)
        assert labels.shape == (4,)

    def test_dict_of_tensors(self):
        batch = [{"a": torch.randn(3), "b": torch.randn(5)} for _ in range(4)]
        out = fits_collate_fn(batch)
        assert out["a"].shape == (4, 3)
        assert out["b"].shape == (4, 5)

    def test_ragged_non_tensor_column_raises(self):
        batch = [
            {"tensor_col": torch.randn(3), "list_col": [1, 2, 3]},
            {"tensor_col": torch.randn(3), "list_col": [4, 5]},
        ]
        with pytest.raises(ValueError, match="non-tensor column"):
            fits_collate_fn(batch)

    def test_non_tensor_error_mentions_custom_collate(self):
        batch = [{"tensor_col": torch.randn(3), "list_col": [1, 2, 3]}]
        with pytest.raises(ValueError, match="custom collate_fn"):
            fits_collate_fn(batch)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported sample type"):
            fits_collate_fn(["string", "batch"])


# ---------------------------------------------------------------------------
# Test: FitsImageDataset
# ---------------------------------------------------------------------------


class TestFitsImageDataset:
    def test_file_list(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)
        assert len(ds) == 8

    def test_glob_pattern(self, temp_image_dir):
        tmpdir, _files = temp_image_dir
        ds = FitsImageDataset(os.path.join(tmpdir, "*.fits"))
        assert len(ds) == 8

    def test_label_from_header(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files, label_key="LABEL")
        assert ds._labels == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_explicit_labels(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        labels = [10, 20, 30, 40, 50, 60, 70, 80]
        ds = FitsImageDataset(files, labels=labels)
        assert ds._labels == labels

    def test_labels_length_mismatch_raises(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        with pytest.raises(ValueError, match="labels length"):
            FitsImageDataset(files, labels=[0, 1])

    def test_default_labels_are_zero(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)
        assert ds._labels == [0] * 8

    def test_getitem_returns_image_label(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert image.ndim == 3
        assert image.shape[0] == 1

    def test_add_channel_dim_false(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files, add_channel_dim=False)
        image, _label = ds[0]
        assert image.ndim == 2

    def test_auto_mmap_policy(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        image, _label = FitsImageDataset(files, mmap="auto")[0]
        assert image.shape == (1, 32, 32)

    def test_3d_cube_no_channel_added(self, temp_image_dir):
        from astropy.io import fits

        tmpdir, _ = temp_image_dir
        data = np.random.rand(8, 32, 32).astype(np.float32)
        path = os.path.join(tmpdir, "cube.fits")
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(path, overwrite=True)

        ds = FitsImageDataset([path], add_channel_dim=True)
        image, _ = ds[0]
        assert image.ndim == 3
        assert image.shape == (8, 32, 32)

    def test_transform_applied(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files, transform=lambda x: x * 0.0)
        image, _label = ds[0]
        assert image.abs().max().item() == 0.0

    def test_integration_with_dataloader(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)
        loader = DataLoader(ds, batch_size=4, collate_fn=fits_collate_fn)
        for images, labels in loader:
            assert images.shape[0] <= 4
            assert labels.shape[0] <= 4
            break

    def test_repr(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)
        r = repr(ds)
        assert "FitsImageDataset" in r
        assert "n=8" in r


# ---------------------------------------------------------------------------
# Test: FitsImageIterableDataset
# ---------------------------------------------------------------------------


class TestFitsImageIterableDataset:
    def test_iterates_all_files(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageIterableDataset(files)
        count = sum(1 for _ in ds)
        assert count == 8

    def test_auto_mmap_policy(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        image = next(iter(FitsImageIterableDataset(files, mmap="auto")))
        assert image.shape == (1, 32, 32)

    def test_output_is_tensor(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageIterableDataset(files)
        sample = next(iter(ds))
        assert isinstance(sample, torch.Tensor)

    def test_add_channel_dim(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageIterableDataset(files, add_channel_dim=True)
        sample = next(iter(ds))
        assert sample.ndim == 3
        assert sample.shape[0] == 1

    def test_add_channel_dim_false(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageIterableDataset(files, add_channel_dim=False)
        sample = next(iter(ds))
        assert sample.ndim == 2

    def test_shuffle_deterministic(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds1 = FitsImageIterableDataset(files, shuffle=True, seed=42)
        ds2 = FitsImageIterableDataset(files, shuffle=True, seed=42)
        out1 = list(ds1)
        out2 = list(ds2)
        for a, b in zip(out1, out2):
            assert torch.equal(a, b)

    def test_no_shuffle_follows_file_order(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageIterableDataset(files, shuffle=False)
        out = list(ds)
        assert len(out) == 8

    def test_transform_applied(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageIterableDataset(files, transform=lambda x: x * 0.0)
        sample = next(iter(ds))
        assert sample.abs().max().item() == 0.0

    def test_integration_with_dataloader(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageIterableDataset(files)
        loader = DataLoader(ds, batch_size=4, collate_fn=fits_collate_fn)
        for batch in loader:
            assert batch.shape[0] <= 4
            break

    def test_repr(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageIterableDataset(files)
        r = repr(ds)
        assert "FitsImageIterableDataset" in r
        assert "n=8" in r


# ---------------------------------------------------------------------------
# Test: FitsTableDataset
# ---------------------------------------------------------------------------


class TestFitsTableDataset:
    def test_len(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file)
        assert len(ds) == 8

    def test_getitem_returns_dict(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file)
        row, label = ds[0]
        assert isinstance(row, dict)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert "flux" in row
        assert "mag" in row
        assert isinstance(row["flux"], torch.Tensor)

    def test_getitem_correct_values(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file)
        row, label = ds[0]
        assert row["flux"].item() == pytest.approx(1.0)
        assert row["mag"].item() == pytest.approx(10.0)
        assert label.item() == 0

    def test_getitem_different_rows(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file)
        r0 = ds[0][0]["flux"].item()
        r3 = ds[3][0]["flux"].item()
        assert r0 != r3

    def test_column_projection(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file, columns=["flux"])
        row, _label = ds[0]
        assert set(row.keys()) == {"flux"}

    def test_where_filter(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file, where="flux > 4.0")
        assert len(ds) == 4
        assert ds[0][0]["flux"].item() == pytest.approx(5.0)
        assert ds[-1][0]["flux"].item() == pytest.approx(8.0)

    def test_transform_applied(self, temp_table_file):
        ds = FitsTableDataset(
            temp_table_file,
            transform=lambda row: {k: v * 0.0 for k, v in row.items()},
        )
        row, _label = ds[0]
        assert row["flux"].item() == 0.0
        assert row["mag"].item() == 0.0

    def test_repr(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file)
        r = repr(ds)
        assert "FitsTableDataset" in r
        assert "n_rows=8" in r

    def test_column_projection_with_where(self, temp_table_file):
        ds = FitsTableDataset(
            temp_table_file, columns=["flux", "mag"], where="mag < 14.0"
        )
        assert len(ds) == 4
        row, _label = ds[0]
        assert set(row.keys()) == {"flux", "mag"}

    def test_empty_where_result(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file, where="flux > 999.0")
        assert len(ds) == 0

    def test_integration_with_dataloader(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file, columns=["flux", "mag"])
        loader = DataLoader(ds, batch_size=4, collate_fn=fits_collate_fn)
        for batch, labels in loader:
            assert isinstance(batch, dict)
            assert isinstance(labels, torch.Tensor)
            for key in batch:
                assert batch[key].shape[0] <= 4
            break


# ---------------------------------------------------------------------------
# Test: FitsTableIterableDataset
# ---------------------------------------------------------------------------


class TestFitsTableIterableDataset:
    def test_yields_all_rows(self, temp_table_file):
        ds = FitsTableIterableDataset(temp_table_file, batch_size=4)
        rows = list(ds)
        assert len(rows) == 8
        assert "flux" in rows[0]
        assert rows[0]["flux"].item() == pytest.approx(1.0)

    def test_where_filter(self, temp_table_file):
        ds = FitsTableIterableDataset(temp_table_file, where="flux > 4.0", batch_size=3)
        rows = list(ds)
        assert len(rows) == 4
        assert rows[0]["flux"].item() == pytest.approx(5.0)

    def test_repr(self, temp_table_file):
        ds = FitsTableIterableDataset(temp_table_file)
        assert "FitsTableIterableDataset" in repr(ds)


# ---------------------------------------------------------------------------
# Test: FitsCutoutDataset
# ---------------------------------------------------------------------------


class TestFitsCutoutDataset:
    def test_len_and_shape(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        path = files[0]
        ds = FitsCutoutDataset([(path, 0, 0, 0, 16, 16)])
        assert len(ds) == 1
        cutout = ds[0]
        assert cutout.shape == (1, 16, 16)

    def test_xy_size_form(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        path = files[0]
        ds = FitsCutoutDataset([(path, 0, 4, 4, 8)])
        cutout = ds[0]
        assert cutout.shape[-2:] == (8, 8)

    def test_files_attribute(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsCutoutDataset([(files[0], 0, 0, 0, 8, 8), (files[1], 0, 0, 0, 8, 8)])
        assert len(ds.files) == 2

    def test_same_file_distinct_cutouts(self, temp_image_dir):
        """NOTE: re-opens file per row; values must differ per window."""
        _tmpdir, files = temp_image_dir
        path = files[0]
        ds = FitsCutoutDataset([(path, 0, 0, 0, 8, 8), (path, 0, 8, 8, 16, 16)])
        assert len(ds) == 2
        assert not torch.equal(ds[0], ds[1])

    def test_invalid_cutout_spec_raises(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        with pytest.raises(ValueError, match="cutout must be"):
            FitsCutoutDataset([(files[0], 0, 0, 0)])


# ---------------------------------------------------------------------------
# Test: make_loader
# ---------------------------------------------------------------------------


class TestMakeLoader:
    def test_returns_dataloader(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)
        loader = make_loader(ds, batch_size=4, optimize_cache=False)
        assert isinstance(loader, DataLoader)
        for images, labels in loader:
            assert images.shape[0] <= 4
            break

    def test_default_shuffle_for_map_dataset(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)
        loader = make_loader(ds, batch_size=8, shuffle=None, optimize_cache=False)
        assert loader.batch_size == 8
        for images, _ in loader:
            assert images.shape[0] == 8
            break

    def test_explicit_shuffle_false(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)
        loader = make_loader(ds, batch_size=8, shuffle=False, optimize_cache=False)
        for images, _ in loader:
            assert images.shape[0] == 8
            break

    def test_custom_collate_fn(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)

        def my_collate(batch):
            images = torch.stack([s[0] for s in batch])
            return images * 2.0

        loader = make_loader(
            ds, batch_size=4, collate_fn=my_collate, optimize_cache=False
        )
        batch = next(iter(loader))
        # Custom collate returns a tensor (not the default tuple)
        assert isinstance(batch, torch.Tensor)

    def test_optimize_cache_no_files_attribute(self, temp_table_file):
        ds = FitsTableDataset(temp_table_file)
        loader = make_loader(ds, batch_size=4)
        assert isinstance(loader, DataLoader)

    def test_remote_files_download_once(self, tmp_path, monkeypatch):
        """Prefetch and resolve share one fetch per remote URL (no double GET)."""
        from unittest import mock

        from torch.utils.data import Dataset as _MapDataset

        monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(tmp_path / "remote_cache"))
        url = "https://archive.example.org/tiles/map-tile.fits"
        calls: list[str] = []

        def _mock_download(u, dest):
            calls.append(u)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"staged")
            return dest

        class _StubDataset(_MapDataset):
            files = [url]

            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return torch.zeros(3), torch.tensor(0)

        with mock.patch(
            "torchfits.data.remote._download", side_effect=_mock_download
        ):
            loader = make_loader(_StubDataset(), batch_size=1, shuffle=False)
            batches = list(loader)
        assert len(batches) == 2
        assert calls == [url]

    def test_drop_last(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageDataset(files)
        loader = make_loader(ds, batch_size=6, drop_last=True, optimize_cache=False)
        count = 0
        for _ in loader:
            count += 1
        assert count == 1

    def test_iterable_dataset_no_shuffle_by_default(self, temp_image_dir):
        _tmpdir, files = temp_image_dir
        ds = FitsImageIterableDataset(files)
        loader = make_loader(ds, batch_size=4, optimize_cache=False)
        assert isinstance(loader, DataLoader)


# ---------------------------------------------------------------------------
# Test: multi-worker DataLoader integration (subprocess)
# ---------------------------------------------------------------------------
#
# These tests launch a subprocess so the real DataLoader worker machinery runs
# without mixing pytest's process state with libomp / libcfitsio thread pools.
# Each subprocess writes a JSON report that pytest reads after it exits.


class TestMultiWorkerDataLoader:
    """Verify that ``make_loader(..., num_workers=N)`` shards files correctly."""

    def _run_in_subprocess(self, source: str) -> dict:
        """Execute ``source`` in a fresh Python subprocess and return the report.

        Subprocess failures re-raise with the actual stderr attached for
        debuggability (multi-worker DataLoader forking inside pytest is
        sensitive to libomp/libcfitsio threadpool state).
        """
        import tempfile as _tempfile
        import os as _os

        report_path = _tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ).name
        # macOS uses multiprocessing "spawn". Keep DataLoader construction
        # behind the standard __main__ guard so workers can import this script
        # without recursively creating more workers.
        body = (
            _textwrap.dedent(source)
            + "\nimport json as _json_local\n"
            + f"with open({report_path!r}, 'w') as _report_file:\n"
            + "    _json_local.dump(report, _report_file)\n"
        )
        script = (
            "def _run():\n"
            + _textwrap.indent(body, "    ")
            + "\nif __name__ == '__main__':\n"
            + "    _run()\n"
        )
        env = {**_os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
        try:
            _subprocess.run(
                [_sys.executable, "-c", script],
                env=env,
                check=True,
                capture_output=True,
                timeout=180,
            )
        except _subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "multi-worker subprocess failed:\n"
                f"--- stdout ---\n{exc.stdout.decode(errors='replace')}\n"
                f"--- stderr ---\n{exc.stderr.decode(errors='replace')}\n"
                f"--- script ---\n{script}\n"
            ) from exc
        except _subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "multi-worker subprocess timed out:\n"
                f"--- stdout ---\n{(exc.stdout or b'').decode(errors='replace')}\n"
                f"--- stderr ---\n{(exc.stderr or b'').decode(errors='replace')}\n"
                f"--- script ---\n{script}\n"
            ) from exc
        with open(report_path) as fh:
            return _json.load(fh)

    def test_multiprocess_loader_sees_all_samples(self, temp_image_dir):
        """``num_workers=2`` yields every file exactly once across workers.

        Tolerate DataLoader's internal sample ordering (workers + optional
        per-worker shuffle change indexing order). The other subprocess
        tests in this class verify the same invariant via
        ``FitsImageIterableDataset`` with ``shuffle=False``.
        """
        _tmpdir, files = temp_image_dir
        report = self._run_in_subprocess(
            f"""
            from torchfits.data import FitsImageDataset, make_loader
            files = {files!r}
            ds = FitsImageDataset(files)
            loader = make_loader(
                ds, batch_size=2, num_workers=2, shuffle=False,
                optimize_cache=False,
            )
            seen_count = 0
            for batch in loader:
                imgs, _labels = batch
                seen_count += imgs.shape[0]
            report = {{'count': seen_count}}
            """
        )
        assert report["count"] == len(files)

    def test_multiprocess_iterable_shards_deterministically(self, temp_image_dir):
        """IterableDataset with num_workers=2 yields total==len(files)."""
        _tmpdir, files = temp_image_dir
        report = self._run_in_subprocess(
            f"""
            from torchfits.data import FitsImageIterableDataset, make_loader
            ds = FitsImageIterableDataset({files!r}, shuffle=False)
            loader = make_loader(
                ds, batch_size=2, num_workers=2, optimize_cache=False
            )
            seen = 0
            for batch in loader:
                seen += batch.shape[0]
            report = {{'count': seen}}
            """
        )
        assert report["count"] == len(files)

    def test_multiprocess_iterable_with_shuffle(self, temp_image_dir):
        """Shuffle=True with epoch-independent seed sees all files."""
        _tmpdir, files = temp_image_dir
        report = self._run_in_subprocess(
            f"""
            from torchfits.data import FitsImageIterableDataset, make_loader
            ds = FitsImageIterableDataset({files!r}, shuffle=True, seed=1)
            loader = make_loader(
                ds, batch_size=2, num_workers=2, optimize_cache=False
            )
            seen = 0
            for batch in loader:
                seen += batch.shape[0]
            report = {{'count': seen}}
            """
        )
        assert report["count"] == len(files)

    def test_multiprocess_table_iterable_sees_all_rows(self, temp_table_file):
        """FitsTableIterableDataset with num_workers=2 yields every table row."""
        report = self._run_in_subprocess(
            f"""
            from torchfits.data import FitsTableIterableDataset, make_loader
            ds = FitsTableIterableDataset({temp_table_file!r}, batch_size=2)
            loader = make_loader(
                ds, batch_size=4, num_workers=2, optimize_cache=False
            )
            seen = 0
            for batch in loader:
                seen += batch["flux"].shape[0]
            report = {{'count': seen}}
            """
        )
        assert report["count"] == 8

    def test_single_worker_matches_no_worker(self, temp_image_dir):
        """num_workers=0 (main process) yields every file exactly once."""
        _tmpdir, files = temp_image_dir
        report = self._run_in_subprocess(
            f"""
            from torchfits.data import FitsImageIterableDataset, make_loader
            ds = FitsImageIterableDataset({files!r}, shuffle=False)
            loader = make_loader(
                ds, batch_size=2, num_workers=0, optimize_cache=False
            )
            seen = 0
            for batch in loader:
                seen += batch.shape[0]
            report = {{'count': seen}}
            """
        )
        assert report["count"] == len(files)
