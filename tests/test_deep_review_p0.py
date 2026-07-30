"""Regression tests for deep-review P0 ship-blockers."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from torchfits import io
from torchfits._io_engine import _read_pipeline
from torchfits._table import _read_where as where_mod


def test_torch_where_filter_skips_large_tables(tmp_path):
    """P0-1: large NAXIS2 must not materialize all rows via torch WHERE."""
    path = str(tmp_path / "large.fits")
    header = {
        "NAXIS2": where_mod._TORCH_WHERE_MAX_ROWS + 1,
        "TFIELDS": 1,
        "TTYPE1": "MAG",
        "TFORM1": "E",
    }

    with (
        mock.patch.object(
            where_mod,
            "_compile_where_to_simple_predicates",
            return_value=[("MAG", "<", 50.0)],
        ),
        mock.patch("torchfits._C") as cpp,
    ):
        cpp.read_nrows.return_value = where_mod._TORCH_WHERE_MAX_ROWS + 1
        cpp.read_fits_table.side_effect = AssertionError(
            "must not full-read large table"
        )
        reader = mock.Mock()
        reader.read_rows.side_effect = AssertionError("must not full-read large table")
        with mock.patch.object(where_mod, "_acquire_cpp_reader", return_value=reader):
            result = where_mod._try_torch_tensor_where_filter(
                pa=mock.Mock(),
                path=path,
                hdu=1,
                columns=["MAG"],
                where="MAG < 50.0",
                row_slice=None,
                rows=None,
                mmap=True,
                decode_bytes=False,
                encoding="ascii",
                strip=True,
                header=header,
            )
    assert result is None
    cpp.read_fits_table.assert_not_called()
    reader.read_rows.assert_not_called()


def test_torch_where_filter_still_runs_for_small_tables(tmp_path):
    """P0-1: small tables keep the torch mask path."""
    path = str(tmp_path / "small.fits")
    header = {
        "NAXIS2": 4,
        "TFIELDS": 1,
        "TTYPE1": "MAG",
        "TFORM1": "E",
    }
    mag = torch.tensor([10.0, 20.0, 60.0, 70.0], dtype=torch.float32)

    with (
        mock.patch.object(
            where_mod,
            "_compile_where_to_simple_predicates",
            return_value=[("MAG", "<", 50.0)],
        ),
        mock.patch.object(
            where_mod, "_can_use_mmap_row_path_for_full_read", return_value=False
        ),
        mock.patch("torchfits._C") as cpp,
    ):
        reader = mock.Mock()
        reader.read_rows.return_value = {"MAG": mag}
        with mock.patch.object(where_mod, "_acquire_cpp_reader", return_value=reader):
            import pyarrow as pa

            result = where_mod._try_torch_tensor_where_filter(
                pa=pa,
                path=path,
                hdu=1,
                columns=["MAG"],
                where="MAG < 50.0",
                row_slice=None,
                rows=None,
                mmap=False,
                decode_bytes=False,
                encoding="ascii",
                strip=True,
                header=header,
            )
    assert result is not None
    assert result.num_rows == 2
    assert result.column("MAG").to_pylist() == pytest.approx([10.0, 20.0])
    reader.read_rows.assert_called_once_with(["MAG"], 1, -1)
    cpp.read_fits_table.assert_not_called()


def test_read_batch_paths_uses_read_exc_types_and_strict():
    """P0-2: batch C++ failures must not bare-except; strict re-raises."""
    cpp = mock.Mock()
    cpp.read_images_batch.side_effect = RuntimeError("batch boom")
    logger = mock.Mock()
    logger.isEnabledFor.return_value = False

    with mock.patch.object(
        _read_pipeline,
        "read_unified",
        side_effect=lambda **kwargs: torch.zeros(2, 2),
    ):
        out = _read_pipeline._read_batch_paths(
            cpp_module=cpp,
            path=["a.fits", "b.fits"],
            hdu=0,
            device="cpu",
            mmap=True,
            fp16=False,
            bf16=False,
            raw_scale=False,
            columns=None,
            start_row=1,
            num_rows=-1,
            cache_capacity=10,
            handle_cache_capacity=16,
            fast_header=True,
            return_header=False,
            mode="auto",
            autodetect_hdu=lambda p, c: 0,
            batch_to_device=lambda xs, d: xs,
            resolve_image_mmap=lambda *a, **k: True,
            read_check_cache=lambda *a, **k: (False, None, None),
            read_header=lambda *a, **k: {},
            debug_scale=False,
            cold_nocache=False,
            read_exc_types=io._READ_EXC_TYPES,
            logger=logger,
            strict=False,
        )
    assert len(out) == 2
    logger.debug.assert_called()

    with pytest.raises(RuntimeError, match="batch boom"):
        _read_pipeline._read_batch_paths(
            cpp_module=cpp,
            path=["a.fits", "b.fits"],
            hdu=0,
            device="cpu",
            mmap=True,
            fp16=False,
            bf16=False,
            raw_scale=False,
            columns=None,
            start_row=1,
            num_rows=-1,
            cache_capacity=10,
            handle_cache_capacity=16,
            fast_header=True,
            return_header=False,
            mode="auto",
            autodetect_hdu=lambda p, c: 0,
            batch_to_device=lambda xs, d: xs,
            resolve_image_mmap=lambda *a, **k: True,
            read_check_cache=lambda *a, **k: (False, None, None),
            read_header=lambda *a, **k: {},
            debug_scale=False,
            cold_nocache=False,
            read_exc_types=io._READ_EXC_TYPES,
            logger=logger,
            strict=True,
        )


def test_read_batch_paths_does_not_swallow_keyboardinterrupt():
    """P0-2: unexpected exceptions (not in read_exc_types) must propagate."""
    cpp = mock.Mock()
    cpp.read_images_batch.side_effect = KeyboardInterrupt()
    logger = mock.Mock()

    with pytest.raises(KeyboardInterrupt):
        _read_pipeline._read_batch_paths(
            cpp_module=cpp,
            path=["a.fits"],
            hdu=0,
            device="cpu",
            mmap=True,
            fp16=False,
            bf16=False,
            raw_scale=False,
            columns=None,
            start_row=1,
            num_rows=-1,
            cache_capacity=10,
            handle_cache_capacity=16,
            fast_header=True,
            return_header=False,
            mode="auto",
            autodetect_hdu=lambda p, c: 0,
            batch_to_device=lambda xs, d: xs,
            resolve_image_mmap=lambda *a, **k: True,
            read_check_cache=lambda *a, **k: (False, None, None),
            read_header=lambda *a, **k: {},
            debug_scale=False,
            cold_nocache=False,
            read_exc_types=io._READ_EXC_TYPES,
            logger=logger,
            strict=False,
        )
