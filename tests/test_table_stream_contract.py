"""Contracts for the table read / stream / persistent-handle surface.

Round 2, unit 2 part 4 -- ``_io_engine/table_api.py``,
``_io_engine/table_reader_api.py`` and ``_io_engine/table_streaming.py``.

Four defects, each pinned by a test that fails without its fix:

* ``read_torch(where=...)`` ran its thin read inside a broad ``except``, so
  the row-window and ``mmap`` checks the unfiltered path enforces were caught
  and re-read as "thin path unavailable"; the fallback then re-derived the
  answer by a different route and returned rows for a request that is
  otherwise rejected.
* The ``where=`` mask gather handled tensors and ndarrays and passed every
  other payload through untouched, so a variable-length column came back at
  the *unfiltered* length next to filtered sibling columns.
* ``TableReaderHandle.read_torch`` moved top-level tensors only, leaving the
  tensors inside a variable-length column on the CPU while its siblings moved.
* ``stream_table`` routed ASCII tables and scaled columns off the raw mmap
  route but not variable-length ones, which that route rejects outright --
  so the default ``mmap=True`` stream raised on a table every other route
  reads.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits
import torchfits._C as cpp
from torchfits._io_engine.table_reader_api import open_table_reader
from torchfits._io_engine.table_streaming import stream_table

# Row payloads of the VLA table, in row order.
_VLA_ROWS = [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10], [11]]


def _write_plain_table(path) -> str:
    cols = [
        fits.Column(name="A", format="J", array=np.arange(5, dtype=np.int32)),
        fits.Column(name="B", format="E", array=np.arange(5, dtype=np.float32)),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        str(path), overwrite=True
    )
    return str(path)


def _write_vla_table(path) -> str:
    cols = [
        fits.Column(name="A", format="J", array=np.arange(5, dtype=np.int32)),
        fits.Column(
            name="P",
            format="PJ(10)",
            array=[np.asarray(row, dtype=np.int32) for row in _VLA_ROWS],
        ),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        str(path), overwrite=True
    )
    return str(path)


def _row_length(value: Any) -> int:
    """Number of rows a column payload carries."""
    if isinstance(value, torch.Tensor):
        return int(value.shape[0])
    return len(value)


def _vla_contents(payload: Any) -> list:
    return [item.tolist() for item in payload]


def _non_cpu_device() -> Optional[str]:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return None


# --- r2-018: where= must not bypass per-request validation ----------------


@pytest.mark.parametrize("bad_mmap", ["sometimes", "auto1", 1, object()], ids=repr)
def test_where_read_rejects_unknown_mmap_mode(tmp_path, bad_mmap):
    path = _write_plain_table(tmp_path / "t.fits")
    with pytest.raises(ValueError, match="mmap must be bool or 'auto'"):
        torchfits.table.read_torch(path, where="A > 1", mmap=bad_mmap)


@pytest.mark.parametrize("bad_start", [0, -1, -5])
def test_where_read_rejects_non_positive_start_row(tmp_path, bad_start):
    path = _write_plain_table(tmp_path / "t.fits")
    with pytest.raises(ValueError, match=r"start_row must be >= 1"):
        torchfits.table.read_torch(path, where="A > 1", start_row=bad_start)


def test_where_read_rejects_num_rows_below_minus_one(tmp_path):
    path = _write_plain_table(tmp_path / "t.fits")
    with pytest.raises(ValueError, match="num_rows must be > 0 or -1"):
        torchfits.table.read_torch(path, where="A > 1", num_rows=-2)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"start_row": 0}, r"start_row must be >= 1"),
        ({"num_rows": -2}, "num_rows must be > 0 or -1"),
        ({"mmap": "sometimes"}, "mmap must be bool or 'auto'"),
    ],
)
def test_unfiltered_read_rejects_the_same_parameters(tmp_path, kwargs, message):
    """The unfiltered path already raised these; hoisting must keep that."""
    path = _write_plain_table(tmp_path / "t.fits")
    with pytest.raises(ValueError, match=message):
        torchfits.table.read_torch(path, **kwargs)


def test_where_read_still_honours_a_valid_row_window(tmp_path):
    """Non-vacuity: a *valid* window must still filter within the window."""
    path = _write_plain_table(tmp_path / "t.fits")
    # Window covers 1-based rows 2..4, i.e. A == [1, 2, 3]; A > 1 keeps [2, 3].
    out = torchfits.table.read_torch(path, where="A > 1", start_row=2, num_rows=3)
    assert out["A"].tolist() == [2, 3]
    assert out["B"].tolist() == [2.0, 3.0]


def test_where_read_accepts_the_default_row_window(tmp_path):
    path = _write_plain_table(tmp_path / "t.fits")
    out = torchfits.table.read_torch(path, where="A > 1")
    assert out["A"].tolist() == [2, 3, 4]


def test_tensor_where_does_not_build_a_python_row_index(tmp_path, monkeypatch):
    """A tensor-only filter must not call ``nonzero().tolist()``.

    That list is only consumed by list columns. Building it for a dense
    numeric filter was slower than the column read.
    """
    path = _write_plain_table(tmp_path / "t.fits")

    def _refuse_index(self, *args, **kwargs):
        raise AssertionError("tensor where built a python row index")

    monkeypatch.setattr(torch.Tensor, "nonzero", _refuse_index)
    out = torchfits.table.read_torch(path, where="A > 1")
    assert out["A"].tolist() == [2, 3, 4]


def test_where_read_accepts_a_zero_row_window(tmp_path):
    """Non-vacuity for the num_rows check.

    ``torchfits.read`` rejects ``num_rows=0``, but the table reader's thin
    path returns empty columns and the where= branch has always agreed with
    it; hoisting the check must not quietly tighten that into an error.
    """
    path = _write_plain_table(tmp_path / "t.fits")
    out = torchfits.table.read_torch(path, where="A > 1", num_rows=0)
    assert out["A"].tolist() == []


# --- r2-019: a mask gather must align every row-aligned payload ----------


def test_where_read_gathers_variable_length_columns(tmp_path):
    path = _write_vla_table(tmp_path / "vla.fits")
    out = torchfits.table.read_torch(path, where="A > 1")
    assert out["A"].tolist() == [2, 3, 4]
    assert isinstance(out["P"], list)
    assert len(out["P"]) == 3
    assert _vla_contents(out["P"]) == [_VLA_ROWS[i] for i in (2, 3, 4)]


def test_where_read_column_row_lengths_all_agree(tmp_path):
    """The mismatch itself: one dict, columns of different lengths."""
    path = _write_vla_table(tmp_path / "vla.fits")
    out = torchfits.table.read_torch(path, where="A >= 1")
    assert {_row_length(value) for value in out.values()} == {4}


def test_where_read_keeps_matching_rows_not_leading_rows(tmp_path):
    """Guards a 'return the first N rows' over-correction."""
    path = _write_vla_table(tmp_path / "vla.fits")
    out = torchfits.table.read_torch(path, where="A == 3")
    assert out["A"].tolist() == [3]
    assert _vla_contents(out["P"]) == [_VLA_ROWS[3]]


def test_where_read_gathers_vla_columns_from_the_buffered_route(tmp_path):
    path = _write_vla_table(tmp_path / "vla.fits")
    out = torchfits.table.read_torch(path, where="A > 1", mmap=False)
    assert out["A"].tolist() == [2, 3, 4]
    assert _vla_contents(out["P"]) == [_VLA_ROWS[i] for i in (2, 3, 4)]


def test_unfiltered_vla_read_is_unaffected(tmp_path):
    path = _write_vla_table(tmp_path / "vla.fits")
    out = torchfits.table.read_torch(path)
    assert _vla_contents(out["P"]) == _VLA_ROWS


def test_row_window_read_still_slices_variable_length_columns(tmp_path):
    """The window path already sliced lists; the mask path now matches it."""
    path = _write_vla_table(tmp_path / "vla.fits")
    out = torchfits.table.read_torch(path, start_row=2, num_rows=2)
    assert out["A"].tolist() == [1, 2]
    assert _vla_contents(out["P"]) == _VLA_ROWS[1:3]


# --- r2-020: the handle must place every payload on the requested device --


def test_table_handle_moves_list_columns_to_the_requested_device(tmp_path):
    device = _non_cpu_device()
    if device is None:
        pytest.skip("no non-CPU torch device available")
    path = _write_vla_table(tmp_path / "vla.fits")
    with open_table_reader(path) as handle:
        out = handle.read_torch(device=device)
    assert out["A"].device.type == device
    assert isinstance(out["P"], list)
    for item in out["P"]:
        assert isinstance(item, torch.Tensor)
        assert item.device.type == device, (
            f"variable-length payload stayed on {item.device} while its "
            f"sibling column moved to {device}"
        )


def test_table_handle_cpu_read_stays_on_cpu(tmp_path):
    """Non-vacuity: the default read is not vacuously 'moved'."""
    path = _write_vla_table(tmp_path / "vla.fits")
    with open_table_reader(path) as handle:
        out = handle.read_torch()
    assert out["A"].device.type == "cpu"
    assert [item.device.type for item in out["P"]] == ["cpu"] * 5


def test_table_handle_matches_cold_read_on_device(tmp_path):
    device = _non_cpu_device()
    if device is None:
        pytest.skip("no non-CPU torch device available")
    path = _write_vla_table(tmp_path / "vla.fits")
    with open_table_reader(path) as handle:
        hot = handle.read_torch(device=device)
    cold = torchfits.table.read_torch(path, device=device)
    assert hot["A"].cpu().tolist() == cold["A"].cpu().tolist()
    assert [_vla_contents(c["P"]) for c in (hot, cold)] == [_VLA_ROWS] * 2


# --- r2-021: the stream route table must include variable-length columns --


def test_stream_table_reads_vla_tables_over_the_mmap_request(tmp_path):
    path = _write_vla_table(tmp_path / "vla.fits")
    chunks = list(stream_table(torchfits.read_header, path, 1, chunk_rows=2, mmap=True))
    assert [chunk["A"].tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]
    assert [len(chunk["P"]) for chunk in chunks] == [2, 2, 1]
    assert [item.tolist() for chunk in chunks for item in chunk["P"]] == _VLA_ROWS


def test_stream_table_mmap_and_buffered_routes_agree_on_vla(tmp_path):
    path = _write_vla_table(tmp_path / "vla.fits")
    mmap_chunks = list(
        stream_table(torchfits.read_header, path, 1, chunk_rows=2, mmap=True)
    )
    buffered = list(
        stream_table(torchfits.read_header, path, 1, chunk_rows=2, mmap=False)
    )
    assert [chunk["A"].tolist() for chunk in mmap_chunks] == [
        chunk["A"].tolist() for chunk in buffered
    ]
    assert [_vla_contents(c["P"]) for c in mmap_chunks] == [
        _vla_contents(c["P"]) for c in buffered
    ]


def _record_mmap_flags(path, chunk_rows=2):
    """Run a stream and return the mmap flag of every read_fits_table_rows call."""
    flags = []
    original = cpp.read_fits_table_rows

    def spy(*args, **kwargs):
        flags.append(args[-1])
        return original(*args, **kwargs)

    cpp.read_fits_table_rows = spy
    try:
        list(
            stream_table(
                torchfits.read_header, path, 1, chunk_rows=chunk_rows, mmap=True
            )
        )
    finally:
        cpp.read_fits_table_rows = original
    return flags


def test_stream_table_never_uses_the_raw_mmap_row_route_for_vla(tmp_path):
    """That route rejects VLA columns outright ("VLA columns not supported
    for mmap"), so reaching it is the defect, independent of what it returns."""
    path = _write_vla_table(tmp_path / "vla.fits")
    flags = _record_mmap_flags(path)
    assert not any(flags), f"raw mmap row route used for a VLA table: {flags}"


def test_stream_table_still_uses_the_raw_mmap_row_route_for_plain_tables(
    tmp_path,
):
    """Non-vacuity: the route must stay reachable for tables it can serve."""
    path = _write_plain_table(tmp_path / "t.fits")
    flags = _record_mmap_flags(path)
    assert flags, "plain table never reached read_fits_table_rows"
    assert all(flags), f"plain table left the raw mmap route: {flags}"


def test_table_hdu_ref_iter_rows_reads_vla_tables(tmp_path):
    """``iter_rows`` defaults to mmap=True and is the caller that broke."""
    path = _write_vla_table(tmp_path / "vla.fits")
    chunks = list(torchfits.open(path)[1].iter_rows(batch_size=2))
    assert [chunk["A"].tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]
    assert [len(chunk["P"]) for chunk in chunks] == [2, 2, 1]
    assert [item.tolist() for chunk in chunks for item in chunk["P"]] == _VLA_ROWS
