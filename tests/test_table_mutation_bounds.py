"""`table.update_rows` must never extend a table (round 2, R2-028).

The mmap writer already refuses a row range past the end of the table
("Row range exceeds table length"). `update_rows` caught that ``RuntimeError``
in its layout-fallback handler -- the same handler that re-raises truncation --
and retried through the CFITSIO writer, which does **not** refuse: it grows
NAXIS2 and writes the payload past the last row.

Measured before the fix, on a 6-row table
(``update_rows(path, {"A": range(6), "B": ...}, slice(4, 10))``):

    before: [0, 10, 20, 30, 40, 50]          (6 rows)
    after : [0, 10, 20, 30, 40, 0, 1, 2, 3, 4, 5]   (10 rows)

Row 5's original value 50 was overwritten by 0 and four rows that never existed
were created -- no exception, on the default ``mmap="auto"`` path. Both sibling
mutations already refuse the same mistake:

    insert_rows(row=99)       -> ValueError: row index 99 is out of range
    delete_rows(slice(99,100)) -> ValueError: row_slice start is out of range

``append_rows`` is the sanctioned way to add rows, and still works.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
from astropy.io import fits as afits

import torchfits
import torchfits._C as cpp


def _write_table(path, n_rows=6, string_column=False) -> str:
    if string_column:
        cols = [
            afits.Column(name="S", format="8A", array=[f"r{i}" for i in range(n_rows)])
        ]
    else:
        cols = [
            afits.Column(name="A", format="J", array=np.arange(n_rows) * 10),
            afits.Column(name="B", format="E", array=np.arange(n_rows) * 1.0),
        ]
    afits.HDUList([afits.PrimaryHDU(), afits.BinTableHDU.from_columns(cols)]).writeto(
        path, overwrite=True
    )
    return path


def _values(path: str) -> list:
    with afits.open(path) as hdul:
        col = "S" if "S" in hdul[1].columns.names else "A"
        raw = hdul[1].data[col].tolist()
        if col == "S":
            return [v.strip() for v in raw]
        return raw


def _n_rows(path: str) -> int:
    with afits.open(path) as hdul:
        return len(hdul[1].data)


def _digest(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _numeric_payload(n: int) -> dict:
    return {"A": list(range(n)), "B": [0.0] * n}


def _typed_payload(n: int) -> dict:
    """Payload whose dtypes match the on-disk columns (int32 'J', float32 'E').

    The forced-mmap writer requires an exact dtype match ('update_rows mmap
    dtype mismatch for A' otherwise); the auto path converts for you.
    """
    return {"A": np.arange(n, dtype=np.int32), "B": np.zeros(n, dtype=np.float32)}


# ---------------------------------------------------------------------------
# the finding: an over-long window fabricated rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row_slice, payload_rows",
    [
        (slice(4, 10), 6),  # ends 4 rows past a 6-row table
        (slice(5, 7), 2),  # ends exactly 1 row past
        ((5, 20), 15),
        (slice(0, 8), 8),  # starts at 0, overruns by 2
    ],
)
def test_update_rows_refuses_a_window_past_the_end(tmp_path, row_slice, payload_rows):
    path = _write_table(str(tmp_path / "t.fits"))
    before = _values(path)
    before_rows = _n_rows(path)

    with pytest.raises(ValueError, match="out of range"):
        torchfits.table.update_rows(path, _numeric_payload(payload_rows), row_slice)

    assert _n_rows(path) == before_rows
    assert _values(path) == before


def test_update_rows_rejection_leaves_the_file_byte_identical(tmp_path):
    """A refused window must not touch a byte -- same inode, same SHA-256."""
    path = _write_table(str(tmp_path / "t.fits"))
    stat_before = os.stat(path)
    digest_before = _digest(path)

    with pytest.raises(ValueError):
        torchfits.table.update_rows(path, _numeric_payload(6), slice(4, 10))

    stat_after = os.stat(path)
    assert stat_after.st_ino == stat_before.st_ino
    assert stat_after.st_size == stat_before.st_size
    assert _digest(path) == digest_before


def test_update_rows_refuses_a_window_starting_past_the_end(tmp_path):
    path = _write_table(str(tmp_path / "t.fits"))
    before = _values(path)
    with pytest.raises(ValueError, match="out of range"):
        torchfits.table.update_rows(path, _numeric_payload(2), slice(6, 8))
    assert _values(path) == before
    assert _n_rows(path) == 6


def test_update_rows_refuses_on_an_empty_table(tmp_path):
    path = _write_table(str(tmp_path / "t.fits"), n_rows=0)
    with pytest.raises(ValueError, match="out of range"):
        torchfits.table.update_rows(path, _numeric_payload(1), slice(0, 1))
    assert _n_rows(path) == 0


def test_update_rows_refuses_forced_mmap_too(tmp_path):
    """mmap=True used to raise RuntimeError from the mmap writer; now the same
    ValueError as the default path, and still no write."""
    path = _write_table(str(tmp_path / "t.fits"))
    digest_before = _digest(path)
    with pytest.raises(ValueError, match="out of range"):
        torchfits.table.update_rows(path, _typed_payload(6), slice(4, 10), mmap=True)
    assert _digest(path) == digest_before


def test_update_rows_refuses_on_a_string_column_table(tmp_path):
    """String columns never take the mmap route, so this is the pure
    CFITSIO-writer case -- the one that grew the table outright."""
    path = _write_table(str(tmp_path / "s.fits"), string_column=True)
    before = _values(path)
    with pytest.raises(ValueError, match="out of range"):
        torchfits.table.update_rows(
            path, {"S": [f"n{i}" for i in range(6)]}, slice(4, 10)
        )
    assert _values(path) == before
    assert _n_rows(path) == 6


# ---------------------------------------------------------------------------
# no regression: windows that fit still update in place
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row_slice, payload_rows, expected_a",
    [
        (slice(0, 6), 6, [0, 1, 2, 3, 4, 5]),
        (slice(4, 6), 2, [0, 10, 20, 30, 0, 1]),
        (slice(5, None), 1, [0, 10, 20, 30, 40, 0]),
        ((2, 4), 2, [0, 10, 0, 1, 40, 50]),
        (slice(3, 3), 0, [0, 10, 20, 30, 40, 50]),  # empty window: a no-op
    ],
)
def test_update_rows_still_updates_an_in_range_window(
    tmp_path, row_slice, payload_rows, expected_a
):
    path = _write_table(str(tmp_path / "t.fits"))
    torchfits.table.update_rows(path, _numeric_payload(payload_rows), row_slice)
    assert _n_rows(path) == 6
    assert _values(path) == expected_a


def test_update_rows_still_updates_an_in_range_window_forced_mmap(tmp_path):
    path = _write_table(str(tmp_path / "t.fits"))
    torchfits.table.update_rows(path, _typed_payload(6), slice(0, 6), mmap=True)
    assert _n_rows(path) == 6
    assert _values(path) == [0, 1, 2, 3, 4, 5]


def test_append_rows_is_still_the_way_to_add_rows(tmp_path):
    """The guard must not make growth impossible -- append_rows is unchanged."""
    path = _write_table(str(tmp_path / "t.fits"))
    torchfits.table.append_rows(path, {"A": [60, 70], "B": [6.0, 7.0]})
    assert _n_rows(path) == 8
    assert _values(path) == [0, 10, 20, 30, 40, 50, 60, 70]


def test_payload_window_mismatch_is_still_its_own_error(tmp_path):
    """The pre-existing payload/window length check must keep its own message."""
    path = _write_table(str(tmp_path / "t.fits"))
    with pytest.raises(ValueError, match="update payload has"):
        torchfits.table.update_rows(path, _numeric_payload(3), slice(0, 6))
    assert _n_rows(path) == 6


# ---------------------------------------------------------------------------
# why the guard has to exist: the writers disagree
# ---------------------------------------------------------------------------


def test_the_cpp_writers_disagree_about_an_over_long_window(tmp_path):
    """Documents the asymmetry the Python boundary now papers over.

    The mmap writer refuses; the CFITSIO writer grows the table. If this test
    ever starts failing because the C++ side learned to refuse, the Python
    guard becomes redundant and can be dropped (but the up-front check is
    still the only thing that keeps the *two routes* consistent).
    """
    path = _write_table(str(tmp_path / "m.fits"))
    payload = {
        "A": np.arange(6, dtype=np.int64),
        "B": np.zeros(6, dtype=np.float32),
    }
    with pytest.raises(RuntimeError, match="Row range exceeds table length"):
        cpp.update_fits_table_rows_mmap(path, 1, payload, 5, 6)

    other = _write_table(str(tmp_path / "c.fits"))
    cpp.update_fits_table_rows(other, 1, payload, 5, 6)
    assert _n_rows(other) == 10, "the CFITSIO writer is expected to grow the table"


def test_update_rows_refusal_names_the_table_size(tmp_path):
    """The error must be actionable: it says what the table actually holds."""
    path = _write_table(str(tmp_path / "t.fits"))
    with pytest.raises(ValueError) as excinfo:
        torchfits.table.update_rows(path, _numeric_payload(6), slice(4, 10))
    message = str(excinfo.value)
    assert "6" in message, message
    assert "update_rows" in message or "update" in message, message


def test_stale_header_row_count_does_not_become_a_grown_table(tmp_path, monkeypatch):
    """The up-front check is not the only guard -- defence in depth.

    If ``_naxis2_row_count`` reports more rows than the file actually holds
    (a stale header is reachable: the path caches carry a signature, and
    R2-014 was about exactly this), the up-front check passes and the mmap
    writer is the last line of defence. It refuses ("Row range exceeds table
    length"), and ``update_rows`` must propagate that refusal rather than
    retry through the writer that grows the table.
    """
    import torchfits._table.mutation as mutation_mod

    path = _write_table(str(tmp_path / "t.fits"))
    digest_before = _digest(path)
    real = mutation_mod._naxis2_row_count
    monkeypatch.setattr(
        mutation_mod,
        "_naxis2_row_count",
        lambda header_map, p: max(real(header_map, p), 10),
    )

    with pytest.raises(RuntimeError, match="Row range exceeds table length"):
        torchfits.table.update_rows(path, _typed_payload(6), slice(4, 10))

    assert _digest(path) == digest_before
    assert _n_rows(path) == 6
