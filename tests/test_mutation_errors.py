"""Mutation-API error contracts (R5 review, slice r5b).

Pins three contracts:

1. Unknown-column errors across the mutation API are ``KeyError`` naming the
   column (dict-like semantics), not ``ValueError``.
2. ``update_rows(..., mmap="auto")`` re-raises non-decode errors (e.g. IO)
   instead of silently swallowing them into a non-mmap fallback.
3. A malformed ``NAXIS2`` row count surfaces loudly; it is never coerced to
   ``0`` rows (which made ``delete_rows`` silently no-op and drop the
   deletion).
"""

from __future__ import annotations

import numpy as np
import pytest

import torchfits


def _write_table(path: str) -> None:
    torchfits.table.write(
        path,
        {
            "A": np.array([1, 2, 3], dtype=np.int32),
            "B": np.array([1.5, 2.5, 3.5], dtype=np.float64),
        },
        overwrite=True,
    )


@pytest.mark.parametrize(
    "op",
    [
        "update_rows",
        "append_rows",
        "insert_rows",
        "replace_column",
        "rename_columns",
        "drop_columns",
    ],
)
def test_unknown_column_raises_keyerror_naming_column(tmp_path, op):
    path = str(tmp_path / "t.fits")
    _write_table(path)
    calls = {
        "update_rows": lambda: torchfits.table.update_rows(
            path, {"ZZZ": np.array([1, 2], dtype=np.int32)}, row_slice=slice(0, 2)
        ),
        "append_rows": lambda: torchfits.table.append_rows(
            path, {"ZZZ": np.array([1], dtype=np.int32)}
        ),
        "insert_rows": lambda: torchfits.table.insert_rows(
            path, {"ZZZ": np.array([1], dtype=np.int32)}, row=0
        ),
        "replace_column": lambda: torchfits.table.replace_column(
            path, "ZZZ", np.array([1, 2, 3], dtype=np.int32)
        ),
        "rename_columns": lambda: torchfits.table.rename_columns(path, {"ZZZ": "Q"}),
        "drop_columns": lambda: torchfits.table.drop_columns(path, ["ZZZ"]),
    }
    with pytest.raises(KeyError, match="ZZZ"):
        calls[op]()


def test_update_rows_mmap_auto_reraises_non_decode_error(tmp_path, monkeypatch):
    """mmap='auto' must re-raise IO / non-decode errors, not silently fall back."""
    path = str(tmp_path / "t.fits")
    _write_table(path)

    import torchfits._C as cpp

    def boom(*args, **kwargs):
        raise OSError("No space left on device (simulated)")

    monkeypatch.setattr(cpp, "update_fits_table_rows_mmap", boom)
    with pytest.raises(OSError):
        torchfits.table.update_rows(
            path,
            {"A": np.array([9, 9], dtype=np.int32)},
            row_slice=slice(0, 2),
            mmap="auto",
        )


def test_update_rows_mmap_auto_still_falls_back_on_decode_error(tmp_path, monkeypatch):
    """mmap='auto' still falls back to the non-mmap writer when the mmap writer
    rejects a column layout (the C++ decode/feature error)."""
    path = str(tmp_path / "t.fits")
    _write_table(path)

    import torchfits._C as cpp

    def unsupported(*args, **kwargs):
        raise RuntimeError("Scaled columns not supported for mmap updates")

    monkeypatch.setattr(cpp, "update_fits_table_rows_mmap", unsupported)
    # Should NOT raise: falls back to the non-mmap writer.
    torchfits.table.update_rows(
        path,
        {"A": np.array([9, 9], dtype=np.int32)},
        row_slice=slice(0, 2),
        mmap="auto",
    )
    assert torchfits.table.read_torch(path, columns=["A"])["A"].tolist() == [9, 9, 3]


def _corrupt_naxis2(monkeypatch):
    """Make read_header report a non-integer NAXIS2 for whatever HDU it reads."""
    import torchfits._C as cpp

    orig = cpp.read_header

    def bad(handle, hdu_num):
        out = []
        for c in orig(handle, hdu_num):
            key = c[0] if isinstance(c, (list, tuple)) else getattr(c, "key", None)
            if key is not None and str(key).strip().upper() == "NAXIS2":
                out.append(("NAXIS2", "garbage", ""))
            else:
                out.append(c)
        return out

    monkeypatch.setattr(cpp, "read_header", bad)


@pytest.mark.parametrize(
    "op",
    ["delete_rows", "insert_rows", "insert_column", "replace_column"],
)
def test_malformed_naxis2_raises_not_silent_noop(tmp_path, monkeypatch, op):
    """A malformed NAXIS2 must raise, not be coerced to 0 rows.

    Before the fix NAXIS2 parse failures became 0 rows: ``delete_rows`` then
    silently returned without deleting (silent data loss) and the others
    misbehaved against a phantom zero-row table.  Now every reader raises.
    """
    path = str(tmp_path / "t.fits")
    _write_table(path)
    _corrupt_naxis2(monkeypatch)
    calls = {
        "delete_rows": lambda: torchfits.table.delete_rows(path, slice(0, 2)),
        "insert_rows": lambda: torchfits.table.insert_rows(
            path, {"A": np.array([9], dtype=np.int32)}, row=0
        ),
        "insert_column": lambda: torchfits.table.insert_column(
            path, "C", np.array([1, 2, 3], dtype=np.int32)
        ),
        "replace_column": lambda: torchfits.table.replace_column(
            path, "A", np.array([1, 2, 3], dtype=np.int32)
        ),
    }
    with pytest.raises((ValueError, TypeError)):
        calls[op]()
