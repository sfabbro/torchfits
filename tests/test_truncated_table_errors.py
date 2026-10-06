"""Truncated FITS tables must raise clean errors instead of SIGBUS."""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits as afits

import torchfits
import torchfits.table as ttable


def _open_fd_count() -> int:
    for fd_dir in ("/dev/fd", "/proc/self/fd"):
        if os.path.isdir(fd_dir):
            return len(os.listdir(fd_dir))
    pytest.skip("no /dev/fd or /proc/self/fd to count descriptors with")


@pytest.fixture()
def truncated_table(tmp_path):
    path = tmp_path / "trunc.fits"
    col = afits.Column(name="A", format="J", array=np.arange(50000, dtype="<i4"))
    afits.BinTableHDU.from_columns([col]).writeto(str(path), overwrite=True)
    raw = path.read_bytes()
    # Drop ~15k rows of tail data: header still claims 50000 rows.
    path.write_bytes(raw[: len(raw) - 60000])
    return str(path)


def test_mmap_read_raises_clean_error(truncated_table):
    with pytest.raises(RuntimeError, match="truncat"):
        torchfits.read(truncated_table, hdu=1, mmap=True)


def test_filtered_read_raises_clean_error(truncated_table):
    with pytest.raises(RuntimeError, match="truncat"):
        ttable.read(truncated_table, hdu=1, where="A > 3")


def test_mmap_update_tail_rows_raises_clean_error(truncated_table):
    payload = {"A": np.zeros(5, dtype=np.int32)}
    with pytest.raises(RuntimeError, match="truncat"):
        ttable.update_rows(
            truncated_table, payload, row_slice=(49995, 50000), mmap=True
        )


def test_intact_file_unaffected(tmp_path):
    path = tmp_path / "ok.fits"
    col = afits.Column(name="A", format="J", array=np.arange(1000, dtype="<i4"))
    afits.BinTableHDU.from_columns([col]).writeto(str(path), overwrite=True)
    out = torchfits.read(str(path), hdu=1, mmap=True)
    assert out["A"].shape[0] == 1000


def test_truncated_vla_read_raises_clean_error(tmp_path):
    """A VLA heap chopped off mid-file must raise, not return partial rows."""
    path = tmp_path / "vla_trunc.fits"
    rng = np.random.default_rng(7)
    vla = np.empty(2000, dtype=object)
    for i in range(2000):
        vla[i] = rng.normal(size=(i % 17 + 1)).astype("<f8")
    cols = [
        afits.Column(name="N", format="J", array=np.arange(2000, dtype="<i4")),
        afits.Column(name="V", format="PD()", array=vla),
    ]
    afits.BinTableHDU.from_columns(cols).writeto(str(path), overwrite=True)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) - 40000])
    with pytest.raises(RuntimeError):
        torchfits.read(str(path), hdu=1)


def test_truncated_projection_read_raises_clean_error(truncated_table):
    """Single-column projections validate the extent just like full reads."""
    with pytest.raises(RuntimeError, match="truncat"):
        torchfits.read(truncated_table, hdu=1, columns=["A"], mmap=True)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param("read", id="mmap-read"),
        pytest.param("update", id="mmap-update"),
    ],
)
def test_rejected_truncated_mmap_call_releases_its_descriptor(
    truncated_table, operation
):
    """A rejected mmap read/update must not leak the descriptor it opened.

    Both mmap paths did ``open`` -> ``fstat`` -> ``ensure_extent_within_file`` ->
    ``mmap``, and the extent check is what rejects a truncated file -- so the
    descriptor had no owner yet at the only point where these calls throw.
    Unwinding does not close a raw ``int``. Measured before the fix: 100
    rejected ``torchfits.read(..., mmap=True)`` calls retained 100 descriptors,
    ``table.read_torch(..., mmap=True)`` 200, and the same on the update path.
    A long job that keeps hitting a half-written file (the reason the check
    exists) exhausts its descriptor table and then fails every open.

    The descriptor is now adopted by an RAII guard as soon as ``open``
    succeeds, and handed to the mapping guard only once the mapping exists.
    """
    import torchfits._C as cpp

    if operation == "read":
        call = lambda: torchfits.read(truncated_table, hdu=1, mmap=True)  # noqa: E731
    else:
        payload = {"A": np.zeros(5, dtype=np.int32)}
        call = lambda: ttable.update_rows(  # noqa: E731
            truncated_table, payload, row_slice=(49995, 50000), mmap=True
        )

    calls = 40
    # Warm up first: the first call may legitimately cache a handle (the
    # per-thread reader and the shared read metadata both hold descriptors).
    with pytest.raises(RuntimeError, match="truncat"):
        call()
    before = _open_fd_count()
    for _ in range(calls):
        with pytest.raises(RuntimeError, match="truncat"):
            call()
    leaked = _open_fd_count() - before
    assert leaked < calls, (
        f"{operation} leaked {leaked} descriptors over {calls} rejected calls"
    )
    # The binding is the same code path the public entry points take; pin it
    # directly too so a future wrapper cannot hide a regression here.
    with pytest.raises(RuntimeError, match="truncat"):
        cpp.read_fits_table_rows(truncated_table, 1, [], 1, -1, True)
