"""`table.scan`/`scan_torch` must validate at call time, like `read` (R2-029).

Both entry points carry the comment

    # Eager guard: a generator body would defer this until first next().
    path = coerce_fits_path(path)
    guard_fits_path(path)

so the intent is explicit: argument validation must not be deferred. Only
``path`` was. ``backend``, ``batch_size`` and ``row_slice`` are checked inside
``_scan_iter``/``_scan_torch_iter``, which are generators, so those checks fired
on the first ``next()`` -- after the caller had already built a pipeline around
the scan. ``table.read`` raised the same errors at call time, so the two entry
points disagreed about when the request was rejected.

Measured before the fix (call the function, do NOT iterate):

    scan(p, backend='bogus')             -> generator, no error at call time
    scan(p, backend=['x'])               -> generator, no error at call time
    scan(p, batch_size=0)                -> generator, no error at call time
    scan(p, batch_size=-5)               -> generator, no error at call time
    scan(p, row_slice=slice(0,10,2))     -> generator, no error at call time
    scan(p, row_slice=slice(0,-1))       -> generator, no error at call time
    scan_torch(p, batch_size=0)          -> generator, no error at call time
    scan_torch(p, row_slice=slice(0,10,2))-> generator, no error at call time
    scan_torch(p, device='bogus')        -> generator, no error at call time

    read(p, backend='bogus')             -> ValueError at CALL time
    read(p, row_slice=slice(0,10,2))     -> ValueError at CALL time

``reader()`` sits on top of ``scan`` and already forced the issue by pulling
the first batch eagerly, so it behaved correctly by accident.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as afits

import torchfits


@pytest.fixture
def table_path(tmp_path):
    path = str(tmp_path / "t.fits")
    afits.HDUList(
        [
            afits.PrimaryHDU(),
            afits.BinTableHDU.from_columns(
                [
                    afits.Column(name="A", format="J", array=np.arange(5)),
                    afits.Column(name="B", format="E", array=np.arange(5) * 1.5),
                ]
            ),
        ]
    ).writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["bogus", "", "CPP", ["x"], ("auto",), 7, None])
def test_scan_rejects_a_bad_backend_at_call_time(table_path, backend):
    """No iteration: the *call* must raise, exactly as table.read does."""
    with pytest.raises(ValueError, match="backend must be one of"):
        torchfits.table.scan(table_path, backend=backend)


@pytest.mark.parametrize("backend", ["bogus", ["x"], 7])
def test_read_and_scan_agree_on_a_bad_backend(table_path, backend):
    """The two entry points must not disagree about *when* they reject."""
    errors = []
    for entry in (torchfits.table.read, torchfits.table.scan):
        with pytest.raises(Exception) as excinfo:
            entry(table_path, backend=backend)
        errors.append(type(excinfo.value))
    assert errors[0] is errors[1], errors


# ---------------------------------------------------------------------------
# batch_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [0, -1, -5])
def test_scan_rejects_a_non_positive_batch_size_at_call_time(table_path, batch_size):
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        torchfits.table.scan(table_path, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [0, -1, -5])
def test_scan_torch_rejects_a_non_positive_batch_size_at_call_time(
    table_path, batch_size
):
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        torchfits.table.scan_torch(table_path, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [0, -5])
def test_read_ignores_batch_size_where_scan_cannot(table_path, batch_size):
    """A deliberate asymmetry, pinned so it cannot drift silently.

    ``read`` has a single-chunk fast path, so for a small table ``batch_size``
    is genuinely unused and it accepts the value; ``scan`` must batch, so a
    non-positive one is an error. This is *not* filed as a finding -- both
    behaviours are defensible and there is no measured wrong answer -- but the
    difference should not be rediscovered by surprise.
    """
    table = torchfits.table.read(table_path, batch_size=batch_size)
    assert table.num_rows == 5
    for entry in (torchfits.table.scan, torchfits.table.scan_torch):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            entry(table_path, batch_size=batch_size)


# ---------------------------------------------------------------------------
# row_slice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row_slice, message",
    [
        (slice(0, 10, 2), "step must be 1"),
        (slice(None, None, -1), "step must be 1"),
        (slice(0, -1), "negative stop is not supported"),
        (slice(-1, 3), "start must be >= 0"),
        ((0, 1, 2), "must be \\(start, stop\\)"),
        ("nope", "must be a slice"),
    ],
)
def test_scan_rejects_a_bad_row_slice_at_call_time(table_path, row_slice, message):
    with pytest.raises(ValueError, match=message):
        torchfits.table.scan(table_path, row_slice=row_slice)


@pytest.mark.parametrize(
    "row_slice, message",
    [
        (slice(0, 10, 2), "step must be 1"),
        (slice(0, -1), "negative stop is not supported"),
        (slice(-1, 3), "start must be >= 0"),
        ("nope", "must be a slice"),
    ],
)
def test_scan_torch_rejects_a_bad_row_slice_at_call_time(
    table_path, row_slice, message
):
    with pytest.raises(ValueError, match=message):
        torchfits.table.scan_torch(table_path, row_slice=row_slice)


@pytest.mark.parametrize(
    "row_slice", [slice(0, 10, 2), slice(0, -1), slice(-1, 3), "nope"]
)
def test_read_agrees_with_scan_on_a_bad_row_slice(table_path, row_slice):
    errors = []
    for entry in (torchfits.table.read, torchfits.table.scan):
        with pytest.raises(Exception) as excinfo:
            entry(table_path, row_slice=row_slice)
        errors.append(type(excinfo.value))
    assert errors[0] is errors[1], errors


# ---------------------------------------------------------------------------
# device (scan_torch only)
# ---------------------------------------------------------------------------


def test_scan_torch_rejects_a_bad_device_at_call_time(table_path):
    with pytest.raises(ValueError, match="device"):
        torchfits.table.scan_torch(table_path, device="not-a-device")


# ---------------------------------------------------------------------------
# no regression: valid requests still return a working, lazy generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"backend": "auto"},
        {"backend": "torch"},
        {"backend": "cpp"},
        {"batch_size": 2},
        {"row_slice": slice(1, 3)},
        {"row_slice": (0, 5)},
        {"row_slice": slice(4, 4)},  # empty window: valid, yields nothing
        {"columns": ["A"]},
        {"where": "A > 2"},
    ],
)
def test_scan_still_accepts_every_valid_request(table_path, kwargs):
    batches = list(torchfits.table.scan(table_path, **kwargs))
    for batch in batches:
        assert "A" in batch.schema.names
    if kwargs.get("columns") == ["A"]:
        for batch in batches:
            assert batch.schema.names == ["A"]


def test_scan_torch_still_accepts_valid_requests(table_path):
    chunks = list(torchfits.table.scan_torch(table_path, batch_size=2))
    assert chunks
    assert torch.is_tensor(chunks[0]["A"])


def test_scan_is_still_lazy(table_path):
    """Eager *validation* must not make the scan eager: no batches are read
    until the caller asks for one."""
    import torchfits._table._read_scan as scan_mod

    calls = []
    real = scan_mod._iter_chunks_cpp_table

    def counted(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    scan_mod._iter_chunks_cpp_table = counted
    try:
        iterator = torchfits.table.scan(table_path)
        assert calls == [], "scan() read a chunk before the first next()"
        next(iterator)
        assert calls, "the first next() did not read anything"
    finally:
        scan_mod._iter_chunks_cpp_table = real


import torch  # noqa: E402  (used by test_scan_torch_still_accepts_valid_requests)
