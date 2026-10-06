"""A list of paths must obey exactly the same read contract as one path (R2-016).

`read_unified` dispatches on `isinstance(path, (list, tuple))` and returns
*before* the validation block, so a list ran none of `_validate_single_path_params`
and none of the `mode='image' + table options` guard. `_read_batch_paths` then
took the C++ `read_images_batch` fast path whenever `mmap is True`, and that
call knows nothing about mode, columns, a row window or `return_header` --
and, measured, does not even fail on a BINTABLE: it returned a zero-length
tensor, so the per-file fallback never got the chance to produce what the
caller asked for.

Every case below therefore compares the list form against the single-path form,
which is the reference contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
import torchfits._C as cpp


@pytest.fixture
def images(tmp_path):
    """Two distinct 4x4 images."""
    paths = []
    for name, fill in (("a.fits", 1.0), ("b.fits", 2.0)):
        p = str(tmp_path / name)
        torchfits.write(p, torch.full((4, 4), fill, dtype=torch.float32))
        paths.append(p)
    return paths


@pytest.fixture
def tables(tmp_path):
    """Two BINTABLE files with columns A (int) and B (float)."""
    paths = []
    for name in ("t1.fits", "t2.fits"):
        p = str(tmp_path / name)
        torchfits.table.write(
            p,
            {
                "A": np.arange(5, dtype=np.int32),
                "B": np.arange(5, dtype=np.float64),
            },
        )
        paths.append(p)
    return paths


def _shape_of(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, tuple):
        return ("pair", _shape_of(value[0]))
    if isinstance(value, dict):
        return ("table", tuple(sorted(value)))
    return type(value).__name__


def _count_batch_calls(call):
    """Run `call` and return (result, times read_images_batch was invoked).

    Proving the batch C++ fast path was skipped is sharper than inspecting the
    answer: the answers for a few of these cases are decided elsewhere, and the
    bug was that a request the fast path cannot express was handed to it.
    """
    calls = []
    original = cpp.read_images_batch

    def spy(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    cpp.read_images_batch = spy
    try:
        try:
            result = call()
        except Exception as exc:  # noqa: BLE001 - a raise is a valid outcome
            result = type(exc).__name__
        return result, len(calls)
    finally:
        cpp.read_images_batch = original


def _both(paths, kwargs):
    """Return (single-form result, list-form result), each as a comparable tag.

    The list form is driven with mmap=True because that is what selects the
    batch C++ fast path -- unless the case under test sets mmap itself, in
    which case that value wins rather than colliding with it.
    Exceptions are compared by type so the two forms have to fail the same way,
    not just return the same thing.
    """
    list_kwargs = {"mmap": True, **kwargs}

    def run(call, per_element=False):
        try:
            value = call()
        except Exception as exc:  # noqa: BLE001 - the tag is the type
            return type(exc).__name__
        if per_element and isinstance(value, list):
            return [_shape_of(item) for item in value]
        return _shape_of(value)

    single = run(lambda: torchfits.read(paths[0], **kwargs))
    listed = run(lambda: torchfits.read(paths, **list_kwargs), per_element=True)
    if isinstance(listed, list) and listed:
        listed = [listed[0], listed[-1]]
    return single, listed


# --- validation must not be skipped just because there are several paths ----


@pytest.mark.parametrize(
    ("label", "paths_fixture", "kwargs", "expected"),
    [
        ("mode='bogus' rejects", "images", {"mode": "bogus"}, "ValueError"),
        ("mmap='bogus' rejects", "images", {"mmap": "bogus"}, "ValueError"),
        ("negative hdu rejects", "images", {"hdu": -1}, "ValueError"),
        (
            "mode='image' + columns rejects",
            "images",
            {"mode": "image", "columns": ["A"]},
            "ValueError",
        ),
        (
            "mode='image' + row window rejects",
            "images",
            {"mode": "image", "start_row": 2, "num_rows": 2},
            "ValueError",
        ),
    ],
)
def test_list_read_rejects_what_a_single_read_rejects(
    request, label, paths_fixture, kwargs, expected
):
    paths = request.getfixturevalue(paths_fixture)
    single, listed = _both(paths, kwargs)
    assert single == expected, f"{label}: single-path form gave {single!r}"
    assert listed == single, (
        f"{label}: the list form diverged from the single-path form "
        f"(single={single!r}, list={listed!r})"
    )


# --- the fast path must not answer for a request it cannot express ---------


def test_return_header_is_honoured_for_a_list(images):
    """The batch C++ call returns bare tensors; return_header must still apply."""
    single, listed = _both(images, {"return_header": True})
    # _shape_of tags a (data, header) pair as ("pair", <data shape>).
    assert single[0] == "pair", f"single-path form returned {single!r}"
    assert listed == [single, single], listed


def test_mode_table_on_images_fails_the_same_way_for_a_list(images):
    """mode='table' on image files must raise, not quietly return pixels."""
    single, listed = _both(images, {"mode": "table"})
    assert single == "RuntimeError", single
    assert listed == single, f"list form returned {listed!r} for mode='table'"


def test_columns_on_tables_is_not_answered_by_the_image_batch_path(tables):
    """A column selection must never be handed to the batch image reader.

    This was the sharpest face of the bug: read_images_batch on a BINTABLE
    returned a zero-length tensor rather than raising, so the per-file loop
    that would have applied the column selection never ran.
    """
    single, listed = _both(tables, {"columns": ["A"]})
    assert listed[0] == single, f"single={single!r} list={listed!r}"
    _, batch_calls = _count_batch_calls(
        lambda: torchfits.read(tables, mmap=True, columns=["A"])
    )
    assert batch_calls == 0, (
        f"the batch image reader ran {batch_calls} time(s) for a column-selected "
        "read; it cannot express columns and does not fail on a BINTABLE"
    )


def test_row_window_on_tables_is_not_answered_by_the_image_batch_path(tables):
    single, listed = _both(tables, {"start_row": 2, "num_rows": 2})
    assert listed[0] == single, f"single={single!r} list={listed!r}"
    _, batch_calls = _count_batch_calls(
        lambda: torchfits.read(tables, mmap=True, start_row=2, num_rows=2)
    )
    assert batch_calls == 0, (
        f"the batch image reader ran {batch_calls} time(s) for a row-windowed "
        "read; it cannot express a row window"
    )


def test_return_header_is_not_answered_by_the_image_batch_path(images):
    """read_images_batch returns bare tensors, so it cannot carry a header."""
    _, batch_calls = _count_batch_calls(
        lambda: torchfits.read(images, mmap=True, return_header=True)
    )
    assert batch_calls == 0, (
        f"the batch image reader ran {batch_calls} time(s) for a "
        "return_header read; it returns bare tensors"
    )


def test_mode_table_is_not_answered_by_the_image_batch_path(images):
    _, batch_calls = _count_batch_calls(
        lambda: torchfits.read(images, mmap=True, mode="table")
    )
    assert batch_calls == 0, (
        f"the batch image reader ran {batch_calls} time(s) for mode='table'"
    )


# --- the fast path must still be taken when it is valid --------------------


def test_plain_image_list_read_still_uses_the_batch_fast_path(images):
    """The fix must not quietly turn every batch read into N single reads."""
    calls = []
    original = cpp.read_images_batch

    def spy(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    cpp.read_images_batch = spy
    try:
        torchfits.read(images, mmap=True)
    finally:
        cpp.read_images_batch = original
    assert len(calls) == 1, f"batch fast path called {len(calls)} time(s)"
    assert list(calls[0]) == images


def test_plain_image_list_read_matches_single_read_values(images):
    """Sanity: the fast path that remains still returns the right pixels."""
    listed = torchfits.read(images, mmap=True)
    assert len(listed) == 2
    for got, path in zip(listed, images):
        expected = torchfits.read(path)
        assert torch.equal(got, expected), f"{path}: batch result differs"
        assert float(got[0, 0]) in (1.0, 2.0)
