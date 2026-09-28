"""Byteswap regression tests.

FITS files are always big-endian on disk.  On little-endian hosts the
torchfits C++ layer must byteswap multi-byte integer and floating-point
values when reading raw data through the mmap / pread fast paths.

These tests write known data, read it back through each code path, and
verify the bytes are correctly decoded.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile

import numpy as np
import pytest
import torch

import torchfits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image(path: str, data: np.ndarray) -> None:
    """Write a numpy array as a FITS primary HDU (uncompressed)."""
    from astropy.io import fits

    fits.PrimaryHDU(data).writeto(path, overwrite=True)


def _write_table(path: str, columns: dict[str, np.ndarray]) -> None:
    """Write a dict of columns as a FITS binary table in extension 1."""
    from astropy.io import fits

    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name=name, format=_np_to_fits_format(arr), array=arr)
            for name, arr in columns.items()
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _np_to_fits_format(arr: np.ndarray) -> str:
    """Return a FITS TFORM string for a numpy array."""
    mapping = {
        np.dtype("bool"): "L",
        np.dtype("uint8"): "B",
        np.dtype("int16"): "I",
        np.dtype(">i2"): "I",
        np.dtype("int32"): "J",
        np.dtype(">i4"): "J",
        np.dtype("int64"): "K",
        np.dtype(">i8"): "K",
        np.dtype("float32"): "E",
        np.dtype(">f4"): "E",
        np.dtype("float64"): "D",
        np.dtype(">f8"): "D",
    }
    for dt, fmt in mapping.items():
        if arr.dtype == dt:
            return fmt
    raise ValueError(f"Unsupported dtype for FITS table: {arr.dtype}")


# ---------------------------------------------------------------------------
# Image byte-swap tests
# ---------------------------------------------------------------------------

IMAGE_DTYPES = [
    ("int16", np.int16, torch.int16),
    ("int32", np.int32, torch.int32),
    ("int64", np.int64, torch.int64),
    ("float32", np.float32, torch.float32),
    ("float64", np.float64, torch.float64),
]


@pytest.mark.parametrize("name,np_dtype,torch_dtype", IMAGE_DTYPES)
@pytest.mark.parametrize("mmap", [True, False])
def test_image_roundtrip_byteswap(name, np_dtype, torch_dtype, mmap):
    """Write a small image with known values, read back, verify exact match."""
    # Use values whose byte patterns are obviously wrong if byteswapped.
    # e.g. int32:  0x01020304  → 0x04030201 if swapped incorrectly
    shape = (17, 23)  # Prime dimensions to avoid accidental alignment bugs
    rng = np.random.default_rng(42)

    if np.issubdtype(np_dtype, np.integer):
        # Exclude values near the dtype boundaries that could overflow.
        info = np.iinfo(np_dtype)
        data = rng.integers(
            max(info.min, -10_000_000),
            min(info.max, 10_000_000),
            size=shape,
            dtype=np_dtype,
        )
    else:
        data = rng.uniform(-1000, 1000, size=shape).astype(np_dtype)

    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    path = handle.name

    try:
        _write_image(path, data)

        tensor = torchfits.read_tensor(path, mmap=mmap)

        assert tensor.dtype == torch_dtype, (
            f"Expected {torch_dtype}, got {tensor.dtype}"
        )
        assert tensor.shape == shape

        np.testing.assert_array_equal(tensor.cpu().numpy(), data)

        # Also test through the unified read path
        tensor2 = torchfits.read(path, mmap=mmap)
        assert tensor2.dtype == torch_dtype
        np.testing.assert_array_equal(tensor2.cpu().numpy(), data)

    finally:
        os.unlink(path)


def test_int16_boundary_values():
    """Test int16 values near type boundaries that would fail with wrong byte order."""
    # Values whose bytes would produce wildly different results if swapped
    values = np.array([0, 1, -1, 32767, -32768, 0x0102, 0x7F01, -256], dtype=np.int16)
    shape = (2, 4)
    data = values.reshape(shape)

    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    path = handle.name

    try:
        _write_image(path, data)

        for mmap in (True, False):
            tensor = torchfits.read_tensor(path, mmap=mmap)
            np.testing.assert_array_equal(
                tensor.cpu().numpy(),
                data,
                err_msg=f"mmap={mmap}: boundary value mismatch",
            )

    finally:
        os.unlink(path)


def test_int32_boundary_values():
    """Test int32 boundary values for byte-swap correctness."""
    values = np.array(
        [0, 1, -1, 0x01020304, 0x7FFFFFFF, -0x80000000, 0x00FF00FF, -0x01020304],
        dtype=np.int32,
    )
    shape = (4, 2)
    data = values.reshape(shape)

    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    path = handle.name

    try:
        _write_image(path, data)

        for mmap in (True, False):
            tensor = torchfits.read_tensor(path, mmap=mmap)
            np.testing.assert_array_equal(
                tensor.cpu().numpy(),
                data,
                err_msg=f"mmap={mmap}: boundary value mismatch",
            )

    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Table column byte-swap tests
# ---------------------------------------------------------------------------

TABLE_DTYPES = [
    ("int16", np.int16, torch.int16),
    ("int32", np.int32, torch.int32),
    ("int64", np.int64, torch.int64),
    ("float32", np.float32, torch.float32),
    ("float64", np.float64, torch.float64),
]


@pytest.mark.parametrize("name,np_dtype,torch_dtype", TABLE_DTYPES)
@pytest.mark.parametrize("mmap", [True, False])
def test_table_column_roundtrip_byteswap(name, np_dtype, torch_dtype, mmap):
    """Write a table column with known values, read back via mmap/non-mmap."""
    nrows = 50
    rng = np.random.default_rng(99)

    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        col = rng.integers(
            max(info.min, -10_000_000),
            min(info.max, 10_000_000),
            size=nrows,
            dtype=np_dtype,
        )
    else:
        col = rng.uniform(-1000, 1000, size=nrows).astype(np_dtype)

    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    path = handle.name

    try:
        _write_table(path, {"COL": col})

        # Legacy read path (parametrized mmap)
        result = torchfits.read(path, hdu=1, mmap=mmap)
        tensor = result["COL"]

        assert tensor.dtype == torch_dtype, (
            f"Expected {torch_dtype}, got {tensor.dtype}"
        )
        assert tensor.shape == (nrows,)
        np.testing.assert_array_equal(tensor.cpu().numpy(), col)

        # Arrow table read path (default mmap)
        arrow_table = torchfits.table.read(path, hdu=1)
        got = np.asarray(arrow_table.column("COL").to_pylist())
        np.testing.assert_array_equal(got, col)

    finally:
        os.unlink(path)


def test_table_multiple_column_types():
    """Write a table with multiple byte widths, read back, verify all correct."""
    nrows = 30
    rng = np.random.default_rng(77)

    cols = {
        "I16": rng.integers(-1000, 1000, size=nrows, dtype=np.int16),
        "I32": rng.integers(-100_000, 100_000, size=nrows, dtype=np.int32),
        "F32": rng.uniform(-500, 500, size=nrows).astype(np.float32),
        "F64": rng.uniform(-500, 500, size=nrows).astype(np.float64),
    }

    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    path = handle.name

    try:
        _write_table(path, cols)

        # Read via torchfits legacy path (both mmap modes)
        for mmap in (True, False):
            result = torchfits.read(path, hdu=1, mmap=mmap)
            for name, expected in cols.items():
                got = result[name].cpu().numpy()
                np.testing.assert_array_equal(
                    got,
                    expected,
                    err_msg=f"Legacy read mmap={mmap}: column {name} mismatch",
                )

        # Read via Arrow table path
        arrow_table = torchfits.table.read(path, hdu=1)
        for name, expected in cols.items():
            got = np.asarray(arrow_table.column(name).to_pylist(), dtype=expected.dtype)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"Arrow read: column {name} mismatch"
            )

    finally:
        os.unlink(path)


def test_float32_special_values():
    """Test float32 NaN/Inf round-trip through byteswap paths."""
    data = np.array(
        [[0.0, -0.0, 1.0, -1.0], [np.nan, np.inf, -np.inf, 3.14159]], dtype=np.float32
    )

    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    path = handle.name

    try:
        _write_image(path, data)

        for mmap in (True, False):
            tensor = torchfits.read_tensor(path, mmap=mmap)
            got = tensor.cpu().numpy()

            # Exact match for non-NaN values
            mask = ~np.isnan(data)
            np.testing.assert_array_equal(
                got[mask], data[mask], err_msg=f"mmap={mmap}: non-NaN mismatch"
            )
            # NaNs should remain NaN
            nan_mask = np.isnan(data)
            assert np.all(np.isnan(got[nan_mask])), f"mmap={mmap}: NaN lost"
            assert not np.signbit(got[0, 0]) and np.signbit(got[0, 1]), (
                f"mmap={mmap}: signed zero lost"
            )

    finally:
        os.unlink(path)


def test_float64_special_values():
    """Test float64 NaN/Inf round-trip through byteswap paths."""
    data = np.array(
        [[0.0, -0.0, 1.0, -1.0], [np.nan, np.inf, -np.inf, 3.14159265358979]],
        dtype=np.float64,
    )

    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    path = handle.name

    try:
        _write_image(path, data)

        for mmap in (True, False):
            tensor = torchfits.read_tensor(path, mmap=mmap)
            got = tensor.cpu().numpy()

            mask = ~np.isnan(data)
            np.testing.assert_array_equal(
                got[mask], data[mask], err_msg=f"mmap={mmap}: non-NaN mismatch"
            )
            nan_mask = np.isnan(data)
            assert np.all(np.isnan(got[nan_mask])), f"mmap={mmap}: NaN lost"
            assert not np.signbit(got[0, 0]) and np.signbit(got[0, 1]), (
                f"mmap={mmap}: signed zero lost"
            )

    finally:
        os.unlink(path)


def test_int64_large_values():
    """Test int64 values that would be garbled by incorrect byte-swap."""
    data = np.array(
        [
            [0, 1, -1, 0x0102030405060708],
            [0x7FFFFFFFFFFFFFFF, -0x8000000000000000, 0x00FF00FF00FF00FF, 42],
        ],
        dtype=np.int64,
    )

    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    path = handle.name

    try:
        _write_image(path, data)

        for mmap in (True, False):
            tensor = torchfits.read_tensor(path, mmap=mmap)
            np.testing.assert_array_equal(
                tensor.cpu().numpy(), data, err_msg=f"mmap={mmap}: int64 mismatch"
            )

    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Direct test of the C++ helpers
# ---------------------------------------------------------------------------


def test_native_bswap_helpers_match_a_byte_wise_reference(tmp_path) -> None:
    """The SIMD helpers themselves, not just what a FITS read produces with them.

    The rest of this file checks byteswap *through the reader*, which means the
    tests only ever see whatever the vector loops happened to produce. These
    helpers had no direct test, and the real-observation suites cannot reach
    them: the unsigned fast paths in ``read_tensor_canonical`` are gated on
    ``!compressed``, while the CFHT corpus is Rice-compressed throughout. So
    neither the unit tests nor the real-data tests could tell a correct shuffle
    from a wrong one on an uncompressed unsigned frame -- and a wrong shuffle is
    silent, producing plausible pixels that are simply byte-reversed.

    A standalone C++ probe compares all five helpers against a byte-wise
    reference at element counts straddling every vector width and the scalar
    tail, and prints which SIMD branch was compiled in, because a branch that
    silently did nothing would still pass a comparison built from it.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = os.path.join(repo_root, "tests", "cpp", "test_bswap_helpers.cpp")
    include = os.path.join(repo_root, "src", "torchfits", "cpp_src")
    executable = str(tmp_path / "test_bswap_helpers")
    # CXX may carry flags as well as a program name ("ccache c++", "c++ -pipe"),
    # so split it the way tests/test_security.py does for the same job: passing
    # the raw string as one argv element raises FileNotFoundError on a perfectly
    # ordinary value.
    compiler = shlex.split(os.environ.get("CXX", "c++"))

    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-I",
            include,
            source,
            "-o",
            executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [executable], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, (
        f"a byteswap helper disagrees with the byte-wise reference:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "match the byte-wise reference" in result.stdout, result.stdout
    # Report which branch was exercised, so a run that silently fell through to
    # the scalar tail is visible in CI output rather than mistaken for coverage.
    print(result.stdout.splitlines()[0])
