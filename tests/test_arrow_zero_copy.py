"""Zero-copy path from native table buffers to PyArrow arrays.

The native reader hands Python a read-only memoryview that owns the decoded
column storage (the extension exports it through a buffer-protocol type).
Arrow can wrap that buffer directly (``Array.from_buffers``); copying it through
``np.frombuffer(...).copy()`` and ``pa.array`` allocated two extra buffers per
column.  These tests pin the buffer identity, not just the values, so a future
change cannot quietly reintroduce the copies.
"""

import gc

import numpy as np
import pyarrow as pa

from torchfits._table.arrow_convert import (
    _numpy_buffer_to_arrow,
    _numpy_to_arrow_array,
    _raw_column_to_arrow_array,
    _raw_column_to_arrow_direct,
    _raw_column_to_numpy,
    _vla_tuple_to_arrow_array,
)


def _readonly_view(flat):
    """A read-only memoryview, like the native transport produces."""
    return memoryview(bytes(flat.tobytes()))


def _raw_fixed(values, dtype):
    flat = np.ascontiguousarray(values, dtype=dtype)
    view = memoryview(bytearray(flat.tobytes()))
    return (
        {
            "kind": "fixed",
            "dtype": np.dtype(dtype).name,
            "shape": list(flat.shape),
            "data": view,
        },
        flat,
    )


def _values_address(arrow_array):
    """Data buffer address; list types keep it on the values child."""
    if (
        pa.types.is_fixed_size_list(arrow_array.type)
        or pa.types.is_list(arrow_array.type)
        or pa.types.is_large_list(arrow_array.type)
    ):
        return arrow_array.values.buffers()[1].address
    return arrow_array.buffers()[1].address


def test_raw_column_to_numpy_does_not_copy():
    raw, flat = _raw_fixed(np.arange(12, dtype=np.int32).reshape(3, 4), np.int32)
    view = _raw_column_to_numpy(raw)
    assert view.shape == (3, 4)
    assert np.shares_memory(view, np.frombuffer(raw["data"], dtype=np.int32))


def test_raw_vla_column_does_not_copy():
    values = np.arange(6, dtype=np.float64)
    offsets = np.array([0, 2, 6], dtype=np.int64)
    raw = {
        "kind": "vla",
        "dtype": "float64",
        "shape": [2],
        "data": memoryview(bytearray(values.tobytes())),
        "offsets": memoryview(bytearray(offsets.tobytes())),
    }
    value_view, offset_view = _raw_column_to_numpy(raw)
    assert np.shares_memory(value_view, np.frombuffer(raw["data"], np.float64))
    assert np.shares_memory(offset_view, np.frombuffer(raw["offsets"], np.int64))


def test_arrow_wraps_the_numpy_buffer_for_every_numeric_dtype():
    for dtype in (
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ):
        raw, _ = _raw_fixed(np.arange(7, dtype=np.int64).astype(dtype), dtype)
        view = _raw_column_to_numpy(raw)
        arrow_array = _numpy_buffer_to_arrow(pa, view)
        assert arrow_array is not None, dtype
        assert _values_address(arrow_array) == view.ctypes.data, dtype
        assert arrow_array.to_pylist() == view.tolist(), dtype


def test_non_native_byte_order_is_not_wrapped():
    view = np.arange(5, dtype=">i4")
    assert not view.dtype.isnative
    assert _numpy_buffer_to_arrow(pa, view) is None


def test_two_dimensional_column_keeps_the_flat_buffer():
    raw, flat = _raw_fixed(np.arange(12, dtype=np.float32).reshape(3, 4), np.float32)
    view = _raw_column_to_numpy(raw)
    arrow_array = _numpy_to_arrow_array(pa, view, False, "ascii", True)
    assert arrow_array.type == pa.list_(pa.float32(), 4)
    assert _values_address(arrow_array) == view.reshape(-1).ctypes.data


def test_null_sentinel_shares_data_and_adds_a_bitmap():
    raw, flat = _raw_fixed(np.array([1, -99, 3, 4], dtype=np.int32), np.int32)
    view = _raw_column_to_numpy(raw)
    arrow_array = _numpy_to_arrow_array(
        pa, view, False, "ascii", True, null_sentinel=-99
    )
    assert arrow_array.to_pylist() == [1, None, 3, 4]
    assert arrow_array.null_count == 1
    assert _values_address(arrow_array) == view.ctypes.data
    assert arrow_array.buffers()[0] is not None


def test_arrow_array_outlives_the_native_result_dict():
    raw, flat = _raw_fixed(np.arange(6, dtype=np.int32).reshape(2, 3), np.int32)
    arrow_array = _numpy_to_arrow_array(
        pa, _raw_column_to_numpy(raw), False, "ascii", True
    )
    del raw, flat
    gc.collect()
    assert arrow_array.to_pylist() == [[0, 1, 2], [3, 4, 5]]


def test_vla_values_are_zero_copy():
    values = np.arange(8, dtype=np.float64)
    offsets = np.array([0, 3, 8], dtype=np.int64)
    raw = {
        "kind": "vla",
        "dtype": "float64",
        "shape": [2],
        "data": memoryview(bytearray(values.tobytes())),
        "offsets": memoryview(bytearray(offsets.tobytes())),
    }
    materialized = _raw_column_to_numpy(raw)
    arrow_array = _vla_tuple_to_arrow_array(pa, materialized)
    assert arrow_array.to_pylist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0, 6.0, 7.0]]
    assert _values_address(arrow_array) == materialized[0].ctypes.data


def test_direct_path_wraps_a_read_only_buffer_without_numpy():
    values = np.arange(12, dtype=np.float64)
    raw = {
        "kind": "fixed",
        "dtype": "float64",
        "shape": [12],
        "data": _readonly_view(values),
    }
    assert raw["data"].readonly
    arrow_array = _raw_column_to_arrow_direct(pa, raw, True, "D")
    assert arrow_array is not None
    assert _values_address(arrow_array) == pa.py_buffer(raw["data"]).address
    assert arrow_array.to_pylist() == values.tolist()


def test_direct_path_keeps_two_dimensional_numeric_columns_flat():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw = {
        "kind": "fixed",
        "dtype": "float32",
        "shape": [3, 4],
        "data": _readonly_view(values),
    }
    arrow_array = _raw_column_to_arrow_array(
        pa, raw, False, "ascii", True, fits_tform="4E"
    )
    assert arrow_array.type == pa.list_(pa.float32(), 4)
    assert _values_address(arrow_array) == pa.py_buffer(raw["data"]).address


def test_direct_path_wraps_a_char_matrix_as_fixed_binary():
    values = np.array([list(b"abc"), list(b"def")], dtype=np.uint8)
    raw = {
        "kind": "fixed",
        "dtype": "uint8",
        "shape": [2, 3],
        "data": _readonly_view(values),
    }
    arrow_array = _raw_column_to_arrow_array(
        pa, raw, False, "ascii", True, fits_tform="3A"
    )
    assert arrow_array.type == pa.binary(3)
    assert arrow_array.to_pylist() == [b"abc", b"def"]


def test_direct_path_defers_masked_and_unsigned_columns():
    values = np.arange(4, dtype=np.int32)
    raw = {
        "kind": "fixed",
        "dtype": "int32",
        "shape": [4],
        "data": _readonly_view(values),
    }
    assert _raw_column_to_arrow_direct(pa, raw, False, "J") is not None
    masked = _raw_column_to_arrow_array(pa, raw, False, "ascii", True, null_sentinel=-1)
    assert masked.to_pylist() == [0, 1, 2, 3]
    widened = _raw_column_to_arrow_array(
        pa, raw, False, "ascii", True, unsigned_dtype="uint32"
    )
    assert widened.type == pa.uint32()


def test_arrow_table_keeps_the_native_column_storage_alive(tmp_path):
    """The buffer exporter owns the decoded tensor; Arrow must outlive it."""
    import torchfits

    values = np.arange(4096, dtype=np.float64)
    path = tmp_path / "zero_copy_lifetime.fits"
    torchfits.table.write(str(path), {"X": values}, overwrite=True)

    arrow_table = torchfits.table.read(str(path))
    gc.collect()
    # Reuse freed heap memory: a dangling Arrow buffer would now read garbage.
    junk = np.full(8 * 1024 * 1024, 7.0)
    try:
        assert arrow_table.column("X").to_pylist() == values.tolist()
    finally:
        del junk
