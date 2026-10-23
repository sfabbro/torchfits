"""Zero-copy path from native table buffers to PyArrow arrays.

The native reader hands Python a writable memoryview over a bytearray.  Arrow
can wrap that buffer directly (``Array.from_buffers``); copying it through
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
    _raw_column_to_numpy,
    _vla_tuple_to_arrow_array,
)


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
