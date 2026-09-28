"""Arrow-native conversion helpers: numpy/torch → pyarrow arrays and record batches."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np

# -- imported from the parent table module (resolved via bottom-of-file import) -----

from .._table.utils import _fits_tform_is_bit, _parse_tform, _require_pyarrow  # noqa: E402

# Dtypes supported by the torch buffer-protocol fast path. Tensor values keep
# their explicit torch boundary; raw native columns use the string map below.
_BUFFER_SUPPORTED_TORCH_DTYPES = frozenset(
    {"float32", "float64", "float16", "int8", "int16", "int32", "int64", "uint8"}
)
_RAW_NUMPY_DTYPES = frozenset(
    {
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "uint16",
        "uint32",
        "uint64",
    }
)


def _is_torch_tensor(value: Any) -> bool:
    """True only for a tensor when torch was already loaded by another path.

    Arrow-only callers must not probe for torch by importing it. A native raw
    column is a dict, so it remains distinguishable without loading the tensor
    runtime.
    """
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _is_raw_column(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("kind") in {"fixed", "vla"}
        and isinstance(value.get("dtype"), str)
        and isinstance(value.get("data"), memoryview)
    )


def _raw_column_to_numpy(value: dict[str, Any]) -> Any:
    """Materialize one native raw column as an owned NumPy array/tuple.

    The native side returns a Python-owned writable memoryview. NumPy provides
    the torch-free typed view used by the established Arrow conversion rules.
    We copy before returning so the result never depends on the lifetime of a
    temporary native result dictionary; the VLA offsets remain exact int64.
    """
    import numpy as np

    dtype_name = str(value.get("dtype", ""))
    if dtype_name not in _RAW_NUMPY_DTYPES:
        raise TypeError(f"unsupported raw table dtype: {dtype_name!r}")
    shape = tuple(int(dim) for dim in value.get("shape", ()))
    data = value.get("data")
    if not isinstance(data, memoryview):
        raise TypeError("raw table column data must be a memoryview")
    dtype = np.dtype(dtype_name)
    kind = value.get("kind")
    if kind == "fixed":
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if data.nbytes != expected:
            raise ValueError(
                f"raw table column buffer has {data.nbytes} bytes; expected {expected}"
            )
        return np.frombuffer(data, dtype=dtype).copy().reshape(shape)

    offsets_data = value.get("offsets")
    if not isinstance(offsets_data, memoryview):
        raise TypeError("raw VLA table column offsets must be a memoryview")
    if offsets_data.nbytes % np.dtype(np.int64).itemsize:
        raise ValueError("raw VLA offsets buffer is not int64-aligned")
    offsets = np.frombuffer(offsets_data, dtype=np.int64).copy()
    if len(shape) != 1 or offsets.size != shape[0] + 1:
        raise ValueError("raw VLA offsets do not match the declared row count")
    if offsets.size == 0 or int(offsets[0]) != 0:
        raise ValueError("raw VLA offsets must start at zero")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("raw VLA offsets are not monotonic")
    expected_values = int(offsets[-1])
    expected = expected_values * dtype.itemsize
    if data.nbytes != expected:
        raise ValueError(
            f"raw VLA values buffer has {data.nbytes} bytes; expected {expected}"
        )
    flat = np.frombuffer(data, dtype=dtype).copy()
    if expected_values != flat.size:
        raise ValueError("raw VLA offsets do not match the values buffer")
    return flat, offsets


# -- low-level Arrow array constructors --------------------------------------------


def _pa_array(pa: Any, value: Any, *, mask: Any = None, type: Any = None) -> Any:
    kwargs: dict[str, Any] = {"from_pandas": False}
    if mask is not None:
        kwargs["mask"] = mask
    if type is not None:
        kwargs["type"] = type
    try:
        return pa.array(value, **kwargs)
    except TypeError:
        kwargs.pop("from_pandas", None)
        return pa.array(value, **kwargs)


def _coerce_null_sentinel(value: "np.ndarray", sentinel: Any) -> Any:
    import numpy as np

    if sentinel is None:
        return None
    arr = np.ascontiguousarray(value)
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    try:
        if arr.dtype.kind in {"i", "u", "b"}:
            return np.array(sentinel, dtype=arr.dtype).item()
        return float(sentinel)
    except (TypeError, ValueError, OverflowError):
        return None


def _column_tnull_from_meta(
    null_meta: Optional[dict[str, dict[str, str]]], name: str
) -> Optional[str]:
    if not null_meta:
        return None
    field = null_meta.get(name)
    if not field:
        return None
    return field.get("fits_tnull")


def _tform_code(tform: Any) -> Optional[str]:
    """TFORM repeat code (``A``, ``B``, ``X``, …) or None when unknown."""
    if not tform:
        return None
    _vla, code, _repeat = _parse_tform(tform)
    return code or None


def _tform_is_scalar(tform: Any) -> bool:
    """True when the TFORM maps to a scalar column (repeat 1, non-VLA).

    String (``A``) and bit (``X``) matrices keep their width semantics;
    1-bit columns are handled as scalar booleans in the bit decoder.
    """
    if not tform:
        return False
    vla, code, repeat = _parse_tform(tform)
    return (
        not vla
        and repeat == 1
        and code in {"L", "B", "I", "J", "K", "E", "D", "C", "M"}
    )


# -- uint8-matrix decode helpers ---------------------------------------------------


def _uint8_matrix_to_fixed_binary(pa: Any, value: "np.ndarray") -> Any:
    import numpy as np

    arr = np.ascontiguousarray(value)
    if arr.ndim != 2:
        return _pa_array(pa, arr)
    width = int(arr.shape[1])
    if width <= 0:
        return _pa_array(pa, [b""] * int(arr.shape[0]))
    byte_view = arr.view(np.dtype(f"S{width}")).reshape(arr.shape[0])
    return _pa_array(pa, byte_view, type=pa.binary(width))


def _uint8_matrix_to_fixed_bool_list(pa: Any, value: "np.ndarray") -> Any:
    import numpy as np

    arr = np.ascontiguousarray(value)
    if arr.ndim != 2:
        return _pa_array(pa, arr.astype(np.bool_, copy=False))
    width = int(arr.shape[1])
    if width <= 0:
        return _pa_array(pa, [[] for _ in range(int(arr.shape[0]))])
    values = _pa_array(pa, arr.astype(np.bool_, copy=False).reshape(-1))
    if width == 1:
        # Repeat-1 bit columns map to scalar booleans.
        return values
    return pa.FixedSizeListArray.from_arrays(values, width)


def _decode_uint8_matrix_to_arrow(
    pa: Any, value: "np.ndarray", encoding: str, strip: bool
) -> Any:
    import numpy as np

    arr = np.ascontiguousarray(value)
    if arr.ndim != 2:
        return _pa_array(pa, arr)
    width = int(arr.shape[1])
    if width <= 0:
        return _pa_array(pa, [""] * int(arr.shape[0]))

    # Vectorized fixed-width bytes -> unicode decode.
    byte_view = arr.view(np.dtype(f"S{width}")).reshape(arr.shape[0])
    if (
        strip
        and encoding.lower() in {"ascii", "utf8", "utf-8"}
        and not np.any(arr[:, -1] == 32)
    ):
        # Fast path: if no row ends with a space, Arrow cast handles NUL trimming correctly.
        try:
            import pyarrow.compute as pc

            return pc.cast(_pa_array(pa, byte_view), pa.string())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
            pass
    if strip:
        # Stripping while still in bytes form is much faster than stripping unicode.
        byte_view = np.char.rstrip(byte_view, b" \x00")
    decoded = np.char.decode(byte_view, encoding=encoding, errors="ignore")
    return _pa_array(pa, decoded)


# -- main numpy/torch → Arrow conversion -------------------------------------------


def _null_mask(arr: "np.ndarray", sentinel: Any) -> Any:
    """True where *arr* matches a FITS TNULL sentinel.

    Scaled integer TNULL is already NaN in the tensor (quantize / TSCAL),
    so a float compare against the raw sentinel would miss those rows.
    """
    import numpy as np

    mask = arr == sentinel
    if arr.dtype.kind == "f":
        mask = np.isnan(arr) | mask
    return mask


def _numpy_to_arrow_array(
    pa: Any,
    value: "np.ndarray",
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    null_sentinel: Any = None,
    *,
    fits_tform: str | None = None,
    unsigned_dtype: str | None = None,
) -> Any:
    import numpy as np

    arr = np.ascontiguousarray(value)
    if unsigned_dtype and arr.dtype.kind == "f":
        arr = arr.astype(np.dtype(unsigned_dtype), copy=False)
    if arr.ndim == 2 and arr.shape[1] == 1 and _tform_is_scalar(fits_tform):
        # Repeat-1 columns are scalar per the schema mapping; a width-1
        # payload must not surface as FixedSizeList<T>[1].
        arr = arr.reshape(-1)
    if arr.ndim <= 1:
        sentinel = _coerce_null_sentinel(arr, null_sentinel)
        if sentinel is None:
            return _pa_array(pa, arr)
        mask = _null_mask(arr, sentinel)
        if mask.any():
            return _pa_array(pa, arr, mask=mask)
        return _pa_array(pa, arr)
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            if _fits_tform_is_bit(fits_tform):
                return _uint8_matrix_to_fixed_bool_list(pa, arr)
            code = _tform_code(fits_tform)
            if code is None or code == "A":
                # Only char ('A') matrices — or schema-less input — follow the
                # strings/bytes contract; numeric byte ('B') vectors fall
                # through to the generic vector mapping below so the data
                # path agrees with the schema (FixedSizeList<uint8>[w]).
                return (
                    _decode_uint8_matrix_to_arrow(pa, arr, encoding, strip)
                    if decode_bytes
                    else _uint8_matrix_to_fixed_binary(pa, arr)
                )
        flat = arr.reshape(-1)
        sentinel = _coerce_null_sentinel(arr, null_sentinel)
        if sentinel is None:
            values = _pa_array(pa, flat)
        else:
            flat_mask = _null_mask(flat, sentinel)
            values = (
                _pa_array(pa, flat, mask=flat_mask)
                if flat_mask.any()
                else _pa_array(pa, flat)
            )
        return pa.FixedSizeListArray.from_arrays(values, int(arr.shape[1]))
    return _pa_array(pa, arr.tolist())


def _tensor_to_arrow_array(
    pa: Any,
    tensor: Any,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    null_sentinel: Any = None,
    *,
    fits_tform: str | None = None,
    unsigned_dtype: str | None = None,
) -> Any:
    from .._tensor_buffer import tensor_to_arrow_array

    # Numpy-free fast path: 1D contiguous CPU tensor with no null/unsigned/
    # multi-dim handling needed. Uses the shared buffer-protocol helper.
    if (
        null_sentinel is None
        and unsigned_dtype is None
        and tensor.dim() <= 1
        and str(tensor.dtype).removeprefix("torch.") in _BUFFER_SUPPORTED_TORCH_DTYPES
    ):
        return tensor_to_arrow_array(tensor, pa)

    # General path: detach, CPU, contiguous, then delegate to numpy-based
    # conversion for null masks, unsigned dtype casting, 2-D decode, etc.
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()

    return _numpy_to_arrow_array(
        pa,
        t.numpy(),
        decode_bytes,
        encoding,
        strip,
        null_sentinel=null_sentinel,
        fits_tform=fits_tform,
        unsigned_dtype=unsigned_dtype,
    )


def _column_value_to_arrow_array(
    pa: Any,
    value: Any,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    null_sentinel: Any = None,
    *,
    fits_tform: str | None = None,
    unsigned_dtype: str | None = None,
) -> Any:
    """Convert one C++ table column value to a PyArrow array."""
    import numpy as np

    if _is_raw_column(value):
        return _raw_column_to_arrow_array(
            pa,
            value,
            decode_bytes,
            encoding,
            strip,
            null_sentinel=null_sentinel,
            fits_tform=fits_tform,
            unsigned_dtype=unsigned_dtype,
        )
    if _is_torch_tensor(value):
        return _tensor_to_arrow_array(
            pa,
            value,
            decode_bytes,
            encoding,
            strip,
            null_sentinel=null_sentinel,
            fits_tform=fits_tform,
            unsigned_dtype=unsigned_dtype,
        )
    if isinstance(value, np.ndarray):
        return _numpy_to_arrow_array(
            pa,
            value,
            decode_bytes,
            encoding,
            strip,
            null_sentinel=null_sentinel,
            fits_tform=fits_tform,
            unsigned_dtype=unsigned_dtype,
        )
    if isinstance(value, list):
        converted = []
        for item in value:
            if _is_torch_tensor(item):
                t = item.detach()
                if t.device.type != "cpu":
                    t = t.cpu()
                if not t.is_contiguous():
                    t = t.contiguous()
                converted.append(t.numpy())
            else:
                converted.append(item)
        return _pa_array(pa, converted)
    if _is_vla_tuple(value):
        return _vla_tuple_to_arrow_array(pa, value, null_sentinel=null_sentinel)
    return _pa_array(pa, value)


# -- native raw-column and VLA helpers ---------------------------------------------


def _raw_column_to_arrow_array(
    pa: Any,
    value: dict[str, Any],
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    null_sentinel: Any = None,
    *,
    fits_tform: str | None = None,
    unsigned_dtype: str | None = None,
) -> Any:
    materialized = _raw_column_to_numpy(value)
    if value["kind"] == "vla":
        return _vla_tuple_to_arrow_array(pa, materialized, null_sentinel=null_sentinel)
    return _numpy_to_arrow_array(
        pa,
        materialized,
        decode_bytes,
        encoding,
        strip,
        null_sentinel=null_sentinel,
        fits_tform=fits_tform,
        unsigned_dtype=unsigned_dtype,
    )


# -- VLA helpers -------------------------------------------------------------------


def _is_vla_tuple(value: Any) -> bool:
    import numpy as np

    if not isinstance(value, tuple) or len(value) != 2:
        return False
    return isinstance(value[0], np.ndarray) and isinstance(value[1], np.ndarray)


def _vla_tuple_to_arrow_array(
    pa: Any, value: tuple[Any, Any], null_sentinel: Any = None
) -> Any:
    import numpy as np

    flat = np.ascontiguousarray(value[0]).reshape(-1)
    offsets64 = np.ascontiguousarray(value[1], dtype=np.int64).reshape(-1)
    if offsets64.size == 0:
        return _pa_array(pa, [])
    sentinel = _coerce_null_sentinel(flat, null_sentinel)
    if sentinel is None:
        values = _pa_array(pa, flat)
    else:
        mask = _null_mask(flat, sentinel)
        values = _pa_array(pa, flat, mask=mask) if mask.any() else _pa_array(pa, flat)

    if int(offsets64[-1]) <= np.iinfo(np.int32).max:
        offsets = offsets64.astype(np.int32, copy=False)
        return pa.ListArray.from_arrays(_pa_array(pa, offsets), values)
    return pa.LargeListArray.from_arrays(_pa_array(pa, offsets64), values)


# -- record-batch builder ----------------------------------------------------------


def _chunk_to_record_batch(
    chunk: dict[str, Any],
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    field_meta: Optional[dict[str, dict[str, str]]] = None,
    table_meta: Optional[dict[str, str]] = None,
    preferred_order: Optional[list[str]] = None,
    null_meta: Optional[dict[str, dict[str, str]]] = None,
    apply_fits_nulls: bool = False,
    column_tforms: Optional[dict[str, str]] = None,
    unsigned_dtypes: Optional[dict[str, str]] = None,
) -> Any:
    import numpy as np

    pa = _require_pyarrow()

    def _tform_for(name: str) -> str | None:
        if column_tforms:
            tf = column_tforms.get(name)
            if tf:
                return tf
        if field_meta and name in field_meta:
            return field_meta[name].get("fits_tform")
        if null_meta and name in null_meta:
            # The FITS field metadata (also used for TNULL) carries the TFORM
            # even when column_tforms is not built (decode_bytes=False reads).
            return null_meta[name].get("fits_tform")
        return None

    def _unsigned_dtype_for(name: str) -> str | None:
        if unsigned_dtypes:
            return unsigned_dtypes.get(name)
        return None

    # Fast path when schema metadata is not requested.
    if not field_meta and not table_meta:
        pydict: dict[str, Any] = {}
        ordered_names: list[str] = []
        if preferred_order:
            for name in preferred_order:
                if name in chunk:
                    ordered_names.append(name)
        for name in chunk.keys():
            if name not in ordered_names:
                ordered_names.append(name)
        if not ordered_names:
            ordered_names = sorted(chunk.keys())

        for name in ordered_names:
            value = chunk[name]
            null_sentinel = (
                _column_tnull_from_meta(null_meta, name) if apply_fits_nulls else None
            )
            if _is_raw_column(value):
                pydict[name] = _raw_column_to_arrow_array(
                    pa,
                    value,
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=null_sentinel,
                    fits_tform=_tform_for(name),
                    unsigned_dtype=_unsigned_dtype_for(name),
                )
            elif _is_torch_tensor(value):
                pydict[name] = _tensor_to_arrow_array(
                    pa,
                    value,
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=null_sentinel,
                    fits_tform=_tform_for(name),
                    unsigned_dtype=_unsigned_dtype_for(name),
                )
            elif isinstance(value, np.ndarray):
                pydict[name] = _numpy_to_arrow_array(
                    pa,
                    value,
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=null_sentinel,
                    fits_tform=_tform_for(name),
                    unsigned_dtype=_unsigned_dtype_for(name),
                )
            elif isinstance(value, list):
                converted = []
                for item in value:
                    if _is_torch_tensor(item):
                        t = item.detach()
                        if t.device.type != "cpu":
                            t = t.cpu()
                        if not t.is_contiguous():
                            t = t.contiguous()
                        converted.append(t.numpy())
                    else:
                        converted.append(item)
                pydict[name] = converted
            elif _is_vla_tuple(value):
                pydict[name] = _vla_tuple_to_arrow_array(
                    pa, value, null_sentinel=null_sentinel
                )
            else:
                pydict[name] = value
        return pa.RecordBatch.from_pydict(pydict)

    arrays: list[Any] = []
    fields: list[Any] = []

    ordered_names = []
    if preferred_order:
        for name in preferred_order:
            if name in chunk:
                ordered_names.append(name)
    for name in chunk.keys():
        if name not in ordered_names:
            ordered_names.append(name)

    for name in ordered_names:
        value = chunk[name]
        null_sentinel = (
            _column_tnull_from_meta(null_meta, name) if apply_fits_nulls else None
        )
        if _is_raw_column(value):
            arr = _raw_column_to_arrow_array(
                pa,
                value,
                decode_bytes,
                encoding,
                strip,
                null_sentinel=null_sentinel,
                fits_tform=_tform_for(name),
                unsigned_dtype=_unsigned_dtype_for(name),
            )
        elif _is_torch_tensor(value):
            arr = _tensor_to_arrow_array(
                pa,
                value,
                decode_bytes,
                encoding,
                strip,
                null_sentinel=null_sentinel,
                fits_tform=_tform_for(name),
                unsigned_dtype=_unsigned_dtype_for(name),
            )
        elif isinstance(value, np.ndarray):
            arr = _numpy_to_arrow_array(
                pa,
                value,
                decode_bytes,
                encoding,
                strip,
                null_sentinel=null_sentinel,
                fits_tform=_tform_for(name),
                unsigned_dtype=_unsigned_dtype_for(name),
            )
        elif isinstance(value, list):
            converted = []
            for item in value:
                if _is_torch_tensor(item):
                    t = item.detach()
                    if t.device.type != "cpu":
                        t = t.cpu()
                    if not t.is_contiguous():
                        t = t.contiguous()
                    converted.append(t.numpy())
                else:
                    converted.append(item)
            arr = _pa_array(pa, converted)
        elif _is_vla_tuple(value):
            arr = _vla_tuple_to_arrow_array(pa, value, null_sentinel=null_sentinel)
        else:
            arr = _pa_array(pa, value)
        arrays.append(arr)
        meta = None
        if field_meta and name in field_meta:
            meta = {
                k.encode("utf-8"): v.encode("utf-8")
                for k, v in field_meta[name].items()
            }
        fields.append(pa.field(name, arr.type, metadata=meta))

    schema_meta = None
    if table_meta:
        schema_meta = {
            k.encode("utf-8"): v.encode("utf-8") for k, v in table_meta.items()
        }
    return pa.RecordBatch.from_arrays(
        arrays, schema=pa.schema(fields, metadata=schema_meta)
    )
