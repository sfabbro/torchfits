"""Arrow-native conversion helpers: numpy/torch → pyarrow arrays and record batches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    import numpy as np

# -- imported from the parent table module (resolved via bottom-of-file import) -----

from ..table import _fits_tform_is_bit, _require_pyarrow  # noqa: E402


# -- low-level Arrow array constructors --------------------------------------------


def _pa_array(pa, value, *, mask=None, type=None):
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
    except Exception:
        return None


def _column_tnull_from_meta(
    null_meta: Optional[dict[str, dict[str, str]]], name: str
) -> Any:
    if not null_meta:
        return None
    field = null_meta.get(name)
    if not field:
        return None
    return field.get("fits_tnull")


# -- uint8-matrix decode helpers ---------------------------------------------------


def _uint8_matrix_to_fixed_binary(pa, value: "np.ndarray"):
    import numpy as np

    arr = np.ascontiguousarray(value)
    if arr.ndim != 2:
        return _pa_array(pa, arr)
    width = int(arr.shape[1])
    if width <= 0:
        return _pa_array(pa, [b""] * int(arr.shape[0]))
    byte_view = arr.view(np.dtype(f"S{width}")).reshape(arr.shape[0])
    return _pa_array(pa, byte_view, type=pa.binary(width))


def _uint8_matrix_to_fixed_bool_list(pa, value: "np.ndarray"):
    import numpy as np

    arr = np.ascontiguousarray(value)
    if arr.ndim != 2:
        return _pa_array(pa, arr.astype(np.bool_, copy=False))
    width = int(arr.shape[1])
    if width <= 0:
        return _pa_array(pa, [[] for _ in range(int(arr.shape[0]))])
    values = _pa_array(pa, arr.astype(np.bool_, copy=False).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(values, width)


def _decode_uint8_matrix_to_arrow(pa, value: "np.ndarray", encoding: str, strip: bool):
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
        except Exception:
            pass
    if strip:
        # Stripping while still in bytes form is much faster than stripping unicode.
        byte_view = np.char.rstrip(byte_view, b" \x00")
    decoded = np.char.decode(byte_view, encoding=encoding, errors="ignore")
    return _pa_array(pa, decoded)


# -- main numpy/torch → Arrow conversion -------------------------------------------


def _numpy_to_arrow_array(
    pa,
    value: "np.ndarray",
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    null_sentinel: Any = None,
    *,
    fits_tform: str | None = None,
    unsigned_dtype: str | None = None,
):
    import numpy as np

    arr = np.ascontiguousarray(value)
    if unsigned_dtype and arr.dtype.kind == "f":
        arr = arr.astype(np.dtype(unsigned_dtype), copy=False)
    if arr.ndim <= 1:
        sentinel = _coerce_null_sentinel(arr, null_sentinel)
        if sentinel is None:
            return _pa_array(pa, arr)
        mask = arr == sentinel
        if mask.any():
            return _pa_array(pa, arr, mask=mask)
        return _pa_array(pa, arr)
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            if _fits_tform_is_bit(fits_tform):
                return _uint8_matrix_to_fixed_bool_list(pa, arr)
            if decode_bytes and not _fits_tform_is_bit(fits_tform):
                return _decode_uint8_matrix_to_arrow(pa, arr, encoding, strip)
            return _uint8_matrix_to_fixed_binary(pa, arr)
        flat = arr.reshape(-1)
        sentinel = _coerce_null_sentinel(arr, null_sentinel)
        if sentinel is None:
            values = _pa_array(pa, flat)
        else:
            flat_mask = flat == sentinel
            values = (
                _pa_array(pa, flat, mask=flat_mask)
                if flat_mask.any()
                else _pa_array(pa, flat)
            )
        return pa.FixedSizeListArray.from_arrays(values, int(arr.shape[1]))
    return _pa_array(pa, arr.tolist())


def _tensor_to_arrow_array(
    pa,
    tensor: torch.Tensor,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    null_sentinel: Any = None,
    *,
    fits_tform: str | None = None,
    unsigned_dtype: str | None = None,
):
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


# -- VLA helpers -------------------------------------------------------------------


def _is_vla_tuple(value: Any) -> bool:
    import numpy as np

    if not isinstance(value, tuple) or len(value) != 2:
        return False
    return isinstance(value[0], np.ndarray) and isinstance(value[1], np.ndarray)


def _vla_tuple_to_arrow_array(pa, value: tuple[Any, Any], null_sentinel: Any = None):
    import numpy as np

    flat = np.ascontiguousarray(value[0]).reshape(-1)
    offsets64 = np.ascontiguousarray(value[1], dtype=np.int64).reshape(-1)
    if offsets64.size == 0:
        return _pa_array(pa, [])
    sentinel = _coerce_null_sentinel(flat, null_sentinel)
    if sentinel is None:
        values = _pa_array(pa, flat)
    else:
        mask = flat == sentinel
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
):
    import numpy as np

    pa = _require_pyarrow()

    def _tform_for(name: str) -> str | None:
        if column_tforms:
            tf = column_tforms.get(name)
            if tf:
                return tf
        if field_meta and name in field_meta:
            return field_meta[name].get("fits_tform")
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
            if isinstance(value, torch.Tensor):
                t = value.detach()
                if t.device.type != "cpu":
                    t = t.cpu()
                if not t.is_contiguous():
                    t = t.contiguous()
                pydict[name] = _numpy_to_arrow_array(
                    pa,
                    t.numpy(),
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
                    if isinstance(item, torch.Tensor):
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
        if isinstance(value, torch.Tensor):
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
                if isinstance(item, torch.Tensor):
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
