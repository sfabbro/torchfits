"""FITS write helpers for the torchfits I/O engine."""

from __future__ import annotations

import os
import stat
import tempfile
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
import torch
from torch import Tensor

from ..hdu import HDUList, Header, TableHDU, TableHDURef, TensorHDU
from .caches import invalidate_path_caches as _invalidate_io_path_caches
from .hdu_api import open_hdulist
from .paths import guard_fits_path
from .quantize import (
    parse_image_quantize_spec,
    parse_table_quantize_spec,
    quantize_int16_robust,
)


def _invalidate_path_caches(path: str) -> None:
    """Invalidate Python-side caches/handles for a path that is being modified."""
    _invalidate_io_path_caches(path)
    import torchfits._C as cpp

    cpp.invalidate_file_cache(path)
    clear_meta = getattr(cpp, "clear_shared_read_meta_cache", None)
    if clear_meta is not None:
        # NOTE: native shared metadata only exposes a global clear today;
        # use per-path invalidation when the extension grows that operation.
        clear_meta()


def _host_tensor_for_fits_write(tensor: Tensor) -> Tensor:
    """Move image tensors to CPU and make them contiguous for CFITSIO writes.

    ``fits_write_img`` consumes ``nelements`` contiguous elements, so a sliced,
    transposed, or negative-stride view must be materialized first (torch handles
    any ndim, unlike the C++ 1D/2D copy helper). GPU tensors are unsupported.
    """
    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    return tensor.contiguous()


def _merge_fits_write_header(
    header: Optional[Dict[str, Any]], extra: Dict[str, Any]
) -> Header:
    """Copy a header and overlay write-time FITS convention metadata."""
    out = Header(header or {})
    for key, value in extra.items():
        out[key] = value
    return out


def _unsigned_image_storage_for_fits_write(
    tensor: Tensor,
) -> tuple[Tensor, Dict[str, Any]]:
    """Convert unsigned image tensors to FITS-standard signed storage.

    FITS image HDUs do not have native uint16/uint32 BITPIX values. Astropy and
    fitsio represent those logical dtypes with signed storage plus BSCALE/BZERO.
    Convert before delegating to the C++ writer so torchfits emits files those
    libraries read back as unsigned data.
    """
    tensor = _host_tensor_for_fits_write(tensor)
    if tensor.dtype == torch.uint16:
        raw: Tensor = (tensor.to(torch.int32) - 32768).to(torch.int16)
        return raw, {"BSCALE": 1.0, "BZERO": 32768.0}
    if tensor.dtype == torch.uint32:
        raw = (tensor.to(torch.int64) - 2147483648).to(torch.int32)
        return raw, {"BSCALE": 1.0, "BZERO": 2147483648.0}
    return tensor, {}


def _apply_image_quantize(
    tensor: Tensor,
    header: Optional[Dict[str, Any]],
    quantize: Any,
) -> tuple[Tensor, Optional[Dict[str, Any]]]:
    """Opt-in robust int16 pack for float images (any NAXIS).

    Returns storage int16 codes plus BSCALE/BZERO. Default ``quantize=None``
    leaves float tensors as BITPIX=-32/-64.
    """
    opts = parse_image_quantize_spec(quantize)
    if opts is None:
        return tensor, header
    if not isinstance(tensor, Tensor):
        raise TypeError("quantize= requires a torch.Tensor image")
    if not tensor.is_floating_point():
        raise TypeError(
            f"quantize= requires a floating-point image tensor, got dtype={tensor.dtype}"
        )
    packed = quantize_int16_robust(
        tensor, lo_q=opts.lo_q, hi_q=opts.hi_q, keep_zero=opts.keep_zero
    )
    extra = {"BSCALE": packed.scale, "BZERO": packed.zero}
    return packed.codes, _merge_fits_write_header(header, extra)


def _image_hdu_dict_for_fits_write(
    tensor: Tensor,
    header: Optional[Dict[str, Any]] = None,
    *,
    quantize: Any = None,
) -> Dict[str, Any]:
    tensor, header = _apply_image_quantize(tensor, header, quantize)
    data, extra_header = _unsigned_image_storage_for_fits_write(tensor)
    hdu_dict: Dict[str, Any] = {"data": data}
    if header or extra_header:
        hdu_dict["header"] = _merge_fits_write_header(header, extra_header)
    return hdu_dict


def _prepare_quantized_table_data_for_write(
    table_dict: Dict[str, Any],
    quantize: Any,
    schema: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Dict[str, Any]]], bool]:
    """Pack selected float columns to int16 + TSCAL/TZERO via robust quantize."""
    if quantize is None or quantize is False:
        return table_dict, schema, False

    columns = [str(name) for name in table_dict.keys()]
    per_col = parse_table_quantize_spec(quantize, columns, data=table_dict)
    if not per_col:
        return table_dict, schema, False

    out: Dict[str, Any] = dict(table_dict)
    prepared_schema: Dict[str, Dict[str, Any]] = {
        str(name): dict(meta or {}) for name, meta in (schema or {}).items()
    }
    changed = False

    for name, opts in per_col.items():
        value = out[name]
        if isinstance(value, Tensor):
            tensor = value
        else:
            import numpy as np

            arr = np.asarray(value)
            if not np.issubdtype(arr.dtype, np.floating):
                raise TypeError(
                    f"quantize column {name!r} must be floating-point, got dtype={arr.dtype}"
                )
            tensor = torch.as_tensor(arr, dtype=torch.float64)
        if not tensor.is_floating_point():
            raise TypeError(
                f"quantize column {name!r} must be floating-point, got dtype={tensor.dtype}"
            )
        packed = quantize_int16_robust(
            tensor, lo_q=opts.lo_q, hi_q=opts.hi_q, keep_zero=opts.keep_zero
        )
        out[name] = packed.codes
        meta = prepared_schema.setdefault(name, {})
        meta["format"] = _unsigned_table_tform(packed.codes, "I")
        meta["bscale"] = float(packed.scale)
        meta["bzero"] = float(packed.zero)
        changed = True

    if not changed:
        return table_dict, schema, False

    ordered_schema: Dict[str, Dict[str, Any]] = {}
    for name in out:
        ordered_schema[name] = prepared_schema.get(name, {})
    return out, ordered_schema, True


def _is_skippable_empty_primary(idx: int, hdu: Any) -> bool:
    """True for a leading empty primary HDU the compressed writer re-creates itself.

    The compressed writer always emits its own empty primary at HDU 0, so a
    leading ``NAXIS=0`` primary in the input must be skipped both when writing and
    when replaying per-HDU header cards; otherwise the output HDU indices shift by
    one and header-card replay runs off the end ("Could not move to HDU").
    """
    if idx != 0:
        return False
    hdr = getattr(hdu, "header", {}) or {}
    try:
        naxis = int(hdr.get("NAXIS", -1))
    except Exception:
        naxis = -1
    if naxis != 0:
        return False
    return not str(hdr.get("XTENSION", "")).strip().upper()


def _write_header_cards_if_supported(
    path: str,
    hdu: int,
    header: Optional[Dict[str, Any]],
) -> None:
    if not header:
        return
    header_obj = header if isinstance(header, Header) else Header(header)
    if not header_obj.cards:
        return
    import torchfits._C as cpp

    writer = getattr(cpp, "write_hdu_header_cards", None)
    if writer is None:
        return
    _invalidate_path_caches(path)
    # Invalidate both before and after the C++ header-card write:
    #  - Before: ensure no stale handle/metadata races with writing new cards.
    #  - After:  ensure the next reader picks up the freshly-written header.
    writer(path, int(hdu), list(header_obj.cards))
    _invalidate_path_caches(path)


def _delete_header_key_if_supported(path: str, hdu: int, key: str) -> None:
    """Delete one non-structural header keyword via CFITSIO ``fits_delete_key``."""
    import torchfits._C as cpp

    deleter = getattr(cpp, "delete_hdu_header_key", None)
    if deleter is None:
        raise RuntimeError("delete_hdu_header_key is unavailable in this build")
    _invalidate_path_caches(path)
    deleter(path, int(hdu), str(key))
    _invalidate_path_caches(path)


def write(
    path: str | os.PathLike[str],
    data: Any,
    header: Optional[Header | Dict[str, Any]] = None,
    overwrite: bool = False,
    compress: Union[bool, str] = False,
    quantize: Any = None,
) -> None:
    """Write data to FITS file.

    Args:
        path: Output file path
        data: Data to write (Tensor, table mapping, or HDUList)
        header: Optional FITS header dictionary
        overwrite: Whether to overwrite existing files
        compress: Whether to use tile compression (Rice algorithm)
        quantize: Opt-in robust int16 packing for float images or table
            columns. ``None`` (default) keeps native float storage.
            For images: ``\"robust\"`` or ``{\"lo_q\", \"hi_q\", \"keep_zero\"}``.
            For dict tables: ``\"robust\"`` (all float columns) or
            ``{\"col\": \"robust\" | opts}``.

    Image tensors on non-CPU devices are detached and copied to CPU before
    the CFITSIO writer runs (in-memory input tensors are not modified).
    """
    path = os.fspath(path)
    guard_fits_path(path)
    path_exists = os.path.exists(path)
    if not overwrite and path_exists:
        raise FileExistsError(
            f"File '{path}' already exists. Use overwrite=True to overwrite."
        )

    if overwrite and path_exists:
        if os.path.isdir(path):
            raise IsADirectoryError(path)
        target = os.path.realpath(path)
        target_dir = os.path.dirname(target) or "."
        original_mode = stat.S_IMODE(os.stat(target).st_mode)
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(target)}.", suffix=".tmp.fits", dir=target_dir
        )
        os.close(fd)
        os.unlink(temp_path)
        try:
            write(
                temp_path,
                data,
                header=header,
                overwrite=False,
                compress=compress,
                quantize=quantize,
            )
            os.chmod(temp_path, original_mode)
            os.replace(temp_path, target)
            _invalidate_path_caches(path)
            if target != path:
                _invalidate_path_caches(target)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return

    # The unified C++ cache and the Python-side handle cache can otherwise return
    # stale views of an overwritten file (mtime/size can be unchanged).
    _invalidate_path_caches(path)

    try:
        import torchfits._C as cpp

        hdus_to_write = []

        if compress:
            compressed_hdus: List[Any] = []
            if isinstance(data, Tensor):
                data, header = _apply_image_quantize(data, header, quantize)
                img, img_header = _unsigned_image_storage_for_fits_write(data)
                compressed_hdus = [
                    TensorHDU(
                        data=img,
                        header=_merge_fits_write_header(header, img_header),
                    )
                ]
            elif isinstance(data, HDUList):
                compressed_hdus = list(getattr(data, "_hdus", []))
            elif isinstance(data, dict):
                if "data" in data:
                    item_hdu = _coerce_compressed_hdu_item(data)
                    compressed_hdus.append(item_hdu)
                else:
                    compressed_hdus.append(
                        _TableHDUWriteProxy(data, Header(header or {}))
                    )
            elif isinstance(data, (list, tuple)):
                for item in data:
                    compressed_hdus.append(_coerce_compressed_hdu_item(item))
            else:
                raise NotImplementedError(
                    "Compressed FITS writing supports tensors, tables, or HDU lists."
                )

            if header and compressed_hdus:
                first = compressed_hdus[0]
                merged = Header(dict(getattr(first, "header", {})))
                merged.update(dict(header))
                if isinstance(first, TensorHDU):
                    first._header = merged
                else:
                    first.header = merged

            _write_hdus_with_optional_compression(
                path, compressed_hdus, compress=compress
            )
            out_hdu = 1
            for idx, item_hdu in enumerate(compressed_hdus):
                if _is_skippable_empty_primary(idx, item_hdu):
                    continue
                _write_header_cards_if_supported(
                    path, out_hdu, getattr(item_hdu, "header", None)
                )
                out_hdu += 1
            return

        if isinstance(data, HDUList):
            _write_hdus_uncompressed(path, list(getattr(data, "_hdus", [])), overwrite)
            return

        if isinstance(data, dict) and "data" not in data:
            data, table_schema, _ = _prepare_unsigned_table_data_for_write(data)
            data, q_schema, q_changed = _prepare_quantized_table_data_for_write(
                data, quantize, table_schema
            )
            if q_changed:
                table_schema = q_schema
            if _can_use_cpp_table_writer(data):
                data = _normalize_cpp_table_data(data)
                header_obj: Header = Header(header) if header else Header()
                cpp.write_fits_table(
                    path,
                    data,
                    header_obj,
                    overwrite,
                    table_schema,
                    "binary",
                )
                _write_header_cards_if_supported(path, 1, header_obj)
                return
            raise ValueError(
                "Dictionary table writes currently require CFITSIO-native column types "
                "(numeric/bool/complex, strings, or VLA lists). Unsupported object/structure "
                "columns should be converted before writing."
            )

        if isinstance(data, Tensor):
            hdus_to_write.append(
                _image_hdu_dict_for_fits_write(data, header, quantize=quantize)
            )

        elif hasattr(data, "__iter__") and not isinstance(data, (str, Tensor)):
            if quantize is not None:
                raise ValueError(
                    "quantize= is supported for a single image tensor or dict table, "
                    "not multi-HDU sequences"
                )
            for item in data:
                if isinstance(item, dict):
                    if "data" in item:
                        payload = item["data"]
                        if isinstance(payload, Tensor):
                            item_merged = dict(item)
                            hdu_dict = _image_hdu_dict_for_fits_write(
                                payload, item_merged.get("header")
                            )
                            item_merged["data"] = hdu_dict["data"]
                            if "header" in hdu_dict:
                                item_merged["header"] = hdu_dict["header"]
                            hdus_to_write.append(item_merged)
                        else:
                            raise TypeError(
                                "HDU dictionary 'data' must be a torch.Tensor"
                            )
                    else:
                        raise ValueError(
                            "HDU dictionaries must contain a 'data' tensor"
                        )
                elif isinstance(item, Tensor):
                    hdus_to_write.append(_image_hdu_dict_for_fits_write(item))
                elif hasattr(item, "data") and isinstance(item.data, Tensor):
                    hdus_to_write.append(
                        _image_hdu_dict_for_fits_write(
                            item.data, getattr(item, "header", None)
                        )
                    )
                else:
                    raise TypeError(f"Unsupported HDU item type: {type(item).__name__}")
        else:
            raise ValueError(f"Unsupported data type for FITS writing: {type(data)}")

        if not hdus_to_write:
            raise ValueError("At least one writable HDU is required")
        cpp.write_fits_file(path, hdus_to_write, overwrite)
        for idx, item in enumerate(hdus_to_write):
            item_header = item.get("header") if isinstance(item, dict) else None
            _write_header_cards_if_supported(path, idx, item_header)

    except Exception as e:
        if not path_exists and os.path.exists(path):
            os.remove(path)
        raise RuntimeError(f"Failed to write FITS file '{path}': {e}") from e


def _can_use_cpp_table_writer(table_dict: Dict[str, Any]) -> bool:
    """Return True when all table columns can use the fast C++ writer."""
    import numpy as np

    if not table_dict:
        return False

    for value in table_dict.values():
        if isinstance(value, torch.Tensor):
            if value.dim() > 2:
                return False
            if value.is_complex():
                if value.dtype not in {torch.complex64, torch.complex128}:
                    return False
                continue
            if value.dtype not in {
                torch.bool,
                torch.uint8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float32,
                torch.float64,
            }:
                return False
            continue

        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
            except ValueError:
                arr = np.asarray(value, dtype=object)
            if arr.dtype != np.object_:
                value = arr
            else:
                for item in value:
                    if item is None:
                        continue
                    if isinstance(item, (str, bytes, np.str_, np.bytes_)):
                        continue
                    if isinstance(item, torch.Tensor):
                        t = item.detach()
                        if t.dim() > 2:
                            return False
                        if t.is_complex():
                            if t.dtype not in {torch.complex64, torch.complex128}:
                                return False
                            continue
                        if t.dtype not in {
                            torch.bool,
                            torch.uint8,
                            torch.int16,
                            torch.int32,
                            torch.int64,
                            torch.float32,
                            torch.float64,
                        }:
                            return False
                        continue
                    arr_item = np.asarray(item)
                    if arr_item.ndim > 2:
                        return False
                    if np.iscomplexobj(arr_item):
                        if arr_item.dtype not in (np.complex64, np.complex128):
                            return False
                        continue
                    kind = arr_item.dtype.kind
                    itemsize = arr_item.dtype.itemsize
                    if kind in {"U", "S"}:
                        continue
                    if kind == "b":
                        continue
                    if kind == "u" and itemsize == 1:
                        continue
                    if kind == "i" and itemsize in (2, 4, 8):
                        continue
                    if kind == "f" and itemsize in (4, 8):
                        continue
                    return False
                continue

        if not isinstance(value, np.ndarray):
            return False
        if value.ndim > 2:
            return False
        if np.iscomplexobj(value):
            if value.dtype not in (np.complex64, np.complex128):
                return False
            continue
        if value.dtype.kind in {"U", "S"}:
            continue
        kind = value.dtype.kind
        itemsize = value.dtype.itemsize
        if kind == "b":
            continue
        if kind == "u" and itemsize == 1:
            continue
        if kind == "i" and itemsize in (2, 4, 8):
            continue
        if kind == "f" and itemsize in (4, 8):
            continue
        return False

    return True


def _normalize_vla_item(item: Any) -> np.ndarray:
    """Normalize a single VLA item."""
    import numpy as np

    if isinstance(item, torch.Tensor):
        t = item.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dim() == 0:
            t = t.reshape(1)
        return np.ascontiguousarray(t.numpy())
    elif isinstance(item, np.ndarray):
        return np.ascontiguousarray(item)
    elif item is None:
        return np.asarray([], dtype=np.float32)
    else:
        return np.asarray(item)


def _normalize_list_sequence(items: list[Any]) -> Any:
    """Normalize a list or tuple sequence."""
    import numpy as np

    if items and all(
        isinstance(item, (str, bytes, np.str_, np.bytes_)) or item is None
        for item in items
    ):
        return items
    if any(isinstance(item, (list, tuple, np.ndarray, torch.Tensor)) for item in items):
        return [_normalize_vla_item(item) for item in items]
    return np.asarray(items)


def _normalize_ndarray_column(value: np.ndarray) -> Any:
    """Normalize an ndarray column."""
    import numpy as np

    if value.dtype == np.object_:
        return list(value)
    if value.dtype.kind in {"U", "S"}:
        return value.astype(str).tolist()
    return value


def _unsigned_table_storage_for_fits_write(value: Any) -> tuple[Any, str, float] | None:
    """Return signed-storage column data plus FITS format/BZERO for uint columns."""
    import numpy as np

    if isinstance(value, torch.Tensor):
        if value.dtype == torch.uint16:
            raw = (value.detach().to(torch.int32) - 32768).to(torch.int16)
            return raw, "I", 32768.0
        if value.dtype == torch.uint32:
            raw = (value.detach().to(torch.int64) - 2147483648).to(torch.int32)
            return raw, "J", 2147483648.0
        return None

    if isinstance(value, np.ndarray):
        if value.dtype == np.uint16:
            raw = (value.astype(np.int32, copy=False) - 32768).astype(np.int16)  # type: ignore[assignment]
            return np.ascontiguousarray(raw), "I", 32768.0
        if value.dtype == np.uint32:
            raw = (value.astype(np.int64, copy=False) - 2147483648).astype(np.int32)  # type: ignore[assignment]
            return np.ascontiguousarray(raw), "J", 2147483648.0
        return None

    return None


def _unsigned_table_tform(value: Any, code: str) -> str:
    """Infer a TFORM repeat for an unsigned column that was converted to signed storage."""
    import numpy as np

    if isinstance(value, torch.Tensor):
        if value.dim() <= 1:
            repeat = 1
        else:
            repeat = 1
            for size in value.shape[1:]:
                repeat *= int(size)
        return code if repeat == 1 else f"{repeat}{code}"

    arr = np.asarray(value)
    if arr.ndim <= 1:
        repeat = 1
    else:
        repeat = int(np.prod(arr.shape[1:]))
    return code if repeat == 1 else f"{repeat}{code}"


def _prepare_unsigned_table_data_for_write(
    table_dict: Dict[str, Any],
    schema: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Dict[str, Any]]], bool]:
    """Convert uint16/uint32 table columns to FITS pseudo-unsigned storage."""
    out: Dict[str, Any] = {}
    prepared_schema: Dict[str, Dict[str, Any]] = {
        str(name): dict(meta or {}) for name, meta in (schema or {}).items()
    }
    changed = False

    for name, value in table_dict.items():
        col_name = str(name)
        converted = _unsigned_table_storage_for_fits_write(value)
        if converted is None:
            out[col_name] = value
            if schema is not None and col_name not in prepared_schema:
                prepared_schema[col_name] = {}
            continue

        raw, code, bzero = converted
        out[col_name] = raw
        meta = prepared_schema.setdefault(col_name, {})
        if "format" not in meta and "tform" not in meta:
            meta["format"] = _unsigned_table_tform(value, code)
        meta.setdefault("bscale", 1.0)
        meta.setdefault("bzero", int(bzero))
        changed = True

    if schema is None and not changed:
        return out, None, False

    # Keep schema column order aligned with input data for callers that did not
    # provide a complete schema.
    ordered_schema: Dict[str, Dict[str, Any]] = {}
    for name in out:
        if name in prepared_schema:
            ordered_schema[name] = prepared_schema[name]
        elif schema is not None:
            ordered_schema[name] = {}
    return out, ordered_schema if (changed or schema is not None) else None, changed


def _table_schema_scale_header_cards(
    schema: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Convert table scaling schema metadata into header cards for HDU-list writes."""
    if not schema:
        return {}
    out: Dict[str, Any] = {}
    for idx, meta in enumerate(schema.values(), start=1):
        if "bscale" in meta:
            out[f"TSCAL{idx}"] = float(meta["bscale"])
        if "bzero" in meta:
            bzero = meta["bzero"]
            out[f"TZERO{idx}"] = int(bzero) if float(bzero).is_integer() else bzero
    return out


def _normalize_cpp_table_data(table_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize table data for the C++ writer (strings/VLA/object arrays)."""
    import numpy as np

    out: Dict[str, Any] = {}
    for name, value in table_dict.items():
        if isinstance(value, (list, tuple)):
            out[name] = _normalize_list_sequence(list(value))
        elif isinstance(value, np.ndarray):
            out[name] = _normalize_ndarray_column(value)
        else:
            out[name] = value
    return out


def _resolve_compression_algorithm(compress: Union[bool, str]) -> Optional[str]:
    """Normalize compress flag to a backend algorithm string or None."""
    if compress is False:
        return None
    if compress is True:
        return "RICE_1"
    if isinstance(compress, str):
        algo = compress.strip()
        return algo if algo else "RICE_1"
    raise TypeError("compress must be bool or compression algorithm string")


def _coerce_compressed_hdu_item(item: Any) -> Any:
    """Normalize compressed-write inputs to TensorHDU/TableHDU objects."""
    if isinstance(item, (TensorHDU, TableHDU)):
        return item
    if isinstance(item, TableHDURef):
        return item.materialize(device="cpu")
    if isinstance(item, Tensor):
        img, img_header = _unsigned_image_storage_for_fits_write(item)
        return TensorHDU(data=img, header=Header(img_header))
    if isinstance(item, dict):
        if "data" in item:
            img = item["data"]
            if not isinstance(img, Tensor):
                try:
                    import numpy as np

                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)
                    elif isinstance(img, (list, tuple)):
                        img = torch.tensor(img)
                    else:
                        img = torch.as_tensor(img)
                except Exception:
                    raise NotImplementedError(
                        "Compressed FITS writing supports tensor image payloads"
                        " for dict HDUs. Could not convert"
                        f" {type(img).__name__} to a tensor."
                    ) from None
            img, img_header = _unsigned_image_storage_for_fits_write(img)
            return TensorHDU(
                data=img,
                header=_merge_fits_write_header(item.get("header", {}), img_header),
            )
        return _TableHDUWriteProxy(item, Header())
    raise NotImplementedError(
        f"Unsupported HDU payload for compressed write: {type(item)}"
    )


class _TableHDUWriteProxy:
    """Small table-HDU proxy for internal writer paths."""

    def __init__(self, raw_data: Dict[str, Any], header: Header):
        prepared, schema, _ = _prepare_unsigned_table_data_for_write(dict(raw_data))
        self._raw_data = _normalize_cpp_table_data(prepared)
        scale_cards = _table_schema_scale_header_cards(schema)
        self.header = _merge_fits_write_header(header, scale_cards)
        self._schema = schema


def _detach_hdus_for_rewrite(path: str) -> List[Any]:
    """Materialize file-backed HDUs so rewrite paths never hold stale handles."""
    with open_hdulist(path) as hdul:
        detached: List[Any] = []
        for hdu in list(hdul._hdus):
            if isinstance(hdu, TensorHDU):
                detached.append(TensorHDU(data=hdu.to_tensor("cpu"), header=hdu.header))
            elif isinstance(hdu, TableHDU):
                detached.append(
                    TableHDU(dict(getattr(hdu, "_raw_data", {})), header=hdu.header)
                )
            elif isinstance(hdu, TableHDURef):
                mat = hdu.materialize(device="cpu")
                detached.append(
                    TableHDU(dict(getattr(mat, "_raw_data", {})), header=hdu.header)
                )
            else:
                detached.append(hdu)
    return detached


def _sanitize_header_for_compressed_write(
    header: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Drop structural/compression keys so CFITSIO can emit canonical metadata."""
    import numpy as np

    if not header:
        return {}

    skip_exact = {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "EXTEND",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "THEAP",
        "BSCALE",
        "BZERO",
        "DATASUM",
        "CHECKSUM",
        "ZIMAGE",
        "ZCMPTYPE",
        "ZBITPIX",
        "ZNAXIS",
        "ZPCOUNT",
        "ZGCOUNT",
        "ZHECKSUM",
        "ZDATASUM",
    }
    skip_prefix = (
        "NAXIS",
        "ZNAXIS",
        "ZTILE",
        "ZNAME",
        "ZVAL",
        "TTYPE",
        "TFORM",
        "TDIM",
        "TSCAL",
        "TZERO",
        "TNULL",
        "TUNIT",
        "TDISP",
    )

    out: Dict[str, Any] = {}
    for key, value in dict(header).items():
        key_str = str(key)
        key_upper = key_str.upper()
        if key_upper in skip_exact or any(
            key_upper.startswith(prefix) for prefix in skip_prefix
        ):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="ignore")
        out[key_str] = value
    return out


def _sanitize_table_header_for_write(
    header: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Drop FITS structural keywords before delegating table writes to CFITSIO."""
    from .._table.write import _TABLE_STRUCTURAL_SKIP_KEYS

    out: Dict[str, Any] = {}
    for key, value in dict(header or {}).items():
        key_upper = str(key).upper()
        if key_upper in _TABLE_STRUCTURAL_SKIP_KEYS or key_upper.startswith("NAXIS"):
            continue
        out[str(key)] = value
    return out


class _TableWriteProxy:
    __slots__ = ("_raw_data", "header", "_schema")

    def __init__(
        self,
        raw_data: Any,
        header: Header,
        schema: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._raw_data = raw_data
        self.header = header
        self._schema = schema


def _write_hdus_uncompressed(path: str, hdus: List[Any], overwrite: bool) -> None:
    """Write an HDU sequence through the uncompressed C++ writer."""
    import torchfits._C as cpp

    guard_fits_path(path)
    payload: List[Any] = []
    for idx, hdu in enumerate(hdus):  # noqa: B007
        if isinstance(hdu, TableHDURef):
            hdu = hdu.materialize(device="cpu")

        if isinstance(hdu, TableHDU):
            raw_data = dict(getattr(hdu, "_raw_data", {}))
            raw_data, schema, _ = _prepare_unsigned_table_data_for_write(raw_data)
            scale_cards = _table_schema_scale_header_cards(schema)
            header = _merge_fits_write_header(
                _sanitize_table_header_for_write(hdu.header), scale_cards
            )
            raw_data = _normalize_cpp_table_data(raw_data)
            payload.append(_TableWriteProxy(raw_data, header, schema))
            continue

        if hasattr(hdu, "_raw_data") and hasattr(hdu, "header"):
            raw_data = dict(getattr(hdu, "_raw_data", {}))
            raw_data, schema, _ = _prepare_unsigned_table_data_for_write(raw_data)
            scale_cards = _table_schema_scale_header_cards(schema)
            tbl_header = _merge_fits_write_header(
                _sanitize_table_header_for_write(hdu.header), scale_cards
            )
            raw_data = _normalize_cpp_table_data(raw_data)
            payload.append(_TableWriteProxy(raw_data, tbl_header, schema))
            continue

        if not isinstance(hdu, TensorHDU):
            raise ValueError(
                f"Unsupported HDU type for write at index {idx}: {type(hdu).__name__}"
            )

        payload.append(
            _image_hdu_dict_for_fits_write(
                hdu.to_tensor("cpu"), getattr(hdu, "header", None)
            )
        )

    _invalidate_path_caches(path)
    cpp.write_fits_file(path, payload, overwrite)


def _write_hdus_with_optional_compression(
    path: str, hdus: List[Any], compress: Union[bool, str] = False
) -> None:
    """Rewrite HDUs, optionally using CFITSIO compressed-image writer."""
    guard_fits_path(path)
    algorithm = _resolve_compression_algorithm(compress)
    if algorithm is None:
        _write_hdus_uncompressed(path, hdus, overwrite=True)
        return

    import torchfits._C as cpp

    payload: list[Any] = []
    for idx, hdu in enumerate(hdus):  # noqa: B007
        if isinstance(hdu, TableHDURef):
            hdu = hdu.materialize(device="cpu")

        if isinstance(hdu, TableHDU):
            raw_data = dict(getattr(hdu, "_raw_data", {}))
            raw_data, schema, _ = _prepare_unsigned_table_data_for_write(raw_data)
            scale_cards = _table_schema_scale_header_cards(schema)
            header = _merge_fits_write_header(
                _sanitize_table_header_for_write(hdu.header), scale_cards
            )
            raw_data = _normalize_cpp_table_data(raw_data)
            payload.append(_TableWriteProxy(raw_data, header, schema))
            continue

        if hasattr(hdu, "_raw_data") and hasattr(hdu, "header"):
            raw_data = dict(getattr(hdu, "_raw_data", {}))
            raw_data, schema, _ = _prepare_unsigned_table_data_for_write(raw_data)
            scale_cards = _table_schema_scale_header_cards(schema)
            tbl_header = _merge_fits_write_header(
                _sanitize_table_header_for_write(hdu.header), scale_cards
            )
            raw_data = _normalize_cpp_table_data(raw_data)
            payload.append(_TableWriteProxy(raw_data, tbl_header, schema))
            continue

        if not isinstance(hdu, TensorHDU):
            raise ValueError(
                f"Unsupported HDU type for rewrite at index {idx}: {type(hdu).__name__}"
            )

        # A compressed FITS file uses an empty primary HDU followed by compressed
        # image extensions; skip this placeholder to avoid duplicating it.
        if _is_skippable_empty_primary(idx, hdu):
            continue

        hdu_dict = _image_hdu_dict_for_fits_write(
            hdu.to_tensor("cpu"), getattr(hdu, "header", None)
        )
        hdr = getattr(hdu, "header", None)
        if hdr:
            hdu_dict["header"] = _sanitize_header_for_compressed_write(
                hdu_dict.get("header", hdr)
            )
        payload.append(hdu_dict)

    _invalidate_path_caches(path)
    cpp.write_fits_file_compressed_images(path, payload, True, algorithm)


def _atomic_rewrite_hdus(
    path: str, hdus: List[Any], compress: Union[bool, str] = False
) -> None:
    """Rewrite an existing HDU sequence without exposing a partial file."""
    target = os.path.realpath(path)
    target_dir = os.path.dirname(target) or "."
    original_mode = stat.S_IMODE(os.stat(target).st_mode)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp.fits", dir=target_dir
    )
    os.close(fd)
    os.unlink(temp_path)
    try:
        _write_hdus_with_optional_compression(temp_path, hdus, compress=compress)
        os.chmod(temp_path, original_mode)
        os.replace(temp_path, target)
        _invalidate_path_caches(path)
        if target != path:
            _invalidate_path_caches(target)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def insert_hdu(
    path: str,
    data: Any,
    index: int = 1,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Insert a new HDU into an existing FITS file."""
    if not isinstance(index, int):
        raise TypeError("index must be an integer HDU position")

    if isinstance(data, TableHDU) or isinstance(data, TensorHDU):
        new_hdu = data
        if header is not None:
            if isinstance(new_hdu, TensorHDU):
                new_hdu._header = Header(header)
            else:
                new_hdu.header = Header(header)
    elif isinstance(data, dict) and "data" not in data:
        new_hdu = TableHDU(data, header=Header(header or {}))
    elif isinstance(data, Tensor):
        new_hdu = TensorHDU(data=data, header=Header(header or {}))
    else:
        raise ValueError(f"Unsupported HDU data type: {type(data)}")

    hdus = _detach_hdus_for_rewrite(path)

    if index < 0 or index > len(hdus):
        raise IndexError(f"index {index} out of range for {len(hdus)} HDUs")
    hdus.insert(index, new_hdu)
    _atomic_rewrite_hdus(path, hdus, compress=compress)


def replace_hdu(
    path: str,
    hdu: Union[int, str],
    data: Any,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Replace an HDU by index or EXTNAME."""
    preserve_header = header is None and not isinstance(data, (TableHDU, TensorHDU))

    if isinstance(data, TableHDU) or isinstance(data, TensorHDU):
        new_hdu = data
        if header is not None:
            if isinstance(new_hdu, TensorHDU):
                new_hdu._header = Header(header)
            else:
                new_hdu.header = Header(header)
    elif isinstance(data, dict) and "data" not in data:
        new_hdu = TableHDU(data, header=Header(header or {}))
    elif isinstance(data, Tensor):
        new_hdu = TensorHDU(data=data, header=Header(header or {}))
    else:
        raise ValueError(f"Unsupported HDU data type: {type(data)}")

    hdus = _detach_hdus_for_rewrite(path)

    if isinstance(hdu, int):
        if hdu < 0 or hdu >= len(hdus):
            raise IndexError(f"hdu index {hdu} out of range for {len(hdus)} HDUs")
        target: int = hdu
    elif isinstance(hdu, str):
        target = -1
        for idx, item in enumerate(hdus):
            if item.header.get("EXTNAME") == hdu:
                target = idx
                break
        if target < 0:
            raise KeyError(f"HDU '{hdu}' not found")
    else:
        raise TypeError("hdu must be an int index or EXTNAME string")

    if preserve_header:
        # Keep the original header (e.g. EXTNAME/WCS) unless the caller overrides it.
        old_header = getattr(hdus[target], "header", None)
        if old_header is not None:
            if isinstance(new_hdu, TensorHDU):
                new_hdu._header = old_header
            else:
                new_hdu.header = old_header

    hdus[target] = new_hdu
    _atomic_rewrite_hdus(path, hdus, compress=compress)


def delete_hdu(
    path: str,
    hdu: Union[int, str],
    compress: Union[bool, str] = False,
) -> None:
    """Delete an HDU by index or EXTNAME."""
    hdus = _detach_hdus_for_rewrite(path)

    if isinstance(hdu, int):
        if hdu < 0 or hdu >= len(hdus):
            raise IndexError(f"hdu index {hdu} out of range for {len(hdus)} HDUs")
        target: int = hdu
    elif isinstance(hdu, str):
        target = -1
        for idx, item in enumerate(hdus):
            if item.header.get("EXTNAME") == hdu:
                target = idx
                break
        if target < 0:
            raise KeyError(f"HDU '{hdu}' not found")
    else:
        raise TypeError("hdu must be an int index or EXTNAME string")

    del hdus[target]
    _atomic_rewrite_hdus(path, hdus, compress=compress)
