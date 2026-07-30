"""HDU rewrite and multi-HDU write paths for FITS I/O."""

from __future__ import annotations

import os
import stat
import tempfile
from typing import Any, Dict, List, Optional, Union

from torch import Tensor

from ..hdu import Header, TableHDU, TableHDURef, TensorHDU
from .hdu_api import open_hdulist
from .paths import guard_fits_path
from ._write_helpers import (
    _image_hdu_dict_for_fits_write,
    _invalidate_path_caches,
    _is_skippable_empty_primary,
    _merge_fits_write_header,
    _normalize_cpp_table_data,
    _prepare_unsigned_table_data_for_write,
    _resolve_compression_algorithm,
    _table_schema_scale_header_cards,
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
        from . import write_api

        write_api._write_hdus_with_optional_compression(
            temp_path, hdus, compress=compress
        )
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
