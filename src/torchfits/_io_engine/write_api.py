"""FITS write helpers for the torchfits I/O engine."""

from __future__ import annotations

import os
import stat
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch

from torch import Tensor

from ..hdu import HDUList, Header, TensorHDU
from .paths import guard_fits_path
from ._hdu_rewrite import (
    _write_hdus_uncompressed,
    _write_hdus_with_optional_compression,
)
from ._write_helpers import (
    _TableHDUWriteProxy,
    _apply_image_quantize,
    _can_use_cpp_table_writer,
    _coerce_compressed_hdu_item,
    _image_hdu_dict_for_fits_write,
    _invalidate_path_caches,
    _is_skippable_empty_primary,
    _merge_fits_write_header,
    _normalize_cpp_table_data,
    _prepare_quantized_table_data_for_write,
    _prepare_unsigned_table_data_for_write,
    _unsigned_image_storage_for_fits_write,
    _write_header_cards_if_supported,
)

# Re-export public and private symbols for backward-compatible imports.
from ._hdu_rewrite import (  # noqa: F401
    _atomic_rewrite_hdus,
    _detach_hdus_for_rewrite,
    _sanitize_header_for_compressed_write,
    _sanitize_table_header_for_write,
    delete_hdu,
    insert_hdu,
    replace_hdu,
)
from ._write_helpers import (  # noqa: F401
    _delete_header_key_if_supported,
    _host_tensor_for_fits_write,
    _normalize_list_sequence,
    _normalize_ndarray_column,
    _normalize_vla_item,
    _resolve_compression_algorithm,
    _table_schema_scale_header_cards,
    _unsigned_table_storage_for_fits_write,
    _unsigned_table_tform,
)

__all__ = [
    "write",
    "insert_hdu",
    "replace_hdu",
    "delete_hdu",
    "_TableHDUWriteProxy",
    "_atomic_rewrite_hdus",
    "_can_use_cpp_table_writer",
    "_coerce_compressed_hdu_item",
    "_delete_header_key_if_supported",
    "_detach_hdus_for_rewrite",
    "_host_tensor_for_fits_write",
    "_merge_fits_write_header",
    "_normalize_cpp_table_data",
    "_normalize_list_sequence",
    "_normalize_ndarray_column",
    "_normalize_vla_item",
    "_prepare_quantized_table_data_for_write",
    "_prepare_unsigned_table_data_for_write",
    "_resolve_compression_algorithm",
    "_sanitize_header_for_compressed_write",
    "_sanitize_table_header_for_write",
    "_table_schema_scale_header_cards",
    "_unsigned_image_storage_for_fits_write",
    "_unsigned_table_storage_for_fits_write",
    "_unsigned_table_tform",
    "_write_hdus_uncompressed",
    "_write_hdus_with_optional_compression",
    "_write_header_cards_if_supported",
]


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
            if isinstance(data, (Tensor, np.ndarray)):
                if isinstance(data, np.ndarray):
                    data = torch.as_tensor(data)
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

        elif isinstance(data, np.ndarray):
            hdus_to_write.append(
                _image_hdu_dict_for_fits_write(
                    torch.as_tensor(data), header, quantize=quantize
                )
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

    except ValueError:
        # Data-validation failures (e.g. unsupported uint64) are actionable
        # as-is; do not bury the message in a generic RuntimeError wrapper.
        if not path_exists and os.path.exists(path):
            os.remove(path)
        raise
    except Exception as e:
        if not path_exists and os.path.exists(path):
            os.remove(path)
        raise RuntimeError(f"Failed to write FITS file '{path}': {e}") from e
