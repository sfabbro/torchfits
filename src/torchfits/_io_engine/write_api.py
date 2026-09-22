"""FITS write helpers for the torchfits I/O engine."""

from __future__ import annotations

import os
import stat
import tempfile
from typing import Any, Dict, List, Optional, Union

import torch

from torch import Tensor

from ..hdu import HDUList, Header, TensorHDU
from .paths import guard_fits_path
from .checksum_api import write_checksums as _write_checksums_impl
from ._hdu_rewrite import (
    _write_hdus_uncompressed,
    _write_hdus_with_optional_compression,
)
from ._write_helpers import (
    QuantizeError,
    UInt64WriteError,
    _TableHDUWriteProxy,
    _apply_image_quantize,
    _can_use_cpp_table_writer,
    _coerce_compressed_hdu_item,
    _cpp_header_mapping,
    _drop_stale_integer_scale_cards,
    _hdu_with_header,
    _image_hdu_dict_for_fits_write,
    _invalidate_path_caches,
    _is_skippable_empty_primary,
    _merge_fits_write_header,
    _merged_write_header,
    _normalize_cpp_table_data,
    _normalize_table_input,
    _payload_replay_header,
    _prepare_quantized_table_data_for_write,
    _prepare_unsigned_table_data_for_write,
    _require_image_hdu_dict_keys,
    _unsigned_image_storage_for_fits_write,
    _write_boundary_header,
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
]


def _write_all_checksums(path: str) -> None:
    """Write DATASUM/CHECKSUM for every HDU of a freshly written file."""
    import torchfits._C as cpp

    n_hdus = int(cpp.read_num_hdus(str(path)))
    for hdu in range(n_hdus):
        _write_checksums_impl(str(path), hdu=hdu)


def write(
    path: str | os.PathLike[str],
    data: Any,
    header: Optional[Header | Dict[str, Any]] = None,
    overwrite: bool = False,
    compress: Union[bool, str] = False,
    quantize: Any = None,
    checksum: bool = False,
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
            For images: ``"robust"`` or ``{"lo_q", "hi_q", "keep_zero"}``.
            For dict tables: ``"robust"`` (all float columns) or
            ``{"col": "robust" | opts}``.
        checksum: When True, compute and write CFITSIO DATASUM/CHECKSUM
            keywords for every HDU after the payload lands (archive-ingest
            friendly). Verify later with :func:`torchfits.verify_checksums`.

    Image tensors on non-CPU devices are detached and copied to CPU before
    the CFITSIO writer runs (in-memory input tensors are not modified).
    """
    path = os.fspath(path)
    guard_fits_path(path)
    if str(path).lower().endswith(".bz2"):
        # CFITSIO would create a plain, uncompressed FITS under this name.
        raise ValueError(
            "Writing bzip2-wrapped FITS files ('.bz2') is not supported; "
            "use compress='BZIP2_1' for tile compression inside the file."
        )
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
                checksum=checksum,
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
        data = _normalize_table_input(data)

        if compress:
            import numpy as np

            compressed_hdus: List[Any] = []
            if isinstance(data, (Tensor, np.ndarray)):
                if isinstance(data, np.ndarray):
                    data = torch.as_tensor(data)
                data, header = _apply_image_quantize(data, header, quantize)
                header = _drop_stale_integer_scale_cards(header, data)
                img, img_header = _unsigned_image_storage_for_fits_write(data)
                compressed_hdus = [
                    TensorHDU(
                        data=img,
                        header=_merge_fits_write_header(header, img_header),
                    )
                ]
            elif isinstance(data, HDUList):
                if quantize is not None:
                    raise QuantizeError(
                        "quantize= is ignored for HDUList writes; quantize each "
                        "image tensor before assembling the list"
                    )
                compressed_hdus = list(getattr(data, "_hdus", []))
                if header and compressed_hdus:
                    compressed_hdus[0] = _hdu_with_header(
                        compressed_hdus[0],
                        _merged_write_header(
                            getattr(compressed_hdus[0], "header", None), header
                        ),
                    )
            elif isinstance(data, dict):
                if "data" in data:
                    item: Dict[str, Any] = dict(data)
                    _require_image_hdu_dict_keys(item)
                    if header:
                        item["header"] = _merged_write_header(
                            item.get("header"), header
                        )
                    if quantize is not None:
                        q_img, q_hdr = _apply_image_quantize(
                            item.get("data"), item.get("header"), quantize
                        )
                        item["data"] = q_img
                        item["header"] = q_hdr
                    compressed_hdus.append(_coerce_compressed_hdu_item(item))
                else:
                    compressed_hdus.append(
                        _TableHDUWriteProxy(data, header, quantize=quantize)
                    )
            elif isinstance(data, (list, tuple)):
                for item in data:
                    compressed_hdus.append(_coerce_compressed_hdu_item(item))
                if header and compressed_hdus:
                    compressed_hdus[0] = _hdu_with_header(
                        compressed_hdus[0],
                        _merged_write_header(
                            getattr(compressed_hdus[0], "header", None), header
                        ),
                    )
            else:
                raise NotImplementedError(
                    "Compressed FITS writing supports tensors, tables, or HDU lists."
                )

            _write_hdus_with_optional_compression(
                path, compressed_hdus, compress=compress
            )
            if checksum:
                _write_all_checksums(path)
            _invalidate_path_caches(path)
            return

        if isinstance(data, HDUList):
            if quantize is not None:
                raise QuantizeError(
                    "quantize= is ignored for HDUList writes; quantize each "
                    "image tensor before assembling the list"
                )
            hdus = list(getattr(data, "_hdus", []))
            if header and hdus:
                hdus[0] = _hdu_with_header(
                    hdus[0],
                    _merged_write_header(getattr(hdus[0], "header", None), header),
                )
            _write_hdus_uncompressed(path, hdus, overwrite)
            if checksum:
                _write_all_checksums(path)
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
                header_obj = _write_boundary_header(header)
                cpp.write_fits_table(
                    path,
                    data,
                    _cpp_header_mapping(header_obj),
                    overwrite,
                    table_schema,
                    "binary",
                )
                _write_header_cards_if_supported(path, 1, header_obj)
                if checksum:
                    _write_all_checksums(path)
                return
            raise ValueError(
                "Dictionary table writes currently require CFITSIO-native column types "
                "(numeric/bool/complex, strings, or VLA lists). Unsupported object/structure "
                "columns should be converted before writing."
            )

        import numpy as np

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

        elif isinstance(data, dict):
            # Image dict-HDU form: exactly {"data": ..., "header": ...}.
            _require_image_hdu_dict_keys(data)
            payload = data["data"]
            if isinstance(payload, np.ndarray):
                payload = torch.as_tensor(payload)
            if not isinstance(payload, Tensor):
                raise TypeError("HDU dictionary 'data' must be a torch.Tensor")
            base = (
                _merged_write_header(data.get("header"), header)
                if header
                else data.get("header")
            )
            hdus_to_write.append(
                _image_hdu_dict_for_fits_write(payload, base, quantize=quantize)
            )

        elif hasattr(data, "__iter__") and not isinstance(data, (str, Tensor)):
            if quantize is not None:
                raise QuantizeError(
                    "quantize= is supported for a single image tensor or dict table, "
                    "not multi-HDU sequences"
                )
            for n, item in enumerate(data):
                overlay = header if (n == 0 and header) else None
                if isinstance(item, dict):
                    if "data" in item:
                        _require_image_hdu_dict_keys(item)
                        payload = item["data"]
                        if isinstance(payload, np.ndarray):
                            payload = torch.as_tensor(payload)
                        if isinstance(payload, Tensor):
                            base = (
                                _merged_write_header(item.get("header"), overlay)
                                if overlay
                                else item.get("header")
                            )
                            hdus_to_write.append(
                                _image_hdu_dict_for_fits_write(payload, base)
                            )
                        else:
                            raise TypeError(
                                "HDU dictionary 'data' must be a torch.Tensor"
                            )
                    else:
                        raise ValueError(
                            "HDU dictionaries must contain a 'data' tensor"
                        )
                elif isinstance(item, Tensor):
                    hdus_to_write.append(
                        _image_hdu_dict_for_fits_write(
                            item, _merged_write_header(None, overlay) if overlay else None
                        )
                    )
                elif hasattr(item, "data") and isinstance(item.data, Tensor):
                    base = (
                        _merged_write_header(getattr(item, "header", None), overlay)
                        if overlay
                        else getattr(item, "header", None)
                    )
                    hdus_to_write.append(_image_hdu_dict_for_fits_write(item.data, base))
                else:
                    raise TypeError(f"Unsupported HDU item type: {type(item).__name__}")
        else:
            raise ValueError(f"Unsupported data type for FITS writing: {type(data)}")

        if not hdus_to_write:
            raise ValueError("At least one writable HDU is required")
        cpp.write_fits_file(path, hdus_to_write, overwrite)
        for idx, item in enumerate(hdus_to_write):
            _write_header_cards_if_supported(
                path, idx, _payload_replay_header(item), invalidate=False
            )
        if checksum:
            _write_all_checksums(path)
        _invalidate_path_caches(path)

    except (UInt64WriteError, QuantizeError):
        if not path_exists and os.path.exists(path):
            os.remove(path)
        raise
    except ValueError as e:
        if not path_exists and os.path.exists(path):
            os.remove(path)
        raise RuntimeError(f"Failed to write FITS file '{path}': {e}") from e
    except Exception as e:
        if not path_exists and os.path.exists(path):
            os.remove(path)
        raise RuntimeError(f"Failed to write FITS file '{path}': {e}") from e
