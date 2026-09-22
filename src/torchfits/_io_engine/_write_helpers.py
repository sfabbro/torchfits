"""Write-path normalization helpers for FITS I/O."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
import torch
from torch import Tensor

from ..hdu import Header, TableHDU, TableHDURef, TensorHDU
from .caches import invalidate_path_caches as _invalidate_io_path_caches
from .quantize import (
    parse_image_quantize_spec,
    parse_table_quantize_spec,
    quantize_int16_robust,
)


class UInt64WriteError(ValueError):
    """FITS has no native uint64 storage (BITPIX=-64 is non-standard)."""


class QuantizeError(ValueError):
    """Invalid quantize= specification or misuse."""


# HISTORY/COMMENT carry no value and legitimately repeat; they must be written
# by the card-replay step (fits_write_history/fits_write_comment). The C++
# dict writers render them as value cards ('HISTORY = ...') and clobber
# CFITSIO's standard COMMENT block with fits_update_key.
_COMMENTARY_HEADER_KEYS = frozenset({"HISTORY", "COMMENT"})

# CHECKSUM/DATASUM describe one payload byte-for-byte; replaying a copy from a
# source header makes a freshly written file fail verification. Fresh keywords
# are stamped by write(..., checksum=True) / restamp after rewrites.
_STALE_CHECKSUM_KEYS = frozenset({"CHECKSUM", "DATASUM"})

# Table cards the writer derives from the payload (column names/formats and
# TSCAL/TZERO rescale). A stale copy in a user header -- typically one read
# from a different table -- must never override the derived cards: a wrong
# TFORM relabels the stored bytes and a wrong NAXIS2 forges rows. TUNIT/TDIM/
# TNULL are annotations the writer does not derive and are kept.
_TABLE_DERIVED_KEYS = frozenset(
    {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "EXTEND",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "THEAP",
    }
)
_TABLE_DERIVED_KEY_PREFIXES = ("NAXIS", "TTYPE", "TFORM", "TSCAL", "TZERO")

# CompImage compression metadata: the compressed writer derives these from the
# tiles it stores. A compressed image extension read back as raw cards carries
# Z* and tile-column TFORM/TTYPE cards of its bintable skeleton; replaying them
# onto a fresh write corrupts the new tile layout (the reader then surfaces the
# mangled extension as a table).
_COMPRESSION_CARD_KEYS = frozenset(
    {
        "ZIMAGE",
        "ZCMPTYPE",
        "ZBITPIX",
        "ZNAXIS",
        "ZPCOUNT",
        "ZGCOUNT",
        "ZCHECKSUM",
        "ZHECKSUM",
        "ZDATASUM",
        "ZQUANTIZ",
        "ZBLANK",
    }
)
_COMPRESSION_CARD_PREFIXES = ("ZNAXIS", "ZTILE", "ZNAME", "ZVAL")

_WRITE_BOUNDARY_DROP_KEYS = (
    _STALE_CHECKSUM_KEYS | _TABLE_DERIVED_KEYS | _COMPRESSION_CARD_KEYS
)
_WRITE_BOUNDARY_DROP_PREFIXES = (
    _TABLE_DERIVED_KEY_PREFIXES + _COMPRESSION_CARD_PREFIXES
)


def _filter_header_cards(
    header: Any,
    drop_keys: frozenset,
    drop_prefixes: tuple = (),
) -> Header:
    """Card-faithful copy of *header* minus the matching cards (duplicates and
    card order preserved)."""
    source = header if isinstance(header, Header) else Header(header or {})
    out = Header()
    for card in source.cards:
        key_u = str(card.key).upper()
        if key_u in drop_keys or key_u.startswith(drop_prefixes):
            continue
        out.append(card)
    return out


def _write_boundary_header(header: Any) -> Header:
    """Write-boundary header for one HDU.

    Cards the writer derives from the payload are dropped so stale copies in
    a user header can never relabel or rescale the new data: the structural
    block, CHECKSUM/DATASUM, CompImage Z* metadata and the table column
    descriptors (TTYPE/TFORM/TSCAL/TZERO). HISTORY/COMMENT cards are kept
    here (the replay step writes them faithfully). Rewrites re-add the source
    TFORM spelling per column via _preserved_table_tform_cards.
    """
    return _filter_header_cards(
        header, _WRITE_BOUNDARY_DROP_KEYS, _WRITE_BOUNDARY_DROP_PREFIXES
    )


def _prepared_tform_code(value: Any) -> Optional[str]:
    """TFORM type char matching a prepared column's storage, or None when the
    storage is not a plain numeric/bool cell (strings, VLA, bits)."""
    import numpy as np

    if isinstance(value, Tensor):
        mapping = {
            torch.int8: "B",
            torch.uint8: "B",
            torch.int16: "I",
            torch.int32: "J",
            torch.int64: "K",
            torch.float32: "E",
            torch.float64: "D",
            torch.bool: "L",
            torch.complex64: "C",
            torch.complex128: "M",
        }
        return mapping.get(value.dtype)
    if isinstance(value, np.ndarray):
        mapping = {
            np.dtype("int8"): "B",
            np.dtype("uint8"): "B",
            np.dtype("int16"): "I",
            np.dtype("int32"): "J",
            np.dtype("int64"): "K",
            np.dtype("float32"): "E",
            np.dtype("float64"): "D",
            np.dtype("bool"): "L",
            np.dtype("complex64"): "C",
            np.dtype("complex128"): "M",
        }
        return mapping.get(value.dtype)
    return None


def _preserved_table_tform_cards(
    source_header: Any, prepared_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Source ``TFORMn`` spellings preserved verbatim for columns whose
    prepared storage still matches them.

    astropy spells repeat-1 formats as bare type chars ('J', 'I'); C++
    re-derivation relabels them '1J'/'1I'. Copies and rewrites must keep the
    source spelling (and exotic spellings like '8X'/'PJ'). A column whose
    materialized data left its storage domain (e.g. TSCAL/TZERO-scaled ints
    decoded to floats) must NOT keep the old TFORM -- the label would
    misdescribe the written cells (see r4b-01).
    """
    import numpy as np

    src = (
        source_header
        if isinstance(source_header, Header)
        else Header(source_header or {})
    )
    out: Dict[str, Any] = {}
    for i, value in enumerate(prepared_data.values(), start=1):
        tform = src.get(f"TFORM{i}")
        if tform is None:
            continue
        letters = "".join(ch for ch in str(tform).upper() if ch.isalpha())
        first = letters[:1]
        if first in {"X", "A", "P", "Q"}:
            # Bit/string/VLA spellings carry layout the writer does not
            # re-derive faithfully; always preserve.
            out[f"TFORM{i}"] = tform
            continue
        expected = _prepared_tform_code(value)
        if expected is None or letters == expected:
            out[f"TFORM{i}"] = tform
    return out


def _cpp_header_mapping(header: Header) -> Dict[str, Any]:
    """Mapping view of a write-boundary header for the C++ dict writers.

    HISTORY/COMMENT are excluded: the C++ ``fits_update_key`` loop mangles
    them into value cards (the replay step writes them properly).
    """
    out = dict(header)
    for key in list(out):
        if str(key).upper() in _COMMENTARY_HEADER_KEYS:
            del out[key]
    return out


def _merged_write_header(base: Any, overlay: Any) -> Header:
    """Card-faithful header merge for write(header=) semantics.

    Overlay value cards replace the base's, HISTORY/COMMENT cards append
    (they legitimately repeat); card order and comments are preserved.
    """
    merged = Header(base) if isinstance(base, Header) else Header(base or {})
    user = overlay if isinstance(overlay, Header) else Header(overlay or {})
    for card in user.cards:
        if str(card.key).upper() in _COMMENTARY_HEADER_KEYS:
            merged.append(card)
        else:
            merged.remove(str(card.key), ignore_missing=True, remove_all=True)
            merged.append(card)
    return merged


def _require_image_hdu_dict_keys(item: Dict[str, Any]) -> None:
    """Image dict-HDUs carry exactly 'data' and 'header'; extra keys would be
    silently dropped from the written file."""
    extra = sorted(str(k) for k in item if k != "data" and k != "header")
    if extra:
        raise ValueError(
            "HDU dictionary may contain only 'data' and 'header' keys; "
            f"unexpected keys: {extra!r}"
        )


def _invalidate_path_caches(path: str) -> None:
    """Invalidate Python-side caches/handles for a path that is being modified."""
    _invalidate_io_path_caches(path)
    import torchfits._C as cpp

    # The native side used to expose a per-path invalidate_file_cache as well;
    # it was a no-op with a misleading name (shared handles were removed), so
    # the live native state is cleared globally here.
    clear_meta = getattr(cpp, "clear_shared_read_meta_cache", None)
    if clear_meta is not None:
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


def _normalize_table_input(data: Any) -> Any:
    """Normalize Table/DataFrame inputs (Astropy, Arrow, Polars, Pandas) into a columnar dict."""
    if isinstance(data, dict):
        return data
    # Astropy Table / QTable
    if hasattr(data, "colnames") and hasattr(data, "columns"):
        return {str(col): data[col] for col in data.colnames}
    # PyArrow Table / RecordBatch
    if hasattr(data, "column_names") and hasattr(data, "column"):
        cols: dict[str, Any] = {}
        for name in data.column_names:
            c = data[name]
            try:
                cols[name] = c.to_numpy(zero_copy_only=False)
            except Exception:
                cols[name] = c.to_pylist()
        return cols
    # Polars DataFrame
    if hasattr(data, "columns") and hasattr(data, "to_dict"):
        try:
            return {col: data[col].to_numpy() for col in data.columns}
        except Exception:
            return dict(data.to_dict(as_series=False))
    # Pandas DataFrame
    if hasattr(data, "columns") and hasattr(data, "to_dict"):
        return {col: data[col].to_numpy() for col in data.columns}
    return data


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
    if tensor.dtype == torch.uint64:
        raise UInt64WriteError(
            "torchfits does not support writing uint64 image tensors: FITS has "
            "no native uint64 storage (BITPIX=-64 is not standard), and a "
            "BZERO=2**64 pseudo-unsigned convention is not interoperable. "
            "Convert to int64 (requires values < 2**63) or float64 before writing."
        )
    if tensor.dtype == torch.uint16:
        raw: Tensor = (tensor.to(torch.int32) - 32768).to(torch.int16)
        return raw, {"BSCALE": 1.0, "BZERO": 32768.0}
    if tensor.dtype == torch.uint32:
        raw = (tensor.to(torch.int64) - 2147483648).to(torch.int32)
        return raw, {"BSCALE": 1.0, "BZERO": 2147483648.0}
    return tensor, {}


_INTEGER_STORAGE_SCALE_KEYS = ("BSCALE", "BZERO", "BLANK")


def _drop_stale_integer_scale_cards(
    header: Optional[Dict[str, Any]],
    tensor: Tensor,
) -> Optional[Dict[str, Any]]:
    """Drop integer-storage scale cards copied onto an already-decoded float.

    CFITSIO applies BSCALE/BZERO to BITPIX=-32 payloads. Replaying a quantized
    source header (BITPIX=16 plus a tiny BSCALE) onto physical floats
    double-scales on the next read. Quantize runs first and leaves int16, so
    those writes keep the cards. Float headers (BITPIX < 0) keep an explicit
    BSCALE.
    """
    if header is None or not tensor.is_floating_point():
        return header
    getter = getattr(header, "get", None)
    bitpix = getter("BITPIX") if getter is not None else None
    try:
        bp = int(bitpix) if bitpix is not None else None
    except (TypeError, ValueError):
        bp = None
    if bp is None or bp < 0:
        return header
    out = Header(header)
    for key in _INTEGER_STORAGE_SCALE_KEYS:
        if key in out:
            del out[key]
    return out


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
        raise QuantizeError("quantize= requires a torch.Tensor image")
    if not tensor.is_floating_point():
        raise QuantizeError(
            f"quantize= requires a floating-point image tensor, got dtype={tensor.dtype}"
        )
    packed = quantize_int16_robust(
        tensor, lo_q=opts.lo_q, hi_q=opts.hi_q, keep_zero=opts.keep_zero
    )
    extra = {"BSCALE": packed.scale, "BZERO": packed.zero}
    if packed.blank_code is not None:
        extra["BLANK"] = int(packed.blank_code)
    return packed.codes, _merge_fits_write_header(header, extra)


def _image_hdu_dict_for_fits_write(
    tensor: Tensor,
    header: Optional[Dict[str, Any]] = None,
    *,
    quantize: Any = None,
) -> Dict[str, Any]:
    tensor, header = _apply_image_quantize(tensor, header, quantize)
    header = _drop_stale_integer_scale_cards(header, tensor)
    data, extra_header = _unsigned_image_storage_for_fits_write(tensor)
    full = _merge_fits_write_header(_write_boundary_header(header), extra_header)
    hdu_dict: Dict[str, Any] = {"data": data}
    if full.cards:
        # C++ dict writers see the mapping (no HISTORY/COMMENT); the replay
        # step writes the full card list faithfully afterwards.
        hdu_dict["header"] = _cpp_header_mapping(full)
        hdu_dict["_replay"] = full
    return hdu_dict


def _payload_replay_header(payload: Any) -> Optional[Header]:
    """Full replay header stashed on a write payload (see _replay)."""
    if isinstance(payload, dict):
        return payload.get("_replay")
    return getattr(payload, "_replay", None)


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
        if packed.blank_code is not None:
            # C++ writer emits TNULLn from this key (B4).
            meta["null"] = int(packed.blank_code)
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
    *,
    invalidate: bool = True,
) -> None:
    """Replay header cards for one HDU via CFITSIO.

    ``invalidate=False`` defers cache invalidation to the caller — multi-HDU
    writers batch the (process-global) flush once after their loop instead of
    paying two full invalidations per HDU.
    """
    if not header:
        return
    header_obj = header if isinstance(header, Header) else Header(header)
    if not header_obj.cards:
        return
    import torchfits._C as cpp

    writer = getattr(cpp, "write_hdu_header_cards", None)
    if writer is None:
        return
    if invalidate:
        # Invalidate both before and after the C++ header-card write:
        #  - Before: ensure no stale handle/metadata races with writing new cards.
        #  - After:  ensure the next reader picks up the freshly-written header.
        _invalidate_path_caches(path)
        writer(path, int(hdu), list(header_obj.cards))
        _invalidate_path_caches(path)
        return
    writer(path, int(hdu), list(header_obj.cards))


def _delete_header_key_if_supported(path: str, hdu: int, key: str) -> None:
    """Delete one non-structural header keyword via CFITSIO ``fits_delete_key``."""
    import torchfits._C as cpp

    deleter = getattr(cpp, "delete_hdu_header_key", None)
    if deleter is None:
        raise RuntimeError("delete_hdu_header_key is unavailable in this build")
    _invalidate_path_caches(path)
    deleter(path, int(hdu), str(key))
    _invalidate_path_caches(path)


def _vla_item_signature(item: Any) -> tuple[Any, ...] | None:
    """Return a comparable dtype signature for a ragged/VLA column item.

    Signatures are ``(kind, itemsize)`` tuples (plus ``("str",)``/``("none",)``
    markers) so callers can enforce a single dtype per VLA column — the C++
    writer infers the CFITSIO column type from the first non-empty row and
    reinterprets any mismatching row buffer byte-wise otherwise. ``None``
    means the item type is unsupported.
    """
    import numpy as np

    if item is None:
        return ("none",)
    if isinstance(item, (str, bytes, np.str_, np.bytes_)):
        return ("str",)
    if isinstance(item, Tensor):
        mapping = {
            torch.bool: ("b", 1),
            torch.uint8: ("u", 1),
            torch.int8: ("i", 1),
            torch.int16: ("i", 2),
            torch.int32: ("i", 4),
            torch.int64: ("i", 8),
            torch.float32: ("f", 4),
            torch.float64: ("f", 8),
            torch.complex64: ("c", 8),
            torch.complex128: ("c", 16),
        }
        if item.dim() > 1:
            return None
        return mapping.get(item.dtype)
    if isinstance(item, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(item)
        except Exception:
            return None
        if arr.ndim > 1 or arr.dtype == np.object_:
            return None
        kind = arr.dtype.kind
        itemsize = arr.dtype.itemsize
        if kind == "b":
            return ("b", 1)
        if kind == "c" and itemsize in (8, 16):
            return ("c", itemsize)
        if kind == "u" and itemsize == 1:
            return ("u", 1)
        if kind == "i" and itemsize in (1, 2, 4, 8):
            return ("i", itemsize)
        if kind == "f" and itemsize in (4, 8):
            return ("f", itemsize)
    return None


def _vla_column_items_ok(items: Any) -> bool:
    """Validate ragged/VLA column items: supported types, 1-D, uniform dtype.

    String items are only allowed when *every* non-None item is a string
    (CFITSIO has no string VLA); array items must all share one dtype so the
    C++ writer never reinterprets a row buffer under a foreign type.
    """
    saw_str = False
    saw_data = False
    sig: tuple[Any, ...] | None = None
    for item in items:
        s = _vla_item_signature(item)
        if s is None:
            return False
        if s == ("none",):
            continue
        if s == ("str",):
            saw_str = True
            if saw_data:
                return False
            continue
        saw_data = True
        if saw_str:
            return False
        if sig is None:
            sig = s
        elif s != sig:
            return False
    return True


def _decode_col_string(x: Any) -> str:
    """Decode an object/string-column entry to a plain str for the C++ writer."""
    import numpy as np

    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("ascii", "replace")
    if isinstance(x, np.bytes_):
        return bytes(x).decode("ascii", "replace")
    return str(x)


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
                torch.int8,
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
                if not _vla_column_items_ok(value):
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
        if value.dtype == np.object_:
            if value.ndim == 1 and _vla_column_items_ok(value):
                continue
            return False
        if value.dtype.kind in {"U", "S"}:
            continue
        kind = value.dtype.kind
        itemsize = value.dtype.itemsize
        if kind == "b":
            continue
        if kind == "u" and itemsize == 1:
            continue
        if kind == "i" and itemsize in (1, 2, 4, 8):
            continue
        if kind == "f" and itemsize in (4, 8):
            continue
        return False

    return True


def _normalize_vla_item(item: Any) -> np.ndarray:
    """Normalize a single VLA item."""
    import numpy as np

    if isinstance(item, Tensor):
        t = item.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dim() == 0:
            t = t.reshape(1)
        out = np.ascontiguousarray(t.numpy())
    elif isinstance(item, np.ndarray):
        out = np.ascontiguousarray(item)
    elif item is None:
        return np.asarray([], dtype=np.float32)
    elif isinstance(item, (str, bytes, np.str_, np.bytes_)):
        raise ValueError(
            "VLA columns accept numeric/bool/complex array items, not strings; "
            "mixed string/array columns are not writable"
        )
    else:
        out = np.asarray(item)
        if out.ndim == 0:
            out = out.reshape(1)
        out = np.ascontiguousarray(out)
    if not out.flags.writeable:
        out = out.copy()
    return out


def _normalize_list_sequence(items: list[Any]) -> Any:
    """Normalize a list or tuple sequence."""
    import numpy as np

    if not items:
        return np.asarray(items)
    has_str = any(isinstance(item, (str, bytes, np.str_, np.bytes_)) for item in items)
    has_array = any(
        isinstance(item, (list, tuple, np.ndarray, Tensor)) for item in items
    )
    if has_str and has_array:
        raise ValueError(
            "Mixed string and array items in one table column are not writable"
        )
    if has_str:
        return [_decode_col_string(item) for item in items]
    if has_array or all(item is None for item in items):
        return [_normalize_vla_item(item) for item in items]
    return np.asarray(items)


def _normalize_ndarray_column(value: np.ndarray) -> Any:
    """Normalize an ndarray column."""
    import numpy as np

    if value.dtype == np.object_:
        if value.ndim == 1 and all(
            isinstance(x, (str, bytes, np.str_, np.bytes_)) or x is None for x in value
        ):
            return [_decode_col_string(x) for x in value]
        return [_normalize_vla_item(item) for item in value]
    if value.dtype.kind in {"U", "S"}:
        return value.astype(str).tolist()
    if not value.flags.writeable:
        return np.ascontiguousarray(value).copy()
    return np.ascontiguousarray(value)


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
        import numpy as np

        is_uint64 = (
            isinstance(value, torch.Tensor) and value.dtype == torch.uint64
        ) or (isinstance(value, np.ndarray) and value.dtype == np.uint64)
        if is_uint64:
            raise UInt64WriteError(
                f"torchfits does not support writing uint64 table column "
                f"{col_name!r}: FITS has no native uint64 storage (BITPIX=-64 "
                f"is not standard), and a BZERO=2**64 pseudo-unsigned "
                f"convention is not interoperable. Convert to int64 (requires "
                f"values < 2**63) or float64 before writing."
            )
        converted = _unsigned_table_storage_for_fits_write(value)
        if converted is None:
            out[col_name] = value
            # Register every column in the synthesized schema: the C++ writer
            # requires schema size == data size when a schema is present.
            prepared_schema.setdefault(col_name, {})
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
        if "null" in meta:
            out[f"TNULL{idx}"] = int(meta["null"])
    return out


def _normalize_cpp_table_data(table_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize table data for the C++ writer (strings/VLA/object arrays)."""
    import numpy as np

    out: Dict[str, Any] = {}
    for name, value in table_dict.items():
        if isinstance(value, (list, tuple)):
            out[name] = _normalize_list_sequence(list(value))
        elif isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                value = value.detach().cpu()
            out[name] = value
        elif isinstance(value, np.ndarray) or hasattr(value, "__array__"):
            arr = np.asarray(value)
            if not arr.flags.writeable or not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr.copy())
            out[name] = _normalize_ndarray_column(arr)
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


class _TableHDUWriteProxy:
    """Small table-HDU proxy for internal writer paths.

    ``header`` is the mapping view for the C++ dict writers; ``_replay``
    carries the full card list for the replay step.
    """

    def __init__(self, raw_data: Dict[str, Any], header: Any, quantize: Any = None):
        prepared, schema, _ = _prepare_unsigned_table_data_for_write(dict(raw_data))
        prepared, schema, _ = _prepare_quantized_table_data_for_write(
            prepared, quantize, schema
        )
        self._raw_data = _normalize_cpp_table_data(prepared)
        self._schema = schema
        full = _merge_fits_write_header(
            _write_boundary_header(header),
            _table_schema_scale_header_cards(schema),
        )
        self._replay = full
        self.header = _cpp_header_mapping(full)

    def _with_header(self, overlay: Any) -> "_TableHDUWriteProxy":
        """Header-replaced shallow copy (never mutates the caller's data)."""
        clone = _TableHDUWriteProxy.__new__(_TableHDUWriteProxy)
        clone._raw_data = self._raw_data
        clone._schema = self._schema
        full = _merge_fits_write_header(
            _write_boundary_header(_merged_write_header(self._replay, overlay)),
            _table_schema_scale_header_cards(self._schema),
        )
        clone._replay = full
        clone.header = _cpp_header_mapping(full)
        return clone


def _hdu_with_header(hdu: Any, header: Any) -> Any:
    """Shallow copy of an HDU carrying a replacement header.

    User HDU objects are never mutated; internal write proxies copy by
    value.
    """
    if isinstance(hdu, TensorHDU):
        return TensorHDU(data=hdu.to_tensor("cpu"), header=header)
    if isinstance(hdu, TableHDU):
        return TableHDU(dict(getattr(hdu, "_raw_data", {})), header=header)
    if isinstance(hdu, TableHDURef):
        mat = hdu.materialize(device="cpu")
        return TableHDU(dict(getattr(mat, "_raw_data", {})), header=header)
    if isinstance(hdu, _TableHDUWriteProxy):
        return hdu._with_header(header)
    if isinstance(hdu, dict):
        out = dict(hdu)
        out["header"] = header
        return out
    raise TypeError(f"cannot replace the header of {type(hdu).__name__}")


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
            _require_image_hdu_dict_keys(item)
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
                header=_merge_fits_write_header(
                    _write_boundary_header(item.get("header", {})),
                    img_header,
                ),
            )
        return _TableHDUWriteProxy(item, Header())
    raise NotImplementedError(
        f"Unsupported HDU payload for compressed write: {type(item)}"
    )
