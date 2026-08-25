"""Table write path: CFITSIO-backed table writes, header management, schema rewrite."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from .._io_engine.caches import invalidate_path_caches as _invalidate_path_caches

# Internal write/mutation helpers re-exported on torchfits.io (not a public API).
# Crossing _table → io → _io_engine is intentional package-internal wiring.
from ..io import _normalize_cpp_table_data, _write_header_cards_if_supported

_log = logging.getLogger(__name__)


def write(
    path: str,
    data: dict[str, Any],
    *,
    schema: dict[str, dict[str, Any]] | None = None,
    header: dict[str, Any] | None = None,
    overwrite: bool = False,
    extname: str | None = None,
    table_type: str = "binary",
    quantize: Any = None,
) -> None:
    """Write a columnar dict as a FITS table.

    ``quantize=\"robust\"`` packs all float columns to ``TFORM=I`` with
    ``TSCAL``/``TZERO``. Pass ``{\"col\": \"robust\"}`` (or per-column option
    dicts) to select columns. Default keeps native float ``TFORM``.
    """
    from .._io_engine._write_helpers import _normalize_table_input
    from .._io_engine.paths import guard_fits_path

    guard_fits_path(path)
    if str(path).lower().endswith(".bz2"):
        # CFITSIO would create a plain, uncompressed FITS under this name.
        raise ValueError(
            "Writing bzip2-wrapped FITS files ('.bz2') is not supported; "
            "use compress='BZIP2_1' for tile compression inside the file."
        )
    data = _normalize_table_input(data)
    if not isinstance(data, dict) or not data:
        raise ValueError(
            "data must be a non-empty dictionary or table/dataframe object"
        )
    table_kind = str(table_type).lower().strip()
    if table_kind not in {"binary", "ascii"}:
        raise ValueError("table_type must be 'binary' or 'ascii'")
    if schema is not None and not isinstance(schema, dict):
        raise TypeError("schema must be a dictionary when provided")
    if quantize is not None and table_kind == "ascii":
        raise ValueError("quantize= is only supported for binary tables")
    hdr: dict[str, Any] | None = None
    if extname is not None:
        hdr = dict(header or {})
        hdr["EXTNAME"] = str(extname)
    else:
        hdr = header
    import torchfits
    from .._io_engine.write_api import (
        _prepare_quantized_table_data_for_write,
        _prepare_unsigned_table_data_for_write,
    )

    data, schema, unsigned_converted = _prepare_unsigned_table_data_for_write(
        data, schema
    )
    data, schema, quantized = _prepare_quantized_table_data_for_write(
        data, quantize, schema
    )

    if schema or unsigned_converted or quantized or table_kind == "ascii":
        import torchfits._C as cpp

        _invalidate_path_caches(path)
        data = _normalize_cpp_table_data(data)
        cpp.write_fits_table(
            path,
            data,
            hdr if hdr else {},
            overwrite,
            schema if schema else None,
            table_kind,
        )
        if hdr:
            _write_header_cards_if_supported(path, 1, hdr)
        _invalidate_path_caches(path)
        return

    torchfits.write(
        path,
        data,
        header=hdr if hdr else None,
        overwrite=overwrite,
        quantize=quantize,
    )


def _header_cards_to_mapping(header_cards: Any) -> dict[str, Any]:
    if isinstance(header_cards, dict):
        return {str(k): v for k, v in header_cards.items()}
    out: dict[str, Any] = {}
    if isinstance(header_cards, (list, tuple)):
        for card in header_cards:
            if hasattr(card, "key") and hasattr(card, "value"):
                out[str(card.key)] = card.value
                continue
            if not isinstance(card, (list, tuple)) or len(card) < 2:
                continue
            out[str(card[0])] = card[1]
    return out


def _column_tform_map(header_map: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    name_by_idx = _column_name_index_map(header_map)
    tform_by_idx: dict[int, str] = {}
    for key, value in header_map.items():
        key_u = str(key).upper()
        if key_u.startswith("TFORM"):
            suffix = key_u[5:]
            if suffix.isdigit():
                tform_by_idx[int(suffix)] = str(value)

    for idx, name in name_by_idx.items():
        out[name] = tform_by_idx.get(idx, "")
    return out


def _column_name_index_map(header_map: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}

    try:
        tfields = int(header_map.get("TFIELDS", 0))
    except (ValueError, TypeError):
        tfields = 0

    if tfields > 0:
        for i in range(1, tfields + 1):
            val = header_map.get(f"TTYPE{i}")
            if val is not None:
                out[i] = str(val)
        return out

    for key, value in header_map.items():
        key_u = str(key).upper()
        if not key_u.startswith("TTYPE"):
            continue
        suffix = key_u[5:]
        if suffix.isdigit():
            out[int(suffix)] = str(value)
    return out


def _extract_table_schema_from_header(
    header_map: dict[str, Any], columns: list[str]
) -> dict[str, dict[str, Any]]:
    name_by_idx = _column_name_index_map(header_map)
    index_by_name = {name: idx for idx, name in name_by_idx.items()}
    schema: dict[str, dict[str, Any]] = {}
    for name in columns:
        idx = index_by_name.get(name)
        if idx is None:
            continue
        meta: dict[str, Any] = {}
        tform = header_map.get(f"TFORM{idx}")
        if tform is not None:
            meta["format"] = str(tform)
        tunit = header_map.get(f"TUNIT{idx}")
        if tunit is not None:
            meta["unit"] = str(tunit)
        tdim = header_map.get(f"TDIM{idx}")
        if tdim is not None:
            meta["dim"] = str(tdim)
        tnull = header_map.get(f"TNULL{idx}")
        if tnull is not None:
            meta["tnull"] = tnull
        tscal = header_map.get(f"TSCAL{idx}")
        if tscal is not None:
            meta["bscale"] = tscal
        tzero = header_map.get(f"TZERO{idx}")
        if tzero is not None:
            meta["bzero"] = tzero
        schema[name] = meta
    return schema


# Structural FITS keywords that must not be copied into rewritten table headers.
_TABLE_STRUCTURAL_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "CHECKSUM",
        "DATASUM",
        "EXTEND",
        "THEAP",
    }
)
_TABLE_COLUMN_SKIP_PREFIXES: tuple[str, ...] = (
    "TTYPE",
    "TFORM",
    "TUNIT",
    "TDIM",
    "TNULL",
    "TSCAL",
    "TZERO",
)


def _sanitize_table_header_for_rewrite(header_map: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in header_map.items():
        key_s = str(key)
        key_u = key_s.upper()
        if key_u in _TABLE_STRUCTURAL_SKIP_KEYS:
            continue
        if key_u.startswith(_TABLE_COLUMN_SKIP_PREFIXES):
            continue
        out[key_s] = value
    return out


def _ordered_dict_for_columns(
    columns: list[str], data_by_name: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in columns:
        out[name] = data_by_name[name]
    return out


def _rewrite_table_hdu_with_schema(
    path: str,
    target_hdu: int,
    data: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    header: dict[str, Any],
    table_type: str,
) -> None:
    import torchfits

    # Always use a temp-file + os.replace() so the rewrite is atomic and
    # never races with CFITSIO's internal file-locking when another handle
    # (e.g. torchfits.open()) is open on the same path.
    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        write(
            tmp_path,
            data=data,
            schema=schema,
            header=header,
            overwrite=True,
            table_type=table_type,
        )
        with torchfits.open(tmp_path) as hdul:
            replacement = hdul[1].materialize(device="cpu")  # type: ignore[union-attr,call-arg]
        torchfits.replace_hdu(path, target_hdu, replacement)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _resolve_table_hdu_index_and_columns(
    path: str, hdu: int | str
) -> tuple[int, dict[str, Any], list[str], dict[str, str]]:
    """Resolve table HDU index + column metadata for write/mutation/read helpers.

    Lives in ``_table.write`` (not ``_table.read``) so ``read`` / ``mutation``
    can import it without a write↔read cycle. Keep it here; do not move to
    ``read.py``.
    """
    import torchfits._C as cpp
    from .._io_engine.paths import guard_fits_path

    guard_fits_path(path)
    handle = cpp.open_fits_file(path, "r")
    try:
        num_hdus = int(cpp.get_num_hdus(handle))
        if num_hdus <= 0:
            raise RuntimeError(f"No HDUs found in '{path}'")

        target_idx: int | None = None
        if isinstance(hdu, int):
            target_idx = hdu
        elif isinstance(hdu, str):
            wanted = hdu.strip().upper()
            if not wanted:
                raise ValueError("hdu name cannot be empty")
            for i in range(num_hdus):
                hdu_type = str(cpp.get_hdu_type(handle, i))
                if hdu_type not in {"BINARY_TABLE", "ASCII_TABLE"}:
                    continue
                header_map = _header_cards_to_mapping(cpp.read_header(handle, i))
                extname = str(header_map.get("EXTNAME", "")).strip().upper()
                if extname == wanted:
                    target_idx = i
                    break
            if target_idx is None:
                raise KeyError(f"Table HDU named '{hdu}' not found in '{path}'")
        else:
            raise TypeError("hdu must be an int index or EXTNAME string")

        if target_idx is None or target_idx < 0 or target_idx >= num_hdus:
            raise IndexError(
                f"hdu index {hdu} out of range for '{path}' (num_hdus={num_hdus})"
            )

        hdu_type = str(cpp.get_hdu_type(handle, target_idx))
        if hdu_type not in {"BINARY_TABLE", "ASCII_TABLE"}:
            raise ValueError(f"HDU {target_idx} is not a table (type={hdu_type})")

        header_map = _header_cards_to_mapping(cpp.read_header(handle, target_idx))
        col_map = _column_name_index_map(header_map)
        columns = [col_map[idx] for idx in sorted(col_map)]
        tform_map = _column_tform_map(header_map)
        return target_idx, header_map, columns, tform_map
    finally:
        try:
            handle.close()
        except Exception as exc:
            _log.warning("table write: failed to close handle for %r: %s", path, exc)
