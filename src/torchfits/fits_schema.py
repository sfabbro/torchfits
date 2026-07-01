"""Shared FITS binary-table header parsing (TFORM, VLA, string, bit, unsigned)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import re
from typing import Any, Optional

import torch

_TFORM_RE = re.compile(r"^\s*(\d+)?\s*([A-Za-z])")
_TFORM_VLA_RE = re.compile(r"^\s*(\d+)?\s*([PQ])\s*([A-Za-z])")
_TFORM_FULL_RE = re.compile(r"(\d*)([PQ]?)([A-Z])")

_UNSIGNED_TZERO_TARGETS: dict[tuple[str, float], torch.dtype] = {
    ("I", 32768.0): torch.uint16,
    ("J", 2147483648.0): torch.uint32,
}


@dataclass(frozen=True)
class TformInfo:
    """Parsed FITS TFORM descriptor."""

    tform: str
    repeat: int
    vla: bool
    code: str | None
    vla_descriptor: str | None = None

    @property
    def is_string(self) -> bool:
        return self.code == "A"

    @property
    def is_bit(self) -> bool:
        return self.code == "X"


@dataclass(frozen=True)
class TableColumnMeta:
    """One binary-table column inferred from header cards."""

    index: int
    name: str
    tform: str
    tform_info: TformInfo
    tdim: str | None = None
    tnull: Any = None
    tscal: float | None = None
    tzero: float | None = None


def parse_tform(tform: str) -> TformInfo:
    """Parse a FITS TFORM string into structured metadata."""
    text = str(tform).strip().upper()
    if not text:
        return TformInfo(tform="", repeat=1, vla=False, code=None)

    m = _TFORM_VLA_RE.match(text)
    if m:
        repeat = int(m.group(1)) if m.group(1) else 1
        descriptor = m.group(2).upper()
        code = m.group(3).upper()
        return TformInfo(
            tform=text,
            repeat=repeat,
            vla=True,
            code=code,
            vla_descriptor=descriptor,
        )

    m = _TFORM_RE.match(text)
    if m:
        repeat = int(m.group(1)) if m.group(1) else 1
        code = m.group(2).upper()
        return TformInfo(tform=text, repeat=repeat, vla=False, code=code)

    return TformInfo(tform=text, repeat=1, vla=False, code=None)


def tform_code_and_repeat(tform: Any) -> tuple[str, int] | None:
    """Return (code, repeat) for a scalar TFORM, or None if unparseable."""
    if not isinstance(tform, str):
        return None
    info = parse_tform(tform)
    if info.vla or info.code is None:
        return None
    return info.code, info.repeat


def tform_is_bit(tform: Any) -> bool:
    parsed = tform_code_and_repeat(tform)
    return parsed is not None and parsed[0] == "X"


def _tfields_count(header: Mapping[str, Any]) -> int:
    try:
        return int(header.get("TFIELDS", 0))
    except (TypeError, ValueError):
        return 0


def _iter_tfields_indexed(
    header: Mapping[str, Any],
) -> Iterator[tuple[int, str, str, str | None]]:
    """Yield (index, name, tform, tdim) using TFIELDS fast-path or card scan."""
    tfields = _tfields_count(header)
    if tfields > 0:
        for i in range(1, tfields + 1):
            name = header.get(f"TTYPE{i}")
            if name is None:
                continue
            tform = header.get(f"TFORM{i}")
            tdim = header.get(f"TDIM{i}")
            yield (
                i,
                str(name),
                str(tform) if tform is not None else "",
                (str(tdim) if tdim is not None else None),
            )
        return

    name_by_idx: dict[int, str] = {}
    tform_by_idx: dict[int, str] = {}
    tdim_by_idx: dict[int, str] = {}
    for key, value in header.items():
        if not isinstance(key, str):
            continue
        key_u = key.upper()
        if key_u.startswith("TTYPE"):
            suffix = key_u[5:]
            if suffix.isdigit():
                name_by_idx[int(suffix)] = str(value)
        elif key_u.startswith("TFORM"):
            suffix = key_u[5:]
            if suffix.isdigit():
                tform_by_idx[int(suffix)] = str(value)
        elif key_u.startswith("TDIM"):
            suffix = key_u[4:]
            if suffix.isdigit():
                tdim_by_idx[int(suffix)] = str(value)

    for idx in sorted(name_by_idx.keys()):
        yield (
            idx,
            name_by_idx[idx],
            tform_by_idx.get(idx, ""),
            tdim_by_idx.get(idx),
        )


def iter_table_columns(
    header: Mapping[str, Any],
    *,
    selected: set[str] | None = None,
) -> Iterator[TableColumnMeta]:
    """Walk table columns from a FITS header mapping."""
    for idx, name, tform, tdim in _iter_tfields_indexed(header):
        if selected is not None and name not in selected:
            continue
        info = parse_tform(tform) if tform else parse_tform("")
        tnull = header.get(f"TNULL{idx}")
        tscal_raw = header.get(f"TSCAL{idx}")
        tzero_raw = header.get(f"TZERO{idx}")
        tscal = float(tscal_raw) if tscal_raw is not None else 1.0
        tzero = float(tzero_raw) if tzero_raw is not None else 0.0
        yield TableColumnMeta(
            index=idx,
            name=name,
            tform=tform,
            tform_info=info,
            tdim=tdim,
            tnull=tnull,
            tscal=tscal,
            tzero=tzero,
        )


def table_columns(
    header: Mapping[str, Any],
    *,
    selected: set[str] | None = None,
) -> list[TableColumnMeta]:
    return list(iter_table_columns(header, selected=selected))


def _tform_might_be_vla(raw: Any) -> bool:
    raw_str = str(raw)
    return any(ch in raw_str for ch in "PpQq")


def table_has_vla(header: Mapping[str, Any]) -> bool:
    """True if any column uses FITS P/Q variable-length array heap storage."""
    for _idx, _name, tform, _tdim in _iter_tfields_indexed(header):
        if not tform or not _tform_might_be_vla(tform):
            continue
        if parse_tform(tform).vla:
            return True
    return False


def column_is_vla(header: Mapping[str, Any], col_name: str) -> bool:
    want = str(col_name)
    for _idx, name, tform, _tdim in _iter_tfields_indexed(header):
        if name != want:
            continue
        if not tform or not _tform_might_be_vla(tform):
            return False
        return parse_tform(tform).vla
    return False


def selected_includes_vla(
    header: Mapping[str, Any],
    columns: Optional[list[str]],
) -> bool:
    """True when the read projection would include at least one VLA column."""
    if columns is None:
        return table_has_vla(header)
    want = set(columns)
    for _idx, name, tform, _tdim in _iter_tfields_indexed(header):
        if name not in want:
            continue
        if tform and _tform_might_be_vla(tform) and parse_tform(tform).vla:
            return True
    return False


def string_column_names(
    header: Mapping[str, Any],
    *,
    selected: set[str] | None = None,
) -> list[str]:
    """Return column names whose TFORM code is character ('A')."""
    out: list[str] = []
    for col in iter_table_columns(header, selected=selected):
        if col.tform_info.is_string:
            out.append(col.name)
    return sorted(set(out))


def bit_column_names(header: Mapping[str, Any]) -> set[str]:
    """Return column names encoded as FITS bit arrays (TFORM ... X)."""
    out: set[str] = set()
    for col in iter_table_columns(header):
        if col.tform_info.is_bit:
            out.add(col.name)
    return out


def unsigned_column_dtypes_from_header(
    header: Mapping[str, Any],
) -> dict[str, torch.dtype]:
    """Map standard unsigned FITS table conventions (TZERO offset) to torch dtypes."""
    out: dict[str, torch.dtype] = {}
    for col in iter_table_columns(header):
        code = col.tform_info.code
        if code is None:
            continue
        if col.tscal != 1.0:
            continue
        target = _UNSIGNED_TZERO_TARGETS.get((code, col.tzero))
        if target is not None:
            out[col.name] = target
    return out


def column_tnull_map(header_map: Mapping[str, Any]) -> dict[str, Any]:
    """Map column name -> TNULL value from a flat header dict."""
    out: dict[str, Any] = {}
    tfields = _tfields_count(header_map)
    if tfields > 0:
        for i in range(1, tfields + 1):
            tnull = header_map.get(f"TNULL{i}")
            if tnull is None:
                continue
            ttype = header_map.get(f"TTYPE{i}")
            if ttype is not None:
                out[str(ttype)] = tnull
        return out

    name_by_idx: dict[int, str] = {}
    tnull_by_idx: dict[int, Any] = {}
    for key, value in header_map.items():
        key_u = str(key).upper()
        if key_u.startswith("TTYPE"):
            suffix = key_u[5:]
            if suffix.isdigit():
                name_by_idx[int(suffix)] = str(value)
        elif key_u.startswith("TNULL"):
            suffix = key_u[5:]
            if suffix.isdigit():
                tnull_by_idx[int(suffix)] = value

    for idx, name in name_by_idx.items():
        if idx in tnull_by_idx:
            out[name] = tnull_by_idx[idx]
    return out


def build_table_schema_dict(
    header: Mapping[str, Any],
    *,
    selected_columns: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Return schema summary used by TableHDU and TableHDURef."""
    if not header:
        return {"columns": [], "string_columns": [], "vla_columns": []}

    selected = set(selected_columns) if selected_columns is not None else None
    columns: list[dict[str, Any]] = []
    string_cols: list[str] = []
    vla_cols: list[str] = []

    for col in iter_table_columns(header, selected=selected):
        info = col.tform_info
        if info.is_string:
            string_cols.append(col.name)
        if info.vla:
            vla_cols.append(col.name)
        entry: dict[str, Any] = {
            "name": col.name,
            "tform": info.tform,
            "repeat": info.repeat,
            "code": info.code,
            "string": info.is_string,
            "vla": info.vla,
            "tdim": col.tdim,
        }
        if info.vla and info.vla_descriptor is not None:
            entry["vla_descriptor"] = info.vla_descriptor
        columns.append(entry)

    return {
        "columns": columns,
        "string_columns": string_cols,
        "vla_columns": vla_cols,
    }
