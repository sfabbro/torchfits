"""Lazy file-backed table handle: metadata-only view with out-of-core streaming."""

from __future__ import annotations

import html
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch import Tensor

from .header import Header
from .table_hdu import TableHDU


class _TableHDURefDataWrapper:
    def __init__(self, parent: "TableHDURef"):
        self._parent = parent

    def __getitem__(self, key: str) -> Any:
        return self._parent[key]

    def __contains__(self, key: str) -> bool:
        return key in self._parent.columns

    def keys(self) -> list[str]:
        return self._parent.columns


class TableHDURef:
    def __init__(
        self,
        *,
        header: Optional[Header] = None,
        source_path: Optional[str] = None,
        source_hdu: Optional[int] = None,
        columns: Optional[List[str]] = None,
        row_slice: Optional[slice | tuple[int, int]] = None,
    ):
        self.header = header or Header()
        self._source_path = source_path
        self._source_hdu = source_hdu
        self._columns: Optional[tuple[str, ...]] = tuple(columns) if columns else None
        self._row_slice = row_slice
        self._all_columns_cache: tuple[str, ...] | None = None

    def _require_source(self) -> tuple[str, int]:
        if not self._source_path or self._source_hdu is None:
            raise RuntimeError("This TableHDURef is not associated with a FITS file")
        return self._source_path, int(self._source_hdu)

    @property
    def num_rows(self) -> int:
        try:
            total = int(self.header.get("NAXIS2", 0))
        except Exception:
            total = 0
        if total <= 0:
            return 0
        if self._row_slice is None:
            return total
        if isinstance(self._row_slice, tuple):
            start, stop = self._row_slice
        else:
            start = 0 if self._row_slice.start is None else int(self._row_slice.start)
            stop = self._row_slice.stop
        start = int(start)
        if start < 0:
            start = 0
        if stop is None:
            return max(0, total - start)
        stop = int(stop)
        stop = min(stop, total)
        return max(0, stop - start)

    def __len__(self) -> int:
        return self.num_rows

    @property
    def columns(self) -> list[str]:
        if self._columns is not None:
            return list(self._columns)

        if self._all_columns_cache is not None:
            return list(self._all_columns_cache)
        try:
            n = int(self.header.get("TFIELDS", 0))
        except Exception:
            n = 0
        out: list[str] = []
        for i in range(1, n + 1):
            name = self.header.get(f"TTYPE{i}")
            if isinstance(name, str) and name:
                out.append(name)
            else:
                out.append(f"COL{i}")
        self._all_columns_cache = tuple(out)
        return out

    @property
    def string_columns(self) -> List[str]:
        from ..fits_schema import string_column_names

        selected = set(self._columns) if self._columns is not None else None
        return string_column_names(self.header, selected=selected)

    @property
    def schema(self) -> Dict[str, Any]:
        from ..fits_schema import build_table_schema_dict

        return build_table_schema_dict(
            self.header,
            selected_columns=(
                list(self._columns) if self._columns is not None else None
            ),
        )

    def select(self, cols: List[str]) -> "TableHDURef":
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise TypeError("cols must be a list[str]")
        return TableHDURef(
            header=self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
            columns=cols,
            row_slice=self._row_slice,
        )

    def head(self, n: int) -> "TableHDURef":
        if n < 0:
            raise ValueError("n must be >= 0")
        return TableHDURef(
            header=self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
            columns=list(self._columns) if self._columns is not None else None,
            row_slice=slice(0, n),
        )

    def filter(self, condition: str) -> "TableHDU":
        return self.materialize().filter(condition)

    def _normalize_row_slice(
        self, row_slice: Optional[slice | tuple[int, int]]
    ) -> tuple[int, int]:
        if row_slice is None:
            return 1, -1
        if isinstance(row_slice, tuple):
            if len(row_slice) != 2:
                raise ValueError("row_slice tuple must be (start, stop)")
            start, stop = row_slice
        else:
            start = 0 if row_slice.start is None else row_slice.start
            stop = row_slice.stop
            if row_slice.step not in (None, 1):
                raise ValueError("row_slice step is not supported")
        start = int(start)
        if start < 0:
            raise ValueError("row_slice start must be >= 0")
        if stop is None:
            return start + 1, -1
        stop = int(stop)
        if stop < start:
            return start + 1, 0
        return start + 1, stop - start

    def _is_ascii_table(self) -> bool:
        try:
            return str(self.header.get("XTENSION", "")).strip().upper() == "TABLE"
        except Exception:
            return False

    def read(
        self,
        *,
        columns: Optional[List[str]] = None,
        row_slice: Optional[slice | tuple[int, int]] = None,
        mmap: bool = True,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        import torchfits

        path, hdu = self._require_source()
        if columns is None:
            columns = list(self._columns) if self._columns is not None else None
        if row_slice is None:
            row_slice = self._row_slice
        start_row, num_rows = self._normalize_row_slice(row_slice)
        effective_mmap = bool(mmap)
        if effective_mmap and self._is_ascii_table():
            effective_mmap = False
        return torchfits.read(  # type: ignore[no-any-return]
            path,
            hdu=hdu,
            columns=columns,
            start_row=start_row,
            num_rows=num_rows,
            mmap=effective_mmap,
            device=device,
            cache_capacity=0,
        )

    def materialize(self, *, mmap: bool = True, device: str = "cpu") -> "TableHDU":
        data = self.read(mmap=mmap, device=device)
        return TableHDU(
            data,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def iter_rows(
        self, batch_size: int = 65536, *, mmap: bool = True
    ) -> Iterator[dict[str, Any]]:
        import torchfits

        path, hdu = self._require_source()
        start_row, num_rows = self._normalize_row_slice(self._row_slice)
        effective_mmap = bool(mmap)
        if effective_mmap and self._is_ascii_table():
            effective_mmap = False

        # Lazy import: table_streaming → hdu is cyclic at module import time.
        from .._io_engine.table_streaming import stream_table as _engine_stream_table

        for chunk in _engine_stream_table(
            torchfits.read_header,
            path,
            hdu=hdu,
            columns=list(self._columns) if self._columns is not None else None,
            start_row=start_row,
            num_rows=num_rows,
            chunk_rows=batch_size,
            mmap=effective_mmap,
        ):
            yield chunk

    def __getitem__(self, col_name: str) -> Any:
        data = self.read(columns=[col_name])
        if col_name not in data:
            raise KeyError(f"Column '{col_name}' not found")
        return data[col_name]

    def get_string_column(
        self, name: str, encoding: str = "ascii", strip: bool = True
    ) -> List[str]:
        value = self[name]
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"Column '{name}' is not a tensor string column")
        if value.dtype != torch.uint8 or value.dim() != 2:
            raise TypeError(
                f"Column '{name}' is not a uint8 (rows,width) encoded string column"
            )
        from .._string_decode import decode_byte_tensor

        return decode_byte_tensor(value, encoding=encoding, strip=strip)

    def get_vla_column(self, name: str) -> List[Tensor]:
        value = self[name]
        if isinstance(value, list):
            return value
        raise KeyError(f"Column '{name}' is not a VLA list")

    def get_vla_lengths(self, name: str) -> List[int]:
        values = self.get_vla_column(name)
        lengths: List[int] = []
        for item in values:
            if isinstance(item, torch.Tensor):
                lengths.append(int(item.numel()))
            elif hasattr(item, "__len__"):
                lengths.append(len(item))
            else:
                lengths.append(1)
        return lengths

    @property
    def vla_lengths(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for col in self.schema.get("vla_columns", []):
            try:
                out[col] = self.get_vla_lengths(col)
            except Exception:
                continue
        return out

    @property
    def data(self) -> _TableHDURefDataWrapper:
        return _TableHDURefDataWrapper(self)

    def to_arrow(self, **kwargs: Any) -> Any:
        import torchfits

        path, hdu = self._require_source()
        return torchfits.table.read(
            path,
            hdu=hdu,
            columns=list(self._columns) if self._columns is not None else None,
            row_slice=self._row_slice,
            **kwargs,
        )

    def scan_arrow(self, **kwargs: Any) -> Any:
        import torchfits

        path, hdu = self._require_source()
        return torchfits.table.scan(
            path,
            hdu=hdu,
            columns=list(self._columns) if self._columns is not None else None,
            row_slice=self._row_slice,
            **kwargs,
        )

    def reader_arrow(self, **kwargs: Any) -> Any:
        import torchfits

        path, hdu = self._require_source()
        return torchfits.table.reader(
            path,
            hdu=hdu,
            columns=list(self._columns) if self._columns is not None else None,
            row_slice=self._row_slice,
            **kwargs,
        )

    def _refresh_file_view(self) -> "TableHDURef":
        import torchfits._C as cpp
        from .._io_engine.caches import invalidate_path_caches
        from .._io_engine.paths import guard_fits_path

        path, hdu = self._require_source()
        guard_fits_path(path)
        # Fresh open after mutation/os.replace: drop Python LRUs and SharedReadMeta
        # (C++ handle-pool invalidate is a no-op after Option A).
        invalidate_path_caches(path)
        clear_meta = getattr(cpp, "clear_shared_read_meta_cache", None)
        if clear_meta is not None:
            clear_meta()
        handle = cpp.open_fits_file(path, "r")
        try:
            header = Header(cpp.read_header(handle, hdu))
        finally:
            try:
                handle.close()
            except Exception:
                pass
        return TableHDURef(header=header, source_path=path, source_hdu=hdu)

    def append_rows_file(self, rows: Dict[str, Any]) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.append_rows(path, rows, hdu=hdu)
        return self._refresh_file_view()

    def insert_column_file(
        self,
        name: str,
        values: Any,
        *,
        index: Optional[int] = None,
        format: Optional[str] = None,
        unit: Optional[str] = None,
        dim: Optional[str] = None,
        tnull: Optional[Any] = None,
        tscal: Optional[float] = None,
        tzero: Optional[float] = None,
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        old_columns = self.columns
        insert_at = index if index is not None else len(old_columns)

        torchfits.table.insert_column(
            path,
            name,
            values,
            hdu=hdu,
            index=insert_at,
            format=format,
            unit=unit,
            dim=dim,
            tnull=tnull,
            tscal=tscal,
            tzero=tzero,
        )
        # Build the updated header from memory instead of reading back from
        # the file (which may return a stale cached handle when the file was
        # atomically replaced underneath an open torchfits.open context).
        new_header = Header(self.header)
        new_columns = list(old_columns)
        new_columns.insert(insert_at, name)
        # Shift existing column metadata to make room for the inserted column
        _COL_PREFIXES = ("TTYPE", "TFORM", "TUNIT", "TDIM", "TNULL", "TSCAL", "TZERO")
        for i in range(len(old_columns), insert_at, -1):
            for prefix in _COL_PREFIXES:
                old_key = f"{prefix}{i}"
                new_key = f"{prefix}{i + 1}"
                if old_key in new_header:
                    new_header[new_key] = new_header.pop(old_key)
        # Set metadata for the newly inserted column
        one_based = insert_at + 1
        new_header[f"TTYPE{one_based}"] = name
        if format is not None:
            new_header[f"TFORM{one_based}"] = format
        if unit is not None:
            new_header[f"TUNIT{one_based}"] = unit
        if dim is not None:
            new_header[f"TDIM{one_based}"] = dim
        if tnull is not None:
            new_header[f"TNULL{one_based}"] = tnull
        if tscal is not None:
            new_header[f"TSCAL{one_based}"] = tscal
        if tzero is not None:
            new_header[f"TZERO{one_based}"] = tzero
        new_header["TFIELDS"] = len(new_columns)
        return TableHDURef(
            header=new_header, source_path=path, source_hdu=hdu, columns=new_columns
        )

    def replace_column_file(
        self,
        name: str,
        values: Any,
        *,
        format: Optional[str] = None,
        unit: Optional[str] = None,
        dim: Optional[str] = None,
        tnull: Optional[Any] = None,
        tscal: Optional[float] = None,
        tzero: Optional[float] = None,
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.replace_column(
            path,
            name,
            values,
            hdu=hdu,
            format=format,
            unit=unit,
            dim=dim,
            tnull=tnull,
            tscal=tscal,
            tzero=tzero,
        )
        # Build the updated header from memory (see insert_column_file for
        # rationale — C++ file cache may return a stale handle after os.replace).
        new_header = Header(self.header)
        new_columns = list(self.columns)
        one_based = new_columns.index(name) + 1
        if unit is not None:
            new_header[f"TUNIT{one_based}"] = unit
        if dim is not None:
            new_header[f"TDIM{one_based}"] = dim
        if tnull is not None:
            new_header[f"TNULL{one_based}"] = tnull
        if tscal is not None:
            new_header[f"TSCAL{one_based}"] = tscal
        if tzero is not None:
            new_header[f"TZERO{one_based}"] = tzero
        return TableHDURef(
            header=new_header, source_path=path, source_hdu=hdu, columns=new_columns
        )

    def insert_rows_file(self, rows: Dict[str, Any], *, row: int) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.insert_rows(path, rows, row=row, hdu=hdu)
        return self._refresh_file_view()

    def delete_rows_file(
        self, row_slice: Union[int, slice, tuple[int, int]]
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.delete_rows(path, row_slice, hdu=hdu)
        return self._refresh_file_view()

    def update_rows_file(
        self,
        rows: Dict[str, Any],
        row_slice: Union[slice, tuple[int, int]],
        *,
        mmap: Union[bool, str] = "auto",
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.update_rows(path, rows, row_slice=row_slice, hdu=hdu, mmap=mmap)
        return self._refresh_file_view()

    def rename_columns_file(self, mapping: Dict[str, str]) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.rename_columns(path, mapping, hdu=hdu)
        return self._refresh_file_view()

    def drop_columns_file(self, columns: List[str]) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.drop_columns(path, columns, hdu=hdu)
        return self._refresh_file_view()

    def __repr__(self) -> str:
        name = self.header.get("EXTNAME", "TABLE")
        proj = f", cols={len(self.columns)}" if self._columns is not None else ""
        return f"TableHDURef(name='{name}', rows={self.num_rows}{proj})"

    def _repr_html_(self) -> str:
        name = html.escape(str(self.header.get("EXTNAME", "TABLE")))
        container_style = (
            "max-height: 400px; overflow: auto; "
            "border: 1px solid rgba(128, 128, 128, 0.3); margin-bottom: 1em;"
        )
        table_style = "border-collapse: collapse; width: 100%; margin: 0;"
        th_col_style = (
            "text-align: left; padding: 8px; position: sticky; top: 0; "
            "background-color: var(--theme-ui-colors-background, white); "
            "border-bottom: 2px solid rgba(128, 128, 128, 0.3); z-index: 1;"
        )
        th_row_style = (
            "font-weight: normal; text-align: left; padding: 8px; "
            "border-bottom: 1px solid rgba(128, 128, 128, 0.2);"
        )
        td_style = (
            "text-align: left; padding: 8px; "
            "border-bottom: 1px solid rgba(128, 128, 128, 0.2);"
        )
        return (
            f'<div tabindex="0" aria-label="TableHDURef" style=\'{container_style}\'>'
            f"<table style='{table_style}'>"
            f"<thead><tr>"
            f"<th scope=\"col\" style='{th_col_style}'>Name</th>"
            f"<th scope=\"col\" style='{th_col_style}'>Rows</th>"
            f"<th scope=\"col\" style='{th_col_style}'>Columns</th>"
            f"</tr></thead>"
            f"<tbody><tr>"
            f"<th scope=\"row\" style='{th_row_style}'>{name}</th>"
            f"<td style='{td_style}'>{self.num_rows}</td>"
            f"<td style='{td_style}'>{len(self.columns)}</td>"
            f"</tr></tbody></table></div>"
        )
