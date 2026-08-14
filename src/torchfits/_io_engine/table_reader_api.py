"""Persistent table reader (open once, many column/row reads)."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .device import to_device, validate_device


class TableReaderHandle:
    """Persistent table reader for repeated column/row reads on one HDU.

    Thin wrapper over nanobind ``cpp.TableReader``: opens the file once and
    keeps the CFITSIO handle alive for multiple :meth:`read_torch` calls,
    instead of reopening the file on every call (the "cold" path used by
    :func:`torchfits.table.read_torch`).

    There is no handle-based filtered/``where=`` read on ``cpp.TableReader``
    (only a path-based ``read_fits_table_filtered`` that reopens the file
    itself). Filtered reads should keep using
    ``torchfits.table.read_torch(path, hdu, where=...)``.
    """

    def __init__(self, path: str, hdu: int | str = 1):
        import torchfits._C as cpp
        from .paths import guard_fits_path

        if not isinstance(path, str):
            raise ValueError("path must be a string")
        guard_fits_path(path)
        if not isinstance(hdu, (int, str)):
            raise ValueError("hdu must be an integer or string")
        if isinstance(hdu, str):
            if hasattr(cpp, "resolve_hdu_name_cached"):
                hdu = int(cpp.resolve_hdu_name_cached(path, hdu))
            else:
                raise ValueError("named HDUs require resolve_hdu_name_cached support")
        if hdu < 0:
            raise ValueError("hdu must be a non-negative integer")
        self._path = path
        self._hdu = int(hdu)
        self._reader = cpp.TableReader(path, self._hdu)
        self._closed = False

    def __enter__(self) -> TableReaderHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._reader = None
        self._closed = True

    @property
    def hdu(self) -> int:
        return self._hdu

    def _ensure_open(self) -> Any:
        if self._closed or self._reader is None:
            raise RuntimeError("TableReaderHandle is closed")
        return self._reader

    def num_rows(self) -> int:
        return int(self._ensure_open().num_rows)

    def read_torch(
        self,
        columns: Optional[list[str]] = None,
        start_row: int = 1,
        num_rows: int = -1,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Read rows for ``columns`` (all columns if ``None``) as tensors."""
        dev_str = validate_device(device)
        reader = self._ensure_open()
        col_names = list(columns) if columns is not None else []
        data = dict(reader.read_rows(col_names, int(start_row), int(num_rows)))
        if dev_str == "cpu":
            return data
        return {
            key: to_device(value, device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }


def open_table_reader(path: str, hdu: int | str = 1) -> TableReaderHandle:
    """Open a reusable table reader for repeated column/row access on one HDU."""
    return TableReaderHandle(path, hdu=hdu)
