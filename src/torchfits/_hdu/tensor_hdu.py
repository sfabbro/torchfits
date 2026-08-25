"""Image/cube HDU with lazy data loading and C++ backend integration."""

from __future__ import annotations

import html
import threading
from typing import Any, Iterator, Optional, Tuple, cast

from torch import Tensor
from .._io_engine.device import to_device
from .header import Header
from .dataview import DataView, _BITPIX_TO_DTYPE


class TensorHDU:
    def __init__(
        self,
        data: Optional[Tensor] = None,
        header: Optional[Header] = None,
        file_handle: Any = None,
        hdu_index: int = 0,
        source_path: Optional[str] = None,
    ):
        self._data = data
        self._header = header or Header()
        self._file_handle = file_handle
        self._hdu_index = hdu_index
        self._source_path = source_path
        self._data_view = (
            DataView(file_handle, hdu_index, header=self._header)
            if file_handle
            else None
        )
        self._closed = False
        self._io_lock = threading.RLock()

    @property
    def data(self) -> DataView:
        if self._data_view is None:
            raise ValueError("No file handle available")
        return self._data_view

    @property
    def header(self) -> Header:
        return self._header

    def mark_closed(self) -> None:
        """Detach file-backed state; safe to call from HDUList.close()."""
        with self._io_lock:
            self._closed = True
            self._file_handle = None
            self._data_view = None

    def to_tensor(self, device: str = "cpu") -> Tensor:
        if self._data is not None:
            return to_device(self._data, device)

        with self._io_lock:
            if self._closed or self._file_handle is None:
                raise RuntimeError(
                    "TensorHDU file handle is closed; cannot read image data"
                )
            import torchfits._C as cpp

            # Prefer a private per-call handle: the HDUList's shared FITSFile
            # keeps CFITSIO cursor state that is not safe under concurrent
            # access (H1). Fall back to the shared handle only when the HDU
            # was constructed without a source path.
            source = self._source_path
            if isinstance(source, str) and source:
                handle = cpp.open_fits_file(source, "r")
                try:
                    return to_device(cpp.read_full(handle, self._hdu_index), device)
                finally:
                    handle.close()

            handle = self._file_handle
            hdu_index = self._hdu_index
            return to_device(cast(Tensor, cpp.read_full(handle, hdu_index)), device)

    def chunks(
        self, chunk_size: Tuple[int, ...]
    ) -> Iterator[Tensor]:
        """Yield row-band slabs of the image lazily (bounded memory).

        ``chunk_size`` follows numpy/torch convention: element 0 is the slab
        height along the first (outermost) axis; remaining elements are
        accepted but always read in full. Each yielded tensor equals the
        corresponding slice of :meth:`to_tensor`.
        """
        if self._data is not None:
            step = max(1, int(chunk_size[0])) if chunk_size else 64
            data = self._data
            for start in range(0, data.shape[0], step):
                yield data[start : start + step]
            return

        with self._io_lock:
            if self._closed or self._file_handle is None:
                raise RuntimeError(
                    "TensorHDU file handle is closed; cannot iterate chunks"
                )
            import torchfits._C as cpp

            source = self._source_path
            if not isinstance(source, str) or not source:
                raise RuntimeError(
                    "TensorHDU.chunks() requires a file-backed HDU opened by "
                    "path (torchfits.open); in-memory handles are unsupported"
                )
            # Private reader per iteration protocol: never shares the
            # HDUList's underlying fitsfile* across threads (H1).
            reader = cpp.SubsetReader(source, int(self._hdu_index))
        try:
            height = int(reader.height)
            step = max(1, int(chunk_size[0])) if chunk_size else 64
            for y0 in range(0, height, step):
                with self._io_lock:
                    closed = self._closed
                if closed:
                    raise RuntimeError(
                        "TensorHDU was closed during chunk iteration"
                    )
                yield cast(Tensor, reader.read(0, y0, int(reader.width), min(y0 + step, height)))
        finally:
            reader.close()

    def _get_shape_str(self) -> str:
        if self._data is not None:
            return str(tuple(self._data.shape))
        elif self._file_handle:
            try:
                naxis = int(self.header.get("NAXIS", 0))
            except (TypeError, ValueError):
                return "unknown"
            if naxis <= 0:
                return "()"
            dims = [str(self.header.get(f"NAXIS{i + 1}", 0)) for i in range(naxis)]
            return f"({', '.join(reversed(dims))})"
        return "()"

    def _get_dtype_str(self) -> str:
        if self._data is not None:
            return str(self._data.dtype).replace("torch.", "")
        elif self._file_handle:
            try:
                bitpix = int(self.header.get("BITPIX", 0))
            except (TypeError, ValueError):
                return "unknown"
            dtype = _BITPIX_TO_DTYPE.get(bitpix)
            if dtype is not None:
                return str(dtype).replace("torch.", "")
            return str(bitpix)
        return "unknown"

    @property
    def shape_str(self) -> str:
        """Public accessor for the human-readable shape string."""
        return self._get_shape_str()

    @property
    def dtype_str(self) -> str:
        """Public accessor for the human-readable dtype string."""
        return self._get_dtype_str()

    def __repr__(self) -> str:
        name = self.header.get("EXTNAME", "PRIMARY")
        return f"TensorHDU(name='{name}', shape={self._get_shape_str()}, dtype={self._get_dtype_str()})"

    def _repr_html_(self) -> str:
        name = html.escape(str(self.header.get("EXTNAME", "PRIMARY")))
        shape = html.escape(self._get_shape_str())
        dtype = html.escape(self._get_dtype_str())
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
            f'<div tabindex="0" aria-label="TensorHDU" style=\'{container_style}\'>'
            f"<table style='{table_style}'>"
            f"<thead><tr>"
            f"<th scope=\"col\" style='{th_col_style}'>Name</th>"
            f"<th scope=\"col\" style='{th_col_style}'>Shape</th>"
            f"<th scope=\"col\" style='{th_col_style}'>Dtype</th>"
            f"</tr></thead>"
            f"<tbody><tr>"
            f"<th scope=\"row\" style='{th_row_style}'>{name}</th>"
            f"<td style='{td_style}'>{shape}</td>"
            f"<td style='{td_style}'>{dtype}</td>"
            f"</tr></tbody></table></div>"
        )
