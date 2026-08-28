"""HDU container: ordered list of HDUs with file-level operations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Type, Union

from torchfits._io_engine.paths import (
    cfitsio_base_path,
    guard_fits_path,
    is_cfitsio_network_url,
)

from .header import Header
from .tensor_hdu import TensorHDU
from .table_hdu import TableHDU
from .table_hdu_ref import TableHDURef


@dataclass(frozen=True)
class _HDUInfo:
    index: int
    type: str
    header: Any


class HDUList:
    def __init__(
        self, hdus: Optional[List[Union[TensorHDU, TableHDU, TableHDURef]]] = None
    ):
        self._hdus: List[Union[TensorHDU, TableHDU, TableHDURef]] = hdus or []
        self._file_handle = None
        self._extname_idx: Optional[dict[str, int]] = None
        self._registry_key: Optional[str] = None

    @classmethod
    def fromfile(cls, path: str, mode: str = "r") -> "HDUList":
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string")

        if mode != "r":
            raise ValueError(
                f"HDUList supports mode='r' only; got {mode!r}. In-place "
                "update is not supported — write modified HDUs to a new "
                "path with hdul.write(path, overwrite=True) or use the "
                "torchfits.insert_hdu / replace_hdu / delete_hdu APIs."
            )

        guard_fits_path(path)
        # Network URLs are opened by CFITSIO; only local paths need exists().
        if (
            mode == "r"
            and not is_cfitsio_network_url(path)
            and not os.path.exists(cfitsio_base_path(path))
        ):
            raise FileNotFoundError(f"FITS file not found: {path}")

        hdul = cls()

        try:
            import torchfits._C as cpp

            try:
                handle, hdu_infos = cpp.open_and_read_headers(
                    path, 0 if mode == "r" else 1
                )
            except AttributeError:
                handle = cpp.open_fits_file(path, mode)
                hdu_infos = []
                num_hdus = cpp.get_num_hdus(handle)
                for i in range(num_hdus):
                    header_dict = cpp.read_header(handle, i)
                    hdu_type = cpp.get_hdu_type(handle, i)

                    hdu_infos.append(
                        _HDUInfo(index=i, type=hdu_type, header=header_dict)
                    )

            hdul._file_handle = handle

            from .._io_engine.caches import _register_open_hdulist

            try:
                _register_open_hdulist(path, handle, hdul)
            except Exception:
                pass

            for info in hdu_infos:
                try:
                    header_cards = cpp.read_header(handle, info.index)
                except Exception:
                    header_cards = info.header
                header = Header(header_cards)

                hdu_type = info.type
                i = info.index

                if hdu_type == "IMAGE":
                    hdu: Any = TensorHDU(
                        header=header, file_handle=handle, hdu_index=i, source_path=path
                    )
                elif hdu_type in ["ASCII_TABLE", "BINARY_TABLE"]:
                    hdu = TableHDURef(header=header, source_path=path, source_hdu=i)
                else:
                    hdu = TensorHDU(header=header)

                hdul._hdus.append(hdu)

            return hdul

        except Exception as e:
            if hdul._file_handle:
                try:
                    hdul._file_handle.close()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to open FITS file '{path}': {str(e)}") from e

    def __len__(self) -> int:
        return len(self._hdus)

    def __getitem__(
        self, key: Union[int, str]
    ) -> Union[TensorHDU, TableHDU, TableHDURef]:
        if isinstance(key, int):
            return self._hdus[key]

        if self._extname_idx is None:
            self._extname_idx = {}
            for i, hdu in enumerate(self._hdus):
                name = hdu.header.get("EXTNAME")
                if name is not None and name not in self._extname_idx:
                    self._extname_idx[name] = i

        idx = self._extname_idx.get(key)
        if idx is not None:
            return self._hdus[idx]

        raise KeyError(f"HDU '{key}' not found")

    def __enter__(self) -> HDUList:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._file_handle:
            # Unregister before closing so the registry doesn't hold a stale entry.
            try:
                from .._io_engine.caches import _open_hdulist_registry

                key = self._registry_key
                if key is not None:
                    entries = _open_hdulist_registry.get(key, [])
                    remaining = [
                        entry for entry in entries if entry[0] is not self._file_handle
                    ]
                    if not remaining:
                        _open_hdulist_registry.pop(key, None)
                    elif len(remaining) != len(entries):
                        _open_hdulist_registry[key] = remaining
                    self._registry_key = None
                else:
                    for real_path, entries in list(_open_hdulist_registry.items()):
                        remaining = [
                            entry
                            for entry in entries
                            if entry[0] is not self._file_handle
                        ]
                        if len(remaining) == len(entries):
                            continue
                        if remaining:
                            _open_hdulist_registry[real_path] = remaining
                        else:
                            _open_hdulist_registry.pop(real_path, None)
                        break
            except Exception:
                pass
            self._file_handle.close()
            self._file_handle = None
        for hdu in self._hdus:
            if isinstance(hdu, TensorHDU):
                hdu.mark_closed()

    def write(self, path: str, overwrite: bool = False) -> None:
        from .._io_engine.write_api import _write_hdus_uncompressed  # type: ignore[attr-defined]

        _write_hdus_uncompressed(path, list(self._hdus), overwrite)

    def append(self, hdu: Union[TensorHDU, TableHDU]) -> None:
        self._hdus.append(hdu)
        self._extname_idx = None

    def validate(self) -> bool:
        try:
            for i, hdu in enumerate(self._hdus):
                if not hdu.header:
                    return False

                if isinstance(hdu, TensorHDU):
                    if hdu._file_handle:
                        _ = hdu.data.shape
                        _ = hdu.data.dtype
                elif isinstance(hdu, TableHDU):
                    _ = hdu.columns
                    _ = hdu.num_rows

            return True
        except Exception:
            return False

    def info(self, output: Any = None) -> None:
        summary = self._get_summary()
        if output is None:
            print(summary)
        else:
            output.write(summary + "\n")

    def _get_summary(self) -> str:
        lines = []
        filename = "(No file associated)"
        if self._file_handle and hasattr(self._file_handle, "name"):
            filename = self._file_handle.name

        lines.append(f"Filename: {filename}")
        lines.append("No.    Name         Type       Cards   Dimensions   Format")

        for idx, hdu in enumerate(self._hdus):
            name = str(hdu.header.get("EXTNAME", "PRIMARY"))

            if isinstance(hdu, (TableHDU, TableHDURef)):
                hdu_type = "TableHDU"
            else:
                if idx == 0 and name == "PRIMARY":
                    hdu_type = "PrimaryHDU"
                else:
                    hdu_type = "ImageHDU"

            cards = (
                len(hdu.header._cards)
                if hasattr(hdu.header, "_cards")
                else len(hdu.header)
            )

            if isinstance(hdu, (TableHDU, TableHDURef)):
                dims = f"({hdu.num_rows}R x {len(hdu.columns)}C)"
                fmt = "Table"
            elif isinstance(hdu, TensorHDU):
                dims = hdu._get_shape_str()
                fmt = hdu._get_dtype_str()
            else:
                dims = ""
                fmt = ""

            lines.append(
                f"{idx:<6d} {name:<12s} {hdu_type:<10s} {cards:<7d} {dims:<12s} {fmt}"
            )

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        html = [
            '<div tabindex="0" aria-label="FITS HDU List" style=\'max-height: 400px; overflow: auto; border: 1px solid rgba(128, 128, 128, 0.3); margin-bottom: 1em;\'>',
            "<table style='border-collapse: collapse; width: 100%; margin: 0;'>",
            "<thead><tr>",
        ]
        headers = ["No.", "Name", "Type", "Cards", "Dimensions", "Format"]
        styles = (
            ["text-align: left;"] * 3
            + ["text-align: right;"]
            + ["text-align: left;"] * 2
        )
        for h, s in zip(headers, styles):
            html.append(
                f'<th scope="col" style=\'{s} padding: 8px; position: sticky; top: 0; '
                f"background-color: var(--theme-ui-colors-background, white); "
                f"border-bottom: 2px solid rgba(128, 128, 128, 0.3); z-index: 1;'>{h}</th>"
            )
        html.append("</tr></thead><tbody>")

        for idx, hdu in enumerate(self._hdus):
            name = str(hdu.header.get("EXTNAME", "PRIMARY"))
            if isinstance(hdu, (TableHDU, TableHDURef)):
                hdu_type = "TableHDU"
                dims = f"({hdu.num_rows}R x {len(hdu.columns)}C)"
                fmt = "Table"
            elif isinstance(hdu, TensorHDU):
                hdu_type = (
                    "PrimaryHDU" if idx == 0 and name == "PRIMARY" else "ImageHDU"
                )
                dims, fmt = hdu._get_shape_str(), hdu._get_dtype_str()
            else:
                hdu_type, dims, fmt = "Unknown", "", ""

            cards = (
                len(hdu.header._cards)
                if hasattr(hdu.header, "_cards")
                else len(hdu.header)
            )

            row = [idx, name, hdu_type, cards, dims, fmt]
            html.append("<tr>")
            import html as pyhtml

            for col_idx, (val, s) in enumerate(zip(row, styles)):
                escaped_val = pyhtml.escape(str(val))
                if col_idx == 0:
                    html.append(
                        f"<th scope=\"row\" style='font-weight: normal; {s} padding: 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>{escaped_val}</th>"
                    )
                else:
                    html.append(
                        f"<td style='{s} padding: 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>{escaped_val}</td>"
                    )
            html.append("</tr>")

        html.append("</tbody></table></div>")
        return "".join(html)

    def __repr__(self) -> str:
        return self._get_summary()
