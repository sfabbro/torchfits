"""HDU container: ordered list of HDUs with file-level operations."""

from __future__ import annotations

import operator
import os
import threading
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

from torchfits._io_engine.paths import (
    cfitsio_base_path,
    guard_fits_path,
    is_cfitsio_network_url,
)

from ._repr import render_html_table
from .card import _reassemble_longstr_cards
from .header import Header

if TYPE_CHECKING:
    from .table_hdu import TableHDU
    from .table_hdu_ref import TableHDURef
    from .tensor_hdu import TensorHDU

# The tensor-backed HDU classes are resolved on first use: ``TableHDU`` imports
# torch, and ``torchfits.open`` on a header must not.  ``TensorHDU``/``TableHDURef``
# are lazy by design and cheap to import, but they live behind the same door so
# the dispatch here stays uniform.
_HDU_MODULES: dict[str, tuple[str, str]] = {
    "TableHDU": (".table_hdu", "TableHDU"),
    "TableHDURef": (".table_hdu_ref", "TableHDURef"),
    "TensorHDU": (".tensor_hdu", "TensorHDU"),
}
_HDU_CLASSES: dict[str, Any] = {}


def _hdu_class(name: str) -> Any:
    """Return an HDU class, importing its module only when first needed."""
    cached = _HDU_CLASSES.get(name)
    if cached is None:
        module_name, attr = _HDU_MODULES[name]
        cached = getattr(import_module(module_name, __package__), attr)
        _HDU_CLASSES[name] = cached
    return cached


def _table_hdu_types() -> tuple[Any, Any]:
    """``(TableHDU, TableHDURef)`` for isinstance dispatch."""
    return _hdu_class("TableHDU"), _hdu_class("TableHDURef")


@dataclass(frozen=True)
class _HDUInfo:
    index: int
    type: str
    header: Any


class _LazyFileHandle:
    """Open the tensor-backed reader on first data access.

    ``torchfits.open`` reads headers through ``torchfits._core``, which does
    not link libtorch, so a header-only open must not import ``torchfits._C``
    (that import preloads PyTorch).  The first HDU that touches its data opens
    the extension once and every file-backed HDU in the list shares it; closing
    the HDUList closes it if it was ever opened.
    """

    def __init__(self, path: str):
        self._path = path
        self._handle: Any = None
        self._lock = threading.Lock()

    def get(self) -> Any:
        with self._lock:
            if self._handle is None:
                import torchfits._C as cpp

                self._handle = cpp.open_fits_file(self._path, "r")
            return self._handle

    def close(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            handle.close()


class HDUList:
    def __init__(
        self, hdus: Optional[List[Union[TensorHDU, TableHDU, TableHDURef]]] = None
    ):
        self._hdus: List[Union[TensorHDU, TableHDU, TableHDURef]] = hdus or []
        self._file_handle: Any = None
        self._lazy_handle: Any = None
        self._source_path: Optional[str] = None
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
        hdul._source_path = path
        lazy = _LazyFileHandle(path)
        hdul._lazy_handle = lazy

        try:
            # Headers come from the torch-free core. Importing ``torchfits._C``
            # here would preload PyTorch (~1 s) just to list HDUs; the handle
            # is opened lazily by the first HDU that touches its data.
            from torchfits import _core as core
            from torchfits.header_parser import fast_parse_header_cards

            num_hdus = int(core.read_num_hdus(path))
            hdu_infos = [
                _HDUInfo(
                    index=i,
                    type=str(core.read_hdu_type(path, i)),
                    # Parse the header string so keyword values keep their FITS
                    # types (NAXIS is an int, not "2"), matching read_header.
                    header=fast_parse_header_cards(core.read_header_string(path, i)),
                )
                for i in range(num_hdus)
            ]

            from .._io_engine.caches import _register_open_hdulist

            try:
                _register_open_hdulist(path, lazy, hdul)
            except Exception:
                pass

            for info in hdu_infos:
                # LONGSTRN '&'+CONTINUE chains are rejoined at construction
                # (r4b-13 root fix): the core keeps the chain markers and
                # detached CONTINUE cards verbatim, so without reassembly every
                # >68-char string value read back truncated and every rewrite
                # through this header corrupted it.
                header = Header(_reassemble_longstr_cards(info.header))

                hdu_type = info.type
                i = info.index

                if hdu_type == "IMAGE":
                    hdu: Any = _hdu_class("TensorHDU")(
                        header=header,
                        hdu_index=i,
                        source_path=path,
                        lazy_handle=lazy,
                    )
                elif hdu_type in ["ASCII_TABLE", "BINARY_TABLE"]:
                    hdu = _hdu_class("TableHDURef")(
                        header=header, source_path=path, source_hdu=i
                    )
                else:
                    hdu = _hdu_class("TensorHDU")(header=header)

                hdul._hdus.append(hdu)

            return hdul

        except Exception as e:
            try:
                lazy.close()
            except Exception:
                pass
            raise RuntimeError(f"Failed to open FITS file '{path}': {str(e)}") from e

    def __len__(self) -> int:
        return len(self._hdus)

    def __getitem__(
        self, key: Union[int, str]
    ) -> Union[TensorHDU, TableHDU, TableHDURef]:
        if isinstance(key, str):
            # Scanned live (a few header.get calls per lookup): a cached name
            # index went stale as soon as a caller renamed EXTNAME on a live
            # header, and HDU counts are tiny.
            for i, hdu in enumerate(self._hdus):
                name = hdu.header.get("EXTNAME")
                if name is not None and name == key:
                    return self._hdus[i]
            raise KeyError(f"HDU '{key}' not found")

        try:
            return self._hdus[operator.index(key)]
        except TypeError:
            raise TypeError(
                f"HDUList indices must be int or EXTNAME str; got {type(key).__name__}"
            ) from None

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
        target = self._file_handle or self._lazy_handle
        if target:
            # Unregister before closing so the registry doesn't hold a stale
            # entry. The removal itself runs under the registry lock (r4c-01):
            # an unlocked read-modify-write here lost concurrent registrations.
            try:
                from .._io_engine.caches import _unregister_open_hdulist

                _unregister_open_hdulist(self._registry_key, target)
                self._registry_key = None
            except Exception:
                pass
            try:
                target.close()
            finally:
                self._file_handle = None
                self._lazy_handle = None
        for hdu in self._hdus:
            if isinstance(hdu, _hdu_class("TensorHDU")):
                hdu.mark_closed()

    def write(self, path: str, overwrite: bool = False) -> None:
        from .._io_engine.write_api import _write_hdus_uncompressed  # type: ignore[attr-defined]

        _write_hdus_uncompressed(path, list(self._hdus), overwrite)

    def append(self, hdu: Union[TensorHDU, TableHDU]) -> None:
        self._hdus.append(hdu)

    def validate(self) -> bool:
        """Whether every HDU in this list can still be read.

        A file-backed HDU is checked by actually reaching its data: one whose
        handle has been detached (``close()`` / ``TensorHDU.mark_closed()``),
        or whose source file has gone away, is not valid. An HDU holding its
        data in memory, or one with nothing to read, is checked on its header
        alone.
        """
        tensor_hdu_type = _hdu_class("TensorHDU")
        table_hdu_type, table_ref_type = _table_hdu_types()
        try:
            for hdu in self._hdus:
                if not hdu.header:
                    return False

                if isinstance(hdu, tensor_hdu_type):
                    if hdu._data is not None:
                        _ = hdu._data.shape
                        _ = hdu._data.dtype
                    elif hdu._file_handle is not None or hdu._lazy_handle is not None:
                        # A lazy handle opens here; a detached one raises.
                        _ = hdu.data.shape
                        _ = hdu.data.dtype
                    elif hdu._source_path:
                        # File-backed with no handle left: mark_closed() ran.
                        return False
                elif isinstance(hdu, (table_hdu_type, table_ref_type)):
                    _ = hdu.columns
                    _ = hdu.num_rows
                    if isinstance(hdu, table_ref_type) and hdu._source_path:
                        # One row: the cheapest read that still opens the file
                        # and parses the HDU. A file-backed table HDU is always
                        # a TableHDURef, which the TableHDU-only isinstance
                        # check skipped, so nothing about a table was ever
                        # validated.
                        hdu.read(row_slice=slice(0, 1))

            return True
        except Exception:
            return False

    @staticmethod
    def _is_table_hdu(hdu: Any) -> bool:
        """Table test that does not import the tensor-backed HDU classes.

        ``isinstance(hdu, TableHDU)`` loads ``table_hdu``, which imports
        torch; ``hdul.info()`` is a header-only operation and must stay
        torch-free. The class name covers in-memory tables, the header covers
        file-backed ones.
        """
        name = type(hdu).__name__
        if name in {"TableHDU", "TableHDURef"}:
            return True
        if name == "TensorHDU":
            # Compressed images live in a BINTABLE but are images to the user.
            return False
        return str(hdu.header.get("XTENSION", "")).strip().upper() in {
            "TABLE",
            "BINTABLE",
        }

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
        elif self._source_path:
            filename = self._source_path

        lines.append(f"Filename: {filename}")
        lines.append("No.    Name         Type       Cards   Dimensions   Format")

        for idx, hdu in enumerate(self._hdus):
            name = str(hdu.header.get("EXTNAME", "PRIMARY"))

            if self._is_table_hdu(hdu):
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

            if self._is_table_hdu(hdu):
                dims = (
                    f"({getattr(hdu, 'num_rows')}R x {len(getattr(hdu, 'columns'))}C)"
                )
                fmt = "Table"
            elif hasattr(hdu, "_get_shape_str"):
                dims = getattr(hdu, "_get_shape_str")()
                fmt = getattr(hdu, "_get_dtype_str")()
            else:
                dims = ""
                fmt = ""

            lines.append(
                f"{idx:<6d} {name:<12s} {hdu_type:<10s} {cards:<7d} {dims:<12s} {fmt}"
            )

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = []
        for idx, hdu in enumerate(self._hdus):
            name = str(hdu.header.get("EXTNAME", "PRIMARY"))
            if self._is_table_hdu(hdu):
                hdu_type = "TableHDU"
                dims = (
                    f"({getattr(hdu, 'num_rows')}R x {len(getattr(hdu, 'columns'))}C)"
                )
                fmt = "Table"
            elif hasattr(hdu, "_get_shape_str"):
                hdu_type = (
                    "PrimaryHDU" if idx == 0 and name == "PRIMARY" else "ImageHDU"
                )
                dims = getattr(hdu, "_get_shape_str")()
                fmt = getattr(hdu, "_get_dtype_str")()
            else:
                hdu_type, dims, fmt = "Unknown", "", ""

            cards = (
                len(hdu.header._cards)
                if hasattr(hdu.header, "_cards")
                else len(hdu.header)
            )
            rows.append((idx, name, hdu_type, cards, dims, fmt))

        return render_html_table(
            "FITS HDU List",
            ["No.", "Name", "Type", "Cards", "Dimensions", "Format"],
            rows,
            aligns=["left", "left", "left", "right", "left", "left"],
        )

    def __repr__(self) -> str:
        return self._get_summary()
