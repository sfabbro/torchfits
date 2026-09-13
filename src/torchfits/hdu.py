"""
Core HDU classes for torchfits.

This module re-exports from ``_hdu/`` sub-modules:
- HDUList: Container for multiple HDUs
- TensorHDU: Image/cube data with lazy loading
- TableHDU: Tensor-backed tabular FITS data
- TableHDURef: Lazy file-backed table handle
- Header: FITS header management
- Card: FITS header card

``Header`` and ``Card`` are pure Python and import eagerly.  The tensor-backed
classes are module attributes resolved on first access, because importing any of
them reaches ``torch`` (~1 s) and ``torchfits.read_header`` must not pay for
that.  ``from torchfits.hdu import TensorHDU`` keeps working: module
``__getattr__`` backs both attribute access and ``from`` imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# -- card / header (torch-free, imported eagerly) -----------------------------------

from ._hdu.card import Card as Card

from ._hdu.header import Header as Header

if TYPE_CHECKING:
    from ._hdu.dataview import DataView as DataView
    from ._hdu.hdu_list import HDUList as HDUList
    from ._hdu.table_hdu import TableDataAccessor as TableDataAccessor
    from ._hdu.table_hdu import TableHDU as TableHDU
    from ._hdu.table_hdu_ref import TableHDURef as TableHDURef
    from ._hdu.tensor_hdu import TensorHDU as TensorHDU

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DataView": ("._hdu.dataview", "DataView"),
    "HDUList": ("._hdu.hdu_list", "HDUList"),
    "TableDataAccessor": ("._hdu.table_hdu", "TableDataAccessor"),
    "TableHDU": ("._hdu.table_hdu", "TableHDU"),
    "TableHDURef": ("._hdu.table_hdu_ref", "TableHDURef"),
    "TensorHDU": ("._hdu.tensor_hdu", "TensorHDU"),
}


def __getattr__(name: str) -> Any:
    """Resolve the tensor-backed HDU classes on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name, __package__), attr)
    globals()[name] = value
    return value


__all__ = [
    "Card",
    "DataView",
    "Header",
    "HDUList",
    "TableHDU",
    "TableDataAccessor",
    "TableHDURef",
    "TensorHDU",
]
