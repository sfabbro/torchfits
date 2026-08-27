"""Deprecated alias for :mod:`torchfits._cpp` (the raw native surface).

The raw nanobind bindings are an implementation detail. Import from
``torchfits`` / ``torchfits.io`` / ``torchfits.table`` instead; every
attribute access through this module emits a :class:`DeprecationWarning`
and will be removed in 2.0.
"""

from __future__ import annotations

from typing import Any

from torchfits import _cpp as _impl

_DEPRECATION = (
    "torchfits.cpp is deprecated and will be removed in 2.0; "
    "the raw binding surface is private (torchfits._cpp). Use the public "
    "Python API instead."
)


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    import warnings

    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_impl)))
