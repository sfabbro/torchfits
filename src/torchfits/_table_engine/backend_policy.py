"""Backend selection policy for FITS table I/O."""

from __future__ import annotations

_TABLE_BACKEND_ORDER = ("auto", "torch", "cpp_numpy")
TABLE_BACKENDS = frozenset(_TABLE_BACKEND_ORDER)


def validate_table_backend(backend: str) -> str:
    """Return a validated table backend name or raise with the public error."""
    if backend not in TABLE_BACKENDS:
        allowed = ", ".join(_TABLE_BACKEND_ORDER)
        raise ValueError(f"backend must be one of: {allowed}")
    return backend
