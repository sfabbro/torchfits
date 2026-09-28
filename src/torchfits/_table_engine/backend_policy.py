"""Backend selection policy for FITS table I/O."""

from __future__ import annotations

_TABLE_BACKEND_ORDER = ("auto", "torch", "cpp")
TABLE_BACKENDS = frozenset(_TABLE_BACKEND_ORDER)


def validate_table_backend(backend: str) -> str:
    """Return a validated table backend name or raise with the public error."""
    # The isinstance guard is load-bearing: ``in`` against a frozenset hashes
    # its argument, so an unhashable backend (a list, a dict) raised a raw
    # "TypeError: unhashable type" from the public table.read/scan entry
    # points instead of the ValueError this function promises. A hashable
    # wrong type (a tuple, bytes, an int) already produced the right error,
    # so the leak depended on hashability rather than on validity.
    if not isinstance(backend, str) or backend not in TABLE_BACKENDS:
        allowed = ", ".join(_TABLE_BACKEND_ORDER)
        raise ValueError(f"backend must be one of: {allowed}")
    return backend
