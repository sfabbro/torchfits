"""Private FITS table I/O implementation modules."""

from .cache import (
    _acquire_cpp_handle,
    _acquire_cpp_reader,
)

__all__ = [
    "_acquire_cpp_handle",
    "_acquire_cpp_reader",
]
