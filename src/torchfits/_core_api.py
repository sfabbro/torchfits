"""Torch-free path to the native metadata surface.

``torchfits._core`` is a nanobind module over ``libtorchfits_core``, the half of
the native stack that links CFITSIO and nothing else. ``torchfits._C`` links
libtorch as well, so touching it costs a ~1 s cold ``dlopen`` on a process that
only wants to look at a header. The functions here are the path-based metadata
probes -- ``read_header_dict``, ``read_nrows``, ``read_colnames``, ``read_keys``,
``read_shape``, ``read_num_hdus``, ``read_hdu_type``, ``read_table_info``,
``read_header_string`` -- routed through the core so ``torchfits.read_header``,
``torchfits.read_nrows`` and ``torchfits.read_colnames`` no longer drag libtorch
into the process.

Handle-based metadata (``FITSFile.get_header`` and friends) stays on ``_C``:
those handles are passed to the tensor readers, so they must be the same Python
type the extension owns. Both modules call the same C++ implementation in
``core/metadata_api.cpp``, so the two paths cannot report different values.

This module is private. It exists so the routing is visible in one place, not
so callers can reach past the public API.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from torchfits._io_engine.paths import guard_fits_path


class _CoreModule(Protocol):
    """The slice of ``torchfits._core`` this shim depends on.

    Spelled out rather than imported: mypy cannot use a module object as an
    annotation, and a protocol makes the dependency explicit -- if a binding
    is renamed or its signature changes, this fails to satisfy the protocol and
    ``mypy`` says so.
    """

    __build_id__: str

    def core_library_build_id(self) -> str: ...
    def thread_count(self) -> int: ...
    def read_header_dict(
        self, filename: str, hdu_num: int
    ) -> list[tuple[str, str, str]]: ...
    def read_header_string(self, filename: str, hdu_num: int) -> str: ...
    def read_num_hdus(self, filename: str) -> int: ...
    def read_hdu_type(self, filename: str, hdu_num: int) -> str: ...
    def read_nrows(self, filename: str, hdu_num: int) -> int: ...
    def read_colnames(self, filename: str, hdu_num: int) -> list[str]: ...
    def read_table_info(self, filename: str, hdu_num: int) -> dict[str, Any]: ...
    def read_keys(
        self, filename: str, hdu_num: int, keys: list[str]
    ) -> dict[str, Any]: ...
    def read_shape(
        self, filename: str, hdu_num: int
    ) -> tuple[int, tuple[int, ...]]: ...


__all__ = [
    "read_colnames",
    "read_header_dict",
    "read_header_string",
    "read_hdu_type",
    "read_keys",
    "read_nrows",
    "read_num_hdus",
    "read_shape",
    "read_table_info",
    "verify_core_link",
]

_lock = threading.Lock()
_module: _CoreModule | None = None


def _core() -> _CoreModule:
    """Import ``torchfits._core`` once, checking it matches ``_C``'s build.

    The guard is what makes a half-rebuilt checkout fail loudly. ``_C`` resolves
    its CFITSIO symbols against ``libtorchfits_core`` at load time, so a stale
    core next to a fresh extension is a cross-version call into a C library:
    exactly the class of bug that shows up as a segfault rather than an
    exception. Compare the ids the build stamped into both and refuse to
    continue if they disagree.
    """
    global _module
    with _lock:
        if _module is not None:
            return _module
        from torchfits import _core as core

        verify_core_link(core)
        _module = core
        return _module


def verify_core_link(core: _CoreModule) -> None:
    """Raise unless the loaded ``libtorchfits_core`` matches the module beside it.

    ``_core.so`` embeds the build id CMake stamped on the core when the *module*
    was compiled. ``core_library_build_id()`` is read out of the shared object
    the process actually loaded. A stale library left next to a fresh module
    passes every compile-time check and only misbehaves once a struct layout
    disagrees -- so compare the two before making a call.

    ``_C`` embeds the same id (``_C.__core_build_id__``);
    :func:`torchfits._ensure_runtime_init` checks that side when the torch-linked
    extension is first loaded, which is why this function must not import ``_C``:
    doing so would load libtorch and defeat the whole point of the core.
    """
    module_id = getattr(core, "__build_id__", None)
    library_id = core.core_library_build_id()
    if not module_id or not library_id:
        raise ImportError(
            "torchfits is missing its native core build id "
            f"(_core={module_id!r}, libtorchfits_core={library_id!r}); rebuild the "
            "extension (pip install -e . --no-build-isolation) rather than mixing "
            "artifacts from different builds."
        )
    if module_id != library_id:
        raise ImportError(
            "torchfits._core and the libtorchfits_core it loaded are from "
            f"different builds:\n  _core: {module_id}\n  library: {library_id}\n"
            "Reinstall the package (pip install -e . --no-build-isolation) so every "
            "native artifact comes from the same build."
        )


def _guarded(path: str) -> str:
    """Apply the Python-side path policy the ``_C`` bindings rely on.

    ``_cpp`` guards every path-taking symbol with this, rejecting
    private/loopback ``http``/``https``/``ftp`` URLs before CFITSIO opens them.
    ``libtorchfits_core`` enforces the CFITSIO-level rules (``|``, ``sh://``)
    in C++, but not this one, so the check has to happen here.
    """
    return guard_fits_path(path)


def read_header_dict(path: str, hdu: int) -> list[tuple[str, str, str]]:
    """Full card list for ``hdu`` (0-based), duplicates preserved."""
    return _core().read_header_dict(_guarded(path), hdu)


def read_header_string(path: str, hdu: int) -> str:
    """Bulk header text for ``hdu`` (0-based), for the Python fast parser."""
    return _core().read_header_string(_guarded(path), hdu)


def read_num_hdus(path: str) -> int:
    """HDU count, with the truncation check that catches a corrupt header tail."""
    return int(_core().read_num_hdus(_guarded(path)))


def read_hdu_type(path: str, hdu: int) -> str:
    """``IMAGE`` / ``ASCII_TABLE`` / ``BINARY_TABLE`` / ``UNKNOWN`` for ``hdu``."""
    return str(_core().read_hdu_type(_guarded(path), hdu))


def read_nrows(path: str, hdu: int) -> int:
    """Table row count via ``fits_get_num_rows`` (no full header dump)."""
    return int(_core().read_nrows(_guarded(path), hdu))


def read_colnames(path: str, hdu: int) -> list[str]:
    """Column names from ``TTYPEn`` (no full header dump)."""
    return list(_core().read_colnames(_guarded(path), hdu))


def read_table_info(path: str, hdu: int) -> dict[str, Any]:
    """``{"nrows", "colnames", "tforms"}`` for a table HDU."""
    return dict(_core().read_table_info(_guarded(path), hdu))


def read_keys(path: str, hdu: int, keys: list[str]) -> dict[str, Any]:
    """Named header keywords, typed the way :func:`torchfits.read_header` types them."""
    return dict(_core().read_keys(_guarded(path), hdu, list(keys)))


def read_shape(path: str, hdu: int) -> tuple[int, tuple[int, ...]]:
    """``(bitpix, shape)`` for an image HDU, shape in torch (row-major) order."""
    bitpix, dims = _core().read_shape(_guarded(path), hdu)
    return int(bitpix), tuple(int(d) for d in dims)
