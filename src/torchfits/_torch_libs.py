"""Map torch's shared libraries before ``torchfits._C`` is loaded.

A pip install leaves ``libc10`` in ``site-packages/torch/lib``. That directory
is not on the dynamic linker's default path. Conda hides this because the
interpreter's rpath includes the prefix lib, and importing the ``torch``
Python package maps the libraries itself. Metadata callers are not allowed to
import ``torch``, and a fresh ``torchfits info`` process never does, so the
extension import failed with ``libc10.so: cannot open shared object file``
before any FITS call.

Loading the libraries by absolute path registers their sonames. It does not
import the ``torch`` Python package.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_DONE = False
_FINDER: _TorchLibFinder | None = None

# Dependency order. CUDA libraries are left out: the published extension is
# CPU-linked, and loading ``libtorch_cuda`` fails when no driver is present.
_LIBS = (
    "torch_global_deps",
    "gomp",
    "omp",
    "c10",
    "torch_cpu",
    "torch",
    "torch_python",
)


def torch_lib_dir() -> Path | None:
    """``torch/lib`` from an already-imported module, else from ``sys.path``."""
    mod = sys.modules.get("torch")
    file = getattr(mod, "__file__", None) if mod is not None else None
    if file:
        lib = Path(file).resolve().parent / "lib"
        if lib.is_dir():
            return lib
    for entry in sys.path:
        if not entry:
            continue
        lib = Path(entry) / "torch" / "lib"
        if lib.is_dir():
            return lib
    return None


def preload() -> None:
    """Map torch's shared libraries once. Missing files are skipped."""
    global _DONE
    if _DONE:
        return
    _DONE = True
    lib = torch_lib_dir()
    if lib is None:
        return
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for name in _LIBS:
        for suffix in (".so", ".dylib"):
            path = lib / f"lib{name}{suffix}"
            if not path.is_file():
                continue
            try:
                ctypes.CDLL(str(path), mode=mode)
            except OSError:
                pass
            break


class _TorchLibFinder:
    """Run :func:`preload` when something imports ``torchfits._C``."""

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> None:
        if fullname == "torchfits._C":
            preload()
        return None


def install() -> None:
    """Register the finder. Importing ``torchfits`` itself does not map libtorch."""
    global _FINDER
    if _FINDER is not None and _FINDER in sys.meta_path:
        return
    _FINDER = _TorchLibFinder()
    sys.meta_path.insert(0, _FINDER)
