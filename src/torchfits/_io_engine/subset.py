"""Image subset/cutout readers for FITS I/O."""

from __future__ import annotations

from typing import Callable, Tuple
from torch import Tensor


def read_subset(
    get_cached_handle: Callable[[str, int], tuple[object, bool]],
    path: str,
    hdu: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    handle_cache_capacity: int = 16,
) -> Tensor:
    """Read a rectangular subset of an image HDU."""
    try:
        file_handle, cached = get_cached_handle(path, handle_cache_capacity)
        try:
            return file_handle.read_subset(hdu, x1, y1, x2, y2)
        finally:
            if not cached:
                try:
                    file_handle.close()
                except Exception:
                    pass
    except Exception as exc:
        raise RuntimeError(f"Failed to read subset from '{path}': {exc}") from exc


class SubsetReader:
    """Persistent subset reader for repeated cutouts on one image HDU."""

    def __init__(self, path: str, hdu: int = 0, device: str = "cpu"):
        import torchfits._C as cpp

        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(hdu, int) or hdu < 0:
            raise ValueError("hdu must be a non-negative integer")
        if device not in ["cpu", "cuda", "mps"] and not str(device).startswith("cuda:"):
            raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

        self._reader = cpp.SubsetReader(path, int(hdu))
        self._device = device

    @property
    def hdu(self) -> int:
        return int(self._reader.hdu)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self._reader.height), int(self._reader.width)

    def read_subset(self, x1: int, y1: int, x2: int, y2: int) -> Tensor:
        out = self._reader.read(int(x1), int(y1), int(x2), int(y2))
        if self._device != "cpu":
            out = out.to(self._device)
        return out

    def close(self) -> None:
        self._reader.close()

    def __call__(self, x1: int, y1: int, x2: int, y2: int) -> Tensor:
        return self.read_subset(x1, y1, x2, y2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def open_subset_reader(path: str, hdu: int = 0, device: str = "cpu") -> SubsetReader:
    """Open a persistent cutout reader for repeated subsets on one HDU."""
    return SubsetReader(path=path, hdu=hdu, device=device)
