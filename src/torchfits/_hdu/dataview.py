"""Lazy data accessor for image/cube HDUs and BITPIX-to-dtype mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, cast

if TYPE_CHECKING:
    import torch
    from torch import Tensor


def _torch() -> Any:
    """Resolve PyTorch lazily.

    Importing it costs about a second, and ``DataView`` is reachable from
    metadata-only paths (``torchfits.open`` on an image), so it must not be
    imported at module scope.
    """
    import torch

    return torch


# Storage BITPIX -> dtype *name*.  Names instead of ``torch.dtype`` objects so
# building this table does not force the import; ``bitpix_to_dtype`` resolves
# them on access.
_BITPIX_TO_KIND: dict[int, str] = {
    8: "uint8",
    16: "int16",
    32: "int32",
    64: "int64",
    -32: "float32",
    -64: "float64",
}


def _dtype(name: str) -> torch.dtype:
    """Resolve a torch dtype by name, importing torch on first use."""
    return cast("torch.dtype", getattr(_torch(), name))


def bitpix_to_dtype(bitpix: int) -> torch.dtype | None:
    """Return the torch dtype tag for a storage BITPIX, or ``None`` if unknown."""
    kind = _BITPIX_TO_KIND.get(bitpix)
    return None if kind is None else _dtype(kind)


class DataView:
    def __init__(
        self,
        file_handle: Any,
        hdu_index: int,
        header: Any = None,
    ):
        self._handle = file_handle
        self._index = hdu_index
        self._header = header

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._handle.get_shape(self._index))

    @property
    def dtype(self) -> torch.dtype:
        bitpix = self._handle.get_dtype(self._index)
        base = bitpix_to_dtype(bitpix)
        if base is None:
            return _dtype("float32")
        # Respect the FITS integer conventions so metadata matches the
        # tensors the readers actually produce (uint16/uint32/int8), not the
        # raw storage BITPIX (L1).
        if self._header is not None and bitpix in (8, 16, 32, 64):
            try:
                bscale = float(self._header.get("BSCALE", 1.0))
                bzero = float(self._header.get("BZERO", 0.0))
            except (TypeError, ValueError):
                return base
            tol = 1e-5
            if bitpix == 8 and abs(bscale - 1.0) < tol and abs(bzero + 128.0) < tol:
                return _dtype("int8")
            if bitpix == 16 and abs(bscale - 1.0) < tol and abs(bzero - 32768.0) < tol:
                return _dtype("uint16")
            if (
                bitpix == 32
                and abs(bscale - 1.0) < tol
                and abs(bzero - 2147483648.0) < tol
            ):
                return _dtype("uint32")
            # Identity integer storage with BLANK is read as scaled float+NaN.
            if "BLANK" in self._header:
                return _dtype("float32")
        return base

    def __getitem__(self, slice_spec: Any) -> Tensor:
        """Read a 2-D rectangular cutout.

        Deliberately *not* a numpy-array emulation: the underlying primitive is
        ``read_subset``, which returns a block, so an integer index becomes a
        length-1 slice (``data[0]`` has shape ``(1, n)`` where numpy would give
        ``(n,)``) and only ``step=1`` slices are accepted. Values are always
        correct; the kept axis is what makes one code path serve both. Use
        ``torchfits.read`` when numpy-style indexing semantics are wanted.
        """
        shape = self.shape
        if len(shape) < 2:
            raise ValueError("Subset reading requires at least 2D data")

        if slice_spec is Ellipsis:
            slice_spec = (slice(None), slice(None))
        elif not isinstance(slice_spec, tuple):
            slice_spec = (slice_spec, slice(None))

        if len(slice_spec) != 2:
            raise ValueError(
                "Subset slicing supports exactly 2 dimensions (y, x); "
                "N-D cubes should use read_subset / open_subset_reader"
            )

        def _normalize_index(s: Any, dim: int) -> tuple[int, int]:
            if isinstance(s, int):
                if s < -dim or s >= dim:
                    raise IndexError(f"index {s} out of range for dimension size {dim}")
                idx = s + dim if s < 0 else s
                return idx, idx + 1
            if isinstance(s, slice):
                if s.step not in (None, 1):
                    raise ValueError("Only step=1 slices are supported")
                start = 0 if s.start is None else s.start
                stop = dim if s.stop is None else s.stop
                if start < 0:
                    start += dim
                if stop < 0:
                    stop += dim
                start = max(0, min(dim, start))
                stop = max(0, min(dim, stop))
                return start, stop
            raise TypeError("Slice spec must be int or slice")

        y1, y2 = _normalize_index(slice_spec[0], shape[0])
        x1, x2 = _normalize_index(slice_spec[1], shape[1])

        return cast("Tensor", self._handle.read_subset(self._index, x1, y1, x2, y2))
