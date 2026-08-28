"""Lazy data accessor for image/cube HDUs and BITPIX-to-dtype mapping."""

from __future__ import annotations

from typing import Any, Tuple, cast

import torch
from torch import Tensor


_BITPIX_TO_DTYPE: dict[int, torch.dtype] = {
    8: torch.uint8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
    -32: torch.float32,
    -64: torch.float64,
}


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
        base = _BITPIX_TO_DTYPE.get(bitpix)
        if base is None:
            return torch.float32
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
                return torch.int8
            if bitpix == 16 and abs(bscale - 1.0) < tol and abs(bzero - 32768.0) < tol:
                return torch.uint16
            if (
                bitpix == 32
                and abs(bscale - 1.0) < tol
                and abs(bzero - 2147483648.0) < tol
            ):
                return torch.uint32
            # Identity integer storage with BLANK is read as scaled float+NaN.
            if "BLANK" in self._header:
                return torch.float32
        return base

    def __getitem__(self, slice_spec: Any) -> Tensor:
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

        return cast(Tensor, self._handle.read_subset(self._index, x1, y1, x2, y2))
