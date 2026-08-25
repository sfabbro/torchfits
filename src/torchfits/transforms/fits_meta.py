from __future__ import annotations


import torch

from .base import FITSTransform
from .helpers import (
    _amin,
    _amax,
)


class FITSHeaderScale(FITSTransform):
    """Apply or remove BSCALE/BZERO scaling using FITS header keywords.

    ``forward`` applies the scaling tensor → physical (BSCALE * tensor + BZERO).
    ``inverse`` removes it: (physical − BZERO) / BSCALE.

    Parameters
    ----------
    bscale : float
        FITS BSCALE keyword value.  Default 1.0.
    bzero : float
        FITS BZERO keyword value.  Default 0.0.

    Example
    -------
    >>> header = {"BSCALE": 0.5, "BZERO": 100.0}
    >>> scaler = FITSHeaderScale.from_header(header)
    >>> physical = scaler(raw_counts)   # raw → physical
    >>> raw = scaler.inverse(physical)  # physical → raw
    """

    def __init__(self, bscale: float = 1.0, bzero: float = 0.0) -> None:
        self.bscale = float(bscale)
        self.bzero = float(bzero)

    @classmethod
    def from_header(cls, header: dict[str, object]) -> FITSHeaderScale:
        """Construct from a FITS header dict-like object."""
        bscale = float(header.get("BSCALE", 1.0))  # type: ignore[arg-type]
        bzero = float(header.get("BZERO", 0.0))  # type: ignore[arg-type]
        return cls(bscale=bscale, bzero=bzero)

    @classmethod
    def from_path(cls, path: str, hdu: int | str = 0) -> FITSHeaderScale:
        """Construct from skinny ``read_keys`` (no full header dump)."""
        import torchfits

        vals: dict[str, float] = {"BSCALE": 1.0, "BZERO": 0.0}
        for name in ("BSCALE", "BZERO"):
            try:
                vals[name] = float(torchfits.read_keys(path, [name], hdu=hdu)[name])
            except (RuntimeError, TypeError, ValueError):
                pass
        return cls(bscale=vals["BSCALE"], bzero=vals["BZERO"])

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.bscale == 1.0 and self.bzero == 0.0:
            return x
        # Functional ops: `.to(float32)` aliases float32 inputs, so in-place
        # mul_/add_ here used to mutate the caller's tensor (M6).
        result = x.to(torch.float32) * self.bscale + self.bzero
        return result.to(x.dtype) if x.dtype != torch.float32 else result

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.bscale == 1.0 and self.bzero == 0.0:
            return x
        result = (x.to(torch.float32) - self.bzero) / self.bscale
        return result.to(x.dtype) if x.dtype != torch.float32 else result

    def __repr__(self) -> str:
        return f"FITSHeaderScale(bscale={self.bscale}, bzero={self.bzero})"


class FITSScaleColumns(FITSTransform):
    """Apply or remove TSCAL/TZERO scaling to table column tensors.

    Reads TSCAL and TZERO keywords for each column from a FITS table header
    and applies ``physical = TSCAL * stored + TZERO``.  Columns with default
    values (TSCAL=1.0, TZERO=0.0) are passed through unchanged.

    ``forward`` applies scaling: stored → physical.
    ``inverse`` removes it: ``(physical - TZERO) / TSCAL``.

    Parameters
    ----------
    header : dict
        FITS table header dict-like with TTYPE*/TFORM*/TSCAL*/TZERO* keywords.

    Example
    -------
    >>> header = {"TFIELDS": 2, "TTYPE1": "FLUX", "TFORM1": "E",
    ...            "TSCAL1": 0.001, "TZERO1": 0.0,
    ...            "TTYPE2": "MAG", "TFORM2": "E",
    ...            "TSCAL2": 1.0, "TZERO2": 25.0}
    >>> scaler = FITSScaleColumns.from_header(header)
    >>> physical = scaler({"FLUX": raw_flux, "MAG": raw_mag})
    >>> raw = scaler.inverse(physical)
    """

    def __init__(self, scales: dict[str, tuple[float, float]]) -> None:
        """
        Parameters
        ----------
        scales : dict[str, tuple[float, float]]
            Mapping of column name → (TSCAL, TZERO).
        """
        self.scales: dict[str, tuple[float, float]] = {
            name: (float(ts), float(tz))
            for name, (ts, tz) in scales.items()
            if ts != 1.0 or tz != 0.0
        }

    @classmethod
    def from_header(cls, header: dict[str, object]) -> FITSScaleColumns:
        """Construct from a FITS table header dict-like object."""
        from ..fits_schema import iter_table_columns  # noqa: PLC0415

        scales: dict[str, tuple[float, float]] = {}
        for col in iter_table_columns(header):
            tscal = float(col.tscal) if col.tscal is not None else 1.0
            tzero = float(col.tzero) if col.tzero is not None else 0.0
            scales[col.name] = (tscal, tzero)
        return cls(scales)

    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if not self.scales:
            return x
        out = dict(x)
        for name, (tscal, tzero) in self.scales.items():
            if name not in out:
                continue
            val = out[name]
            if tscal == 1.0 and tzero == 0.0:
                continue
            # Functional ops: never mutate the caller's tensor (M6).
            result = val.to(torch.float32) * tscal + tzero
            out[name] = result.to(val.dtype) if val.dtype != torch.float32 else result
        return out

    def inverse(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if not self.scales:
            return x
        out = dict(x)
        for name, (tscal, tzero) in self.scales.items():
            if name not in out:
                continue
            val = out[name]
            if tscal == 1.0 and tzero == 0.0:
                continue
            result = (val.to(torch.float32) - tzero) / tscal
            out[name] = result.to(val.dtype) if val.dtype != torch.float32 else result
        return out

    def __repr__(self) -> str:
        items = ", ".join(
            f"{n!r}: ({ts}, {tz})" for n, (ts, tz) in sorted(self.scales.items())
        )
        return f"FITSScaleColumns({{{items}}})"


class TNullToNan(FITSTransform):
    """Replace FITS TNULL sentinel values with NaN.

    Reads TNULL keywords from a FITS table header and replaces the
    corresponding sentinel values in each tensor column with NaN.
    Integer columns are promoted to float32 so NaN can be represented.

    ``inverse`` is not available — null replacement is lossy.

    Parameters
    ----------
    header : dict
        FITS table header dict-like with TTYPE*/TNULL* keywords.

    Example
    -------
    >>> header = {"TFIELDS": 1, "TTYPE1": "FLUX", "TFORM1": "J", "TNULL1": -999}
    >>> nuller = TNullToNan.from_header(header)
    >>> clean = nuller({"FLUX": torch.tensor([1, -999, 3], dtype=torch.int32)})
    >>> # FLUX is now float32 with NaN at position 1
    """

    def __init__(self, nulls: dict[str, float]) -> None:
        """
        Parameters
        ----------
        nulls : dict[str, float]
            Mapping of column name → TNULL value.
        """
        self.nulls: dict[str, float] = {name: float(v) for name, v in nulls.items()}

    @classmethod
    def from_header(cls, header: dict[str, object]) -> TNullToNan:
        """Construct from a FITS table header dict-like object."""
        from ..fits_schema import column_tnull_map  # noqa: PLC0415

        nulls = column_tnull_map(header)
        return cls(nulls)

    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if not self.nulls:
            return x
        out = dict(x)
        for name, tnull in self.nulls.items():
            if name not in out:
                continue
            val = out[name]
            # Promote integer columns to float32 so NaN is representable
            if val.dtype not in (torch.float32, torch.float64):
                val = val.to(torch.float32)
            null_mask = val.eq(tnull)
            out[name] = torch.where(
                null_mask,
                torch.tensor(float("nan"), dtype=val.dtype, device=val.device),
                val,
            )
        return out

    def inverse(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        raise RuntimeError(
            "TNullToNan.inverse() is not available — null replacement is lossy."
        )

    def __repr__(self) -> str:
        items = ", ".join(f"{n!r}: {v}" for n, v in sorted(self.nulls.items()))
        return f"TNullToNan({{{items}}})"


class FITSHeaderNormalize(FITSTransform):
    """Auto-detect and apply normalization from FITS header keywords.

    Inspects BITPIX, BSCALE, and BZERO to determine the best
    normalization strategy:

    - **Integer types** (BITPIX 8/16/32): scales to [0, 1] using the
      integer range, optionally compensating for BZERO offset.
    - **Float types** (BITPIX -32/-64): applies no scaling by default
      (floats are already in physical units).  Set *scale_floats=True*
      to normalize to [0, 1] via min-max.

    ``inverse`` reverses the normalization using the cached parameters.

    Parameters
    ----------
    header : dict
        FITS header dict-like with BITPIX, BSCALE, BZERO keywords.
    scale_floats : bool
        If True, min-max normalize floating-point data.  Default False.
    """

    # BITPIX → (dtype, signed, bits)
    _BITPIX_MAP: dict[int, tuple[torch.dtype, bool, int]] = {
        8: (torch.uint8, False, 8),
        16: (torch.int16, True, 16),
        32: (torch.int32, True, 32),
        64: (torch.int64, True, 64),
        -32: (torch.float32, False, 32),
        -64: (torch.float64, False, 64),
    }

    def __init__(self, header: dict[str, object], scale_floats: bool = False) -> None:
        self.bitpix = int(header.get("BITPIX", -32))  # type: ignore[call-overload]
        self.bscale = float(header.get("BSCALE", 1.0))  # type: ignore[arg-type]
        self.bzero = float(header.get("BZERO", 0.0))  # type: ignore[arg-type]
        self.scale_floats = bool(scale_floats)

        info = self._BITPIX_MAP.get(self.bitpix)
        self._is_integer = info is not None and info[1]
        self._is_unsigned = info is not None and not info[1] and self.bitpix > 0
        self._bits = info[2] if info else 32
        self._in_range: tuple[float, float] | None = None

        # Pre-compute the physical value range for integer types
        if self._is_integer:
            raw_min = -(2 ** (self._bits - 1))
            raw_max = (2 ** (self._bits - 1)) - 1
            phys_min = raw_min * self.bscale + self.bzero
            phys_max = raw_max * self.bscale + self.bzero
            self._in_range = (phys_min, phys_max)
        elif self._is_unsigned and self.bitpix == 8:
            phys_min = self.bzero
            phys_max = 255.0 * self.bscale + self.bzero
            self._in_range = (phys_min, phys_max)

    @classmethod
    def from_path(
        cls, path: str, hdu: int | str = 0, *, scale_floats: bool = False
    ) -> FITSHeaderNormalize:
        """Construct from skinny ``read_keys`` (no full header dump)."""
        import torchfits

        keys: dict[str, object] = {}
        for name, default in (("BITPIX", -32), ("BSCALE", 1.0), ("BZERO", 0.0)):
            try:
                keys[name] = torchfits.read_keys(path, [name], hdu=hdu)[name]
            except RuntimeError:
                keys[name] = default
        return cls(keys, scale_floats=scale_floats)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._is_integer or self._is_unsigned:
            vmin, vmax = self._in_range  # type: ignore[misc]
            if vmax == vmin:
                return torch.zeros_like(x)
            return (x - vmin) / (vmax - vmin)
        if self.scale_floats:
            vmin = _amin(x, tuple(range(x.ndim)), mask=mask)
            vmax = _amax(x, tuple(range(x.ndim)), mask=mask)
            self._in_range = (float(vmin.item()), float(vmax.item()))
            if vmax == vmin:
                return torch.zeros_like(x)
            return (x - vmin) / (vmax - vmin)
        # Float types, no scaling requested — identity
        return x

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._is_integer or self._is_unsigned or self.scale_floats:
            if self._in_range is None:
                raise RuntimeError(
                    "FITSHeaderNormalize.inverse() requires a prior forward() pass "
                    "when scale_floats=True."
                )
            vmin, vmax = self._in_range
            return x * (vmax - vmin) + vmin
        return x

    def __repr__(self) -> str:
        return (
            f"FITSHeaderNormalize(bitpix={self.bitpix}, "
            f"bscale={self.bscale}, bzero={self.bzero}, "
            f"scale_floats={self.scale_floats})"
        )
