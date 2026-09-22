from __future__ import annotations

from typing import Any, cast

import torch

from .base import FITSTransform
from .helpers import (
    _ThreadedAttr,
    _amin,
    _amax,
)
from .state import (
    SCALABLE,
    DataState,
    as_state,
    check_state,
    is_payload,
    set_state,
    stamp_state,
)


def _linear_apply(x: torch.Tensor, scale: float, zero: float) -> torch.Tensor:
    """Compute ``scale * x + zero`` at FITS-appropriate precision.

    float64/int64 compute in float64, everything else in float32;
    float16/bfloat16 compute in float32 and are cast back.  Integer inputs
    are promoted to float and **stay** float: squeezing physical values back
    into the storage dtype wraps around (BZERO=32768 on int16 overflows to
    negative) and truncates fractional BSCALE.
    """
    dtype = x.dtype
    if dtype.is_floating_point:
        if dtype in (torch.float16, torch.bfloat16):
            return (x.float() * scale + zero).to(dtype)
        return x * scale + zero
    up = x.double() if dtype == torch.int64 else x.float()
    return up * scale + zero


def _linear_remove(x: torch.Tensor, scale: float, zero: float) -> torch.Tensor:
    """Compute ``(x - zero) / scale`` — inverse of :func:`_linear_apply`."""
    dtype = x.dtype
    if dtype.is_floating_point:
        if dtype in (torch.float16, torch.bfloat16):
            return ((x.float() - zero) / scale).to(dtype)
        return (x - zero) / scale
    up = x.double() if dtype == torch.int64 else x.float()
    return (up - zero) / scale


def _table_linear_apply(x: torch.Tensor, scale: float, zero: float) -> torch.Tensor:
    """Compute ``TSCAL * x + TZERO`` at the reader's table convention.

    ``table.read_torch`` delivers TSCAL/TZERO columns computed in **float64**
    (integer codes above 2**24 stay exact); hand-built column dicts must replay
    that bit-for-bit or they disagree with reader output. Image scaling stays
    float32 per :func:`_linear_apply` — the two readers' conventions really do
    differ.
    """
    return x.double() * scale + zero


def _table_linear_remove(x: torch.Tensor, scale: float, zero: float) -> torch.Tensor:
    """Compute ``(x - zero) / TSCAL`` — inverse of :func:`_table_linear_apply`."""
    return (x.double() - zero) / scale


def _require_columns(x: Any, who: str) -> Any:
    """Column transforms act on ``{column: tensor}`` dicts only."""
    if not isinstance(x, dict):
        raise TypeError(
            f"{who} expects a dict of column tensors (like table.read_torch "
            f"output), got {type(x)}"
        )
    return x


def _read_header_floats(
    path: str,
    hdu: int | str,
    defaults: tuple[tuple[str, float], ...],
) -> dict[str, float]:
    """Skinny per-key header reads with standard defaults for absent keys.

    ``read_keys`` raises ``RuntimeError`` both for an absent keyword (legal —
    BSCALE/BZERO default to 1.0/0.0 per the FITS standard) and for IO/HDU
    failures, which must never be papered over with identity scaling. A
    mandatory keyword (BITPIX) distinguishes the two without matching error
    strings: if it reads back, the HDU is fine and only the requested key is
    absent. Non-numeric keyword values raise :class:`ValueError`.
    """
    import torchfits

    out: dict[str, float] = {}
    for name, default in defaults:
        try:
            raw = torchfits.read_keys(path, [name], hdu=hdu)[name]
        except RuntimeError:
            torchfits.read_keys(path, ["BITPIX"], hdu=hdu)  # raises on IO errors
            out[name] = default
            continue
        try:
            out[name] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"header keyword {name}={raw!r} in {path} is not numeric"
            ) from exc
    return out


class FITSHeaderScale(FITSTransform):
    """Apply or remove BSCALE/BZERO scaling using FITS header keywords.

    ``forward`` applies the scaling tensor → physical (BSCALE * tensor + BZERO).
    ``inverse`` removes it: (physical − BZERO) / BSCALE.

    .. warning::
       This transform expects **stored** (unscaled) values. ``read_tensor``
       already applies BSCALE/BZERO by default, so applying this to its output
       would scale twice. Read with ``raw_scale=True`` for stored codes, or use
       :class:`FITSHeaderNormalize`, which is written for physical values.
       A payload that explicitly declares ``state="physical"`` is rejected.

    Parameters
    ----------
    bscale : float
        FITS BSCALE keyword value.  Default 1.0.
    bzero : float
        FITS BZERO keyword value.  Default 0.0.
    state : str or DataState or None
        Declared state of the ``forward()`` input, when it is not carried by a
        payload. A payload that carries its own state must agree with it;
        ``inverse()`` inputs are validated against the state ``forward()``
        produced instead.

    Example
    -------
    >>> header = {"BSCALE": 0.5, "BZERO": 100.0}
    >>> scaler = FITSHeaderScale.from_header(header)
    >>> physical = scaler(raw_counts)   # raw stored codes → physical
    >>> raw = scaler.inverse(physical)  # physical → stored codes
    """

    expects = frozenset({DataState.STORED})
    produces = DataState.PHYSICAL
    propagates_ivar = True

    def __init__(
        self,
        bscale: float = 1.0,
        bzero: float = 0.0,
        *,
        state: DataState | str | None = None,
    ) -> None:
        self.bscale = float(bscale)
        self.bzero = float(bzero)
        self.state = as_state(state)

    @classmethod
    def from_header(
        cls, header: dict[str, object], *, state: DataState | str | None = None
    ) -> FITSHeaderScale:
        """Construct from a FITS header dict-like object."""
        bscale = float(header.get("BSCALE", 1.0))  # type: ignore[arg-type]
        bzero = float(header.get("BZERO", 0.0))  # type: ignore[arg-type]
        return cls(bscale=bscale, bzero=bzero, state=state)

    @classmethod
    def from_path(
        cls,
        path: str,
        hdu: int | str = 0,
        *,
        state: DataState | str | None = None,
    ) -> FITSHeaderScale:
        """Construct from skinny ``read_keys`` (no full header dump).

        Absent BSCALE/BZERO cards default to 1.0 / 0.0 (the FITS standard);
        IO failures and non-numeric keyword values raise instead of silently
        producing an identity scaler.
        """
        vals = _read_header_floats(path, hdu, (("BSCALE", 1.0), ("BZERO", 0.0)))
        return cls(bscale=vals["BSCALE"], bzero=vals["BZERO"], state=state)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        check_state(x, self.expects, type(self).__name__, state=self.state)
        if self.bscale == 1.0 and self.bzero == 0.0:
            return x
        if is_payload(x):
            view = self.view(x, state=self.state)
            return view.replace(
                _linear_apply(view.flux, self.bscale, self.bzero),
                ivar=self.scale_ivar(view.ivar, self.bscale),
            )
        # Functional ops: out-of-place arithmetic only.
        return _linear_apply(x, self.bscale, self.bzero)

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        # No state= prior here: the prior describes forward() inputs, while an
        # inverse() input carries whatever forward() produced. Injecting the
        # forward prior would reject the transform's own output.
        check_state(x, self.inverse_expects(), type(self).__name__)
        if self.bscale == 1.0 and self.bzero == 0.0:
            return stamp_state(x, DataState.STORED)
        if is_payload(x):
            view = self.view(x, expects=self.inverse_expects())
            return stamp_state(
                view.replace(
                    _linear_remove(view.flux, self.bscale, self.bzero),
                    ivar=self.scale_ivar(view.ivar, 1.0 / self.bscale),
                ),
                DataState.STORED,
            )
        return _linear_remove(x, self.bscale, self.bzero)

    def __repr__(self) -> str:
        state = self.state.value if self.state is not None else None
        return (
            f"FITSHeaderScale(bscale={self.bscale}, bzero={self.bzero}, "
            f"state={state!r})"
        )


class FITSScaleColumns(FITSTransform):
    """Apply or remove TSCAL/TZERO scaling to table column tensors.

    Reads TSCAL and TZERO keywords for each column from a FITS table header
    and applies ``physical = TSCAL * stored + TZERO``.  Columns with default
    values (TSCAL=1.0, TZERO=0.0) are passed through unchanged.  Scaling
    computes in **float64**, matching ``table.read_torch``'s table convention
    bit-for-bit (the image convention of :class:`FITSHeaderScale` is float32).

    ``forward`` applies scaling: stored → physical.
    ``inverse`` removes it: ``(physical - TZERO) / TSCAL``.

    .. warning::
       Expects **stored** column values. ``table.read_torch`` already returns
       physical values (TSCAL/TZERO applied), so guard with ``state="stored"``
       semantics: the transform only acts on data that has not been scaled yet.

    Parameters
    ----------
    scales : dict[str, tuple[float, float]]
        Mapping of column name → (TSCAL, TZERO).
    state : str or DataState or None
        Declared state of the ``forward()`` input, when it is not carried by a
        payload. A payload that carries its own state must agree with it;
        ``inverse()`` inputs are validated against the state ``forward()``
        produced instead.
    """

    expects = frozenset({DataState.STORED})
    produces = DataState.PHYSICAL
    propagates_ivar = True

    def __init__(
        self,
        scales: dict[str, tuple[float, float]],
        *,
        state: DataState | str | None = None,
    ) -> None:
        self.scales: dict[str, tuple[float, float]] = {
            name: (float(ts), float(tz))
            for name, (ts, tz) in scales.items()
            if ts != 1.0 or tz != 0.0
        }
        self.state = as_state(state)

    @classmethod
    def from_header(
        cls, header: dict[str, object], *, state: DataState | str | None = None
    ) -> FITSScaleColumns:
        """Construct from a FITS table header dict-like object."""
        from ..fits_schema import iter_table_columns  # noqa: PLC0415

        scales: dict[str, tuple[float, float]] = {}
        for col in iter_table_columns(header):
            tscal = float(col.tscal) if col.tscal is not None else 1.0
            tzero = float(col.tzero) if col.tzero is not None else 0.0
            scales[col.name] = (tscal, tzero)
        return cls(scales, state=state)

    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        x = _require_columns(x, type(self).__name__)
        check_state(x, self.expects, type(self).__name__, state=self.state)
        if not self.scales:
            return x
        out = dict(x)
        for name, (tscal, tzero) in self.scales.items():
            if name not in out:
                continue
            # Functional ops: never mutate the caller's tensor.
            out[name] = _table_linear_apply(out[name], tscal, tzero)
        return out

    def inverse(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        # No state= prior here: it describes forward() inputs, while an
        # inverse() input carries whatever forward() produced.
        x = _require_columns(x, type(self).__name__)
        check_state(x, self.inverse_expects(), type(self).__name__)
        if not self.scales:
            return cast("dict[str, torch.Tensor]", stamp_state(x, DataState.STORED))
        out = dict(x)
        for name, (tscal, tzero) in self.scales.items():
            if name not in out:
                continue
            out[name] = _table_linear_remove(out[name], tscal, tzero)
        return cast("dict[str, torch.Tensor]", stamp_state(out, DataState.STORED))

    def __repr__(self) -> str:
        items = ", ".join(
            f"{n!r}: ({ts}, {tz})" for n, (ts, tz) in sorted(self.scales.items())
        )
        return f"FITSScaleColumns({{{items}}})"


class TNullToNan(FITSTransform):
    """Replace FITS TNULL sentinel values with NaN.

    Reads TNULL keywords from a FITS table header and replaces the
    corresponding sentinel values in each tensor column with NaN.
    Integer columns are promoted to float64 so NaN can be represented and the
    surrounding codes stay exact — the same convention ``table.read_torch``
    uses for its NaN-carrying columns. Floating columns keep their dtype.
    The sentinel comparison runs in the column's own dtype first, so valid
    rows that round onto the sentinel in a narrower float cannot be NaNed.

    .. note::
       The table reader already turns TNULL into missing values, so this is a
       **stored**-state utility for hand-built column dicts or foreign readers.

    Parameters
    ----------
    nulls : dict[str, float or int]
        Mapping of column name → TNULL value.
    """

    expects = frozenset({DataState.STORED})

    def __init__(
        self,
        nulls: dict[str, Any],
        *,
        state: DataState | str | None = None,
    ) -> None:
        self.nulls: dict[str, Any] = {}
        for name, value in nulls.items():
            float(value)  # reject non-numeric sentinels at construction
            self.nulls[name] = value
        self.state = as_state(state)

    @classmethod
    def from_header(
        cls, header: dict[str, object], *, state: DataState | str | None = None
    ) -> TNullToNan:
        """Construct from a FITS table header dict-like object."""
        from ..fits_schema import column_tnull_map  # noqa: PLC0415

        nulls = column_tnull_map(header)
        return cls(nulls, state=state)

    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        x = _require_columns(x, type(self).__name__)
        check_state(x, self.expects, type(self).__name__, state=self.state)
        if not self.nulls:
            return x
        out = dict(x)
        for name, tnull in self.nulls.items():
            if name not in out:
                continue
            val = out[name]
            # Sentinel comparison runs in the column's own dtype: promoting
            # first rounds int32/int64 codes into each other (float32 has a
            # 24-bit mantissa) and would NaN valid rows that round onto the
            # sentinel — or miss the sentinel itself.
            if val.dtype.is_floating_point or val.dtype.is_complex:
                null_mask = val.eq(float(tnull))
            else:
                null_mask = val.eq(int(tnull))
                # Reader convention: NaN-carrying table columns are float64,
                # keeping integer codes exact up to 2**53.
                val = val.double()
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

    This transform is written for **physical** values — the shape of
    ``read_tensor``'s output, including the unsigned-convention case where it
    returns ``uint16``/``uint32`` directly. It rescales flux, so it accepts
    every state except ``CONTINUUM_NORMALIZED`` (see ``SCALABLE``).

    Inverse limits computed by ``forward`` (``scale_floats=True`` on float
    headers) are cached **per thread**: one instance is safe to share across
    ``-J`` worker threads, but ``inverse()`` must run on the thread whose
    ``forward()`` produced its input.

    Parameters
    ----------
    header : dict
        FITS header dict-like with BITPIX, BSCALE, BZERO keywords.
    scale_floats : bool
        If True, min-max normalize floating-point data.  Default False.
    """

    expects = SCALABLE
    produces: DataState | None = DataState.NORMALIZED

    # BITPIX → (dtype, signed, bits)
    _BITPIX_MAP: dict[int, tuple[torch.dtype, bool, int]] = {
        8: (torch.uint8, False, 8),
        16: (torch.int16, True, 16),
        32: (torch.int32, True, 32),
        64: (torch.int64, True, 64),
        -32: (torch.float32, False, 32),
        -64: (torch.float64, False, 64),
    }

    # forward() limits for the scale_floats path: per-(instance, thread)
    # storage so one instance stays correct under -J worker threads, __call__
    # never mutates instance state, and instances stay picklable (the cache is
    # transient and intentionally not pickled — a fresh worker re-runs
    # forward()).
    _fit_range = _ThreadedAttr()

    def __init__(self, header: dict[str, object], scale_floats: bool = False) -> None:
        self.bitpix = int(header.get("BITPIX", -32))  # type: ignore[call-overload]
        self.bscale = float(header.get("BSCALE", 1.0))  # type: ignore[arg-type]
        self.bzero = float(header.get("BZERO", 0.0))  # type: ignore[arg-type]
        self.scale_floats = bool(scale_floats)

        info = self._BITPIX_MAP.get(self.bitpix)
        self._is_integer = info is not None and info[1]
        self._is_unsigned = info is not None and not info[1] and self.bitpix > 0
        self._bits = info[2] if info else 32
        # Header-derived physical range — a per-header constant, immutable
        # after construction.
        self._in_range: tuple[float, float] | None = None
        # A float header without scale_floats is an identity pass: it does not
        # normalize anything, so it must not relabel the payload.
        self.produces = (
            DataState.NORMALIZED
            if (self._is_integer or self._is_unsigned or self.scale_floats)
            else None
        )

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
        """Construct from skinny ``read_keys`` (no full header dump).

        Absent BSCALE/BZERO cards default to 1.0 / 0.0 (the FITS standard);
        IO failures and non-numeric keyword values raise instead of silently
        producing an identity normalization.
        """
        keys = _read_header_floats(
            path, hdu, (("BITPIX", -32.0), ("BSCALE", 1.0), ("BZERO", 0.0))
        )
        return cls(keys, scale_floats=scale_floats)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        flux = view.flux
        if self._is_integer or self._is_unsigned:
            vmin, vmax = self._in_range  # type: ignore[misc]
            if vmax == vmin:
                return view.replace(torch.zeros_like(flux))
            # float32 mantissa is 24 bits: int32/int64 counts above 2**24
            # round, so the [0, 1] map collapses neighboring integers.
            xf = (
                flux.double()
                if (not flux.dtype.is_floating_point and flux.element_size() >= 4)
                else flux
            )
            span = vmax - vmin
            out = (xf - vmin) / span
            return view.replace(out, ivar=self.divide_ivar(view.ivar, span))
        if self.scale_floats:
            vmin = _amin(flux, tuple(range(flux.ndim)), mask=view.effective_mask(mask))
            vmax = _amax(flux, tuple(range(flux.ndim)), mask=view.effective_mask(mask))
            finite = bool(torch.isfinite(vmin) and torch.isfinite(vmax))
            if not finite:
                return view.replace(torch.full_like(flux, float("nan")))
            self._fit_range = (float(vmin.item()), float(vmax.item()))
            if vmax == vmin:
                return view.replace(torch.zeros_like(flux))
            span = vmax - vmin
            out = (flux - vmin) / span
            return view.replace(out, ivar=self.divide_ivar(view.ivar, span))
        # Float types, no scaling requested — identity
        return x

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if not (self._is_integer or self._is_unsigned or self.scale_floats):
            return x
        limits = self._in_range if self._in_range is not None else self._fit_range
        if limits is None:
            raise RuntimeError(
                "FITSHeaderNormalize.inverse() requires a prior forward() pass "
                "on this thread when scale_floats=True."
            )
        view = self.view(x, expects=self.inverse_expects())
        vmin, vmax = limits
        span = vmax - vmin
        out = view.flux * span + vmin
        # Undoing a normalization cannot claim any particular state; drop the
        # declared one rather than leaving a stale "normalized" label behind.
        return set_state(
            view.replace(out, ivar=self.divide_ivar(view.ivar, 1.0 / span)), None
        )

    def __repr__(self) -> str:
        return (
            f"FITSHeaderNormalize(bitpix={self.bitpix}, "
            f"bscale={self.bscale}, bzero={self.bzero}, "
            f"scale_floats={self.scale_floats})"
        )
