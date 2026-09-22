from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from .base import FITSTransform
from .helpers import (
    _ThreadedAttr,
    _amax,
    _amin,
    _median,
    _quantile,
    _stats_upcast,
    _weighted_dispersion,
    _weighted_quantile,
    estimate_background,
    zscale_limits,
)
from .state import SCALABLE, DataState


def _nan_like(reference: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A scalar NaN broadcastable against *reference*."""
    return torch.full((), float("nan"), dtype=dtype, device=reference.device)


class AffineTransform(FITSTransform):
    """Explicit ``flux * scale + offset`` with exact IVAR propagation.

    The linear building block every normalizer reduces to. Because the mapping
    is affine, ``ivar`` transforms exactly: ``ivar / scale**2`` (an offset does
    not change the variance).

    Parameters
    ----------
    scale : float
        Multiplicative factor. Must be non-zero.
    offset : float
        Additive factor (default 0).
    """

    propagates_ivar = True

    def __init__(self, scale: float = 1.0, offset: float = 0.0) -> None:
        self.scale = float(scale)
        if self.scale == 0.0:
            raise ValueError("AffineTransform scale must be non-zero")
        self.offset = float(offset)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        out = view.flux * self.scale + self.offset
        return view.replace(out, ivar=self.scale_ivar(view.ivar, self.scale))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        out = (view.flux - self.offset) / self.scale
        return view.replace(out, ivar=self.scale_ivar(view.ivar, 1.0 / self.scale))

    def __repr__(self) -> str:
        return f"AffineTransform(scale={self.scale}, offset={self.offset})"


class ZScaleNormalize(FITSTransform):
    """IRAF zscale auto-contrast normalisation.

    ``forward`` maps data to [0, 1] using dynamically computed limits.
    ``inverse`` uses the limits from the most recent forward pass.

    Parameters
    ----------
    contrast : float
        IRAF contrast; smaller = tighter range.
    dim : tuple
        Dimensions the limits are computed over.
    algorithm : str
        ``"proxy"`` (default) — ``median ± MAD/contrast``, fast.
        ``"iraf"`` — the iterative line-fit IRAF algorithm (astropy parity).
    weighted : bool
        Use inverse-variance weighted limits when the payload carries ``ivar``.
    """

    propagates_ivar = True
    expects = SCALABLE
    produces = DataState.NORMALIZED
    _last_state = _ThreadedAttr()

    def __init__(
        self,
        contrast: float = 0.25,
        dim: Tuple[int, ...] = (-2, -1),
        *,
        algorithm: str = "proxy",
        weighted: bool = False,
    ) -> None:
        self.contrast = float(contrast)
        self.dim = tuple(dim)
        self.algorithm = algorithm
        self.weighted = bool(weighted)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        flux = view.flux
        z1, z2 = zscale_limits(
            flux,
            contrast=self.contrast,
            dim=self.dim,
            mask=view.effective_mask(mask),
            ivar=view.ivar,
            weighted=self.weighted,
            algorithm=self.algorithm,
        )
        self._last_state = (z1, z2)
        span = z2 - z1
        out = (flux - z1).div_(span)
        return view.replace(out, ivar=self.divide_ivar(view.ivar, span))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._last_state is None:
            raise RuntimeError(
                "ZScaleNormalize.inverse() requires a prior forward() pass "
                "to capture the per-image limits."
            )
        view = self.view(x)
        z1, z2 = self._last_state
        span = z2 - z1
        # Functional: inverses never mutate their input.
        out = view.flux * span + z1
        return view.replace(out, ivar=self.divide_ivar(view.ivar, 1.0 / span))

    def __repr__(self) -> str:
        return (
            f"ZScaleNormalize(contrast={self.contrast}, dim={self.dim}, "
            f"algorithm={self.algorithm!r}, weighted={self.weighted})"
        )


class RobustNormalize(FITSTransform):
    """Normalise by subtracting the median and dividing by MAD-derived std.

    ``forward`` → ~zero median, unit MAD scale.
    ``inverse`` reverses using the cached statistics.
    """

    propagates_ivar = True
    expects = SCALABLE
    produces = DataState.NORMALIZED
    _last_med = _ThreadedAttr()
    _last_std = _ThreadedAttr()

    def __init__(
        self,
        dim: Tuple[int, ...] = (-2, -1),
        *,
        weighted: bool = False,
    ) -> None:
        self.dim = tuple(dim)
        self.weighted = bool(weighted)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        med, std = estimate_background(
            view.flux,
            dim=self.dim,
            mask=view.effective_mask(mask),
            ivar=view.ivar,
            weighted=self.weighted,
        )
        self._last_med = med
        self._last_std = std
        safe_std = torch.clamp_min(std, 1e-9)
        out = (view.flux - med).div_(safe_std)
        return view.replace(out, ivar=self.divide_ivar(view.ivar, safe_std))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._last_med is None or self._last_std is None:
            raise RuntimeError(
                "RobustNormalize.inverse() requires a prior forward() pass."
            )
        view = self.view(x)
        safe_std = torch.clamp_min(self._last_std, 1e-9)
        out = view.flux * safe_std + self._last_med
        return view.replace(out, ivar=self.divide_ivar(view.ivar, 1.0 / safe_std))

    def __repr__(self) -> str:
        return f"RobustNormalize(dim={self.dim}, weighted={self.weighted})"


class BackgroundSubtract(FITSTransform):
    """Subtract the estimated background (median).

    Pure offset: companion ``ivar`` is unchanged (variance is shift-invariant).
    """

    propagates_ivar = True

    expects = SCALABLE
    _last_bg = _ThreadedAttr()

    def __init__(
        self,
        dim: Tuple[int, ...] = (-2, -1),
        *,
        weighted: bool = False,
    ) -> None:
        self.dim = tuple(dim)
        self.weighted = bool(weighted)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        bg, _ = estimate_background(
            view.flux,
            dim=self.dim,
            mask=view.effective_mask(mask),
            ivar=view.ivar,
            weighted=self.weighted,
        )
        self._last_bg = bg
        return view.replace(view.flux - bg)

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._last_bg is None:
            raise RuntimeError(
                "BackgroundSubtract.inverse() requires a prior forward() pass."
            )
        view = self.view(x)
        return view.replace(view.flux + self._last_bg)

    def __repr__(self) -> str:
        return f"BackgroundSubtract(dim={self.dim}, weighted={self.weighted})"


class PercentileClipNormalize(FITSTransform):
    """Clip to [lower_pct, upper_pct] percentile range, then normalise to [0, 1].

    ``inverse()`` is **approximate**: pixels outside the percentile range were
    clamped and cannot be recovered. Because clipping is nonlinear, a companion
    ``ivar`` is passed through unchanged (and flagged with a warning).

    Parameters
    ----------
    lower_pct : float
        Lower percentile (0–100).
    upper_pct : float
        Upper percentile (0–100).
    dim :
        Dimensions along which percentiles are computed jointly.
    weighted : bool
        Use inverse-variance weighted percentiles when ``ivar`` is present.
    """

    propagates_ivar = False
    expects = SCALABLE
    produces = DataState.NORMALIZED
    _last_state = _ThreadedAttr()

    def __init__(
        self,
        lower_pct: float = 1.0,
        upper_pct: float = 99.0,
        dim: Tuple[int, ...] = (-2, -1),
        *,
        weighted: bool = False,
    ) -> None:
        self.lower_pct = lower_pct / 100.0
        self.upper_pct = upper_pct / 100.0
        self.dim = tuple(dim)
        self.weighted = bool(weighted)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        flux = view.flux
        if view.ivar is not None:
            self._warn_ivar_not_propagated()
        effective = view.effective_mask(mask)
        with torch.no_grad():
            if self.weighted and view.ivar is not None:
                lower = _weighted_quantile(
                    flux, self.lower_pct, self.dim, mask=effective, ivar=view.ivar
                )
                upper = _weighted_quantile(
                    flux, self.upper_pct, self.dim, mask=effective, ivar=view.ivar
                )
            else:
                lower = _quantile(flux, self.lower_pct, self.dim, mask=effective)
                upper = _quantile(flux, self.upper_pct, self.dim, mask=effective)

        self._last_state = (lower, upper)
        clipped = torch.clamp(flux, lower, upper)
        denom = torch.where(upper == lower, torch.ones_like(upper), upper - lower)
        return view.replace((clipped - lower).div_(denom))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._last_state is None:
            raise RuntimeError(
                "PercentileClipNormalize.inverse() requires a prior forward() pass."
            )
        view = self.view(x)
        lower, upper = self._last_state
        return view.replace(view.flux * (upper - lower) + lower)

    def __repr__(self) -> str:
        return (
            f"PercentileClipNormalize("
            f"lower_pct={self.lower_pct * 100:.0f}, "
            f"upper_pct={self.upper_pct * 100:.0f}, "
            f"dim={self.dim}, weighted={self.weighted})"
        )


class MinMaxNormalize(FITSTransform):
    """Normalise to [0, 1] using per-image min / max.

    A group with no valid pixels (all masked or NaN) yields NaN rather than an
    infinity-derived value; masked/NaN pixels themselves stay NaN.
    """

    propagates_ivar = True
    expects = SCALABLE
    produces = DataState.NORMALIZED
    _last_state = _ThreadedAttr()
    _last_span = _ThreadedAttr()

    def __init__(self, dim: Tuple[int, ...] = (-2, -1)) -> None:
        self.dim = tuple(dim)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        flux = view.flux
        effective = view.effective_mask(mask)
        with torch.no_grad():
            vmin = _amin(flux, self.dim, mask=effective)
            vmax = _amax(flux, self.dim, mask=effective)
            # All-masked / all-NaN groups come back as +inf / -inf.
            invalid = ~(torch.isfinite(vmin) & torch.isfinite(vmax))
            vmin = torch.where(invalid, torch.zeros_like(vmin), vmin)
            vmax = torch.where(invalid, torch.zeros_like(vmax), vmax)
            # Data-relative epsilon to avoid float32 underflow on constant images.
            _eps = torch.maximum(
                torch.tensor(1e-6, device=flux.device, dtype=vmin.dtype),
                vmin.abs() * 1e-6,
            )
            span = torch.where(vmin == vmax, _eps, vmax - vmin)
        self._last_state = (vmin, vmax)
        self._last_span = span
        out = (flux - vmin).div_(span)
        out = torch.where(invalid, _nan_like(out, out.dtype), out)
        return view.replace(out, ivar=self.divide_ivar(view.ivar, span))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._last_state is None or self._last_span is None:
            raise RuntimeError(
                "MinMaxNormalize.inverse() requires a prior forward() pass."
            )
        view = self.view(x)
        vmin = self._last_state[0]
        span = self._last_span
        out = view.flux * span + vmin
        return view.replace(out, ivar=self.divide_ivar(view.ivar, 1.0 / span))

    def __repr__(self) -> str:
        return f"MinMaxNormalize(dim={self.dim})"


class GlobalScalarNorm(FITSTransform):
    """Normalise by dividing by a global scalar statistic.

    The simplest linear transform — used by virtually all astronomical
    foundation models (AstroCLIP, SpecFormer, SpecHub) as the only
    preprocessing step.  A neural network's first layer can implicitly
    un-learn this through gradient descent.

    ``inverse`` multiplies by the cached scalar.

    Parameters
    ----------
    stat : str
        Statistic to compute: ``"median"`` (default, robust), ``"max"``,
        ``"mean"``, or ``"rms"``.
    dim :
        Dimensions over which to compute the statistic.  Default ``None``
        (all dims — a single scalar for the whole tensor).
    weighted : bool
        Use inverse-variance weights for ``median`` / ``mean`` / ``rms``.
    """

    propagates_ivar = True
    expects = SCALABLE
    produces = DataState.NORMALIZED
    _scalar = _ThreadedAttr()

    def __init__(
        self,
        stat: str = "median",
        dim: Optional[Tuple[int, ...]] = None,
        *,
        weighted: bool = False,
    ) -> None:
        if stat not in ("median", "max", "mean", "rms"):
            raise ValueError("stat must be 'median', 'max', 'mean', or 'rms'")
        self.stat = stat
        self.dim = dim
        self.weighted = bool(weighted)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        flux = view.flux
        effective = view.effective_mask(mask)
        dim = self.dim if self.dim is not None else tuple(range(flux.ndim))
        dims = tuple(dim)
        with torch.no_grad():
            if self.stat == "median":
                scalar = (
                    _weighted_quantile(flux, 0.5, dims, mask=effective, ivar=view.ivar)
                    if self.weighted and view.ivar is not None
                    else _median(flux, dims, mask=effective)
                )
            elif self.stat == "max":
                scalar = _amax(flux, dims, mask=effective)
            else:  # mean / rms
                xf = _stats_upcast(flux)
                # Exclude user-masked AND non-finite values from the
                # statistic: a single NaN must not poison the whole frame.
                valid = torch.isfinite(xf)
                if effective is not None:
                    valid = valid & effective.to(torch.bool)
                if self.weighted and view.ivar is not None:
                    w = _stats_upcast(view.ivar).to(xf.dtype)
                    w = torch.where(torch.isfinite(w) & (w > 0), w, torch.zeros_like(w))
                    w = torch.where(valid, w, torch.zeros_like(w))
                else:
                    w = valid.to(xf.dtype)
                total_w = w.sum(dim=dims, keepdim=True)
                has_data = total_w > 0
                if self.stat == "mean":
                    data = xf
                else:  # rms
                    data = xf * xf
                numer = (torch.where(valid, data, torch.zeros_like(xf)) * w).sum(
                    dim=dims, keepdim=True
                )
                denom = torch.where(has_data, total_w, torch.ones_like(total_w))
                # Keep the statistic in the stats dtype: casting it to an
                # integer flux dtype truncates the divisor (and the NaN fill
                # below cannot even be built in an integer dtype).
                scalar = numer / denom
                if self.stat == "rms":
                    scalar = torch.sqrt(torch.clamp_min(scalar, 0.0))
                scalar = torch.where(has_data, scalar, _nan_like(scalar, scalar.dtype))
            # Sign-preserving floor: a negative statistic (e.g. max of a
            # negative background) must divide by itself — clamping it up to
            # +1e-30 produced ~1e30-scale garbage. Non-finite statistics
            # (all-masked groups) fall back to 1.0 (identity) instead.
            floored = torch.where(
                scalar < 0,
                torch.clamp(scalar, max=-1e-30),
                torch.clamp_min(scalar, 1e-30),
            )
            divisor = torch.where(
                torch.isfinite(scalar), floored, torch.ones_like(scalar)
            )
        # Cache the divisor actually used so inverse() round-trips exactly.
        self._scalar = divisor
        return view.replace(flux / divisor, ivar=self.divide_ivar(view.ivar, divisor))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._scalar is None:
            raise RuntimeError(
                "GlobalScalarNorm.inverse() requires a prior forward() pass."
            )
        view = self.view(x)
        return view.replace(
            view.flux * self._scalar,
            ivar=self.divide_ivar(view.ivar, 1.0 / self._scalar),
        )

    def __repr__(self) -> str:
        return (
            f"GlobalScalarNorm(stat={self.stat!r}, dim={self.dim}, "
            f"weighted={self.weighted})"
        )


class SigmaNormalize(FITSTransform):
    """Divide by the robust per-group background dispersion (sigma).

    The normalization foundation models actually use for images: rescale so
    sky noise is unit variance while preserving relative flux and, in
    zero-preserving mode, colour ratios.

    Parameters
    ----------
    dim :
        Dimensions over which sigma is estimated.
    stat : str
        ``"mad"`` (default) — MAD × 1.4826 from :func:`estimate_background`;
        ``"std"`` — population RMS about the median.
    zero_preserving : bool
        ``True`` (default) divides ``flux / sigma`` without subtracting an
        offset, so zero flux and band ratios survive exactly. ``False``
        centers first: ``(flux - median) / sigma``.
    eps : float
        Floor on the divisor for constant groups.
    weighted : bool
        Use inverse-variance weights for the dispersion.
    """

    propagates_ivar = True
    expects = SCALABLE
    produces = DataState.NORMALIZED
    _last_scale = _ThreadedAttr()
    _last_offset = _ThreadedAttr()

    def __init__(
        self,
        dim: Tuple[int, ...] = (-2, -1),
        *,
        stat: str = "mad",
        zero_preserving: bool = True,
        eps: float = 1e-12,
        weighted: bool = False,
    ) -> None:
        if stat not in ("mad", "std"):
            raise ValueError("stat must be 'mad' or 'std'")
        self.dim = tuple(dim)
        self.stat = stat
        self.zero_preserving = bool(zero_preserving)
        self.eps = float(eps)
        self.weighted = bool(weighted)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        flux = view.flux
        effective = view.effective_mask(mask)
        med, mad = estimate_background(
            flux,
            dim=self.dim,
            mask=effective,
            ivar=view.ivar,
            weighted=self.weighted,
        )
        if self.stat == "mad":
            sigma = mad
        else:
            sigma = _weighted_dispersion(
                flux,
                med,
                self.dim,
                mask=effective,
                ivar=view.ivar if self.weighted else None,
            )
        scale = torch.clamp_min(sigma, self.eps)
        offset = None if self.zero_preserving else med
        self._last_scale = scale
        self._last_offset = offset
        numer = flux if offset is None else (flux - offset)
        return view.replace(numer / scale, ivar=self.divide_ivar(view.ivar, scale))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._last_scale is None:
            raise RuntimeError(
                "SigmaNormalize.inverse() requires a prior forward() pass."
            )
        view = self.view(x)
        scale = self._last_scale.to(dtype=view.flux.dtype, device=view.flux.device)
        out = view.flux * scale
        if self._last_offset is not None:
            out = out + self._last_offset.to(
                dtype=view.flux.dtype, device=view.flux.device
            )
        return view.replace(out, ivar=self.divide_ivar(view.ivar, 1.0 / scale))

    def __repr__(self) -> str:
        return (
            f"SigmaNormalize(dim={self.dim}, stat={self.stat!r}, "
            f"zero_preserving={self.zero_preserving}, weighted={self.weighted})"
        )


class InterquantileScale(FITSTransform):
    """Zero-preserving or centered interquantile scale normalisation.

    Computes a robust spread:
        s = Q(q_high) - Q(q_low)
    (e.g., IQR with q=(0.25, 0.75) or 90% range with q=(0.05, 0.95)) over the
    specified reduction dimensions.

    In zero-preserving mode (default), the transform divides by *s* directly:
    ``x / s``.  Because no additive offset is subtracted, zero-flux stays at
    zero and relative channel ratios (astronomical colours: g - r, r - i)
    remain strictly invariant when *dim* includes the channel axis (e.g.
    ``dim=None`` or ``dim=(-3, -2, -1)``).

    When ``zero_preserving=False``, data is centered by subtracting the
    median: ``(x - median) / s``.

    Supports raw :class:`torch.Tensor` inputs, companion dictionary
    payloads ``{"flux": Tensor, "ivar"?: Tensor, "mask"?: Tensor}`` and
    :class:`~torchfits.transforms.state.Payload`.

    Parameters
    ----------
    q_low : float, default 0.05
        Lower quantile (0.0 to 1.0).
    q_high : float, default 0.95
        Upper quantile (0.0 to 1.0). Must be strictly greater than *q_low*.
    dim : tuple[int, ...] or None, default None
        Dimensions over which quantiles are computed jointly.
    zero_preserving : bool, default True
        If ``True``, scale without subtracting an offset.
    eps : float, default 1e-9
        Minimum divisor floor.
    weighted : bool, default False
        Use inverse-variance weighted quantiles when ``ivar`` is present.
    """

    propagates_ivar = True
    expects = SCALABLE
    produces = DataState.NORMALIZED
    _last_scale = _ThreadedAttr()
    _last_offset = _ThreadedAttr()

    def __init__(
        self,
        q_low: float = 0.05,
        q_high: float = 0.95,
        dim: Optional[Tuple[int, ...]] = None,
        zero_preserving: bool = True,
        eps: float = 1e-9,
        *,
        weighted: bool = False,
    ) -> None:
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError(
                f"Expected 0.0 <= q_low < q_high <= 1.0, got q_low={q_low}, q_high={q_high}"
            )
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.dim = tuple(dim) if dim is not None else None
        self.zero_preserving = bool(zero_preserving)
        self.eps = float(eps)
        self.weighted = bool(weighted)

    def _forward_flux(
        self,
        flux: torch.Tensor,
        mask: torch.Tensor | None,
        ivar: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xf = _stats_upcast(flux)
        dim = self.dim if self.dim is not None else tuple(range(xf.ndim))
        with torch.no_grad():
            if self.weighted and ivar is not None:
                q_lo = _weighted_quantile(xf, self.q_low, dim, mask=mask, ivar=ivar)
                q_hi = _weighted_quantile(xf, self.q_high, dim, mask=mask, ivar=ivar)
            else:
                q_lo = _quantile(xf, self.q_low, dim, mask=mask)
                q_hi = _quantile(xf, self.q_high, dim, mask=mask)
            scale = torch.clamp_min(q_hi - q_lo, self.eps).to(xf.dtype)
            offset = (
                None
                if self.zero_preserving
                else _median(xf, dim, mask=mask).to(xf.dtype)
            )

        self._last_scale = scale
        self._last_offset = offset
        if offset is not None:
            return (xf - offset) / scale, scale
        return xf / scale, scale

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        scaled, scale = self._forward_flux(
            view.flux, view.effective_mask(mask), view.ivar
        )
        return view.replace(scaled, ivar=self.divide_ivar(view.ivar, scale))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._last_scale is None:
            raise RuntimeError(
                "InterquantileScale.inverse() requires a prior forward() pass."
            )
        view = self.view(x)
        flux = view.flux
        scale = self._last_scale.to(dtype=flux.dtype, device=flux.device)
        restored = flux * scale
        if self._last_offset is not None:
            restored = restored + self._last_offset.to(
                dtype=flux.dtype, device=flux.device
            )
        return view.replace(restored, ivar=self.divide_ivar(view.ivar, 1.0 / scale))

    def __repr__(self) -> str:
        return (
            f"InterquantileScale(q_low={self.q_low}, q_high={self.q_high}, "
            f"dim={self.dim}, zero_preserving={self.zero_preserving}, "
            f"weighted={self.weighted})"
        )


InterquantileNormalize = InterquantileScale
