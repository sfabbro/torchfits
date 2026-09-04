from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from .base import FITSTransform
from .helpers import (
    _median,
    _amin,
    _amax,
    _quantile,
    _stats_upcast,
    estimate_background,
    zscale_limits,
)


class ZScaleNormalize(FITSTransform):
    """IRAF zscale auto-contrast normalisation.

    ``forward`` maps data to [0, 1] using dynamically computed limits.
    ``inverse`` uses the limits from the most recent forward pass.
    """

    def __init__(self, contrast: float = 0.25, dim: Tuple[int, ...] = (-2, -1)) -> None:
        self.contrast = float(contrast)
        self.dim = tuple(dim)
        self._last_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        z1, z2 = zscale_limits(x, contrast=self.contrast, dim=self.dim, mask=mask)
        self._last_state = (z1, z2)
        return (x - z1).div_(z2 - z1)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._last_state is None:
            raise RuntimeError(
                "ZScaleNormalize.inverse() requires a prior forward() pass "
                "to capture the per-image limits."
            )
        z1, z2 = self._last_state
        # Functional: inverses never mutate their input (M6).
        return x * (z2 - z1) + z1

    def __repr__(self) -> str:
        return f"ZScaleNormalize(contrast={self.contrast}, dim={self.dim})"


class RobustNormalize(FITSTransform):
    """Normalise by subtracting the median and dividing by MAD-derived std.

    ``forward`` → ~zero median, unit MAD scale.
    ``inverse`` reverses using the cached statistics.
    """

    def __init__(self, dim: Tuple[int, ...] = (-2, -1)) -> None:
        self.dim = tuple(dim)
        self._last_med: Optional[torch.Tensor] = None
        self._last_std: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        med, std = estimate_background(x, dim=self.dim, mask=mask)
        self._last_med = med
        self._last_std = std
        return (x - med).div_(torch.clamp_min(std, 1e-9))

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._last_med is None or self._last_std is None:
            raise RuntimeError(
                "RobustNormalize.inverse() requires a prior forward() pass."
            )
        return x * self._last_std + self._last_med

    def __repr__(self) -> str:
        return f"RobustNormalize(dim={self.dim})"


class BackgroundSubtract(FITSTransform):
    """Subtract the estimated background (median)."""

    def __init__(self, dim: Tuple[int, ...] = (-2, -1)) -> None:
        self.dim = tuple(dim)
        self._last_bg: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bg, _ = estimate_background(x, dim=self.dim, mask=mask)
        self._last_bg = bg
        return x - bg

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._last_bg is None:
            raise RuntimeError(
                "BackgroundSubtract.inverse() requires a prior forward() pass."
            )
        return x + self._last_bg

    def __repr__(self) -> str:
        return f"BackgroundSubtract(dim={self.dim})"


class PercentileClipNormalize(FITSTransform):
    """Clip to [lower_pct, upper_pct] percentile range, then normalise to [0, 1].

    Parameters
    ----------
    lower_pct : float
        Lower percentile (0–100).
    upper_pct : float
        Upper percentile (0–100).
    dim :
        Dimensions along which percentiles are computed jointly.
    """

    def __init__(
        self,
        lower_pct: float = 1.0,
        upper_pct: float = 99.0,
        dim: Tuple[int, ...] = (-2, -1),
    ) -> None:
        self.lower_pct = lower_pct / 100.0
        self.upper_pct = upper_pct / 100.0
        self.dim = tuple(dim)
        self._last_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            lower = _quantile(x, self.lower_pct, self.dim, mask=mask)
            upper = _quantile(x, self.upper_pct, self.dim, mask=mask)

        self._last_state = (lower, upper)
        clipped = torch.clamp(x, lower, upper)
        denom = torch.where(upper == lower, torch.ones_like(upper), upper - lower)
        return (clipped - lower).div_(denom)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._last_state is None:
            raise RuntimeError(
                "PercentileClipNormalize.inverse() requires a prior forward() pass."
            )
        lower, upper = self._last_state
        return x * (upper - lower) + lower

    def __repr__(self) -> str:
        return (
            f"PercentileClipNormalize("
            f"lower_pct={self.lower_pct * 100:.0f}, "
            f"upper_pct={self.upper_pct * 100:.0f}, "
            f"dim={self.dim})"
        )


class MinMaxNormalize(FITSTransform):
    """Normalise to [0, 1] using per-image min / max."""

    def __init__(self, dim: Tuple[int, ...] = (-2, -1)) -> None:
        self.dim = tuple(dim)
        self._last_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            vmin = _amin(x, self.dim, mask=mask)
            vmax = _amax(x, self.dim, mask=mask)
            # Data-relative epsilon to avoid float32 underflow on constant images.
            _eps = torch.maximum(
                torch.tensor(1e-6, device=x.device, dtype=vmin.dtype),
                vmin.abs() * 1e-6,
            )
            vmax = torch.where(vmin == vmax, vmin + _eps, vmax)
        self._last_state = (vmin, vmax)
        return (x - vmin).div_(vmax - vmin)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._last_state is None:
            raise RuntimeError(
                "MinMaxNormalize.inverse() requires a prior forward() pass."
            )
        vmin, vmax = self._last_state
        return x * (vmax - vmin) + vmin

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
    """

    def __init__(
        self, stat: str = "median", dim: Optional[Tuple[int, ...]] = None
    ) -> None:
        if stat not in ("median", "max", "mean", "rms"):
            raise ValueError("stat must be 'median', 'max', 'mean', or 'rms'")
        self.stat = stat
        self.dim = dim
        self._scalar: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        dim = self.dim if self.dim is not None else tuple(range(x.ndim))
        with torch.no_grad():
            if self.stat == "median":
                scalar = _median(x, dim, mask=mask)
            elif self.stat == "max":
                scalar = _amax(x, dim, mask=mask)
            else:  # mean / rms
                xf = x.float() if x.dtype != torch.int64 else x.double()
                # Exclude user-masked AND non-finite values from the
                # statistic: a single NaN must not poison the whole frame.
                valid = torch.isfinite(xf)
                if mask is not None:
                    valid = valid & mask.to(torch.bool)
                if self.stat == "mean":
                    total = torch.where(valid, xf, torch.zeros_like(xf)).sum(
                        dim=dim, keepdim=True
                    )
                    count = valid.to(xf.dtype).sum(dim=dim, keepdim=True)
                    scalar = (total / torch.clamp_min(count, 1.0)).to(x.dtype)
                else:  # rms
                    sq = torch.where(valid, xf * xf, torch.zeros_like(xf))
                    count = valid.to(xf.dtype).sum(dim=dim, keepdim=True)
                    scalar = torch.sqrt(
                        sq.sum(dim=dim, keepdim=True) / torch.clamp_min(count, 1.0)
                    ).to(x.dtype)
            # Sign-preserving floor: a negative statistic (e.g. max of a
            # negative background) must divide by itself — clamping it up to
            # +1e-30 produced ~1e30-scale garbage.
            divisor = torch.where(
                scalar < 0,
                torch.clamp(scalar, max=-1e-30),
                torch.clamp_min(scalar, 1e-30),
            )
        # Cache the divisor actually used so inverse() round-trips exactly.
        self._scalar = divisor
        return x / divisor

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._scalar is None:
            raise RuntimeError(
                "GlobalScalarNorm.inverse() requires a prior forward() pass."
            )
        return x * self._scalar

    def __repr__(self) -> str:
        return f"GlobalScalarNorm(stat={self.stat!r}, dim={self.dim})"


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

    Supports both raw :class:`torch.Tensor` inputs and companion dictionary
    payloads ``{"flux": Tensor, "ivar"?: Tensor, "mask"?: Tensor}``.
    When an inverse-variance companion is present, it is scaled by ``s**2``
    so that signal-to-noise ratios are preserved.

    Parameters
    ----------
    q_low : float, default 0.05
        Lower quantile (0.0 to 1.0).
    q_high : float, default 0.95
        Upper quantile (0.0 to 1.0). Must be strictly greater than *q_low*.
    dim : tuple[int, ...] or None, default None
        Dimensions over which quantiles are computed jointly. Default ``None``
        pools across all dimensions of the input tensor.
    zero_preserving : bool, default True
        If ``True``, scale without subtracting an offset (preserves zero-point
        and colour flux ratios). If ``False``, subtracts the median.
    eps : float, default 1e-9
        Minimum divisor floor to avoid division by zero on constant images.
    """

    def __init__(
        self,
        q_low: float = 0.05,
        q_high: float = 0.95,
        dim: Optional[Tuple[int, ...]] = None,
        zero_preserving: bool = True,
        eps: float = 1e-9,
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
        self._last_scale: Optional[torch.Tensor] = None
        self._last_offset: Optional[torch.Tensor] = None

    def _forward_tensor(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xf = _stats_upcast(x)
        dim = self.dim if self.dim is not None else tuple(range(xf.ndim))
        with torch.no_grad():
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
        if isinstance(x, dict):
            flux = x.get("flux")
            if flux is None or not torch.is_tensor(flux):
                raise ValueError("Dict payload requires a torch.Tensor 'flux' field")
            m = mask if mask is not None else x.get("mask")
            m_t = m if torch.is_tensor(m) else None
            scaled_flux, scale = self._forward_tensor(flux, mask=m_t)
            out = {**x, "flux": scaled_flux}
            if "ivar" in x and x["ivar"] is not None and torch.is_tensor(x["ivar"]):
                ivar_t = x["ivar"]
                scale_for_ivar = scale.to(device=ivar_t.device, dtype=ivar_t.dtype)
                out["ivar"] = ivar_t * (scale_for_ivar**2)
            return out

        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor or dict, got {type(x)}")
        scaled_x, _ = self._forward_tensor(x, mask=mask)
        return scaled_x

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._last_scale is None:
            raise RuntimeError(
                "InterquantileScale.inverse() requires a prior forward() pass."
            )
        if isinstance(x, dict):
            flux = x.get("flux")
            if flux is None or not torch.is_tensor(flux):
                raise ValueError("Dict payload requires a torch.Tensor 'flux' field")
            scale = self._last_scale.to(dtype=flux.dtype, device=flux.device)
            restored = flux * scale
            if self._last_offset is not None:
                restored = restored + self._last_offset.to(
                    dtype=flux.dtype, device=flux.device
                )
            out = {**x, "flux": restored}
            if "ivar" in x and x["ivar"] is not None and torch.is_tensor(x["ivar"]):
                ivar_t = x["ivar"]
                ivar_scale = scale.to(device=ivar_t.device, dtype=ivar_t.dtype)
                out["ivar"] = ivar_t / (ivar_scale**2)
            return out

        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor or dict, got {type(x)}")
        scale = self._last_scale.to(dtype=x.dtype, device=x.device)
        restored = x * scale
        if self._last_offset is not None:
            restored = restored + self._last_offset.to(dtype=x.dtype, device=x.device)
        return restored

    def __repr__(self) -> str:
        return (
            f"InterquantileScale(q_low={self.q_low}, q_high={self.q_high}, "
            f"dim={self.dim}, zero_preserving={self.zero_preserving})"
        )


InterquantileNormalize = InterquantileScale
