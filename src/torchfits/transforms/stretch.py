from __future__ import annotations

import math
from typing import cast

import torch

from .base import FITSTransform
from .helpers import _stretch_dtype, _upcast_for_precision, safe_arcsinh


class ArcsinhStretch(FITSTransform):
    """Lupton+ (2004) arcsinh stretch — the standard for high-DR astronomy.

    ``forward`` computes ``arcsinh(a * x) / arcsinh(a)``, which maps
    ``[0, 1] → [0, 1]`` exactly (``f(0) = 0``, ``f(1) = 1``) and keeps
    brighter values distinguishable instead of saturating them.

    Integer inputs are promoted to float32 — the stretch is *never* silently
    truncated back to the storage dtype (which would collapse the result to
    0/1/2).

    Parameters
    ----------
    a : float
        Softening parameter.  Smaller values = more linear near zero.
    propagate_ivar : bool
        Propagate a companion ``ivar`` through the stretch by the delta method
        (``d/dx arcsinh(a x) = a / sqrt(1 + (a x)**2)``), so the uncertainty
        stays consistent with the stretched flux.  Off by default, where
        ``ivar`` is passed through unchanged and a warning is emitted.
    """

    propagates_ivar = False

    def __init__(self, a: float = 1.0, *, propagate_ivar: bool = False) -> None:
        self.a = float(a)
        if self.a <= 0:
            raise ValueError(f"ArcsinhStretch 'a' must be > 0, got {self.a}")
        self._norm = math.asinh(self.a)
        self.propagate_ivar = bool(propagate_ivar)
        self.propagates_ivar = self.propagate_ivar

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        view = self.view(x)
        out = safe_arcsinh(view.flux, self.a).div_(self._norm)
        if view.ivar is None:
            return cast(torch.Tensor, view.replace(out))
        if not self.propagate_ivar:
            self._warn_ivar_not_propagated()
            return cast(torch.Tensor, view.replace(out))
        # d/dx [asinh(a x) / asinh(a)] = a / (sqrt(1 + (a x)^2) asinh(a))
        ax = _upcast_for_precision(view.flux) * self.a
        slope = self.a / (torch.sqrt(1.0 + ax * ax) * self._norm)
        return cast(
            torch.Tensor,
            view.replace(out, ivar=self.delta_ivar(view.ivar, slope)),
        )

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        view = self.view(x)
        flux = view.flux
        return cast(
            torch.Tensor,
            view.replace(
                torch.sinh(_upcast_for_precision(flux) * self._norm)
                .div_(self.a)
                .to(_stretch_dtype(flux))
            ),
        )

    def __repr__(self) -> str:
        return f"ArcsinhStretch(a={self.a})"


class LogStretch(FITSTransform):
    """Logarithmic stretch, safe for heavy-tailed flux distributions.

    .. note::
       Negative values are silently clamped to zero.  If your data may
       contain negatives (e.g. after sky subtraction), consider applying
       :class:`BackgroundSubtract` with an appropriate sky estimate first.

    Integer inputs return float32 (never a truncated integer result).

    Parameters
    ----------
    a : float
        Scale factor applied before the log.  Larger ``a`` compresses the
        low-flux region more gently.
    eps : float
        Floor value to prevent ``log(0)``. Since the argument is
        ``1 + a * max(x, 0) >= 1`` this only matters for pathological inputs.
    propagate_ivar : bool
        Propagate a companion ``ivar`` through the stretch by the delta method.
        Off by default, where ``ivar`` is passed through unchanged and a
        warning is emitted.
    """

    propagates_ivar = False

    def __init__(
        self,
        a: float = 1000.0,
        eps: float = 1e-9,
        *,
        propagate_ivar: bool = False,
    ) -> None:
        self.a = float(a)
        if self.a <= 0:
            raise ValueError(f"LogStretch 'a' must be > 0, got {self.a}")
        self.eps = float(eps)
        self._norm = math.log10(1.0 + self.a)
        self.propagate_ivar = bool(propagate_ivar)
        self.propagates_ivar = self.propagate_ivar

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        view = self.view(x)
        flux = view.flux
        # Upcast BEFORE 1 + a*x: in float16, a*x overflows to inf for x > ~65
        # (a=1000) before safe_log's internal upcast could take effect.
        orig_dtype = _stretch_dtype(flux)
        xu = torch.clamp_min(_upcast_for_precision(flux), 0.0)
        out = torch.log10(torch.clamp_min(1.0 + self.a * xu, self.eps))
        out = out.div_(self._norm).to(orig_dtype)
        if view.ivar is None:
            return cast(torch.Tensor, view.replace(out))
        if not self.propagate_ivar:
            self._warn_ivar_not_propagated()
            return cast(torch.Tensor, view.replace(out))
        # d/dx [log10(1 + a x) / log10(1 + a)] = a / ((1 + a x) ln(1 + a)).
        # x < 0 is clamped flat, so its slope — and its information — is zero.
        slope = torch.where(
            flux > 0,
            self.a / ((1.0 + self.a * xu) * math.log1p(self.a)),
            torch.zeros_like(xu),
        )
        return cast(
            torch.Tensor,
            view.replace(out, ivar=self.delta_ivar(view.ivar, slope)),
        )

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        view = self.view(x)
        flux = view.flux
        orig_dtype = _stretch_dtype(flux)
        x_up = _upcast_for_precision(flux)
        # Clamp the exponent to avoid overflow: 10^37.9 leaves headroom for
        # float32 (max ~3.4e38), 10^308 for float64.
        max_safe_exp = 37.9 if x_up.dtype == torch.float32 else 308.0
        exponent = (x_up * self._norm).clamp_max(max_safe_exp)
        val = torch.pow(10.0, exponent).sub_(1.0)
        return cast(torch.Tensor, view.replace(val.div_(self.a).to(orig_dtype)))

    def __repr__(self) -> str:
        return f"LogStretch(a={self.a}, eps={self.eps})"


class SqrtStretch(FITSTransform):
    """Square-root stretch — stabilises Poisson variance.

    .. note::
       Negative values are silently clamped to zero.

    Because ``sqrt`` is variance-stabilising for Poisson counts (for which
    ``var(x) = x``), enabling ``propagate_ivar`` yields a nearly constant
    inverse variance of ``4`` across the frame — the classic result that the
    stretched data have unit-ish noise regardless of flux level.

    Integer inputs return float32 on both passes (never a truncated or
    wrapped integer result), matching :class:`ArcsinhStretch` and
    :class:`LogStretch`.

    Parameters
    ----------
    propagate_ivar : bool
        Propagate a companion ``ivar`` through the stretch by the delta method
        (``d/dx sqrt(x) = 1 / (2 sqrt(x))``).  Off by default, where ``ivar``
        is passed through unchanged and a warning is emitted.
    """

    propagates_ivar = False

    def __init__(self, *, propagate_ivar: bool = False) -> None:
        self.propagate_ivar = bool(propagate_ivar)
        self.propagates_ivar = self.propagate_ivar

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        view = self.view(x)
        flux = view.flux
        out = torch.sqrt(torch.clamp_min(_upcast_for_precision(flux), 0.0)).to(
            _stretch_dtype(flux)
        )
        if view.ivar is None:
            return cast(torch.Tensor, view.replace(out))
        if not self.propagate_ivar:
            self._warn_ivar_not_propagated()
            return cast(torch.Tensor, view.replace(out))
        # d/dx sqrt(x) = 1 / (2 sqrt(x)). The clamp makes x <= 0 flat; keep the
        # denominator off zero so a clamped pixel cannot emit inf/NaN slopes.
        xu = torch.clamp_min(_upcast_for_precision(flux), 0.0)
        denom = torch.sqrt(torch.clamp_min(xu, torch.finfo(xu.dtype).tiny))
        slope = torch.where(flux > 0, 0.5 / denom, torch.zeros_like(xu))
        return cast(
            torch.Tensor,
            view.replace(out, ivar=self.delta_ivar(view.ivar, slope)),
        )

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        view = self.view(x)
        flux = view.flux
        val = torch.square(_upcast_for_precision(flux))
        return cast(torch.Tensor, view.replace(val.to(_stretch_dtype(flux))))

    def __repr__(self) -> str:
        return "SqrtStretch()"
