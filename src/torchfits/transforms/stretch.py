from __future__ import annotations

import math

import torch

from .base import FITSTransform
from .helpers import (
    _upcast_for_precision,
    safe_arcsinh,
)


class ArcsinhStretch(FITSTransform):
    """Lupton+ (2004) arcsinh stretch — the standard for high-DR astronomy.

    ``forward`` computes ``arcsinh(a * x) / arcsinh(a)``, which maps
    [0, 1] → [0, 1] for non-negative inputs when *a* is tuned to the
    data range.  For general inputs the output is not clamped.

    Parameters
    ----------
    a : float
        Softening parameter.  Smaller values = more linear near zero.
    """

    def __init__(self, a: float = 1.0) -> None:
        self.a = float(a)
        if self.a <= 0:
            raise ValueError(f"ArcsinhStretch 'a' must be > 0, got {self.a}")
        self._norm = math.asinh(self.a)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return safe_arcsinh(x, self.a).div_(self._norm)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return (
            torch.sinh(_upcast_for_precision(x) * self._norm).div_(self.a).to(x.dtype)
        )

    def __repr__(self) -> str:
        return f"ArcsinhStretch(a={self.a})"


class LogStretch(FITSTransform):
    """Logarithmic stretch, safe for heavy-tailed flux distributions.

    .. note::
       Negative values are silently clamped to zero.  If your data may
       contain negatives (e.g. after sky subtraction), consider applying
       :class:`BackgroundSubtract` with an appropriate sky estimate first.

    Parameters
    ----------
    a : float
        Scale factor applied before the log.  Larger ``a`` compresses the
        low-flux region more gently.
    eps : float
        Floor value to prevent ``log(0)``.
    """

    def __init__(self, a: float = 1000.0, eps: float = 1e-9) -> None:
        self.a = float(a)
        if self.a <= 0:
            raise ValueError(f"LogStretch 'a' must be > 0, got {self.a}")
        self.eps = float(eps)
        self._norm = math.log10(1.0 + self.a)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Upcast BEFORE 1 + a*x: in float16, a*x overflows to inf for x > ~65
        # (a=1000) before safe_log's internal upcast could take effect.
        orig_dtype = x.dtype
        xu = torch.clamp_min(_upcast_for_precision(x), 0.0)
        out = torch.log10(torch.clamp_min(1.0 + self.a * xu, self.eps))
        return out.div_(self._norm).to(orig_dtype)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_up = _upcast_for_precision(x)
        # Clamp the exponent to avoid overflow: 10^37.9 leaves headroom for
        # float32 (max ~3.4e38), 10^308 for float64.
        max_safe_exp = 37.9 if x_up.dtype == torch.float32 else 308.0
        exponent = (x_up * self._norm).clamp_max(max_safe_exp)
        val = torch.pow(10.0, exponent).sub_(1.0)
        return val.div_(self.a).to(orig_dtype)

    def __repr__(self) -> str:
        return f"LogStretch(a={self.a}, eps={self.eps})"


class SqrtStretch(FITSTransform):
    """Square-root stretch — stabilises Poisson variance.

    .. note::
       Negative values are silently clamped to zero.
    """

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(x, 0.0))

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return x.pow(2)

    def __repr__(self) -> str:
        return "SqrtStretch()"


# ---------------------------------------------------------------------------
# Normalizers (data-dependent — cache state for inverse)
# ---------------------------------------------------------------------------
