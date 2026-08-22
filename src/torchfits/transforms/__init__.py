"""Machine-learning friendly transformations for FITS images and high-DR data.

All transforms implement the :class:`FITSTransform` callable protocol
(``forward`` / ``inverse`` / ``__call__``). They are **not**
``torch.nn.Module`` subclasses; wrap with :func:`as_module` for
``nn.Sequential``.

Inverse state is **instance-local**. Construct one pipeline per DataLoader
worker when ``num_workers > 0``.

Import from this package only::

    from torchfits.transforms import ArcsinhStretch, Compose, ZScaleNormalize
"""

from __future__ import annotations

from .base import AsModule, Compose, FITSTransform, as_module
from .clip import AsymmetricSigmaClip, SigmaClip
from .fits_meta import (
    FITSHeaderNormalize,
    FITSHeaderScale,
    FITSScaleColumns,
    TNullToNan,
)
from .helpers import estimate_background, safe_arcsinh, safe_log, zscale_limits
from .normalize import (
    BackgroundSubtract,
    GlobalScalarNorm,
    MinMaxNormalize,
    PercentileClipNormalize,
    RobustNormalize,
    ZScaleNormalize,
)
from .rgb import lupton_rgb, rgb
from .stretch import ArcsinhStretch, LogStretch, SqrtStretch

__all__ = [
    "FITSTransform",
    "Compose",
    "AsModule",
    "as_module",
    "ArcsinhStretch",
    "LogStretch",
    "SqrtStretch",
    "lupton_rgb",
    "rgb",
    "ZScaleNormalize",
    "RobustNormalize",
    "BackgroundSubtract",
    "PercentileClipNormalize",
    "MinMaxNormalize",
    "GlobalScalarNorm",
    "FITSHeaderScale",
    "FITSScaleColumns",
    "TNullToNan",
    "FITSHeaderNormalize",
    "SigmaClip",
    "AsymmetricSigmaClip",
    "safe_arcsinh",
    "safe_log",
    "estimate_background",
    "zscale_limits",
]
