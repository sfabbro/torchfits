"""Machine-learning friendly transformations for FITS images and high-DR data.

All transforms implement the :class:`FITSTransform` callable protocol
(``forward`` / ``inverse`` / ``__call__``). They are **not**
``torch.nn.Module`` subclasses; wrap with :func:`as_module` for
``nn.Sequential``.

Inputs may be a bare tensor or a companion payload
(``{"flux", "ivar"?, "mask"?}`` or :class:`Payload`). Transforms propagate
``ivar`` exactly through linear operations and declare the processing state
they expect, so calibrated data is never scaled twice.

Inverse state is **instance-local**. Construct one pipeline per DataLoader
worker when ``num_workers > 0``.

Import from this package only::

    from torchfits.transforms import ArcsinhStretch, Compose, ZScaleNormalize
"""

from __future__ import annotations

from .background import MeshBackgroundSubtract
from .base import AsModule, Compose, FITSTransform, as_module
from .clip import AsymmetricSigmaClip, SigmaClip
from .fits_meta import (
    FITSHeaderNormalize,
    FITSHeaderScale,
    FITSScaleColumns,
    TNullToNan,
)
from .helpers import estimate_background, safe_arcsinh, safe_log, zscale_limits
from .mask import (
    apply_mask,
    combine_masks,
    mask_from_dq,
    mask_from_ivar,
    mask_from_nan,
)
from .normalize import (
    AffineTransform,
    BackgroundSubtract,
    GlobalScalarNorm,
    InterquantileNormalize,
    InterquantileScale,
    MinMaxNormalize,
    PercentileClipNormalize,
    RobustNormalize,
    SigmaNormalize,
    ZScaleNormalize,
)
from .rgb import lupton_rgb, rgb
from .state import (
    DataState,
    DataStateError,
    Payload,
    calibration_state,
)
from .stretch import ArcsinhStretch, LogStretch, SqrtStretch

__all__ = [
    # Protocol / composition
    "FITSTransform",
    "Compose",
    "AsModule",
    "as_module",
    # State contract
    "DataState",
    "DataStateError",
    "Payload",
    "calibration_state",
    # Stretches
    "ArcsinhStretch",
    "LogStretch",
    "SqrtStretch",
    # Normalizers
    "AffineTransform",
    "ZScaleNormalize",
    "RobustNormalize",
    "SigmaNormalize",
    "BackgroundSubtract",
    "MeshBackgroundSubtract",
    "PercentileClipNormalize",
    "MinMaxNormalize",
    "GlobalScalarNorm",
    "InterquantileScale",
    "InterquantileNormalize",
    # Outlier rejection
    "SigmaClip",
    "AsymmetricSigmaClip",
    # FITS metadata
    "FITSHeaderScale",
    "FITSScaleColumns",
    "TNullToNan",
    "FITSHeaderNormalize",
    # Masks
    "mask_from_dq",
    "mask_from_ivar",
    "mask_from_nan",
    "combine_masks",
    "apply_mask",
    # RGB
    "lupton_rgb",
    "rgb",
    # Helpers
    "safe_arcsinh",
    "safe_log",
    "estimate_background",
    "zscale_limits",
]
