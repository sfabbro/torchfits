# Type stubs for transforms_reference — reference implementation for parity testing.

from __future__ import annotations

import numpy as np
import torch

def sigma_clip_naive(
    x: torch.Tensor,
    n_sigma: float,
    max_iter: int,
    dims: tuple[int, ...],
    fill: str,
) -> torch.Tensor: ...
def weighted_quantile_naive(
    x: torch.Tensor,
    q: float,
    dim: tuple[int, ...],
    mask: torch.Tensor | None = None,
    ivar: torch.Tensor | None = None,
) -> torch.Tensor: ...
def iraf_zscale_naive(
    image: np.ndarray,
    contrast: float = 0.25,
    *,
    n_samples: int = 1000,
    max_reject: float = 0.5,
    min_npixels: int = 5,
    krej: float = 2.5,
    max_iterations: int = 5,
) -> tuple[float, float]: ...
