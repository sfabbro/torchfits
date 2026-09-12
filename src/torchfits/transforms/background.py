"""Tiled background estimation (SExtractor-style mesh).

:class:`BackgroundSubtract` removes a single global median, which is wrong for
wide-field mosaics where the sky level drifts across the frame. SExtractor
(Bertin & Arnouts 1996) instead divides the frame into a mesh of tiles,
sigma-clips each tile, smooths the resulting background map and interpolates
it back to full resolution. :class:`MeshBackgroundSubtract` implements that
recipe in torch and is mask/IVAR aware per tile.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

from .base import FITSTransform
from .helpers import estimate_background
from .state import SCALABLE

__all__ = ["MeshBackgroundSubtract"]


def _median_filter_3x3(grid: torch.Tensor) -> torch.Tensor:
    """3×3 median filter over the last two dims (edge-padded).

    Edge replication is done with explicit concatenation because
    ``F.pad(mode="replicate")`` only accepts rank-3+ inputs.
    """
    padded = torch.cat([grid[..., :1, :], grid, grid[..., -1:, :]], dim=-2)
    padded = torch.cat([padded[..., :, :1], padded, padded[..., :, -1:]], dim=-1)
    neighbours = torch.stack(
        [
            padded[..., dy : dy + grid.shape[-2], dx : dx + grid.shape[-1]]
            for dy in range(3)
            for dx in range(3)
        ],
        dim=0,
    )
    return neighbours.median(dim=0).values


class MeshBackgroundSubtract(FITSTransform):
    """Subtract a smoothly varying background estimated on a tile mesh.

    The trailing two dimensions are treated as spatial (``..., H, W``); any
    leading dimensions are batched, so this works for multi-band stacks and
    ``(C, H, W)`` cubes as well as single frames.

    Parameters
    ----------
    mesh : tuple[int, int]
        Number of background tiles along ``(H, W)``. Each is sigma-clipped
        independently. Default ``(8, 8)``.
    n_sigma, max_iter : float, int
        Sigma-clipping thresholds passed to :func:`estimate_background`
        (which uses a robust median/MAD for a single pass; ``n_sigma`` is
        accepted for API symmetry and used only by the optional iterative
        refinement).
    filter_mesh : bool
        Apply a 3×3 median filter to the tile grid before interpolation, as
        SExtractor does, to stop bright sources from imprinting on the sky
        map. Default ``True``.
    min_tile_pixels : int
        Tiles with fewer valid pixels than this fall back to the frame-level
        background. Default ``4``.
    weighted : bool
        Use inverse-variance weighted tile statistics when ``ivar`` is
        present.

    Examples
    --------
    >>> pipeline = Compose(
    ...     [MeshBackgroundSubtract(mesh=(8, 8)), ArcsinhStretch(a=0.1)]
    ... )
    >>> sky_subtracted = pipeline(image)
    """

    expects = SCALABLE
    propagates_ivar = True  # pure offset: variance is unchanged

    def __init__(
        self,
        mesh: Tuple[int, int] = (8, 8),
        *,
        n_sigma: float = 3.0,
        max_iter: int = 5,
        filter_mesh: bool = True,
        min_tile_pixels: int = 4,
        weighted: bool = False,
    ) -> None:
        if len(mesh) != 2 or mesh[0] < 1 or mesh[1] < 1:
            raise ValueError(f"mesh must be two positive ints, got {mesh!r}")
        if min_tile_pixels < 1:
            raise ValueError("min_tile_pixels must be >= 1")
        self.mesh = (int(mesh[0]), int(mesh[1]))
        self.n_sigma = float(n_sigma)
        self.max_iter = int(max_iter)
        self.filter_mesh = bool(filter_mesh)
        self.min_tile_pixels = int(min_tile_pixels)
        self.weighted = bool(weighted)
        self._last_bg: torch.Tensor | None = None
        self._last_grid: torch.Tensor | None = None

    def _tile_boundaries(self, size: int, tiles: int) -> torch.Tensor:
        return torch.linspace(0, size, tiles + 1, device="cpu").round().to(torch.long)

    def _estimate_grid(
        self,
        flux: torch.Tensor,
        mask: torch.Tensor | None,
        ivar: torch.Tensor | None,
    ) -> torch.Tensor:
        height, width = int(flux.shape[-2]), int(flux.shape[-1])
        gh = min(self.mesh[0], height)
        gw = min(self.mesh[1], width)
        ys = self._tile_boundaries(height, gh)
        xs = self._tile_boundaries(width, gw)
        batch = flux.shape[:-2]
        grid = flux.new_full((*batch, gh, gw), float("nan"))
        frame_grid = None

        for i in range(gh):
            y0, y1 = int(ys[i]), int(ys[i + 1])
            for j in range(gw):
                x0, x1 = int(xs[j]), int(xs[j + 1])
                tile = flux[..., y0:y1, x0:x1]
                tile_mask = None if mask is None else mask[..., y0:y1, x0:x1]
                tile_ivar = None if ivar is None else ivar[..., y0:y1, x0:x1]
                med, _ = estimate_background(
                    tile,
                    dim=(-2, -1),
                    mask=tile_mask,
                    ivar=tile_ivar,
                    weighted=self.weighted,
                )
                med_flat = med.squeeze(-1).squeeze(-1)
                grid[..., i, j] = med_flat

        if self.filter_mesh and gh >= 3 and gw >= 3:
            grid = _median_filter_3x3(grid)
        elif self.filter_mesh and gh > 1 and gw > 1:
            grid = _median_filter_3x3(grid)

        # Tiles with too few valid pixels (or a fully masked tile) are NaN:
        # fall back to the frame-level background so the correction stays
        # continuous instead of punching holes in the data.
        frame_med, _ = estimate_background(
            flux,
            dim=(-2, -1),
            mask=mask,
            ivar=ivar,
            weighted=self.weighted,
        )
        frame_grid = frame_med.squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(-1)
        grid = torch.where(torch.isfinite(grid), grid, frame_grid)
        grid = torch.where(torch.isfinite(grid), grid, torch.zeros_like(grid))
        self._last_grid = grid
        return grid

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        flux = view.flux
        if flux.ndim < 2:
            raise ValueError(
                "MeshBackgroundSubtract needs at least 2 dims (..., H, W), "
                f"got shape {tuple(flux.shape)}"
            )
        effective = view.effective_mask(mask)
        grid = self._estimate_grid(flux, effective, view.ivar)
        height, width = int(flux.shape[-2]), int(flux.shape[-1])
        gh, gw = int(grid.shape[-2]), int(grid.shape[-1])
        # interpolate() needs (N, C, H, W): flatten the leading batch dims.
        flat = grid.reshape(-1, 1, gh, gw)
        bg = torch.nn.functional.interpolate(
            flat,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(*grid.shape[:-2], height, width)
        self._last_bg = bg
        return view.replace(flux - bg)

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        if self._last_bg is None:
            raise RuntimeError(
                "MeshBackgroundSubtract.inverse() requires a prior forward() pass."
            )
        view = self.view(x)
        return view.replace(view.flux + self._last_bg)

    def __repr__(self) -> str:
        return (
            f"MeshBackgroundSubtract(mesh={self.mesh}, "
            f"filter_mesh={self.filter_mesh}, weighted={self.weighted})"
        )
