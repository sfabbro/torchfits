"""Auto-adaptive ``rgb()`` compositor."""

from __future__ import annotations

import math

import pytest
import torch

from torchfits.transforms.rgb import rgb


def _sky_scene(
    height: int = 64,
    width: int = 64,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Empty-ish field: sky + a red core + a faint midtone patch."""
    gen = torch.Generator().manual_seed(seed)
    sky = 10.0 + 0.4 * torch.randn(height, width, generator=gen)
    g = sky.clone()
    r = sky.clone()
    i = sky.clone()
    g[20:28, 20:28] += 2.0
    r[20:28, 20:28] += 2.0
    i[20:28, 20:28] += 2.0
    # Reddest core (i) — colour coupling test.
    i[8:12, 8:12] += 80.0
    return g, r, i


def test_rgb_flux_scale_invariance() -> None:
    g, r, i = _sky_scene()
    a = rgb(g, r, i)
    b = rgb(100 * g, 100 * r, 100 * i)
    c = rgb(g, r, i, calibrated=True)
    d = rgb(100 * g, 100 * r, 100 * i, calibrated=True)
    assert a.shape == (64, 64, 3)
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)
    assert torch.allclose(c, d, atol=1e-3, rtol=1e-3)


def test_rgb_uncalibrated_equalize_vs_calibrated_colour() -> None:
    g, r, i = _sky_scene()
    base = rgb(g, r, i, saturation=1.0)
    loud_blue = rgb(100 * g, r, i, saturation=1.0)
    cal_loud = rgb(100 * g, r, i, saturation=1.0, calibrated=True)
    restored = rgb(
        100 * g,
        r,
        i,
        saturation=1.0,
        zeropoints=(27.5, 22.5, 22.5),
    )
    equal = rgb(g, r, i, saturation=1.0, calibrated=True)
    # Uncalibrated: ×100 on blue is absorbed by MAD equalize.
    assert torch.allclose(base, loud_blue, atol=2e-2, rtol=2e-2)
    # Calibrated: the loud blue band paints the image.
    assert float(cal_loud[..., 2].mean()) > float(equal[..., 2].mean()) + 0.02
    # Δm = +5 on the ×100 band restores nanomaggies.
    assert torch.allclose(restored, equal, atol=2e-2, rtol=2e-2)


def test_rgb_uncalibrated_ignores_sky_pedestal() -> None:
    g, r, i = _sky_scene()
    base = rgb(g, r, i, saturation=1.0)
    lifted = rgb(g + 80.0, r, i + 25.0, saturation=1.0)
    cal = rgb(g + 80.0, r, i, saturation=1.0, calibrated=True)
    assert torch.allclose(base, lifted, atol=3e-2, rtol=3e-2)
    assert float(cal[..., 2].mean()) > float(base[..., 2].mean()) + 0.05


def test_rgb_four_band_increases_red() -> None:
    g, r, i = _sky_scene()
    z = i + 5.0 * torch.ones_like(i)
    three = rgb(g, r, i, calibrated=True, saturation=1.0)
    four = rgb(g, r, i, z, calibrated=True, saturation=1.0)
    assert float(four[..., 0].mean()) > float(three[..., 0].mean())


def test_rgb_red_core_stays_chromatic() -> None:
    g, r, i = _sky_scene()
    out = rgb(g, r, i, saturation=1.0, calibrated=True)
    core = out[8:12, 8:12]
    # Reddest (i) maps to R; core must not go white.
    assert float(core[..., 0].mean()) > float(core[..., 2].mean()) + 0.05
    assert float(core[..., 2].mean()) < 0.85


def test_rgb_bright_star_does_not_crush_midtones() -> None:
    g, r, i = _sky_scene()
    g = g.clone()
    g[0, 0] = 1.0e6
    out = rgb(g, r, i)
    mid = float(out[40:50, 40:50].mean())
    star = float(out[0, 0].max())
    assert star > 0.5
    assert mid > 0.02


def test_rgb_filled_disk_not_near_black() -> None:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 64),
        torch.linspace(-1.0, 1.0, 64),
        indexing="ij",
    )
    disk = torch.where(xx * xx + yy * yy < 0.85, 50.0 + 8.0 * xx, 1.0)
    out = rgb(disk * 0.9, disk, disk * 1.1, scene="auto")
    assert float(out.mean()) > 0.12


def test_rgb_one_band_is_grey() -> None:
    g, _r, _i = _sky_scene()
    out = rgb(g, saturation=1.0)
    assert torch.allclose(out[..., 0], out[..., 1], atol=1e-6)
    assert torch.allclose(out[..., 1], out[..., 2], atol=1e-6)


def test_rgb_cube_matches_unpacked() -> None:
    g, r, i = _sky_scene()
    cube = torch.stack((g, r, i), dim=0)
    assert torch.allclose(rgb(cube), rgb(g, r, i), atol=1e-6)


def test_rgb_empty_nan_zero_size() -> None:
    empty = rgb(torch.zeros(0, 5), torch.zeros(0, 5), torch.zeros(0, 5))
    assert empty.shape == (0, 5, 3)
    nan = rgb(torch.full((8, 8), math.nan), torch.zeros(8, 8), torch.zeros(8, 8))
    assert torch.isfinite(nan).all()
    z = rgb(torch.zeros(8, 8), torch.zeros(8, 8), torch.zeros(8, 8))
    assert z.shape == (8, 8, 3)


def test_rgb_nan_mosaic_holes_stay_black() -> None:
    """NaN mosaic holes must not zero-fill into a fake filled-scene stretch."""
    g0, r0, i0 = _sky_scene()
    g, r, i = g0.clone(), r0.clone(), i0.clone()
    g[:6, :] = math.nan
    r[:6, :] = math.nan
    i[:6, :] = math.nan
    out = rgb(g, r, i)
    assert torch.isfinite(out).all()
    assert float(out[:6].mean()) < 0.05
    assert float(out[20:50, 20:50].max()) > 0.2


def test_rgb_rejects_bad_knobs() -> None:
    g, r, i = _sky_scene(8, 8)
    with pytest.raises(ValueError, match="brightness"):
        rgb(g, r, i, brightness=0.0)
    with pytest.raises(ValueError, match="scene"):
        rgb(g, r, i, scene="planet")
    with pytest.raises(ValueError, match="zeropoints"):
        rgb(g, r, i, zeropoints=(22.5, 22.5))
    with pytest.raises(ValueError, match="8"):
        rgb(torch.zeros(8, 4, 4))
