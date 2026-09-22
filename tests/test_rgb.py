"""Auto-adaptive ``rgb()`` compositor."""

from __future__ import annotations

import math

import pytest
import torch

from torchfits.transforms.rgb import lupton_rgb, rgb


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


def test_rgb_channel_maps_normalized_and_neutral() -> None:
    """Every mix row sums ~1 so band count never shifts brightness."""
    from torchfits.transforms.rgb import _SCARLET_MAPS

    for channels, rows in _SCARLET_MAPS.items():
        tolerance = 1e-12 if channels == 7 else 2e-3
        for row in rows:
            # Upstream scarlet values are approximate (C=2 -> 0.9985,
            # C=4 -> 1.00075); our 7-band extension is exact.
            assert abs(sum(row) - 1.0) < tolerance
        ones = torch.ones(1, channels)
        mixed = torch.tensor(rows).matmul(ones.T)
        assert torch.allclose(mixed, torch.ones(3, 1), atol=6e-3)


def test_write_rgb_image_roundtrip(tmp_path) -> None:
    """PNG writer emits spec-shaped scanlines decodable without Pillow."""
    import struct as _struct
    import zlib as _zlib

    from torchfits.transforms.rgb import write_rgb_image

    image = torch.rand(6, 9, 3)
    path = tmp_path / "out.png"
    write_rgb_image(str(path), image)
    blob = path.read_bytes()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = _struct.unpack(">II", blob[16:24])
    assert (width, height) == (9, 6)
    idat = blob.find(b"IDAT")
    size = _struct.unpack(">I", blob[idat - 4 : idat])[0]
    raw = _zlib.decompress(blob[idat + 4 : idat + 4 + size])
    assert len(raw) == height * (1 + width * 3)
    assert all(raw[i * (1 + width * 3)] == 0 for i in range(height))
    stride = 1 + width * 3
    pixels = (
        image.clamp(0, 1).mul(255).round().to(torch.uint8).reshape(-1).numpy().tobytes()
    )
    stripped = b"".join(raw[i * stride + 1 : (i + 1) * stride] for i in range(height))
    assert stripped == pixels


# ---------------------------------------------------------------------------
# dtype= contract and Astropy float-precision parity (R2 review)
# ---------------------------------------------------------------------------


def test_rgb_dtype_is_exact_on_empty_and_full_images() -> None:
    """``dtype=`` must control the output precision on every code path."""
    for dtype in (torch.float16, torch.float32, torch.float64):
        empty = rgb(
            torch.zeros(0, 5), torch.zeros(0, 5), torch.zeros(0, 5), dtype=dtype
        )
        assert empty.dtype == dtype
        full = rgb(torch.ones(4, 4), torch.ones(4, 4), torch.ones(4, 4), dtype=dtype)
        assert full.dtype == dtype


def test_lupton_rgb_dtype_is_exact() -> None:
    for dtype in (torch.float16, torch.float32, torch.float64):
        out = lupton_rgb(
            torch.rand(4, 4), torch.rand(4, 4), torch.rand(4, 4), dtype=dtype
        )
        assert out.dtype == dtype


def test_integer_dtype_request_raises_not_black_image() -> None:
    """Casting the [0, 1] floats to an integer dtype silently truncates the
    whole image to zeros — a typed error instead."""
    bands = (torch.rand(4, 4), torch.rand(4, 4), torch.rand(4, 4))
    with pytest.raises(TypeError, match="floating"):
        lupton_rgb(*bands, dtype=torch.uint8)
    with pytest.raises(TypeError, match="floating"):
        rgb(*bands, dtype=torch.int16)


def test_rgb_nan_inf_inputs_stay_finite_in_range() -> None:
    g = torch.full((8, 8), math.inf)
    g[0, 0] = math.nan
    out = rgb(g, torch.zeros(8, 8), torch.zeros(8, 8), dtype=torch.float32)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_lupton_rgb_float_parity_with_astropy() -> None:
    """Float-precision parity with astropy's make_lupton_rgb Lupton pipeline.

    Covers the Q softening clamps astropy applies (near-zero Q floors to 0.1,
    Q above 1e10 caps at 1e10) plus a saturated star, negative pixels and a
    non-zero minimum.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("astropy")
    from astropy.visualization import make_lupton_rgb

    rng = np.random.default_rng(0)
    r = np.abs(rng.normal(0, 40, (24, 24)))
    r[0, 0] = 1e6
    r[1, 1] = 0.0
    g = np.abs(rng.normal(0, 30, (24, 24)))
    g[2, 2] = -5.0
    b = np.abs(rng.normal(0, 20, (24, 24)))
    cases = [
        (8.0, 0.5, 0.0),
        (8.0, 5.0, 0.0),
        (3.0, 1.5, 2.0),
        (0.0, 0.5, 0.0),  # near-zero Q floor
        (1e12, 0.5, 0.0),  # Q above astropy's 1e10 cap
    ]
    for q, stretch, minimum in cases:
        ref = make_lupton_rgb(
            r, g, b, minimum=minimum, stretch=stretch, Q=q, output_dtype=np.float64
        )
        ours = lupton_rgb(
            torch.from_numpy(r),
            torch.from_numpy(g),
            torch.from_numpy(b),
            Q=q,
            stretch=stretch,
            minimum=minimum,
        ).numpy()
        assert np.allclose(ours, ref, rtol=0, atol=1e-12), (
            q,
            stretch,
            np.abs(ours - ref).max(),
        )


def test_write_rgb_image_rejects_empty_images(tmp_path) -> None:
    """Zero-width/height PNGs are invalid per spec — raise instead of writing."""
    from torchfits.transforms.rgb import write_rgb_image

    with pytest.raises(ValueError, match="empty"):
        write_rgb_image(str(tmp_path / "e1.png"), torch.zeros(0, 5, 3))
    with pytest.raises(ValueError, match="empty"):
        write_rgb_image(str(tmp_path / "e2.png"), torch.zeros(5, 0, 3))


def test_write_rgb_image_nan_pixels_are_black(tmp_path) -> None:
    """NaN pixels map to black deterministically (no float->uint8 UB)."""
    import struct as _struct
    import zlib as _zlib

    from torchfits.transforms.rgb import write_rgb_image

    image = torch.zeros(1, 2, 3)
    image[0, 0, 0] = math.nan
    path = tmp_path / "nan.png"
    write_rgb_image(str(path), image)
    blob = path.read_bytes()
    idat = blob.find(b"IDAT")
    size = _struct.unpack(">I", blob[idat - 4 : idat])[0]
    raw = _zlib.decompress(blob[idat + 4 : idat + 4 + size])
    assert list(raw[1:4]) == [0, 0, 0]
