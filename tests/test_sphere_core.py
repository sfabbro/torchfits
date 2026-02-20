import math

import numpy as np
import torch

from torchfits.sphere.core import (
    fit_monopole_dipole,
    great_circle_distance,
    interpolate_wavelength_axis,
    pairwise_angular_distance,
    remove_monopole_dipole,
    sample_multiband_healpix,
    sample_multiwavelength_healpix,
    slerp_lonlat,
    wrap_longitude,
)
from torchfits.sphere.geom import (
    convex_polygon_contains,
    query_ellipse,
    spherical_polygon_area,
    spherical_triangle_area,
)
from torchfits.wcs import healpix as hp


def test_wrap_longitude_domains() -> None:
    lon = torch.tensor(
        [-540.0, -190.0, -10.0, 0.0, 10.0, 359.9, 720.0], dtype=torch.float64
    )
    wrapped_360 = wrap_longitude(lon, center_deg=180.0)
    wrapped_180 = wrap_longitude(lon, center_deg=0.0)
    assert torch.all((wrapped_360 >= 0.0) & (wrapped_360 < 360.0))
    assert torch.all((wrapped_180 >= -180.0) & (wrapped_180 < 180.0))


def test_great_circle_distance_contracts() -> None:
    d0 = great_circle_distance(0.0, 0.0, 0.0, 0.0)
    d1 = great_circle_distance(0.0, 0.0, 90.0, 0.0)
    d2 = great_circle_distance(0.0, 0.0, 180.0, 0.0)
    d3 = great_circle_distance(12.0, -30.0, 145.0, 20.0)
    d4 = great_circle_distance(145.0, 20.0, 12.0, -30.0)
    torch.testing.assert_close(
        d0, torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        d1, torch.tensor(90.0, dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        d2, torch.tensor(180.0, dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(d3, d4, atol=1e-12, rtol=0.0)


def test_pairwise_distance_symmetry() -> None:
    lon = torch.tensor([0.0, 45.0, 180.0, 270.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 10.0, 0.0, -30.0], dtype=torch.float64)
    d = pairwise_angular_distance(lon, lat)
    torch.testing.assert_close(d, d.transpose(0, 1), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        torch.diagonal(d), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0.0
    )


def test_slerp_endpoints_and_midpoint() -> None:
    lon0, lat0 = slerp_lonlat(0.0, 0.0, 90.0, 0.0, t=0.0)
    lon1, lat1 = slerp_lonlat(0.0, 0.0, 90.0, 0.0, t=1.0)
    lonm, latm = slerp_lonlat(0.0, 0.0, 90.0, 0.0, t=0.5)
    torch.testing.assert_close(
        lon0, torch.tensor(0.0, dtype=lon0.dtype), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        lat0, torch.tensor(0.0, dtype=lat0.dtype), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        lon1, torch.tensor(90.0, dtype=lon1.dtype), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        lat1, torch.tensor(0.0, dtype=lat1.dtype), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        lonm, torch.tensor(45.0, dtype=lonm.dtype), atol=1e-10, rtol=0.0
    )
    torch.testing.assert_close(
        latm, torch.tensor(0.0, dtype=latm.dtype), atol=1e-10, rtol=0.0
    )


def test_spherical_area_known_octant() -> None:
    area_tri = spherical_triangle_area(0.0, 0.0, 90.0, 0.0, 0.0, 90.0, degrees=False)
    np.testing.assert_allclose(float(area_tri), math.pi / 2.0, atol=1e-12, rtol=0.0)

    area_poly = spherical_polygon_area(
        torch.tensor([0.0, 90.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 90.0], dtype=torch.float64),
        degrees=False,
    )
    np.testing.assert_allclose(float(area_poly), math.pi / 2.0, atol=1e-12, rtol=0.0)


def test_convex_polygon_contains() -> None:
    poly_lon = torch.tensor([0.0, 90.0, 0.0], dtype=torch.float64)
    poly_lat = torch.tensor([0.0, 0.0, 90.0], dtype=torch.float64)
    inside = convex_polygon_contains(20.0, 20.0, poly_lon, poly_lat)
    outside = convex_polygon_contains(130.0, 20.0, poly_lon, poly_lat)
    assert bool(inside)
    assert not bool(outside)


def test_query_ellipse_subset_of_disc() -> None:
    nside = 64
    lon0, lat0 = 37.5, -15.0
    major, minor = 8.0, 3.0
    e = query_ellipse(nside, lon0, lat0, major, minor, pa_deg=25.0, nest=False)
    d = hp.query_circle(nside, lon0, lat0, major, degrees=True, nest=False)

    assert e.dtype == torch.int64
    assert e.ndim == 1
    assert set(e.cpu().tolist()).issubset(set(d.cpu().tolist()))


def test_sample_multiband_matches_healpix_interp() -> None:
    rng = np.random.default_rng(42)
    nside = 16
    npix = hp.nside2npix(nside)
    cube = torch.from_numpy(rng.normal(size=(4, npix))).to(torch.float64)
    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=128)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=128)))).to(
        torch.float64
    )

    sampled = sample_multiband_healpix(
        cube, nside, lon, lat, nest=False, interpolation="bilinear"
    )
    assert sampled.shape == (4, 128)

    expected = torch.stack(
        [
            hp.get_interp_val(cube[i], lon, lat, nest=False, lonlat=True)
            for i in range(cube.shape[0])
        ],
        dim=0,
    )
    torch.testing.assert_close(sampled, expected, atol=1e-12, rtol=0.0)


def test_interpolate_wavelength_axis_linear() -> None:
    values = torch.tensor(
        [[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]], dtype=torch.float64
    )
    out = interpolate_wavelength_axis(
        values,
        source_wavelength=[1.0, 2.0, 3.0],
        target_wavelength=[1.5, 2.5],
        axis=1,
    )
    expected = torch.tensor([[15.0, 25.0], [150.0, 250.0]], dtype=torch.float64)
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=0.0)


def test_sample_multiwavelength_resampling_matches_manual() -> None:
    rng = np.random.default_rng(9)
    nside = 8
    npix = hp.nside2npix(nside)
    cube = torch.from_numpy(rng.normal(size=(3, npix))).to(torch.float64)
    lon = torch.tensor([12.0, 45.0, 300.0], dtype=torch.float64)
    lat = torch.tensor([-10.0, 5.0, 30.0], dtype=torch.float64)
    src = torch.tensor([400.0, 500.0, 600.0], dtype=torch.float64)
    tgt = torch.tensor([450.0, 550.0], dtype=torch.float64)

    got = sample_multiwavelength_healpix(
        cube,
        nside,
        lon,
        lat,
        source_wavelength=src,
        target_wavelength=tgt,
        interpolation="nearest",
    )
    base = sample_multiband_healpix(cube, nside, lon, lat, interpolation="nearest")
    expected = interpolate_wavelength_axis(base, src, tgt, axis=0)
    torch.testing.assert_close(got, expected, atol=1e-12, rtol=0.0)


def test_fit_remove_monopole_dipole_roundtrip() -> None:
    nside = 16
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    x, y, z = hp.pix2vec(nside, pix, nest=False)

    mono_true = torch.tensor(2.5, dtype=torch.float64)
    dip_true = torch.tensor([0.3, -0.15, 0.07], dtype=torch.float64)
    signal = mono_true + (dip_true[0] * x) + (dip_true[1] * y) + (dip_true[2] * z)

    mono_fit, dip_fit = fit_monopole_dipole(signal, nside, nest=False)
    torch.testing.assert_close(mono_fit, mono_true, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(dip_fit, dip_true, atol=1e-10, rtol=0.0)

    residual = remove_monopole_dipole(signal, nside, nest=False)
    torch.testing.assert_close(
        residual, torch.zeros_like(residual), atol=1e-10, rtol=0.0
    )
