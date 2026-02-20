import math

import numpy as np
import torch
import pytest

from torchfits.sphere.geom import (
    PixelizedRegion,
    SphericalMultiPolygon,
    SphericalPolygon,
    query_polygon_general,
    spherical_polygon_area,
    spherical_polygon_contains,
    spherical_polygons_intersect,
    spherical_polygon_signed_area,
)
from torchfits.wcs import healpix as hp

try:
    from spherical_geometry.polygon import SphericalPolygon as SGP  # type: ignore
except Exception:  # pragma: no cover - optional comparator
    SGP = None


def _concave_polygon() -> tuple[torch.Tensor, torch.Tensor]:
    # Concave L-shaped polygon near the equator.
    lon = torch.tensor([-20.0, 20.0, 20.0, 0.0, 0.0, -20.0], dtype=torch.float64)
    lat = torch.tensor([-20.0, -20.0, 0.0, 0.0, 20.0, 20.0], dtype=torch.float64)
    return lon, lat


def test_spherical_polygon_signed_area_orientation() -> None:
    lon = torch.tensor([0.0, 90.0, 0.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 90.0], dtype=torch.float64)
    a_pos = spherical_polygon_signed_area(lon, lat)
    a_neg = spherical_polygon_signed_area(
        torch.flip(lon, dims=[0]), torch.flip(lat, dims=[0])
    )
    np.testing.assert_allclose(float(a_pos), math.pi / 2.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(float(a_neg), -math.pi / 2.0, atol=1e-12, rtol=0.0)


def test_spherical_polygon_contains_concave_shape() -> None:
    lon_poly, lat_poly = _concave_polygon()
    inside_a = spherical_polygon_contains(-10.0, 10.0, lon_poly, lat_poly)
    inside_b = spherical_polygon_contains(10.0, -10.0, lon_poly, lat_poly)
    outside_notch = spherical_polygon_contains(10.0, 10.0, lon_poly, lat_poly)
    outside_far = spherical_polygon_contains(80.0, 0.0, lon_poly, lat_poly)
    assert bool(inside_a)
    assert bool(inside_b)
    assert not bool(outside_notch)
    assert not bool(outside_far)


def test_query_polygon_general_matches_convex_baseline() -> None:
    nside = 64
    lon = torch.tensor([0.0, 20.0, 20.0, 0.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 20.0, 20.0], dtype=torch.float64)

    got = query_polygon_general(nside, lon, lat, nest=False)
    verts = hp.ang2vec(lon, lat, lonlat=True)
    expected = hp.query_polygon(nside, verts, nest=False)
    torch.testing.assert_close(got, expected)


def test_spherical_polygon_pixel_boolean_ops() -> None:
    nside = 32
    lon1 = torch.tensor([0.0, 25.0, 25.0, 0.0], dtype=torch.float64)
    lat1 = torch.tensor([0.0, 0.0, 25.0, 25.0], dtype=torch.float64)
    lon2 = torch.tensor([15.0, 40.0, 40.0, 15.0], dtype=torch.float64)
    lat2 = torch.tensor([10.0, 10.0, 35.0, 35.0], dtype=torch.float64)

    r1 = SphericalPolygon(lon1, lat1).pixelize(nside=nside, nest=False)
    r2 = SphericalPolygon(lon2, lat2).pixelize(nside=nside, nest=False)

    u = r1.union(r2)
    i = r1.intersection(r2)
    d = r1.difference(r2)

    assert isinstance(u, PixelizedRegion)
    assert isinstance(i, PixelizedRegion)
    assert isinstance(d, PixelizedRegion)
    assert i.pixels.numel() <= min(r1.pixels.numel(), r2.pixels.numel())
    assert d.pixels.numel() <= r1.pixels.numel()
    assert u.pixels.numel() >= max(r1.pixels.numel(), r2.pixels.numel())
    assert i.area() <= r1.area() + 1e-12


def test_spherical_polygon_class_contracts() -> None:
    lon, lat = _concave_polygon()
    poly = SphericalPolygon(lon, lat)
    area = poly.area(degrees=False)
    signed = poly.signed_area(degrees=False)
    assert float(area) > 0.0
    assert float(abs(signed)) > 0.0

    # Reversing winding flips signed area and preserves absolute area.
    poly_rev = SphericalPolygon(torch.flip(lon, dims=[0]), torch.flip(lat, dims=[0]))
    signed_rev = poly_rev.signed_area(degrees=False)
    area_rev = poly_rev.area(degrees=False)
    np.testing.assert_allclose(float(signed), -float(signed_rev), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(float(area), float(area_rev), atol=1e-12, rtol=0.0)

    assert bool(poly.contains(-10.0, 10.0))
    assert not bool(poly.contains(10.0, 10.0))


def test_spherical_polygon_area_units() -> None:
    lon = torch.tensor([0.0, 90.0, 0.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 90.0], dtype=torch.float64)
    sr = spherical_polygon_area(lon, lat, degrees=False)
    deg2 = spherical_polygon_area(lon, lat, degrees=True)
    np.testing.assert_allclose(
        float(deg2), float(sr) * (180.0 / math.pi) ** 2, atol=1e-10, rtol=0.0
    )


def test_spherical_polygons_intersect_basic_cases() -> None:
    lon1 = torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64)
    lat1 = torch.tensor([30.0, 30.0, 0.0, 0.0], dtype=torch.float64)

    lon2 = torch.tensor([-5.0, 15.0, 15.0, -5.0], dtype=torch.float64)
    lat2 = torch.tensor([20.0, 20.0, -10.0, -10.0], dtype=torch.float64)

    lon3 = torch.tensor([-20.0, 20.0, 20.0, -20.0], dtype=torch.float64)
    lat3 = torch.tensor([-20.0, -20.0, -10.0, -10.0], dtype=torch.float64)

    assert spherical_polygons_intersect(lon1, lat1, lon2, lat2)
    assert not spherical_polygons_intersect(lon1, lat1, lon3, lat3)


def test_spherical_polygon_inside_reference_flip() -> None:
    # Square around the north pole. Inside reference at equator should flip semantics.
    lon = torch.tensor([0.0, 90.0, 180.0, 270.0], dtype=torch.float64)
    lat = torch.tensor([80.0, 80.0, 80.0, 80.0], dtype=torch.float64)

    north = spherical_polygon_contains(0.0, 89.0, lon, lat)
    south_default = spherical_polygon_contains(0.0, 0.0, lon, lat)
    south_flipped = spherical_polygon_contains(
        0.0,
        0.0,
        lon,
        lat,
        inside_lon_deg=0.0,
        inside_lat_deg=0.0,
    )
    assert bool(north)
    assert not bool(south_default)
    assert bool(south_flipped)


def test_spherical_polygon_intersects_method_and_multipolygon() -> None:
    p1 = SphericalPolygon(
        torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64),
        torch.tensor([30.0, 30.0, 0.0, 0.0], dtype=torch.float64),
    )
    p2 = SphericalPolygon(
        torch.tensor([-5.0, 15.0, 15.0, -5.0], dtype=torch.float64),
        torch.tensor([20.0, 20.0, -10.0, -10.0], dtype=torch.float64),
    )
    p3 = SphericalPolygon(
        torch.tensor([120.0, 140.0, 140.0, 120.0], dtype=torch.float64),
        torch.tensor([-20.0, -20.0, -40.0, -40.0], dtype=torch.float64),
    )
    assert p1.intersects(p2)
    assert not p1.intersects(p3)

    mp = SphericalMultiPolygon((p1, p3))
    assert mp.intersects(p2)
    assert bool(mp.contains(0.0, 10.0))


@pytest.mark.skipif(SGP is None, reason="spherical-geometry not available")
def test_spherical_polygon_contains_matches_spherical_geometry() -> None:
    lon_poly = np.array(
        [106.0, 114.0, 130.0, 114.0, 106.0, 98.0, 82.0, 98.0, 106.0], dtype=np.float64
    )
    lat_poly = np.array(
        [-20.0, -15.0, -20.0, -25.0, -32.0, -25.0, -20.0, -15.0, -20.0],
        dtype=np.float64,
    )
    rng = np.random.default_rng(123)
    q_lon = rng.uniform(70.0, 140.0, size=2000)
    q_lat = rng.uniform(-40.0, -8.0, size=2000)

    if hasattr(SGP, "from_lonlat"):
        sg_poly = SGP.from_lonlat(lon_poly, lat_poly, degrees=True)
        sg_contains = np.asarray(
            [
                sg_poly.contains_lonlat(float(lo), float(la), degrees=True)
                for lo, la in zip(q_lon, q_lat, strict=False)
            ],
            dtype=bool,
        )
    elif hasattr(SGP, "from_radec"):
        sg_poly = SGP.from_radec(lon_poly, lat_poly, degrees=True)
        sg_contains = np.asarray(
            [
                sg_poly.contains_radec(float(lo), float(la), degrees=True)
                for lo, la in zip(q_lon, q_lat, strict=False)
            ],
            dtype=bool,
        )
    else:
        pytest.skip("unsupported spherical-geometry API")

    tf_contains = (
        spherical_polygon_contains(
            torch.from_numpy(q_lon),
            torch.from_numpy(q_lat),
            torch.from_numpy(lon_poly),
            torch.from_numpy(lat_poly),
            inclusive=False,
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_array_equal(tf_contains, sg_contains)
