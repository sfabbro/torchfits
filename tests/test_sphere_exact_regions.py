import numpy as np
import pytest
import torch

from torchfits.sphere.geom import NativeExactSphericalRegion, SphericalCap, SphericalPolygon, to_exact_region

try:
    from spherical_geometry.polygon import SphericalPolygon as SGP  # type: ignore
except Exception:  # pragma: no cover
    SGP = None


@pytest.mark.skipif(SGP is None, reason="spherical-geometry not available")
def test_exact_region_area_matches_spherical_geometry() -> None:
    lon = torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64)
    lat = torch.tensor([-5.0, -5.0, 5.0, 5.0], dtype=torch.float64)
    reg = to_exact_region(SphericalPolygon(lon, lat))

    if hasattr(SGP, "from_lonlat"):
        sg = SGP.from_lonlat(lon.numpy(), lat.numpy(), degrees=True)
    else:
        sg = SGP.from_radec(lon.numpy(), lat.numpy(), degrees=True)

    np.testing.assert_allclose(reg.area(degrees=False), float(sg.area()), rtol=0.0, atol=1e-12)


@pytest.mark.skipif(SGP is None, reason="spherical-geometry not available")
def test_exact_boolean_ops_area_ordering() -> None:
    p1 = SphericalPolygon(
        torch.tensor([-8.0, 8.0, 8.0, -8.0], dtype=torch.float64),
        torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64),
    )
    p2 = SphericalPolygon(
        torch.tensor([0.0, 14.0, 14.0, 0.0], dtype=torch.float64),
        torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64),
    )

    u = p1.exact_union(p2)
    i = p1.exact_intersection(p2)
    d = p1.exact_difference(p2)

    assert not u.is_empty
    assert not i.is_empty
    assert not d.is_empty

    assert u.area() >= p1.to_exact().area()
    assert i.area() <= p1.to_exact().area()
    np.testing.assert_allclose(d.area() + i.area(), p1.to_exact().area(), rtol=0.0, atol=5e-8)


@pytest.mark.skipif(SGP is None, reason="spherical-geometry not available")
def test_exact_cap_intersection_contains() -> None:
    cap = SphericalCap(0.0, 0.0, 8.0)
    box = SphericalPolygon(
        torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64),
        torch.tensor([-4.0, -4.0, 4.0, 4.0], dtype=torch.float64),
    )
    reg = cap.exact_intersection(box)

    assert bool(reg.contains(0.0, 0.0))
    assert not bool(reg.contains(0.0, 7.5))


@pytest.mark.skipif(SGP is None, reason="spherical-geometry not available")
def test_boolean_region_to_exact_conversion() -> None:
    p1 = SphericalPolygon(
        torch.tensor([-8.0, 8.0, 8.0, -8.0], dtype=torch.float64),
        torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64),
    )
    p2 = SphericalPolygon(
        torch.tensor([0.0, 14.0, 14.0, 0.0], dtype=torch.float64),
        torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64),
    )
    reg_bool = p1.union(p2).difference(SphericalCap(3.0, 0.0, 3.0))
    reg_exact = reg_bool.to_exact(cap_steps=256)

    assert reg_exact.area() > 0.0
    mp = reg_exact.to_multipolygon()
    assert len(mp.polygons) >= 1


def test_native_exact_region_basic_area_contains() -> None:
    poly = SphericalPolygon(
        torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64),
        torch.tensor([-5.0, -5.0, 5.0, 5.0], dtype=torch.float64),
    )
    reg = to_exact_region(poly, backend="native", nsides=(64, 128, 256))
    assert isinstance(reg, NativeExactSphericalRegion)
    assert reg.area() > 0.0
    assert bool(reg.contains(0.0, 0.0))
    assert not bool(reg.contains(60.0, 60.0))


def test_native_exact_region_boolean_area_ordering() -> None:
    p1 = SphericalPolygon(
        torch.tensor([-8.0, 8.0, 8.0, -8.0], dtype=torch.float64),
        torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64),
    )
    p2 = SphericalPolygon(
        torch.tensor([0.0, 14.0, 14.0, 0.0], dtype=torch.float64),
        torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64),
    )
    r1 = to_exact_region(p1, backend="native", nsides=(64, 128, 256))
    u = r1.union(p2)
    i = r1.intersection(p2)
    d = r1.difference(p2)

    assert u.area() >= r1.area()
    assert i.area() <= r1.area()
    np.testing.assert_allclose(d.area() + i.area(), r1.area(), rtol=0.0, atol=5e-3)
