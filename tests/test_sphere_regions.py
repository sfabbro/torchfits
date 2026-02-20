import numpy as np
import pytest
import torch

from torchfits.sphere.geom import (
    SphericalBooleanRegion,
    SphericalCap,
    SphericalPolygon,
)


def _poly1() -> SphericalPolygon:
    return SphericalPolygon(
        torch.tensor([-5.0, 5.0, 5.0, -5.0], dtype=torch.float64),
        torch.tensor([-5.0, -5.0, 5.0, 5.0], dtype=torch.float64),
    )


def _poly2() -> SphericalPolygon:
    return SphericalPolygon(
        torch.tensor([0.0, 10.0, 10.0, 0.0], dtype=torch.float64),
        torch.tensor([-5.0, -5.0, 5.0, 5.0], dtype=torch.float64),
    )


def test_boolean_region_contains_contracts() -> None:
    p1 = _poly1()
    p2 = _poly2()

    union = p1.union(p2)
    inter = p1.intersection(p2)
    diff = p1.difference(p2)

    assert bool(union.contains(-2.0, 0.0))
    assert bool(union.contains(8.0, 0.0))

    assert bool(inter.contains(2.0, 0.0))
    assert not bool(inter.contains(-2.0, 0.0))

    assert bool(diff.contains(-2.0, 0.0))
    assert not bool(diff.contains(2.0, 0.0))


def test_boolean_region_pixelize_matches_manual_set_algebra() -> None:
    p1 = _poly1()
    p2 = _poly2()
    nside = 128

    got_union = p1.union(p2).pixelize(nside=nside, nest=False)
    got_inter = p1.intersection(p2).pixelize(nside=nside, nest=False)
    got_diff = p1.difference(p2).pixelize(nside=nside, nest=False)

    r1 = p1.pixelize(nside=nside, nest=False)
    r2 = p2.pixelize(nside=nside, nest=False)
    exp_union = r1.union(r2)
    exp_inter = r1.intersection(r2)
    exp_diff = r1.difference(r2)

    torch.testing.assert_close(got_union.pixels, exp_union.pixels)
    torch.testing.assert_close(got_inter.pixels, exp_inter.pixels)
    torch.testing.assert_close(got_diff.pixels, exp_diff.pixels)


def test_region_area_estimate_tracks_analytic_polygon_area() -> None:
    poly = SphericalPolygon(
        torch.tensor([10.0, 40.0, 40.0, 10.0], dtype=torch.float64),
        torch.tensor([-20.0, -20.0, 0.0, 0.0], dtype=torch.float64),
    )
    est = poly.area_estimate(nsides=(256, 512, 1024), nest=False)
    analytic = float(poly.area(degrees=False))
    rel_err = abs(est.area_sr - analytic) / analytic

    assert est.nside == 1024
    assert est.convergence_rel is not None
    assert est.convergence_rel < 0.25
    assert rel_err < 0.12


def test_boolean_region_area_estimate_metadata() -> None:
    region = _poly1().union(_poly2())
    est = region.area_estimate(nsides=(128, 256, 512), nest=False)

    assert est.nside == 512
    assert est.pixels > 0
    assert set(est.area_by_nside_sr.keys()) == {"128", "256", "512"}
    assert set(est.pixels_by_nside.keys()) == {"128", "256", "512"}


def test_region_invalid_nside_ladder_raises() -> None:
    poly = _poly1()
    with pytest.raises(ValueError):
        _ = poly.area_estimate(nsides=(128, 96), nest=False)


def test_cap_polygon_boolean_region() -> None:
    cap = SphericalCap(0.0, 0.0, 8.0)
    poly = SphericalPolygon(
        torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64),
        torch.tensor([-4.0, -4.0, 4.0, 4.0], dtype=torch.float64),
    )
    region = cap.intersection(poly)

    assert isinstance(region, SphericalBooleanRegion)
    assert bool(region.contains(0.0, 0.0))
    assert not bool(region.contains(0.0, 7.5))

    pix = region.query_pixels(128, nest=False)
    assert pix.ndim == 1
    assert pix.dtype == torch.int64
    assert pix.numel() > 0


def test_chained_boolean_regions() -> None:
    p1 = _poly1()
    p2 = _poly2()
    cap = SphericalCap(2.0, 0.0, 6.0)
    region = p1.union(p2).intersection(cap)
    assert isinstance(region, SphericalBooleanRegion)

    vals = np.array([
        bool(region.contains(-2.0, 0.0)),
        bool(region.contains(2.0, 0.0)),
        bool(region.contains(9.0, 0.0)),
    ])
    assert vals.tolist() == [True, True, False]
