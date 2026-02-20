import numpy as np
import torch

from torchfits.sphere.sparse import SparseHealpixMap
from torchfits.wcs import healpix as hp


def test_sparse_from_dense_roundtrip() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    dense = torch.full((npix,), float(hp.UNSEEN), dtype=torch.float64)
    keep = torch.arange(0, npix, 7, dtype=torch.int64)
    dense[keep] = torch.linspace(0.0, 1.0, steps=keep.numel(), dtype=torch.float64)

    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)
    back = sp.to_dense()
    torch.testing.assert_close(back, dense)
    assert sp.pixels.numel() == keep.numel()


def test_sparse_nearest_interpolation_on_centers() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    dense = torch.full((npix,), float(hp.UNSEEN), dtype=torch.float64)
    keep = torch.arange(0, npix, 5, dtype=torch.int64)
    dense[keep] = keep.to(torch.float64)

    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)
    lon, lat = hp.pix2ang(nside, keep, nest=False, lonlat=True)
    got = sp.interpolate(lon, lat, method="nearest")
    torch.testing.assert_close(got, dense[keep], atol=0.0, rtol=0.0)


def test_sparse_bilinear_matches_dense_when_fully_covered() -> None:
    nside = 16
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(5)
    dense = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)

    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=64)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=64)))).to(torch.float64)

    got = sp.interpolate(lon, lat, method="bilinear")
    exp = hp.get_interp_val(dense, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)


def test_sparse_ud_grade_workflow() -> None:
    nside = 16
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(17)
    dense = torch.full((npix,), float(hp.UNSEEN), dtype=torch.float64)
    keep = torch.from_numpy(rng.choice(npix, size=npix // 3, replace=False)).to(torch.int64)
    dense[keep] = torch.from_numpy(rng.normal(size=keep.numel())).to(torch.float64)

    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)
    sp2 = sp.ud_grade(8)
    assert sp2.nside == 8
    assert sp2.pixels.numel() > 0


def test_sparse_multiband_support() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    dense = torch.full((3, npix), float(hp.UNSEEN), dtype=torch.float64)
    keep = torch.arange(0, npix, 9, dtype=torch.int64)
    for b in range(3):
        dense[b, keep] = (b + 1) * 10.0 + keep.to(torch.float64)

    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)
    assert sp.values.shape[0] == 3

    lon, lat = hp.pix2ang(nside, keep[:12], nest=False, lonlat=True)
    got = sp.interpolate(lon, lat, method="nearest")
    exp = dense[:, keep[:12]]
    torch.testing.assert_close(got, exp, atol=0.0, rtol=0.0)


def test_sparse_ud_grade_full_coverage_matches_dense_ud_grade() -> None:
    rng = np.random.default_rng(1234)
    for nest in (False, True):
        nside = 8
        npix = hp.nside2npix(nside)
        dense = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
        sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=nest)
        order = "NEST" if nest else "RING"
        for nside_out in (4, 16):
            for power in (None, 1.0, -2.0):
                got = sp.ud_grade(nside_out, power=power).to_dense()
                exp = hp.ud_grade(
                    dense,
                    nside_out,
                    pess=False,
                    order_in=order,
                    order_out=order,
                    power=power,
                )
                torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)


def test_sparse_ud_grade_partial_coverage_pess_semantics() -> None:
    # NSIDE=4 -> NSIDE=2 (parent_mult=4 in NEST hierarchy).
    pixels = torch.tensor([12, 13, 28, 29, 30, 31], dtype=torch.int64)
    values = torch.tensor([1.0, 3.0, 10.0, 14.0, 18.0, 22.0], dtype=torch.float64)
    sp = SparseHealpixMap(nside=4, nest=True, pixels=pixels, values=values)

    got = sp.ud_grade(2, pess=False)
    torch.testing.assert_close(got.pixels, torch.tensor([3, 7], dtype=torch.int64), atol=0, rtol=0)
    torch.testing.assert_close(got.values, torch.tensor([2.0, 16.0], dtype=torch.float64), atol=1e-12, rtol=0.0)

    got_pess = sp.ud_grade(2, pess=True)
    torch.testing.assert_close(got_pess.pixels, torch.tensor([7], dtype=torch.int64), atol=0, rtol=0)
    torch.testing.assert_close(got_pess.values, torch.tensor([16.0], dtype=torch.float64), atol=1e-12, rtol=0.0)

    got_power = sp.ud_grade(2, pess=False, power=1.0)
    torch.testing.assert_close(
        got_power.values,
        torch.tensor([1.0, 8.0], dtype=torch.float64),  # scale by (nside_out/nside_in)^power = 0.5
        atol=1e-12,
        rtol=0.0,
    )


def test_sparse_coverage_mask_and_any_coverage_mode() -> None:
    nside = 4
    npix = hp.nside2npix(nside)
    dense = torch.full((2, npix), float(hp.UNSEEN), dtype=torch.float64)
    dense[0, 3] = 1.0
    dense[1, 7] = 2.0

    sp_all = SparseHealpixMap.from_dense(dense, nside=nside, nest=True, coverage_mode="all")
    assert sp_all.pixels.numel() == 0

    sp_any = SparseHealpixMap.from_dense(dense, nside=nside, nest=True, coverage_mode="any")
    torch.testing.assert_close(sp_any.pixels, torch.tensor([3, 7], dtype=torch.int64), atol=0, rtol=0)
    cov = sp_any.coverage_mask
    assert cov.dtype == torch.bool
    assert bool(cov[3]) and bool(cov[7]) and (int(cov.sum().item()) == 2)


def test_sparse_ud_grade_dense_equivalence_with_bad_values() -> None:
    nside_in = 8
    nside_out = 4
    npix = hp.nside2npix(nside_in)
    bad = -12345.0
    rng = np.random.default_rng(91)

    dense = torch.full((2, npix), bad, dtype=torch.float64)
    keep = torch.from_numpy(rng.choice(npix, size=npix // 2, replace=False)).to(torch.int64)
    dense[:, keep] = torch.from_numpy(rng.normal(size=(2, keep.numel()))).to(torch.float64)
    dense[0, keep[:20]] = torch.nan
    dense[1, keep[20:40]] = torch.inf

    sp = SparseHealpixMap.from_dense(dense, nside=nside_in, nest=False, fill_value=bad, coverage_mode="any")
    got = sp.ud_grade(nside_out, pess=False, power=2.0).to_dense()
    exp = hp.ud_grade(
        dense,
        nside_out,
        pess=False,
        badval=bad,
        order_in="RING",
        order_out="RING",
        power=2.0,
    )
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0, equal_nan=True)
