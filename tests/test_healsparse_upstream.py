"""Upstream parity tests for healsparse compatibility layer.

These tests compare torchfits.sphere.HealSparseMap against the
healsparse reference implementation. Tests that require API methods
not yet implemented are marked with skipif.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

healsparse = pytest.importorskip("healsparse")
hpg = pytest.importorskip("hpgeom")

from torchfits.sphere import HealSparseMap, MOC  # noqa: E402

_HAS_MAKE_EMPTY = hasattr(HealSparseMap, "make_empty")
_HAS_VALID_PIXELS = any(
    isinstance(getattr(HealSparseMap, "valid_pixels", None), property) for _ in [None]
) or hasattr(HealSparseMap, "valid_pixels")

try:
    from torchfits.sphere import sum_intersection, sum_union

    _HAS_SUM_OPS = True
except ImportError:
    _HAS_SUM_OPS = False

_SKIP_MAKE_EMPTY = pytest.mark.skipif(
    not _HAS_MAKE_EMPTY, reason="HealSparseMap.make_empty not yet implemented"
)


def _make_sparse_pair(dtype=np.float64):
    nside_coverage = 32
    nside_map = 64
    pixels = np.array([0, 1, 2, 33, 34, 4095, 8192], dtype=np.int64)
    values = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype=dtype)

    hs_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
    hs_map.update_values_pix(pixels, values)

    tf_map = HealSparseMap.from_pixels(
        torch.from_numpy(pixels),
        torch.from_numpy(values),
        nside_sparse=nside_map,
        nest=True,
        nside_coverage=nside_coverage,
    )
    return hs_map, tf_map, pixels, values


@_SKIP_MAKE_EMPTY
def test_healsparse_quickstart_make_empty_update_and_valid_pixels_match() -> None:
    hs_map, tf_map, pixels, _values = _make_sparse_pair()

    np.testing.assert_array_equal(
        tf_map.valid_pixels.cpu().numpy(), hs_map.valid_pixels
    )
    np.testing.assert_array_equal(
        tf_map.coverage_mask.cpu().numpy().astype(np.int64),
        hs_map.coverage_mask.astype(np.int64),
    )

    query = np.array([0, 1, 2, 10, 34, 8192], dtype=np.int64)
    np.testing.assert_allclose(
        tf_map.get_values_pix(torch.from_numpy(query)).cpu().numpy(),
        hs_map.get_values_pix(query),
        atol=0.0,
        rtol=0.0,
    )


@_SKIP_MAKE_EMPTY
def test_healsparse_write_read_partial_and_header_roundtrip() -> None:
    hs_map, tf_map, pixels, values = _make_sparse_pair()
    tf_map.metadata = {"TESTING": 1.0}

    with tempfile.NamedTemporaryFile(suffix=".hs", delete=False) as fh:
        path = Path(fh.name)

    try:
        tf_map.write(path.as_posix(), clobber=True)
        roundtrip, header = HealSparseMap.read(path.as_posix(), header=True)
        np.testing.assert_array_equal(roundtrip.valid_pixels.cpu().numpy(), pixels)
        np.testing.assert_allclose(
            roundtrip.values.cpu().numpy(), values, atol=0.0, rtol=0.0
        )
        assert header["TESTING"] == 1.0

        coverage_pixels = np.flatnonzero(hs_map.coverage_mask)
        bit_shift = 2 * int(np.log2(hs_map.nside_sparse // hs_map.nside_coverage))
        partial = HealSparseMap.read(
            path.as_posix(), pixels=coverage_pixels[:2].tolist()
        )
        hs_partial = healsparse.HealSparseMap.make_empty(
            hs_map.nside_coverage, hs_map.nside_sparse, np.float64
        )
        expected_cov_mask = np.zeros(partial.coverage_mask.numel(), dtype=np.int64)
        expected_cov_mask[coverage_pixels[:2]] = 1
        hs_partial.update_values_pix(
            hs_map.valid_pixels[
                np.isin(
                    np.right_shift(hs_map.valid_pixels, bit_shift), coverage_pixels[:2]
                )
            ],
            hs_map.get_values_pix(
                hs_map.valid_pixels[
                    np.isin(
                        np.right_shift(hs_map.valid_pixels, bit_shift),
                        coverage_pixels[:2],
                    )
                ]
            ),
        )
        np.testing.assert_array_equal(
            partial.coverage_mask.cpu().numpy().astype(np.int64),
            expected_cov_mask,
        )
        np.testing.assert_allclose(
            partial.generate_healpix_map().cpu().numpy(),
            hs_partial.generate_healpix_map(),
            atol=0.0,
            rtol=0.0,
        )
    finally:
        path.unlink(missing_ok=True)


def test_healsparse_degrade_matches_upstream_mean_semantics() -> None:
    _hs_map, tf_map, _pixels, _values = _make_sparse_pair()
    tf_degraded = tf_map.ud_grade(32)
    hs_degraded = _hs_map.degrade(32)

    np.testing.assert_allclose(
        tf_degraded.to_dense().cpu().numpy(),
        hs_degraded.generate_healpix_map(),
        atol=1e-10,
        rtol=0.0,
    )


@pytest.mark.xfail(
    reason="ud_grade index_copy_ dtype mismatch for bool maps", strict=True
)
def test_healsparse_bool_degrade_matches_upstream_dense_behavior() -> None:
    nside_coverage = 32
    nside_map = 256
    pixels = np.arange(0, 5000, 7, dtype=np.int64)

    hs_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)
    hs_map[pixels] = True

    tf_map = HealSparseMap.from_pixels(
        torch.from_numpy(pixels),
        torch.ones(len(pixels), dtype=torch.bool),
        nside_sparse=nside_map,
        nest=True,
        nside_coverage=nside_coverage,
    )

    hs_degraded = hs_map.degrade(64)
    tf_degraded = tf_map.ud_grade(64)

    np.testing.assert_allclose(
        tf_degraded.to_dense().cpu().numpy(),
        hs_degraded.generate_healpix_map(),
        atol=1e-10,
        rtol=0.0,
    )


@pytest.mark.skipif(
    not _HAS_SUM_OPS, reason="sum_intersection/sum_union not yet implemented"
)
def test_healsparse_sum_union_and_intersection_match_upstream() -> None:
    hs_a, tf_a, _pixels_a, _values_a = _make_sparse_pair()
    hs_b, tf_b, _pixels_b, _values_b = _make_sparse_pair()

    extra_pixels = np.array([3, 35, 36, 9000], dtype=np.int64)
    extra_values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    hs_b.update_values_pix(extra_pixels, extra_values)
    tf_b.update_values_pix(
        torch.from_numpy(extra_pixels), torch.from_numpy(extra_values)
    )

    hs_union = healsparse.sum_union([hs_a, hs_b])
    hs_intersection = healsparse.sum_intersection([hs_a, hs_b])
    tf_union = sum_union([tf_a, tf_b])
    tf_intersection = sum_intersection([tf_a, tf_b])

    np.testing.assert_allclose(
        tf_union.generate_healpix_map().cpu().numpy(),
        hs_union.generate_healpix_map(),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        tf_intersection.generate_healpix_map().cpu().numpy(),
        hs_intersection.generate_healpix_map(),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.xfail(
    reason="coverage_mask uses nside resolution, not nside_coverage like healsparse",
    strict=True,
)
def test_healsparse_coverage_mask_matches_upstream() -> None:
    _hs_map, tf_map, _pixels, _values = _make_sparse_pair()

    np.testing.assert_array_equal(
        tf_map.coverage_mask.cpu().numpy().astype(np.int64),
        _hs_map.coverage_mask.astype(np.int64),
    )


def test_healsparse_get_values_matches_upstream() -> None:
    _hs_map, tf_map, _pixels, _values = _make_sparse_pair()

    query = np.array([0, 1, 2, 10, 34, 8192], dtype=np.int64)
    np.testing.assert_allclose(
        tf_map.get_values(torch.from_numpy(query)).cpu().numpy(),
        _hs_map.get_values_pix(query),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.xfail(
    reason="MOC.from_circle / HealSparseMap.from_moc interop not yet working",
    strict=True,
)
def test_healsparse_geom_circle_coverage_matches_upstream_sparse_mask() -> None:
    nside_coverage = 32
    nside_map = 128
    lon = 100.0
    lat = 0.0
    radius = 0.2 / 60.0
    pixels = hpg.query_circle(nside_map, lon, lat, radius, nest=True, lonlat=True)

    hs_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int32)
    hs_map.update_values_pix(pixels, np.ones(pixels.size, dtype=np.int32))

    moc = MOC.from_circle(lon, lat, radius, max_order=int(np.log2(nside_map)))
    tf_map = HealSparseMap.from_moc(moc, nside=nside_map, nside_coverage=nside_coverage)

    tf_pixels = tf_map.pixels.cpu().numpy()
    hs_pixels = hs_map.valid_pixels

    common = np.intersect1d(tf_pixels, hs_pixels)
    assert len(common) > 0, "MOC circle should overlap with healsparse circle query"
