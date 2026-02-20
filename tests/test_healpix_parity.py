import numpy as np
import pytest
import torch

from torchfits.wcs.healpix import (
    ang2vec,
    boundaries,
    get_map_size,
    get_nside,
    get_all_neighbours,
    get_interp_val,
    get_interp_weights,
    isnpixok,
    isnsideok,
    max_pixrad,
    nside2order,
    ud_grade,
    query_disc,
    query_polygon,
    query_strip,
    reorder,
    vec2ang,
    vec2pix,
    pix2vec,
    ang2pix_nested,
    ang2pix_ring,
    nest2ring,
    pix2ang_nested,
    pix2ang_ring,
    ring2nest,
)

try:
    import healpy
except Exception:  # pragma: no cover - optional comparator
    healpy = None

pytestmark = pytest.mark.skipif(healpy is None, reason="healpy not available")


def _random_lonlat(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return ra, dec


def _random_convex_polygon_xyz(rng: np.random.Generator, n_vertices: int = 5) -> np.ndarray:
    ra = float(rng.uniform(0.0, 360.0))
    dec = float(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0))))
    center = np.array(healpy.ang2vec(ra, dec, lonlat=True), dtype=np.float64)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(ref, center))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e1 = np.cross(ref, center)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(center, e1)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_vertices))
    radius = float(rng.uniform(0.03, 0.3))
    verts: list[np.ndarray] = []
    for a in angles:
        direction = np.cos(a) * e1 + np.sin(a) * e2
        v = np.cos(radius) * center + np.sin(radius) * direction
        v /= np.linalg.norm(v)
        verts.append(v)
    return np.asarray(verts, dtype=np.float64)


@pytest.mark.parametrize("nside,n", [(1, 5000), (2, 5000), (8, 20000), (64, 20000), (1024, 20000)])
def test_ang2pix_ring_matches_healpy(nside: int, n: int) -> None:
    ra, dec = _random_lonlat(n, seed=11 + nside)

    got = ang2pix_ring(
        nside,
        torch.from_numpy(ra).to(torch.float64),
        torch.from_numpy(dec).to(torch.float64),
    ).cpu().numpy()
    expected = healpy.ang2pix(nside, ra, dec, lonlat=True, nest=False)

    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nside,n", [(1, 5000), (2, 5000), (8, 20000), (64, 20000), (1024, 20000)])
def test_ang2pix_nested_matches_healpy(nside: int, n: int) -> None:
    ra, dec = _random_lonlat(n, seed=97 + nside)

    got = ang2pix_nested(
        nside,
        torch.from_numpy(ra).to(torch.float64),
        torch.from_numpy(dec).to(torch.float64),
    ).cpu().numpy()
    expected = healpy.ang2pix(nside, ra, dec, lonlat=True, nest=True)

    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nside,n", [(1, 2000), (2, 2000), (8, 20000), (64, 20000), (1024, 20000)])
def test_pix2ang_ring_matches_healpy(nside: int, n: int) -> None:
    rng = np.random.default_rng(123 + nside)
    pix = rng.integers(0, 12 * nside * nside, size=n, dtype=np.int64)

    ra_t, dec_t = pix2ang_ring(nside, torch.from_numpy(pix))
    ra_h, dec_h = healpy.pix2ang(nside, pix, nest=False, lonlat=True)

    dra = np.abs(((ra_t.cpu().numpy() - ra_h + 180.0) % 360.0) - 180.0)
    ddec = np.abs(dec_t.cpu().numpy() - dec_h)

    np.testing.assert_allclose(dra, 0.0, atol=1e-10)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-10)


@pytest.mark.parametrize("nside,n", [(1, 2000), (2, 2000), (8, 20000), (64, 20000), (1024, 20000)])
def test_pix2ang_nested_matches_healpy(nside: int, n: int) -> None:
    rng = np.random.default_rng(321 + nside)
    pix = rng.integers(0, 12 * nside * nside, size=n, dtype=np.int64)

    ra_t, dec_t = pix2ang_nested(nside, torch.from_numpy(pix))
    ra_h, dec_h = healpy.pix2ang(nside, pix, nest=True, lonlat=True)

    dra = np.abs(((ra_t.cpu().numpy() - ra_h + 180.0) % 360.0) - 180.0)
    ddec = np.abs(dec_t.cpu().numpy() - dec_h)

    np.testing.assert_allclose(dra, 0.0, atol=1e-10)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-10)


@pytest.mark.parametrize("nside,n", [(1, 2000), (2, 2000), (8, 20000), (64, 20000), (1024, 20000)])
def test_ring_nest_conversions_match_healpy(nside: int, n: int) -> None:
    rng = np.random.default_rng(777 + nside)
    ring = rng.integers(0, 12 * nside * nside, size=n, dtype=np.int64)
    nest = rng.integers(0, 12 * nside * nside, size=n, dtype=np.int64)

    ring_to_nest = ring2nest(nside, torch.from_numpy(ring)).cpu().numpy()
    nest_to_ring = nest2ring(nside, torch.from_numpy(nest)).cpu().numpy()

    expected_ring_to_nest = healpy.ring2nest(nside, ring)
    expected_nest_to_ring = healpy.nest2ring(nside, nest)

    np.testing.assert_array_equal(ring_to_nest, expected_ring_to_nest)
    np.testing.assert_array_equal(nest_to_ring, expected_nest_to_ring)


@pytest.mark.parametrize("nside", [1, 2, 8, 64, 1024])
def test_scalar_validators_and_order_match_healpy(nside: int) -> None:
    assert bool(isnsideok(nside)) == bool(healpy.isnsideok(nside))
    npix = 12 * nside * nside
    assert bool(isnpixok(npix)) == bool(healpy.isnpixok(npix))
    assert nside2order(nside) == healpy.nside2order(nside)
    np.testing.assert_allclose(max_pixrad(nside), healpy.max_pixrad(nside), atol=1e-12, rtol=0.0)


def test_vector_angle_wrappers_match_healpy() -> None:
    rng = np.random.default_rng(888)
    lon = rng.uniform(0.0, 360.0, size=4096)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=4096)))

    vec_t = ang2vec(torch.from_numpy(lon), torch.from_numpy(lat), lonlat=True).cpu().numpy()
    vec_h = healpy.ang2vec(lon, lat, lonlat=True)
    np.testing.assert_allclose(vec_t, vec_h, atol=1e-12, rtol=0.0)

    lon_t, lat_t = vec2ang(torch.from_numpy(vec_h), lonlat=True)
    lon_h, lat_h = healpy.vec2ang(vec_h, lonlat=True)
    np.testing.assert_allclose(lon_t.cpu().numpy(), lon_h, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(lat_t.cpu().numpy(), lat_h, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    "nside_in,nside_out,order_in,order_out,power",
    [
        (8, 4, "RING", "RING", None),
        (8, 4, "RING", "NEST", None),
        (8, 16, "RING", "RING", None),
        (8, 16, "NEST", "NEST", None),
        (8, 4, "RING", "RING", -2.0),
        (8, 4, "RING", "RING", 2.0),
    ],
)
def test_ud_grade_matches_healpy(
    nside_in: int, nside_out: int, order_in: str, order_out: str, power: float | None
) -> None:
    rng = np.random.default_rng(990 + nside_in + nside_out)
    npix_in = 12 * nside_in * nside_in
    m = rng.normal(size=npix_in).astype(np.float64)
    if order_in.upper().startswith("NEST"):
        m = healpy.reorder(m, r2n=True)

    got = ud_grade(
        torch.from_numpy(m),
        nside_out=nside_out,
        order_in=order_in,
        order_out=order_out,
        power=power,
    ).cpu().numpy()
    expected = healpy.ud_grade(m, nside_out, order_in=order_in, order_out=order_out, power=power)

    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)
    assert get_nside(torch.from_numpy(got)) == nside_out
    assert get_map_size(torch.from_numpy(got)) == 12 * nside_out * nside_out


def test_ud_grade_multimap_matches_healpy() -> None:
    rng = np.random.default_rng(991)
    nside_in = 8
    nside_out = 4
    npix = 12 * nside_in * nside_in
    mm = rng.normal(size=(2, npix)).astype(np.float64)

    got = ud_grade(torch.from_numpy(mm), nside_out=nside_out, order_in="RING", order_out="RING").cpu().numpy()
    expected = healpy.ud_grade(mm, nside_out, order_in="RING", order_out="RING")
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("pess", [False, True])
@pytest.mark.parametrize("order_in,order_out", [("RING", "RING"), ("RING", "NEST"), ("NEST", "RING")])
def test_ud_grade_badmask_and_pess_matches_healpy(pess: bool, order_in: str, order_out: str) -> None:
    rng = np.random.default_rng(2190 + int(pess))
    nside_in = 16
    nside_out = 4
    npix = 12 * nside_in * nside_in
    badval = healpy.UNSEEN

    m = rng.normal(size=npix).astype(np.float64)
    m[np.arange(0, npix, 97)] = np.nan
    m[np.arange(11, npix, 113)] = np.inf
    m[np.arange(23, npix, 127)] = badval
    if order_in == "NEST":
        m = healpy.reorder(m, r2n=True)

    got = ud_grade(
        torch.from_numpy(m),
        nside_out=nside_out,
        order_in=order_in,
        order_out=order_out,
        pess=pess,
        power=2.0,
    ).cpu().numpy()
    expected = healpy.ud_grade(
        m,
        nside_out,
        order_in=order_in,
        order_out=order_out,
        pess=pess,
        power=2.0,
    )

    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10, equal_nan=True)


def test_ud_grade_custom_badval_is_respected() -> None:
    nside_in = 8
    nside_out = 4
    npix_in = 12 * nside_in * nside_in
    custom_bad = -12345.0

    m = np.ones(npix_in, dtype=np.float64)
    m[:4] = custom_bad  # one degraded output pixel gets only bad inputs

    out = ud_grade(
        torch.from_numpy(m),
        nside_out=nside_out,
        order_in="NEST",
        order_out="NEST",
        pess=False,
        badval=custom_bad,
    ).cpu().numpy()
    assert out[0] == custom_bad
    np.testing.assert_allclose(out[1:], 1.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("nside", [8, 16, 32])
def test_query_disc_matches_healpy(nside: int) -> None:
    rng = np.random.default_rng(202 + nside)
    for nest in (False, True):
        for _ in range(8):
            ra = float(rng.uniform(0.0, 360.0))
            dec = float(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0))))
            radius = float(rng.uniform(0.01, 0.5))
            x, y, z = healpy.ang2vec(ra, dec, lonlat=True)

            got = query_disc(nside, np.array([x, y, z]), radius, nest=nest).cpu().numpy()
            expected = healpy.query_disc(nside, np.array([x, y, z]), radius, nest=nest, inclusive=False)
            np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nside", [8, 16, 32])
def test_query_polygon_matches_healpy(nside: int) -> None:
    rng = np.random.default_rng(303 + nside)
    npix = 12 * nside * nside
    sample_pix = rng.integers(0, npix, size=24, dtype=np.int64)
    for nest in (False, True):
        for pix in sample_pix:
            verts = boundaries(nside, torch.tensor(int(pix)), step=1, nest=nest).cpu().numpy().T
            got = query_polygon(nside, verts, nest=nest).cpu().numpy()
            expected = healpy.query_polygon(nside, verts, nest=nest, inclusive=False)
            np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nside", [8, 16, 32])
def test_query_polygon_random_convex_matches_healpy(nside: int) -> None:
    rng = np.random.default_rng(909 + nside)
    for nest in (False, True):
        for _ in range(12):
            verts = _random_convex_polygon_xyz(rng, n_vertices=5)
            got = query_polygon(nside, verts, nest=nest).cpu().numpy()
            expected = healpy.query_polygon(nside, verts, nest=nest, inclusive=False)
            np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nside", [8, 16, 32])
def test_query_strip_matches_healpy(nside: int) -> None:
    rng = np.random.default_rng(404 + nside)
    for nest in (False, True):
        for _ in range(8):
            t1 = float(rng.uniform(0.0, np.pi))
            t2 = float(rng.uniform(0.0, np.pi))
            got = np.sort(query_strip(nside, t1, t2, nest=nest).cpu().numpy())
            expected = np.sort(healpy.query_strip(nside, min(t1, t2), max(t1, t2), nest=nest, inclusive=False))
            np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nside,n", [(1, 256), (8, 4096), (64, 4096)])
def test_get_all_neighbours_matches_healpy(nside: int, n: int) -> None:
    rng = np.random.default_rng(606 + nside)
    pix = rng.integers(0, 12 * nside * nside, size=n, dtype=np.int64)
    pix_t = torch.from_numpy(pix)

    for nest in (False, True):
        got = get_all_neighbours(nside, pix_t, nest=nest).cpu().numpy()
        expected = healpy.get_all_neighbours(nside, pix, nest=nest)
        np.testing.assert_array_equal(got, expected)

        ra = rng.uniform(0.0, 360.0, size=n)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
        got_ang = get_all_neighbours(
            nside,
            torch.from_numpy(ra).to(torch.float64),
            torch.from_numpy(dec).to(torch.float64),
            nest=nest,
            lonlat=True,
        ).cpu().numpy()
        exp_ang = healpy.get_all_neighbours(nside, ra, dec, nest=nest, lonlat=True)
        np.testing.assert_array_equal(got_ang, exp_ang)


@pytest.mark.parametrize("nside,n", [(1, 128), (8, 2048), (64, 2048)])
def test_get_interp_weights_matches_healpy(nside: int, n: int) -> None:
    rng = np.random.default_rng(707 + nside)
    ra = rng.uniform(0.0, 360.0, size=n)
    dec = np.degrees(np.arcsin(rng.uniform(-0.99999, 0.99999, size=n)))
    ra_t = torch.from_numpy(ra).to(torch.float64)
    dec_t = torch.from_numpy(dec).to(torch.float64)
    pix = rng.integers(0, 12 * nside * nside, size=n, dtype=np.int64)
    pix_t = torch.from_numpy(pix)

    for nest in (False, True):
        got_p, got_w = get_interp_weights(nside, ra_t, dec_t, nest=nest, lonlat=True)
        exp_p, exp_w = healpy.get_interp_weights(nside, ra, dec, nest=nest, lonlat=True)
        np.testing.assert_array_equal(got_p.cpu().numpy(), exp_p)
        np.testing.assert_allclose(got_w.cpu().numpy(), exp_w, atol=1e-12, rtol=0.0)

        got_p2, got_w2 = get_interp_weights(nside, pix_t, nest=nest)
        exp_p2, exp_w2 = healpy.get_interp_weights(nside, pix, nest=nest)
        got_p2_np = got_p2.cpu().numpy()
        got_w2_np = got_w2.cpu().numpy()
        for j in range(pix.size):
            keep_g = got_w2_np[:, j] > 1e-6
            keep_e = exp_w2[:, j] > 1e-6
            idx_g = got_p2_np[keep_g, j]
            idx_e = exp_p2[keep_e, j]
            w_g = got_w2_np[keep_g, j]
            w_e = exp_w2[keep_e, j]
            order_g = np.argsort(idx_g)
            order_e = np.argsort(idx_e)
            np.testing.assert_array_equal(idx_g[order_g], idx_e[order_e])
            np.testing.assert_allclose(w_g[order_g], w_e[order_e], atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("nside,n", [(1, 128), (8, 4096), (64, 4096)])
def test_get_interp_val_matches_healpy(nside: int, n: int) -> None:
    rng = np.random.default_rng(808 + nside)
    npix = 12 * nside * nside
    m = rng.normal(size=npix).astype(np.float64)
    ra = rng.uniform(0.0, 360.0, size=n)
    dec = np.degrees(np.arcsin(rng.uniform(-0.99999, 0.99999, size=n)))

    ra_t = torch.from_numpy(ra).to(torch.float64)
    dec_t = torch.from_numpy(dec).to(torch.float64)
    m_t = torch.from_numpy(m).to(torch.float64)

    for nest in (False, True):
        got = get_interp_val(m_t, ra_t, dec_t, nest=nest, lonlat=True).cpu().numpy()
        expected = healpy.get_interp_val(m, ra, dec, nest=nest, lonlat=True)
        np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("nside", [1, 8, 64])
def test_interp_edge_cases_match_healpy(nside: int) -> None:
    rng = np.random.default_rng(1808 + nside)
    npix = 12 * nside * nside
    m = rng.normal(size=npix).astype(np.float64)

    lon_base = np.array(
        [-720.0, -360.0, -180.0, -1.0e-12, 0.0, 1.0e-12, 45.0, 180.0, 359.999999999999, 360.0, 720.0],
        dtype=np.float64,
    )
    lat_base = np.array(
        [-90.0, -89.999999999, -60.0, -1.0e-12, 0.0, 1.0e-12, 60.0, 89.999999999, 90.0],
        dtype=np.float64,
    )
    lon_grid, lat_grid = np.meshgrid(lon_base, lat_base, indexing="ij")
    lon = lon_grid.reshape(-1)
    lat = lat_grid.reshape(-1)

    lon_t = torch.from_numpy(lon).to(torch.float64)
    lat_t = torch.from_numpy(lat).to(torch.float64)
    m_t = torch.from_numpy(m).to(torch.float64)

    pix_edges = np.array(
        sorted(
            {
                0,
                1,
                2,
                max(npix // 2 - 1, 0),
                npix // 2,
                min(npix // 2 + 1, npix - 1),
                npix - 3,
                npix - 2,
                npix - 1,
            }
        ),
        dtype=np.int64,
    )
    pix_t = torch.from_numpy(pix_edges)

    for nest in (False, True):
        got_p, got_w = get_interp_weights(nside, lon_t, lat_t, nest=nest, lonlat=True)
        exp_p, exp_w = healpy.get_interp_weights(nside, lon, lat, nest=nest, lonlat=True)
        got_p_np = got_p.cpu().numpy()
        got_w_np = got_w.cpu().numpy()
        for j in range(lon.size):
            keep_g = got_w_np[:, j] > 1e-8
            keep_e = exp_w[:, j] > 1e-8
            idx_g = got_p_np[keep_g, j]
            idx_e = exp_p[keep_e, j]
            w_g = got_w_np[keep_g, j]
            w_e = exp_w[keep_e, j]
            order_g = np.argsort(idx_g)
            order_e = np.argsort(idx_e)
            np.testing.assert_array_equal(idx_g[order_g], idx_e[order_e])
            np.testing.assert_allclose(w_g[order_g], w_e[order_e], atol=1e-6, rtol=0.0)

        got_v = get_interp_val(m_t, lon_t, lat_t, nest=nest, lonlat=True).cpu().numpy()
        exp_v = healpy.get_interp_val(m, lon, lat, nest=nest, lonlat=True)
        np.testing.assert_allclose(got_v, exp_v, atol=1e-10, rtol=1e-10)

        got_p2, got_w2 = get_interp_weights(nside, pix_t, nest=nest)
        exp_p2, exp_w2 = healpy.get_interp_weights(nside, pix_edges, nest=nest)
        got_p2_np = got_p2.cpu().numpy()
        got_w2_np = got_w2.cpu().numpy()
        for j in range(pix_edges.size):
            keep_g = got_w2_np[:, j] > 1e-8
            keep_e = exp_w2[:, j] > 1e-8
            idx_g = got_p2_np[keep_g, j]
            idx_e = exp_p2[keep_e, j]
            w_g = got_w2_np[keep_g, j]
            w_e = exp_w2[keep_e, j]
            order_g = np.argsort(idx_g)
            order_e = np.argsort(idx_e)
            np.testing.assert_array_equal(idx_g[order_g], idx_e[order_e])
            np.testing.assert_allclose(w_g[order_g], w_e[order_e], atol=1e-6, rtol=0.0)

@pytest.mark.parametrize("nside", [1, 8, 32])
def test_reorder_matches_healpy(nside: int) -> None:
    rng = np.random.default_rng(505 + nside)
    npix = 12 * nside * nside
    m = rng.normal(size=npix).astype(np.float64)
    m_t = torch.from_numpy(m)

    got_r2n = reorder(m_t, r2n=True).cpu().numpy()
    exp_r2n = healpy.reorder(m, r2n=True)
    np.testing.assert_allclose(got_r2n, exp_r2n, atol=0.0, rtol=0.0)

    got_n2r = reorder(torch.from_numpy(exp_r2n), n2r=True).cpu().numpy()
    exp_n2r = healpy.reorder(exp_r2n, n2r=True)
    np.testing.assert_allclose(got_n2r, exp_n2r, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("nside,n", [(1, 1000), (2, 1000), (8, 10000), (64, 10000), (1024, 10000)])
def test_vec2pix_matches_healpy(nside: int, n: int) -> None:
    rng = np.random.default_rng(555 + nside)
    pix = rng.integers(0, 12 * nside * nside, size=n, dtype=np.int64)

    for nest in (False, True):
        x_h, y_h, z_h = healpy.pix2vec(nside, pix, nest=nest)
        x_t = torch.from_numpy(np.asarray(x_h, dtype=np.float64))
        y_t = torch.from_numpy(np.asarray(y_h, dtype=np.float64))
        z_t = torch.from_numpy(np.asarray(z_h, dtype=np.float64))

        got = vec2pix(nside, x_t, y_t, z_t, nest=nest).cpu().numpy()
        expected = healpy.vec2pix(nside, x_h, y_h, z_h, nest=nest)
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nside,n", [(1, 1000), (2, 1000), (8, 10000), (64, 10000), (1024, 10000)])
def test_pix2vec_matches_healpy(nside: int, n: int) -> None:
    rng = np.random.default_rng(7770 + nside)
    pix = rng.integers(0, 12 * nside * nside, size=n, dtype=np.int64)
    pix_t = torch.from_numpy(pix)

    for nest in (False, True):
        x_t, y_t, z_t = pix2vec(nside, pix_t, nest=nest)
        x_h, y_h, z_h = healpy.pix2vec(nside, pix, nest=nest)
        np.testing.assert_allclose(x_t.cpu().numpy(), x_h, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(y_t.cpu().numpy(), y_h, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(z_t.cpu().numpy(), z_h, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("nside,step,n", [(1, 1, 64), (1, 4, 64), (8, 1, 256), (8, 4, 256), (64, 1, 512), (64, 4, 512)])
def test_boundaries_matches_healpy(nside: int, step: int, n: int) -> None:
    rng = np.random.default_rng(8890 + nside + step)
    pix = rng.integers(0, 12 * nside * nside, size=n, dtype=np.int64)
    pix_t = torch.from_numpy(pix)

    for nest in (False, True):
        got = boundaries(nside, pix_t, step=step, nest=nest).cpu().numpy()
        expected = healpy.boundaries(nside, pix, step=step, nest=nest)
        np.testing.assert_allclose(got, expected, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("nside", [8, 64, 1024])
def test_ring2nest_equatorial_band_matches_healpy(nside: int) -> None:
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    eq_start = ncap
    eq_stop = npix - ncap

    rng = np.random.default_rng(9000 + nside)
    core = rng.integers(eq_start, eq_stop, size=20000, dtype=np.int64)
    anchors = np.array(
        [
            eq_start,
            eq_start + 1,
            eq_start + (4 * nside) - 1,
            eq_start + (4 * nside),
            eq_stop - (4 * nside) - 1,
            eq_stop - (4 * nside),
            eq_stop - 2,
            eq_stop - 1,
        ],
        dtype=np.int64,
    )
    ring = np.concatenate([anchors, core])

    got = ring2nest(nside, torch.from_numpy(ring)).cpu().numpy()
    expected = healpy.ring2nest(nside, ring)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nside", [1, 2, 4, 8])
def test_center_roundtrip_for_all_pixels(nside: int) -> None:
    npix = 12 * nside * nside
    pix = torch.arange(npix, dtype=torch.int64)

    ra_r, dec_r = pix2ang_ring(nside, pix)
    back_r = ang2pix_ring(nside, ra_r, dec_r)
    torch.testing.assert_close(back_r, pix)

    ra_n, dec_n = pix2ang_nested(nside, pix)
    back_n = ang2pix_nested(nside, ra_n, dec_n)
    torch.testing.assert_close(back_n, pix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu_small_sample() -> None:
    nside = 256
    n = 20000
    ra, dec = _random_lonlat(n, seed=2718)
    ring = np.random.default_rng(2719).integers(0, 12 * nside * nside, size=n, dtype=np.int64)

    ra_cpu = torch.from_numpy(ra).to(torch.float64)
    dec_cpu = torch.from_numpy(dec).to(torch.float64)
    ring_cpu = torch.from_numpy(ring)

    ra_gpu = ra_cpu.cuda()
    dec_gpu = dec_cpu.cuda()
    ring_gpu = ring_cpu.cuda()

    pix_r_cpu = ang2pix_ring(nside, ra_cpu, dec_cpu)
    pix_r_gpu = ang2pix_ring(nside, ra_gpu, dec_gpu).cpu()
    torch.testing.assert_close(pix_r_gpu, pix_r_cpu)

    pix_n_cpu = ang2pix_nested(nside, ra_cpu, dec_cpu)
    pix_n_gpu = ang2pix_nested(nside, ra_gpu, dec_gpu).cpu()
    torch.testing.assert_close(pix_n_gpu, pix_n_cpu)

    ra_r_cpu, dec_r_cpu = pix2ang_ring(nside, ring_cpu)
    ra_r_gpu, dec_r_gpu = pix2ang_ring(nside, ring_gpu)
    torch.testing.assert_close(ra_r_gpu.cpu(), ra_r_cpu, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(dec_r_gpu.cpu(), dec_r_cpu, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_mps_matches_cpu_small_sample() -> None:
    nside = 256
    n = 20000
    ra, dec = _random_lonlat(n, seed=4242)
    ring = np.random.default_rng(4243).integers(0, 12 * nside * nside, size=n, dtype=np.int64)

    ra_cpu = torch.from_numpy(ra).to(torch.float64)
    dec_cpu = torch.from_numpy(dec).to(torch.float64)
    ring_cpu = torch.from_numpy(ring)

    ra_mps = ra_cpu.to(torch.float32).to("mps")
    dec_mps = dec_cpu.to(torch.float32).to("mps")
    ring_mps = ring_cpu.to("mps")

    pix_r_cpu = ang2pix_ring(nside, ra_cpu, dec_cpu)
    pix_r_mps = ang2pix_ring(nside, ra_mps, dec_mps).cpu()
    torch.testing.assert_close(pix_r_mps, pix_r_cpu)

    pix_n_cpu = ang2pix_nested(nside, ra_cpu, dec_cpu)
    pix_n_mps = ang2pix_nested(nside, ra_mps, dec_mps).cpu()
    torch.testing.assert_close(pix_n_mps, pix_n_cpu)

    ra_r_cpu, dec_r_cpu = pix2ang_ring(nside, ring_cpu)
    ra_r_mps, dec_r_mps = pix2ang_ring(nside, ring_mps)
    torch.testing.assert_close(ra_r_mps.cpu(), ra_r_cpu.to(torch.float32), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(dec_r_mps.cpu(), dec_r_cpu.to(torch.float32), atol=1e-4, rtol=0.0)
