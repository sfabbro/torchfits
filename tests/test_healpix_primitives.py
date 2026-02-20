import numpy as np
import pytest
import torch
import torchfits.wcs.healpix as healpix_mod

from torchfits.wcs.healpix import (
    ang2pix,
    ang2vec,
    ang2pix_ring,
    angular_distance_deg,
    boundaries,
    vec2ang,
    get_all_neighbours,
    get_interp_val,
    get_interp_weights,
    max_pixrad,
    max_pixel_radius,
    pixel_resolution_to_nside,
    query_box,
    query_circle,
    isnpixok,
    isnsideok,
    nside2order,
    lonlat_to_xyz,
    get_map_size,
    get_nside,
    mask_bad,
    nest_children,
    nest_parent,
    nest2ring,
    nside2npix,
    nside2pixarea,
    nside2resol,
    npix2nside,
    order2nside,
    pixel_ranges_to_pixels,
    pixels_to_pixel_ranges,
    pix2ang,
    pix2ang_ring,
    pix2vec,
    spherical_fourier_features,
    query_circle_vec,
    query_polygon_vec,
    ring2nest,
    upgrade_pixel_ranges,
    ud_grade,
    vec2pix,
    xyz_to_lonlat,
)


def _ra_delta_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a - b + 180.0, 360.0) - 180.0


def test_lonlat_xyz_roundtrip() -> None:
    rng = np.random.default_rng(123)
    ra = torch.from_numpy(rng.uniform(0.0, 360.0, size=10000)).to(torch.float64)
    dec = torch.from_numpy(
        np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=10000)))
    ).to(torch.float64)

    x, y, z = lonlat_to_xyz(ra, dec)
    ra2, dec2 = xyz_to_lonlat(x, y, z)

    dra = torch.abs(_ra_delta_deg(ra2, ra))
    ddec = torch.abs(dec2 - dec)
    torch.testing.assert_close(dra, torch.zeros_like(dra), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(ddec, torch.zeros_like(ddec), atol=1e-12, rtol=0.0)


def test_angular_distance_known_points() -> None:
    d0 = angular_distance_deg(
        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    )
    d1 = angular_distance_deg(
        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(90.0), torch.tensor(0.0)
    )
    d2 = angular_distance_deg(
        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(180.0), torch.tensor(0.0)
    )
    torch.testing.assert_close(
        d0, torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        d1, torch.tensor(90.0, dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        d2, torch.tensor(180.0, dtype=torch.float64), atol=1e-12, rtol=0.0
    )


def test_spherical_fourier_features_shape_and_grad() -> None:
    ra = torch.linspace(0.0, 359.0, steps=257, dtype=torch.float64, requires_grad=True)
    dec = torch.linspace(
        -80.0, 80.0, steps=257, dtype=torch.float64, requires_grad=True
    )
    feats = spherical_fourier_features(
        ra,
        dec,
        num_frequencies=5,
        min_frequency=0.5,
        base=2.0,
        include_xyz=True,
        include_lonlat=True,
    )
    assert feats.shape == (257, 53)

    loss = feats.square().mean()
    loss.backward()
    assert ra.grad is not None
    assert dec.grad is not None
    assert torch.isfinite(ra.grad).all()
    assert torch.isfinite(dec.grad).all()


def test_nest_parent_children_roundtrip() -> None:
    pix = torch.tensor([0, 1, 7, 15, 31, 127, 1024, 1234567], dtype=torch.int64)
    children = nest_children(pix, levels=2)
    assert children.shape == (pix.numel(), 16)

    parent = nest_parent(children, levels=2)
    expected = pix.unsqueeze(-1).expand_as(parent)
    torch.testing.assert_close(parent, expected)


def test_nest_children_offsets() -> None:
    pix = torch.tensor([11, 42], dtype=torch.int64)
    got = nest_children(pix, levels=1)
    expected = torch.tensor([[44, 45, 46, 47], [168, 169, 170, 171]], dtype=torch.int64)
    torch.testing.assert_close(got, expected)


def test_nside_scalar_helpers() -> None:
    assert nside2npix(8) == 768
    assert order2nside(5) == 32
    assert nside2order(32) == 5
    assert npix2nside(12288) == 32
    assert bool(isnsideok(32))
    assert not bool(isnsideok(12))
    assert bool(isnpixok(12288))
    assert not bool(isnpixok(12345))

    area = nside2pixarea(16)
    area_deg = nside2pixarea(16, degrees=True)
    resol = nside2resol(16)
    resol_arcmin = nside2resol(16, arcmin=True)

    assert area > 0.0
    assert area_deg > 0.0
    assert resol > 0.0
    assert resol_arcmin > 0.0
    assert max_pixrad(16) == max_pixel_radius(16)
    np.testing.assert_allclose(
        area_deg, area * (180.0 / np.pi) ** 2, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        resol_arcmin, resol * 60.0 * 180.0 / np.pi, rtol=0.0, atol=1e-12
    )


def test_map_helpers_and_ud_grade_shapes() -> None:
    nside = 8
    npix = 12 * nside * nside
    m = torch.arange(npix, dtype=torch.float64)
    assert get_nside(m) == nside
    assert get_map_size(m) == npix

    bad = m.clone()
    bad[3] = -1.6375e30
    mb = mask_bad(bad)
    assert bool(mb[3].item())
    assert not bool(mb[4].item())

    up = ud_grade(m, nside_out=16, order_in="RING", order_out="RING")
    dn = ud_grade(m, nside_out=4, order_in="RING", order_out="RING")
    assert up.shape == (12 * 16 * 16,)
    assert dn.shape == (12 * 4 * 4,)

    mm = torch.stack([m, m + 1.0], dim=0)
    out = ud_grade(mm, nside_out=4, order_in="RING", order_out="NEST")
    assert out.shape == (2, 12 * 4 * 4)


def test_ang2pix_pix2ang_compat_wrappers() -> None:
    rng = np.random.default_rng(99)
    ra = torch.from_numpy(rng.uniform(0.0, 360.0, size=4096)).to(torch.float64)
    dec = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=4096)))).to(
        torch.float64
    )

    theta = torch.deg2rad(90.0 - dec)
    phi = torch.deg2rad(ra)

    pix_a = ang2pix(64, theta, phi, nest=False, lonlat=False)
    pix_b = ang2pix_ring(64, ra, dec)
    torch.testing.assert_close(pix_a, pix_b)

    theta2, phi2 = pix2ang(64, pix_a, nest=False, lonlat=False)
    ra2, dec2 = pix2ang_ring(64, pix_a)
    np.testing.assert_allclose(
        theta2.cpu().numpy(),
        np.deg2rad(90.0 - dec2.cpu().numpy()),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        phi2.cpu().numpy(), np.deg2rad(ra2.cpu().numpy()), atol=1e-12, rtol=0.0
    )


def test_cpp_torch_bindings_core_healpix_match_python() -> None:
    cpp = pytest.importorskip("torchfits.cpp")
    required = [
        "healpix_ring2nest_torch_cpu",
        "healpix_nest2ring_torch_cpu",
        "healpix_ang2pix_ring_torch_cpu",
        "healpix_ang2pix_nested_torch_cpu",
        "healpix_pix2ang_ring_torch_cpu",
        "healpix_pix2ang_nested_torch_cpu",
    ]
    if not all(hasattr(cpp, name) for name in required):
        pytest.skip("torch-native HEALPix C++ bindings unavailable")

    nside = 64
    rng = np.random.default_rng(321)
    ra = torch.from_numpy(rng.uniform(0.0, 360.0, size=1024)).to(torch.float64)
    dec = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=1024)))).to(
        torch.float64
    )

    pix_ring_cpp = cpp.healpix_ang2pix_ring_torch_cpu(nside, ra, dec)
    pix_ring_py = ang2pix_ring(nside, ra, dec)
    torch.testing.assert_close(pix_ring_cpp, pix_ring_py)

    pix_nest_cpp = cpp.healpix_ang2pix_nested_torch_cpu(nside, ra, dec)
    pix_nest_py = ang2pix(
        nside, torch.deg2rad(90.0 - dec), torch.deg2rad(ra), nest=True, lonlat=False
    )
    torch.testing.assert_close(pix_nest_cpp, pix_nest_py)

    pix_nest_from_ring_cpp = cpp.healpix_ring2nest_torch_cpu(nside, pix_ring_cpp)
    pix_nest_from_ring_py = ring2nest(nside, pix_ring_cpp)
    torch.testing.assert_close(pix_nest_from_ring_cpp, pix_nest_from_ring_py)
    torch.testing.assert_close(
        cpp.healpix_nest2ring_torch_cpu(nside, pix_nest_cpp),
        nest2ring(nside, pix_nest_cpp),
    )

    ra_r_cpp, dec_r_cpp = cpp.healpix_pix2ang_ring_torch_cpu(nside, pix_ring_cpp)
    ra_r_py, dec_r_py = pix2ang_ring(nside, pix_ring_cpp)
    torch.testing.assert_close(ra_r_cpp, ra_r_py, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(dec_r_cpp, dec_r_py, atol=1e-12, rtol=0.0)

    ra_n_cpp, dec_n_cpp = cpp.healpix_pix2ang_nested_torch_cpu(nside, pix_nest_cpp)
    ra_n_py, dec_n_py = pix2ang(nside, pix_nest_cpp, nest=True, lonlat=True)
    torch.testing.assert_close(ra_n_cpp, ra_n_py, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(dec_n_cpp, dec_n_py, atol=1e-12, rtol=0.0)


def test_cpp_torch_bindings_interp_val_match_python() -> None:
    cpp = pytest.importorskip("torchfits.cpp")
    required = [
        "healpix_get_interp_val_ring_torch_cpu",
        "healpix_get_interp_val_nested_torch_cpu",
    ]
    if not all(hasattr(cpp, name) for name in required):
        pytest.skip("torch-native HEALPix interp C++ bindings unavailable")

    nside = 32
    npix = nside2npix(nside)
    rng = np.random.default_rng(987)
    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=256)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=256)))).to(
        torch.float64
    )
    m_ring = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
    m_nest = m_ring[ring2nest(nside, torch.arange(npix, dtype=torch.int64))]

    got_r = cpp.healpix_get_interp_val_ring_torch_cpu(nside, m_ring, lon, lat)
    exp_r = get_interp_val(m_ring, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(got_r, exp_r, atol=1e-12, rtol=1e-12)

    got_n = cpp.healpix_get_interp_val_nested_torch_cpu(nside, m_nest, lon, lat)
    exp_n = get_interp_val(m_nest, lon, lat, nest=True, lonlat=True)
    torch.testing.assert_close(got_n, exp_n, atol=1e-12, rtol=1e-12)

    m_stack = torch.stack([m_ring, m_ring * 0.5 + 0.2], dim=0)
    got_stack = cpp.healpix_get_interp_val_ring_torch_cpu(nside, m_stack, lon, lat)
    exp_stack = get_interp_val(m_stack, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(got_stack, exp_stack, atol=1e-12, rtol=1e-12)


def test_vec2pix_pix2vec_center_roundtrip() -> None:
    nside = 8
    npix = nside2npix(nside)
    for nest in (False, True):
        pix = torch.arange(npix, dtype=torch.int64)
        x, y, z = pix2vec(nside, pix, nest=nest)
        back = vec2pix(nside, x, y, z, nest=nest)
        torch.testing.assert_close(back, pix)


def test_ang2vec_vec2ang_roundtrip() -> None:
    rng = np.random.default_rng(13)
    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=4096)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=4096)))).to(
        torch.float64
    )
    vec = ang2vec(lon, lat, lonlat=True)
    lon2, lat2 = vec2ang(vec, lonlat=True)

    dlon = torch.abs(_ra_delta_deg(lon2, lon))
    dlat = torch.abs(lat2 - lat)
    torch.testing.assert_close(dlon, torch.zeros_like(dlon), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(dlat, torch.zeros_like(dlat), atol=1e-12, rtol=0.0)


def test_boundaries_shapes_and_unit_norm() -> None:
    b_scalar = boundaries(8, 1, step=4, nest=False)
    assert b_scalar.shape == (3, 16)

    pix = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    b_vec = boundaries(8, pix, step=4, nest=False)
    assert b_vec.shape == (5, 3, 16)

    norms = torch.sqrt((b_vec * b_vec).sum(dim=1))
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-12, rtol=0.0)


def test_resolution_helpers_and_query_shapes() -> None:
    nside = pixel_resolution_to_nside(nside2resol(64))
    assert nside == 64
    assert max_pixel_radius(64, degrees=True) > 0.0

    q1 = query_circle(32, 10.0, -20.0, 5.0, degrees=True, nest=False)
    assert q1.ndim == 1
    assert q1.dtype == torch.int64

    q2 = query_box(32, 350.0, -10.0, 15.0, 10.0, nest=True)
    assert q2.ndim == 1
    assert q2.dtype == torch.int64


def test_query_vec_helpers_match_hpgeom() -> None:
    hpg = pytest.importorskip("hpgeom")
    nside = 32
    lon = torch.tensor(14.0, dtype=torch.float64)
    lat = torch.tensor(-11.0, dtype=torch.float64)
    x, y, z = lonlat_to_xyz(lon, lat)
    vec = torch.stack([x, y, z]).to(torch.float64)

    got = query_circle_vec(nside, vec, np.deg2rad(4.0), inclusive=False, nest=True)
    exp = torch.as_tensor(
        hpg.query_circle_vec(
            nside, vec.numpy(), np.deg2rad(4.0), inclusive=False, nest=True
        )
    )
    torch.testing.assert_close(got, exp.to(torch.int64))
    got_inc = query_circle_vec(nside, vec, np.deg2rad(4.0), inclusive=True, nest=True)
    exp_inc = torch.as_tensor(
        hpg.query_circle_vec(
            nside, vec.numpy(), np.deg2rad(4.0), inclusive=True, nest=True
        )
    )
    assert set(exp_inc.to(torch.int64).tolist()).issubset(set(got_inc.tolist()))

    verts = torch.tensor(
        [[0.8, 0.1, 0.6], [0.7, 0.3, 0.65], [0.6, 0.2, 0.75], [0.75, 0.0, 0.66]],
        dtype=torch.float64,
    )
    verts = verts / torch.linalg.norm(verts, dim=1, keepdim=True)
    got_poly = query_polygon_vec(nside, verts, inclusive=False, nest=False)
    exp_poly = torch.as_tensor(
        hpg.query_polygon_vec(nside, verts.numpy(), inclusive=False, nest=False)
    )
    torch.testing.assert_close(got_poly, exp_poly.to(torch.int64))


def test_query_disc_cpp_matches_python_fallback() -> None:
    cpp = pytest.importorskip("torchfits.cpp")
    if not hasattr(cpp, "healpix_query_disc_torch_cpu"):
        pytest.skip("torch-native HEALPix query_disc C++ binding unavailable")

    nside = 64
    lon = 13.7
    lat = -21.2
    radius_deg = 6.0

    got_ring = query_circle(
        nside, lon, lat, radius_deg, degrees=True, nest=False, inclusive=True
    )
    got_nest = query_circle(
        nside, lon, lat, radius_deg, degrees=True, nest=True, inclusive=False
    )

    old_cpp = healpix_mod._cpp
    healpix_mod._cpp = None
    try:
        exp_ring = query_circle(
            nside, lon, lat, radius_deg, degrees=True, nest=False, inclusive=True
        )
        exp_nest = query_circle(
            nside, lon, lat, radius_deg, degrees=True, nest=True, inclusive=False
        )
    finally:
        healpix_mod._cpp = old_cpp

    torch.testing.assert_close(torch.sort(got_ring).values, torch.sort(exp_ring).values)
    torch.testing.assert_close(torch.sort(got_nest).values, torch.sort(exp_nest).values)


def test_pixel_ranges_helpers_match_hpgeom() -> None:
    hpg = pytest.importorskip("hpgeom")
    ranges = torch.tensor([[10, 13], [20, 22]], dtype=torch.int64)
    got_excl = pixel_ranges_to_pixels(ranges, inclusive=False)
    got_incl = pixel_ranges_to_pixels(ranges, inclusive=True)
    exp_excl = torch.as_tensor(
        hpg.pixel_ranges_to_pixels(ranges.numpy(), inclusive=False), dtype=torch.int64
    )
    exp_incl = torch.as_tensor(
        hpg.pixel_ranges_to_pixels(ranges.numpy(), inclusive=True), dtype=torch.int64
    )
    torch.testing.assert_close(got_excl, exp_excl)
    torch.testing.assert_close(got_incl, exp_incl)

    up = upgrade_pixel_ranges(2, ranges, 8)
    exp_up = torch.as_tensor(
        hpg.upgrade_pixel_ranges(2, ranges.numpy(), 8), dtype=torch.int64
    )
    torch.testing.assert_close(up, exp_up)

    pix = torch.tensor([7, 8, 9, 15, 16, 18], dtype=torch.int64)
    got_ranges = pixels_to_pixel_ranges(pix)
    torch.testing.assert_close(
        got_ranges, torch.tensor([[7, 10], [15, 17], [18, 19]], dtype=torch.int64)
    )


def test_neighbors_and_interp_shapes() -> None:
    nside = 32
    pix = torch.tensor([0, 1, 2, 10, 100], dtype=torch.int64)
    neigh = get_all_neighbours(nside, pix, nest=False)
    assert neigh.shape == (8, pix.numel())
    assert neigh.dtype == torch.int64

    ra = torch.tensor([10.0, 45.0, 180.0], dtype=torch.float64)
    dec = torch.tensor([-20.0, 0.0, 35.0], dtype=torch.float64)
    ip, w = get_interp_weights(nside, ra, dec, nest=False, lonlat=True)
    assert ip.shape == (4, 3)
    assert w.shape == (4, 3)
    torch.testing.assert_close(
        w.sum(dim=0), torch.ones(3, dtype=torch.float64), atol=1e-12, rtol=0.0
    )

    m = torch.arange(12 * nside * nside, dtype=torch.float64)
    vals = get_interp_val(m, ra, dec, nest=False, lonlat=True)
    assert vals.shape == (3,)
    assert vals.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lonlat_xyz_cuda_matches_cpu() -> None:
    rng = np.random.default_rng(7)
    ra_cpu = torch.from_numpy(rng.uniform(0.0, 360.0, size=4096)).to(torch.float64)
    dec_cpu = torch.from_numpy(
        np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=4096)))
    ).to(torch.float64)
    ra_gpu = ra_cpu.cuda()
    dec_gpu = dec_cpu.cuda()

    x_cpu, y_cpu, z_cpu = lonlat_to_xyz(ra_cpu, dec_cpu)
    x_gpu, y_gpu, z_gpu = lonlat_to_xyz(ra_gpu, dec_gpu)
    torch.testing.assert_close(x_gpu.cpu(), x_cpu, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(y_gpu.cpu(), y_cpu, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(z_gpu.cpu(), z_cpu, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_lonlat_xyz_mps_matches_cpu() -> None:
    rng = np.random.default_rng(8)
    ra_cpu = torch.from_numpy(rng.uniform(0.0, 360.0, size=4096)).to(torch.float64)
    dec_cpu = torch.from_numpy(
        np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=4096)))
    ).to(torch.float64)
    ra_mps = ra_cpu.to(torch.float32).to("mps")
    dec_mps = dec_cpu.to(torch.float32).to("mps")

    x_cpu, y_cpu, z_cpu = lonlat_to_xyz(ra_cpu, dec_cpu)
    x_mps, y_mps, z_mps = lonlat_to_xyz(ra_mps, dec_mps)
    torch.testing.assert_close(
        x_mps.cpu(), x_cpu.to(torch.float32), atol=5e-6, rtol=0.0
    )
    torch.testing.assert_close(
        y_mps.cpu(), y_cpu.to(torch.float32), atol=5e-6, rtol=0.0
    )
    torch.testing.assert_close(
        z_mps.cpu(), z_cpu.to(torch.float32), atol=5e-6, rtol=0.0
    )
