import numpy as np
import torch
import pytest

from torchfits.sphere import compat
from torchfits.sphere.geom import query_ellipse as geom_query_ellipse
from torchfits.wcs import healpix as hp


def test_compat_ang2pix_pix2ang_roundtrip_matches_healpix() -> None:
    rng = np.random.default_rng(123)
    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=1024)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=1024)))).to(
        torch.float64
    )

    pix = compat.ang2pix(64, lon, lat, nest=False, lonlat=True)
    expected = hp.ang2pix(64, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(pix, expected)

    lon1, lat1 = compat.pix2ang(64, pix, nest=False, lonlat=True)
    lon2, lat2 = hp.pix2ang(64, pix, nest=False, lonlat=True)
    torch.testing.assert_close(lon1, lon2, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(lat1, lat2, atol=1e-12, rtol=0.0)


def test_compat_neighbors_aliases_identical() -> None:
    pix = torch.tensor([0, 1, 10, 42], dtype=torch.int64)
    a = compat.get_all_neighbors(32, pix, nest=False)
    b = compat.get_all_neighbours(32, pix, nest=False)
    torch.testing.assert_close(a, b)


def test_compat_neighbors_angle_form_matches_healpix() -> None:
    lon = torch.tensor([12.0, 200.0], dtype=torch.float64)
    lat = torch.tensor([-10.0, 40.0], dtype=torch.float64)
    got = compat.get_all_neighbors(64, lon, lat, nest=False, lonlat=True)
    expected = hp.get_all_neighbours(64, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(got, expected)


def test_compat_query_circle_strict_mode_matches_hpgeom() -> None:
    hpg = pytest.importorskip("hpgeom")
    with compat.strict_mode(True):
        out = compat.query_circle(
            32, 25.0, -15.0, 5.0, degrees=True, inclusive=True, nest=True, fact=4
        )
    expected = torch.as_tensor(
        hpg.query_circle(
            32,
            25.0,
            -15.0,
            5.0,
            inclusive=True,
            fact=4,
            nest=True,
            lonlat=True,
            degrees=True,
        ),
        dtype=torch.int64,
    )
    torch.testing.assert_close(out, expected)


def test_compat_query_circle_vec_strict_mode_matches_hpgeom() -> None:
    hpg = pytest.importorskip("hpgeom")
    lon = 25.0
    lat = -15.0
    radius = np.deg2rad(5.0)
    x, y, z = hp.lonlat_to_xyz(torch.tensor(lon), torch.tensor(lat))
    vec = torch.stack([x, y, z]).to(torch.float64)
    with compat.strict_mode(True):
        out = compat.query_circle_vec(
            32, vec, radius, inclusive=True, nest=True, fact=4
        )
    expected = torch.as_tensor(
        hpg.query_circle_vec(
            32, vec.numpy(), radius, inclusive=True, fact=4, nest=True
        ),
        dtype=torch.int64,
    )
    torch.testing.assert_close(out, expected)


def test_compat_query_ellipse_calls_geometry_implementation() -> None:
    out_compat = compat.query_ellipse(
        32, 10.0, -20.0, 6.0, 2.5, pa_deg=35.0, nest=False, backend="torch"
    )
    out_geom = geom_query_ellipse(32, 10.0, -20.0, 6.0, 2.5, pa_deg=35.0, nest=False)
    torch.testing.assert_close(out_compat, out_geom)


def test_compat_query_ellipse_strict_mode_matches_hpgeom() -> None:
    hpg = pytest.importorskip("hpgeom")
    expected = torch.as_tensor(
        hpg.query_ellipse(
            32,
            10.0,
            -20.0,
            6.0,
            2.5,
            35.0,
            inclusive=False,
            fact=4,
            nest=False,
            lonlat=True,
            degrees=True,
        ),
        dtype=torch.int64,
    )
    with compat.strict_mode(True):
        out = compat.query_ellipse(32, 10.0, -20.0, 6.0, 2.5, pa_deg=35.0, nest=False)
    torch.testing.assert_close(out, expected)


def test_compat_query_ellipse_auto_backend_matches_hpgeom_when_available() -> None:
    hpg = pytest.importorskip("hpgeom")
    out = compat.query_ellipse(
        32, 10.0, -20.0, 6.0, 2.5, pa_deg=35.0, nest=True, inclusive=True, fact=4
    )
    expected = torch.as_tensor(
        hpg.query_ellipse(
            32,
            10.0,
            -20.0,
            6.0,
            2.5,
            35.0,
            inclusive=True,
            fact=4,
            nest=True,
            lonlat=True,
            degrees=True,
        ),
        dtype=torch.int64,
    )
    torch.testing.assert_close(out, expected)


def test_compat_query_ellipse_return_ranges_and_theta_phi_units() -> None:
    hpg = pytest.importorskip("hpgeom")
    nside = 32
    lon = 10.0
    lat = -20.0
    major = 6.0
    minor = 2.5
    pa = 35.0

    got_ranges = compat.query_ellipse(
        nside,
        lon,
        lat,
        major,
        minor,
        pa_deg=pa,
        nest=True,
        inclusive=True,
        fact=4,
        return_pixel_ranges=True,
    )
    exp_ranges = torch.as_tensor(
        hpg.query_ellipse(
            nside,
            lon,
            lat,
            major,
            minor,
            pa,
            inclusive=True,
            fact=4,
            nest=True,
            lonlat=True,
            degrees=True,
            return_pixel_ranges=True,
        ),
        dtype=torch.int64,
    ).reshape(-1, 2)
    torch.testing.assert_close(got_ranges, exp_ranges)

    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)
    got_tf = compat.query_ellipse(
        nside,
        theta,
        phi,
        np.deg2rad(major),
        np.deg2rad(minor),
        alpha=np.deg2rad(pa),
        nest=True,
        inclusive=False,
        fact=4,
        lonlat=False,
        degrees=False,
        backend="torch",
    )
    exp_tf = torch.as_tensor(
        hpg.query_ellipse(
            nside,
            theta,
            phi,
            np.deg2rad(major),
            np.deg2rad(minor),
            np.deg2rad(pa),
            inclusive=False,
            fact=4,
            nest=True,
            lonlat=False,
            degrees=False,
        ),
        dtype=torch.int64,
    )
    # Torch backend should stay near-canonical even without hpgeom dispatch.
    sym = len(
        set(got_tf.cpu().tolist()).symmetric_difference(set(exp_tf.cpu().tolist()))
    )
    assert sym <= 8


def test_compat_query_ellipse_inclusive_fact_refines_overlap() -> None:
    # Thin, long ellipse where coarse overlap approximations can miss edge pixels.
    nside = 64
    lon = 246.75536890809175
    lat = 42.359164144309915
    major = 19.620377411573955
    minor = 0.44602023784123124
    pa = 159.9907917472977

    coarse = compat.query_ellipse(
        nside, lon, lat, major, minor, pa_deg=pa, nest=False, inclusive=True, fact=1
    )
    fine = compat.query_ellipse(
        nside, lon, lat, major, minor, pa_deg=pa, nest=False, inclusive=True, fact=8
    )
    assert fine.numel() != coarse.numel()
    hpg = pytest.importorskip("hpgeom")
    exp = torch.as_tensor(
        hpg.query_ellipse(
            nside,
            lon,
            lat,
            major,
            minor,
            pa,
            inclusive=True,
            fact=8,
            nest=False,
            lonlat=True,
            degrees=True,
        ),
        dtype=torch.int64,
    )
    c_sym = len(
        set(coarse.cpu().tolist()).symmetric_difference(set(exp.cpu().tolist()))
    )
    f_sym = len(set(fine.cpu().tolist()).symmetric_difference(set(exp.cpu().tolist())))
    assert f_sym <= c_sym


def test_compat_query_ellipse_native_reasonable_vs_hpgeom_on_hard_case() -> None:
    hpg = pytest.importorskip("hpgeom")
    nside = 64
    lon = 32.82103279179087
    lat = 61.05973037738469
    major = 19.277723008371698
    minor = 11.918411640241988
    pa = -43.690581832574374

    got = compat.query_ellipse(
        nside, lon, lat, major, minor, pa_deg=pa, nest=False, inclusive=True, fact=8
    )
    exp = torch.as_tensor(
        hpg.query_ellipse(
            nside,
            lon,
            lat,
            major,
            minor,
            pa,
            inclusive=True,
            fact=8,
            nest=False,
            lonlat=True,
            degrees=True,
        ),
        dtype=torch.int64,
    )
    got_set = set(got.cpu().tolist())
    exp_set = set(exp.cpu().tolist())
    assert len(got_set.symmetric_difference(exp_set)) <= 24


def test_compat_fit_remove_dipole() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    x, y, z = hp.pix2vec(nside, pix, nest=False)
    mono = torch.tensor(1.2, dtype=torch.float64)
    dip = torch.tensor([0.05, -0.03, 0.02], dtype=torch.float64)
    m = mono + dip[0] * x + dip[1] * y + dip[2] * z

    mono_fit, dip_fit = compat.fit_dipole(m, nside=nside, nest=False)
    torch.testing.assert_close(mono_fit, mono, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(dip_fit, dip, atol=1e-10, rtol=0.0)

    cleaned = compat.remove_dipole(m, nside=nside, nest=False)
    torch.testing.assert_close(cleaned, torch.zeros_like(cleaned), atol=1e-10, rtol=0.0)


def test_compat_pol_spectral_api_shapes() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(222)
    maps = torch.from_numpy(rng.normal(size=(3, npix))).to(torch.float64)

    alms = compat.map2alm(maps, nside=nside, lmax=8, pol=True)
    assert alms.shape[0] == 3

    rec = compat.alm2map(alms, nside=nside, lmax=8, pol=True)
    assert rec.shape == maps.shape

    cl = compat.anafast(maps, nside=nside, lmax=8, pol=True)
    assert cl.shape == (6, 9)

    beam = compat.gaussian_beam(np.deg2rad(1.0), lmax=8, pol=True)
    assert beam.shape == (9, 4)


def test_compat_map2alm_lsq_scalar_api() -> None:
    nside = 8
    lmax = 6
    npix = hp.nside2npix(nside)
    expected_nalm = (lmax + 1) * (lmax + 2) // 2
    rng = np.random.default_rng(507)
    m = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
    alm, rel_res, n_iter = compat.map2alm_lsq(
        m, lmax=lmax, mmax=lmax, pol=False, maxiter=4, nside=nside
    )
    assert alm.shape == (expected_nalm,)
    assert isinstance(rel_res, float)
    assert np.isfinite(rel_res)
    assert 0 <= n_iter <= 4


def test_compat_sphtfunc_aliases() -> None:
    lmax = 6
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = torch.ones((nalm,), dtype=torch.complex128)
    fl = torch.linspace(1.0, 2.0, steps=lmax + 1, dtype=torch.float64)

    scaled = compat.almxfl(alm, fl)
    assert scaled.shape == alm.shape

    cl = compat.alm2cl(alm, lmax=lmax, mmax=lmax)
    assert cl.shape == (lmax + 1,)

    b1 = compat.gaussian_beam(np.deg2rad(1.0), lmax=lmax, pol=False)
    b2 = compat.gauss_beam(np.deg2rad(1.0), lmax=lmax, pol=False)
    torch.testing.assert_close(b1, b2, atol=1e-12, rtol=0.0)

    nside = 8
    npix = hp.nside2npix(nside)
    m = torch.linspace(-1.0, 1.0, steps=npix, dtype=torch.float64)
    sm = compat.smoothing(m, fwhm=np.deg2rad(30.0 / 60.0), lmax=lmax, backend="torch")
    assert sm.shape == m.shape


def test_compat_syn_and_beam_aliases() -> None:
    lmax = 6
    nside = 8
    cl = torch.zeros((lmax + 1,), dtype=torch.float64)

    alm = compat.synalm(cl, lmax=lmax, mmax=lmax)
    assert alm.shape == ((lmax + 1) * (lmax + 2) // 2,)

    m = compat.synfast(cl, nside=nside, lmax=lmax, mmax=lmax, pol=True)
    assert m.shape == (hp.nside2npix(nside),)

    theta = torch.linspace(0.0, 0.2, steps=64, dtype=torch.float64)
    bl = compat.gauss_beam(np.deg2rad(1.0), lmax=lmax, pol=False)
    beam = compat.bl2beam(bl, theta)
    bl_back = compat.beam2bl(beam, theta, lmax=lmax)
    assert beam.shape == theta.shape
    assert bl_back.shape == bl.shape


def test_compat_pixwin_api() -> None:
    nside = 8
    lmax = 10
    pw = compat.pixwin(nside, pol=False, lmax=lmax)
    assert pw.shape == (lmax + 1,)
    pwt, pwp = compat.pixwin(nside, pol=True, lmax=lmax)
    assert pwt.shape == (lmax + 1,)
    assert pwp.shape == (lmax + 1,)


def test_compat_hpgeom_vec_queries_and_ranges() -> None:
    hpg = pytest.importorskip("hpgeom")
    nside = 32
    lon = 25.0
    lat = -15.0
    radius = np.deg2rad(5.0)
    x, y, z = hp.lonlat_to_xyz(torch.tensor(lon), torch.tensor(lat))
    vec = torch.stack([x, y, z]).to(torch.float64)

    got_circle = compat.query_circle_vec(nside, vec, radius, nest=True, inclusive=False)
    exp_circle = torch.as_tensor(
        hpg.query_circle_vec(nside, vec.numpy(), radius, inclusive=False, nest=True),
        dtype=torch.int64,
    )
    torch.testing.assert_close(got_circle, exp_circle)
    got_circle_inc = compat.query_circle_vec(
        nside, vec, radius, nest=True, inclusive=True
    )
    exp_circle_inc = torch.as_tensor(
        hpg.query_circle_vec(nside, vec.numpy(), radius, inclusive=True, nest=True),
        dtype=torch.int64,
    )
    assert set(exp_circle_inc.tolist()).issubset(set(got_circle_inc.tolist()))

    v_lon = np.array([10.0, 20.0, 20.0, 10.0])
    v_lat = np.array([-5.0, -5.0, 5.0, 5.0])
    vx, vy, vz = hp.lonlat_to_xyz(torch.as_tensor(v_lon), torch.as_tensor(v_lat))
    vxyz = torch.stack([vx, vy, vz], dim=1).to(torch.float64)
    got_poly = compat.query_polygon_vec(nside, vxyz, nest=False, inclusive=False)
    exp_poly = torch.as_tensor(
        hpg.query_polygon_vec(nside, vxyz.numpy(), inclusive=False, nest=False),
        dtype=torch.int64,
    )
    torch.testing.assert_close(got_poly, exp_poly)

    ranges = torch.tensor([[10, 13], [20, 22]], dtype=torch.int64)
    torch.testing.assert_close(
        compat.pixel_ranges_to_pixels(ranges, inclusive=False),
        torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64),
    )
    torch.testing.assert_close(
        compat.pixel_ranges_to_pixels(ranges, inclusive=True),
        torch.tensor([10, 11, 12, 13, 20, 21, 22], dtype=torch.int64),
    )
    torch.testing.assert_close(
        compat.upgrade_pixel_ranges(2, ranges, 8),
        torch.as_tensor(
            hpg.upgrade_pixel_ranges(2, ranges.numpy(), 8), dtype=torch.int64
        ),
    )


def test_compat_pixels_to_ranges_roundtrip() -> None:
    pixels = torch.tensor([8, 9, 10, 15, 17, 18, 18], dtype=torch.int64)
    ranges = compat.pixels_to_pixel_ranges(pixels)
    torch.testing.assert_close(
        ranges, torch.tensor([[8, 11], [15, 16], [17, 19]], dtype=torch.int64)
    )
    restored = compat.pixel_ranges_to_pixels(ranges, inclusive=False)
    torch.testing.assert_close(
        restored, torch.tensor([8, 9, 10, 15, 17, 18], dtype=torch.int64)
    )


def test_compat_fit_remove_monopole_matches_healpy() -> None:
    healpy = pytest.importorskip("healpy")
    nside = 8
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    _, lat = hp.pix2ang(nside, pix, nest=False, lonlat=True)
    m = (1.5 + 0.01 * lat).to(torch.float64)
    m[5] = hp.UNSEEN

    got = compat.fit_monopole(m, nest=False, bad=hp.UNSEEN, gal_cut=10.0)
    exp = torch.tensor(
        float(healpy.fit_monopole(m.numpy(), nest=False, bad=hp.UNSEEN, gal_cut=10.0)),
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)

    got_removed, got_fit = compat.remove_monopole(
        m, nest=False, bad=hp.UNSEEN, gal_cut=10.0, fitval=True, copy=True
    )
    exp_removed, exp_fit = healpy.remove_monopole(
        m.numpy(),
        nest=False,
        bad=hp.UNSEEN,
        gal_cut=10.0,
        fitval=True,
        copy=True,
        verbose=False,
    )
    torch.testing.assert_close(
        got_fit, torch.tensor(float(exp_fit), dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        got_removed,
        torch.as_tensor(exp_removed, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_compat_fit_remove_dipole_matches_healpy() -> None:
    healpy = pytest.importorskip("healpy")
    nside = 8
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    x, y, z = hp.pix2vec(nside, pix, nest=False)
    m = (1.1 + 0.2 * x - 0.05 * y + 0.08 * z).to(torch.float64)
    m[7] = hp.UNSEEN

    got_mono, got_dip = compat.fit_dipole(m, nest=False, bad=hp.UNSEEN, gal_cut=12.0)
    exp_mono, exp_dip = healpy.fit_dipole(
        m.numpy(), nest=False, bad=hp.UNSEEN, gal_cut=12.0
    )
    torch.testing.assert_close(
        got_mono,
        torch.tensor(float(exp_mono), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        got_dip, torch.as_tensor(exp_dip, dtype=torch.float64), atol=1e-12, rtol=0.0
    )

    got_removed, got_mono2, got_dip2 = compat.remove_dipole(
        m,
        nest=False,
        bad=hp.UNSEEN,
        gal_cut=12.0,
        fitval=True,
        copy=True,
    )
    exp_removed, exp_mono2, exp_dip2 = healpy.remove_dipole(
        m.numpy(),
        nest=False,
        bad=hp.UNSEEN,
        gal_cut=12.0,
        fitval=True,
        copy=True,
        verbose=False,
    )
    exp_removed_t = torch.as_tensor(exp_removed.filled(hp.UNSEEN), dtype=torch.float64)
    torch.testing.assert_close(got_removed, exp_removed_t, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        got_mono2,
        torch.tensor(float(exp_mono2), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        got_dip2, torch.as_tensor(exp_dip2, dtype=torch.float64), atol=1e-12, rtol=0.0
    )
