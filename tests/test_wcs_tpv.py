import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from torchfits.wcs.core import WCS


def _tpv_header() -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 512.0
    header["CRVAL1"] = 200.0
    header["CRVAL2"] = -20.0
    header["CD1_1"] = -2.8e-4
    header["CD1_2"] = 1.2e-7
    header["CD2_1"] = -1.1e-7
    header["CD2_2"] = 2.8e-4

    # Keep a mix of linear and non-linear terms; this remains stable for forward parity.
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_2"] = 0.0
    header["PV1_4"] = 2.0e-4
    header["PV1_5"] = -3.0e-4
    header["PV1_6"] = 1.5e-4
    header["PV1_7"] = 2.0e-6
    header["PV1_39"] = 2.0e-11

    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.0
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4
    header["PV2_5"] = 2.5e-4
    header["PV2_6"] = -2.0e-4
    header["PV2_7"] = -1.5e-6
    header["PV2_39"] = -1.0e-11

    return header


def _tpv_affine_linear_header() -> fits.Header:
    header = _tpv_header()
    # Pure affine TPV (non-identity, with offsets and cross terms)
    for k in list(header.keys()):
        if k.startswith("PV1_") or k.startswith("PV2_"):
            del header[k]
    header["PV1_0"] = 0.05
    header["PV1_1"] = 1.12
    header["PV1_2"] = -0.08
    # Axis-2 TPV uses swapped arguments: PV2_2 contributes u, PV2_1 contributes v
    header["PV2_0"] = -0.03
    header["PV2_1"] = 0.04
    header["PV2_2"] = 0.91
    return header


def _sample_pixels() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(430.0, 590.0, 9, dtype=np.float64)
    y = np.linspace(420.0, 600.0, 9, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def test_tpv_forward_parity_with_astropy() -> None:
    header = _tpv_header()
    awcs = AstropyWCS(header)
    twcs = WCS(dict(header))

    x, y = _sample_pixels()

    ra_ast, dec_ast = awcs.all_pix2world(x, y, 0)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=0
    )

    np.testing.assert_allclose(_ra_delta(ra_t.cpu().numpy(), ra_ast), 0.0, atol=1e-8)
    np.testing.assert_allclose(dec_t.cpu().numpy() - dec_ast, 0.0, atol=1e-8)


def test_tpv_analytic_jacobian_matches_finite_difference() -> None:
    twcs = WCS(dict(_tpv_header()))
    assert twcs.tpv is not None

    # Probe a range of tangent-plane coordinates (degrees), including near-origin
    # values where TPV radial terms are most numerically delicate.
    u = torch.tensor(
        [0.0, 1e-6, -2e-6, 1e-3, -2e-3, 0.1, -0.2],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [0.0, -3e-6, 4e-6, -2e-3, 3e-3, -0.15, 0.25],
        dtype=torch.float64,
    )

    xi, eta, j11, j12, j21, j22 = twcs.tpv._distort_and_jacobian(u, v)
    eps = 1e-7
    xi_dx, eta_dx = twcs.tpv.distort(u + eps, v)
    xi_dy, eta_dy = twcs.tpv.distort(u, v + eps)

    np.testing.assert_allclose(
        j11.cpu().numpy(), ((xi_dx - xi) / eps).cpu().numpy(), atol=5e-8
    )
    np.testing.assert_allclose(
        j12.cpu().numpy(), ((xi_dy - xi) / eps).cpu().numpy(), atol=5e-8
    )
    np.testing.assert_allclose(
        j21.cpu().numpy(), ((eta_dx - eta) / eps).cpu().numpy(), atol=5e-8
    )
    np.testing.assert_allclose(
        j22.cpu().numpy(), ((eta_dy - eta) / eps).cpu().numpy(), atol=5e-8
    )


def test_tpv_high_order_terms_change_solution() -> None:
    base_header = _tpv_header()
    reduced_header = _tpv_header()

    # Remove high-order terms from comparison model.
    for key in ["PV1_7", "PV2_7", "PV1_39", "PV2_39"]:
        reduced_header[key] = 0.0

    full = WCS(dict(base_header))
    reduced = WCS(dict(reduced_header))

    x = torch.tensor([540.0, 560.0], dtype=torch.float64)
    y = torch.tensor([520.0, 545.0], dtype=torch.float64)

    ra_full, dec_full = full.pixel_to_world(x, y, origin=0)
    ra_reduced, dec_reduced = reduced.pixel_to_world(x, y, origin=0)

    assert torch.max(torch.abs(ra_full - ra_reduced)) > 0.0
    assert torch.max(torch.abs(dec_full - dec_reduced)) > 0.0


def test_tpv_inverse_trace_records_iterations() -> None:
    twcs = WCS(dict(_tpv_header()))
    assert twcs.tpv is not None

    x, y = _sample_pixels()
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=0
    )

    twcs.tpv.set_invert_trace(True)
    x_rt, y_rt = twcs.world_to_pixel(ra_t, dec_t, origin=0)
    trace = twcs.tpv.get_last_invert_trace()
    twcs.tpv.set_invert_trace(False)
    x_ref, y_ref = twcs.world_to_pixel(ra_t, dec_t, origin=0)

    assert trace is not None
    assert int(trace.get("n_points", 0)) == x.size
    assert 0 <= int(trace.get("converged", -1)) <= x.size
    assert float(trace.get("mean_point_iterations", 0.0)) >= 0.0
    assert (trace.get("active_counts") or trace.get("active_counts_sum")) is not None
    np.testing.assert_allclose(x_rt.cpu().numpy(), x_ref.cpu().numpy(), atol=1.0)
    np.testing.assert_allclose(y_rt.cpu().numpy(), y_ref.cpu().numpy(), atol=1.0)


def test_tpv_affine_seed_exact_for_affine_only_tpv() -> None:
    twcs = WCS(dict(_tpv_affine_linear_header()))
    assert twcs.tpv is not None
    twcs.tpv.set_cpp_invert_max_points(0)

    x, y = _sample_pixels()
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    ra_t, dec_t = twcs.pixel_to_world(x_t, y_t, origin=0)

    twcs.tpv.set_invert_trace(True)
    x_rt, y_rt = twcs.world_to_pixel(ra_t, dec_t, origin=0)
    trace = twcs.tpv.get_last_invert_trace()
    twcs.tpv.set_invert_trace(False)

    np.testing.assert_allclose(x_rt.cpu().numpy(), x, atol=2e-5)
    np.testing.assert_allclose(y_rt.cpu().numpy(), y, atol=2e-5)
    assert trace is not None
    # Pure affine TPV should solve in a single Newton evaluation from the affine seed.
    assert int(trace.get("iterations", trace.get("iterations_max", 0))) <= 1
    assert int(trace.get("final_active", 1)) == 0
