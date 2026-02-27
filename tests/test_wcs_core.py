import numpy as np
import torch

from torchfits.wcs.core import WCS


def _tan_header() -> dict[str, float | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 256,
        "NAXIS2": 256,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 128.0,
        "CRPIX2": 128.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }


def test_tan_sip_projection_suffix_parsing() -> None:
    header = _tan_header()
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["A_ORDER"] = 2
    header["B_ORDER"] = 2
    header["A_2_0"] = 1.0e-4
    header["B_0_2"] = -2.0e-4

    wcs = WCS(header)

    assert wcs._proj_code == "TAN"
    assert wcs.sip is not None


def test_header_unit_and_pole_keywords_preserved() -> None:
    header = _tan_header()
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["LONPOLE"] = 123.0
    header["LATPOLE"] = 77.0

    wcs = WCS(header)

    assert wcs.wcs_params["CUNIT1"] == "deg"
    assert wcs.wcs_params["CUNIT2"] == "deg"
    assert wcs.phi_p == 123.0
    assert wcs.theta_p == 77.0


def test_cd_matrix_takes_priority_over_cdelt() -> None:
    header = _tan_header()
    header["CDELT1"] = -0.2
    header["CDELT2"] = 0.2

    wcs = WCS(header)

    assert torch.allclose(
        wcs.cd, torch.tensor([[-0.01, 0.0], [0.0, 0.01]], dtype=torch.float64)
    )


def test_world_to_pixel_default_origin_zero() -> None:
    wcs = WCS(_tan_header())

    world = torch.tensor([[180.0, 0.0]], dtype=torch.float64)
    pixel = wcs.world_to_pixel(world)

    expected = torch.tensor([[127.0, 127.0]], dtype=torch.float64)
    assert torch.allclose(pixel, expected, atol=1e-9)


def test_dtype_is_promoted_to_float64_for_transform_inputs() -> None:
    wcs = WCS(_tan_header())

    pixels = torch.tensor([[127.0, 127.0], [140.0, 150.0]], dtype=torch.float32)
    world = wcs.pixel_to_world(pixels)

    assert world.dtype == torch.float64

    pixels_back = wcs.world_to_pixel(world.to(torch.float32))
    assert pixels_back.dtype == torch.float64


def test_to_moves_all_cached_tensors_and_tpv_coefficients() -> None:
    header = _tan_header()
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["PV1_1"] = 1.0
    header["PV2_2"] = 1.0
    header["PV1_7"] = 1.0e-7
    header["PV2_7"] = 1.0e-7

    wcs = WCS(header)

    wcs.to("cpu")
    assert wcs.crpix.device.type == "cpu"
    assert wcs.cd.device.type == "cpu"
    assert wcs._sin_dec_p.device.type == "cpu"
    assert wcs.tpv is not None
    assert wcs.tpv.c1.device.type == "cpu"
    assert wcs.tpv.c2.device.type == "cpu"

    if torch.cuda.is_available():
        wcs.to("cuda")
        assert wcs.crpix.device.type == "cuda"
        assert wcs.cd.device.type == "cuda"
        assert wcs._sin_dec_p.device.type == "cuda"
        assert wcs.tpv.c1.device.type == "cuda"
        assert wcs.tpv.c2.device.type == "cuda"

        # Smoke check transform path on CUDA tensors.
        x = torch.tensor([120.0, 140.0], device="cuda", dtype=torch.float64)
        y = torch.tensor([120.0, 140.0], device="cuda", dtype=torch.float64)
        ra, dec = wcs.pixel_to_world(x, y)
        assert torch.isfinite(ra).all()
        assert torch.isfinite(dec).all()


def test_origin_zero_and_one_consistent_shift() -> None:
    wcs = WCS(_tan_header())

    x = np.array([120.0, 140.0], dtype=np.float64)
    y = np.array([120.0, 140.0], dtype=np.float64)

    w0 = wcs.pixel_to_world(torch.from_numpy(x), torch.from_numpy(y), origin=0)
    w1 = wcs.pixel_to_world(
        torch.from_numpy(x + 1.0), torch.from_numpy(y + 1.0), origin=1
    )

    np.testing.assert_allclose(w0[0].cpu().numpy(), w1[0].cpu().numpy(), atol=1e-12)
    np.testing.assert_allclose(w0[1].cpu().numpy(), w1[1].cpu().numpy(), atol=1e-12)


def test_cpp_helpers_match_python_fallbacks_for_inverse_rotation_and_tan_intermediate() -> (
    None
):
    wcs = WCS(_tan_header()).to("cpu")

    if (
        wcs._cpp_inverse_spherical_rotation_pole is None
        or wcs._cpp_tan_intermediate_from_radec is None
    ):
        return

    ra = torch.tensor([179.9, 180.0, 180.3, 181.2], dtype=torch.float64)
    dec = torch.tensor([-0.4, 0.0, 0.2, 0.8], dtype=torch.float64)

    phi_cpp, theta_cpp = wcs._inverse_spherical_rotation(ra, dec, "pole")
    xi_cpp, eta_cpp = wcs._tan_intermediate_from_radec(ra, dec)

    old_inv = wcs._cpp_inverse_spherical_rotation_pole
    old_tan = wcs._cpp_tan_intermediate_from_radec
    try:
        wcs._cpp_inverse_spherical_rotation_pole = None
        wcs._cpp_tan_intermediate_from_radec = None
        phi_py, theta_py = wcs._inverse_spherical_rotation(ra, dec, "pole")
        xi_py, eta_py = wcs._tan_intermediate_from_radec(ra, dec)
    finally:
        wcs._cpp_inverse_spherical_rotation_pole = old_inv
        wcs._cpp_tan_intermediate_from_radec = old_tan

    assert torch.allclose(phi_cpp, phi_py, atol=1e-12, rtol=0.0)
    assert torch.allclose(theta_cpp, theta_py, atol=1e-12, rtol=0.0)
    assert torch.allclose(xi_cpp, xi_py, atol=1e-12, rtol=0.0)
    assert torch.allclose(eta_cpp, eta_py, atol=1e-12, rtol=0.0)
