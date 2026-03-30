import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from torchfits.wcs.core import WCS as TorchWCS
from torchfits.wcs.sip import SIP


def _sip_header() -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 512.0
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 5.0
    header["CD1_1"] = -2.8e-4
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 2.8e-4

    header["A_ORDER"] = 3
    header["B_ORDER"] = 3
    header["A_2_0"] = 3.0e-6
    header["A_1_1"] = -2.0e-6
    header["A_0_2"] = 1.0e-6
    header["B_2_0"] = -2.5e-6
    header["B_1_1"] = 2.0e-6
    header["B_0_2"] = 2.5e-6

    # Approximate inverse terms so Astropy and Torch run a similar inverse path.
    header["AP_ORDER"] = 2
    header["BP_ORDER"] = 2
    header["AP_2_0"] = -2.0e-6
    header["AP_1_1"] = 1.5e-6
    header["BP_0_2"] = -2.0e-6
    header["BP_1_1"] = -1.5e-6

    return header


def _sample_pixels() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(350.0, 675.0, 9, dtype=np.float64)
    y = np.linspace(340.0, 690.0, 9, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def test_sip_forward_parity_with_astropy() -> None:
    header = _sip_header()
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x, y = _sample_pixels()

    ra_ast, dec_ast = awcs.all_pix2world(x, y, 0)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=0
    )

    np.testing.assert_allclose(_ra_delta(ra_t.cpu().numpy(), ra_ast), 0.0, atol=1e-8)
    np.testing.assert_allclose(dec_t.cpu().numpy() - dec_ast, 0.0, atol=1e-8)


def test_sip_inverse_roundtrip_parity_with_astropy() -> None:
    header = _sip_header()
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x, y = _sample_pixels()
    ra_ast, dec_ast = awcs.all_pix2world(x, y, 0)

    x_ast, y_ast = awcs.all_world2pix(ra_ast, dec_ast, 0)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_ast),
        torch.from_numpy(dec_ast),
        origin=0,
    )

    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=2e-5)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=2e-5)
    np.testing.assert_allclose(x_t.cpu().numpy(), x, atol=2e-5)
    np.testing.assert_allclose(y_t.cpu().numpy(), y, atol=2e-5)


def test_sip_polynomial_evaluation():
    # A dummy SIP header to test polynomial evaluation independently of WCS
    header = {
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 0.5,
        "A_1_1": 0.3,
        "A_0_2": 0.1,
        "B_2_0": 0.2,
        "B_1_1": 0.4,
        "B_0_2": 0.6,
        "AP_ORDER": 1,
        "BP_ORDER": 1,
        "AP_1_0": 0.1,
        "BP_0_1": 0.2,
    }

    sip = SIP(header)

    u = torch.tensor([1.0, 2.0], dtype=torch.float64)
    v = torch.tensor([1.0, 3.0], dtype=torch.float64)

    # Forward evaluation
    # f(u, v) = A_2_0 * u^2 + A_1_1 * u * v + A_0_2 * v^2
    # g(u, v) = B_2_0 * u^2 + B_1_1 * u * v + B_0_2 * v^2
    # Point 1 (u=1, v=1):
    # f(1, 1) = 0.5 * 1 + 0.3 * 1 + 0.1 * 1 = 0.9
    # g(1, 1) = 0.2 * 1 + 0.4 * 1 + 0.6 * 1 = 1.2
    # Result: u + f = 1 + 0.9 = 1.9, v + g = 1 + 1.2 = 2.2

    # Point 2 (u=2, v=3):
    # f(2, 3) = 0.5 * 4 + 0.3 * 6 + 0.1 * 9 = 2.0 + 1.8 + 0.9 = 4.7
    # g(2, 3) = 0.2 * 4 + 0.4 * 6 + 0.6 * 9 = 0.8 + 2.4 + 5.4 = 8.6
    # Result: u + f = 2 + 4.7 = 6.7, v + g = 3 + 8.6 = 11.6

    xd, yd = sip.distort(u, v)

    assert torch.allclose(xd, torch.tensor([1.9, 6.7], dtype=torch.float64))
    assert torch.allclose(yd, torch.tensor([2.2, 11.6], dtype=torch.float64))

    # Inverse evaluation
    # AP_1_0 = 0.1 -> u + 0.1 * u
    # BP_0_1 = 0.2 -> v + 0.2 * v
    xu, yu = sip.undistort(u, v)

    assert torch.allclose(xu, torch.tensor([1.1, 2.2], dtype=torch.float64))
    assert torch.allclose(yu, torch.tensor([1.2, 3.6], dtype=torch.float64))


def test_sip_edge_cases():
    # Test empty dictionaries / missing orders
    sip_empty = SIP({})
    u = torch.tensor([1.0], dtype=torch.float64)
    v = torch.tensor([2.0], dtype=torch.float64)

    # Should just return inputs if no SIP coeffs
    xd, yd = sip_empty.distort(u, v)
    assert torch.allclose(xd, u)
    assert torch.allclose(yd, v)

    xu, yu = sip_empty.undistort(u, v)
    assert torch.allclose(xu, u)
    assert torch.allclose(yu, v)

    # Test distort with empty tensors
    u_empty = torch.tensor([], dtype=torch.float64)
    v_empty = torch.tensor([], dtype=torch.float64)

    sip_full = SIP(
        {
            "A_ORDER": 2,
            "B_ORDER": 2,
            "A_2_0": 0.5,
            "B_2_0": 0.5,
        }
    )
    xd, yd = sip_full.distort(u_empty, v_empty)
    assert xd.numel() == 0
    assert yd.numel() == 0

    xu, yu = sip_full.undistort(u_empty, v_empty)
    assert xu.numel() == 0
    assert yu.numel() == 0


def test_sip_invert_distortion():
    # Test invert distortion using newton solver
    header = {
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 1e-4,
        "A_1_1": -1e-4,
        "A_0_2": 2e-4,
        "B_2_0": -2e-4,
        "B_1_1": 1e-4,
        "B_0_2": 1e-4,
    }
    sip = SIP(header)

    u = torch.tensor([10.0, -5.0, 0.0], dtype=torch.float64)
    v = torch.tensor([5.0, 10.0, 0.0], dtype=torch.float64)

    # Forward
    xd, yd = sip.distort(u, v)

    # Invert back to original
    u_inv, v_inv = sip.invert_distortion(xd, yd, max_iter=10, tol=1e-8)

    assert torch.allclose(u_inv, u, atol=1e-6)
    assert torch.allclose(v_inv, v, atol=1e-6)


def test_sip_device_transfer():
    header = {
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 0.5,
        "B_2_0": 0.5,
    }
    sip = SIP(header)

    # Check current device
    assert sip._a_c.device == torch.device("cpu")

    # Move to cpu again
    sip = sip.to(torch.device("cpu"))
    assert sip._a_c.device == torch.device("cpu")

    # If cuda is available, test moving to cuda
    if torch.cuda.is_available():
        sip = sip.to(torch.device("cuda"))
        assert sip._a_c.device.type == "cuda"

        # Test distort on cuda
        u = torch.tensor([1.0], device="cuda", dtype=torch.float64)
        v = torch.tensor([2.0], device="cuda", dtype=torch.float64)
        xd, yd = sip.distort(u, v)
        assert xd.device.type == "cuda"
        assert yd.device.type == "cuda"
