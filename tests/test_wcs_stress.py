import torch
import numpy as np
from astropy.wcs import WCS as AstropyWCS
import torchfits.wcs as wcs


def create_high_order_sip_header(order=10):
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRPIX1": 1024.0,
        "CRPIX2": 1024.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
        "A_ORDER": order,
        "B_ORDER": order,
    }

    # Fill with some non-zero coefficients
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if i + j > 1:
                # Decaying coefficients to avoid extreme singularity
                # coeff * 1024^10 should be reasonable
                # 1024^10 ~ 10^30. 1e-35 * 10^30 = 1e-5.
                header[f"A_{i}_{j}"] = 1e-25 * (1e-1 ** (i + j - 2))
                header[f"B_{i}_{j}"] = 1e-25 * (1e-1 ** (i + j - 2))

    return header


def test_sip_high_order_precision():
    order = 10
    header = create_high_order_sip_header(order)

    # Astropy WCS
    awcs = AstropyWCS(header)

    # TorchFits WCS
    twcs = wcs.WCS(header)

    # Test on a large grid
    pixels = torch.cartesian_prod(
        torch.linspace(0, 2048, 10), torch.linspace(0, 2048, 10)
    ).to(torch.float64)

    # Forward: Pixel -> World
    expected_world = awcs.all_pix2world(pixels.numpy(), 0)
    actual_world = twcs.pixel_to_world(pixels, origin=0).numpy()

    # We expect extreme precision (double precision)
    np.testing.assert_allclose(actual_world, expected_world, atol=1e-10)
    print(f"SIP Order {order} Forward Transform Passed!")

    # Inverse: World -> Pixel (Newton-Raphson)
    actual_pix = twcs.world_to_pixel(torch.from_numpy(expected_world), origin=0).numpy()
    np.testing.assert_allclose(actual_pix, pixels.numpy(), atol=1e-5)
    print(f"SIP Order {order} Inverse Transform Passed!")


def create_zpn_header():
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---ZPN",
        "CTYPE2": "DEC--ZPN",
        "CRPIX1": 1024.0,
        "CRPIX2": 1024.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
        # Standard ZPN PVs (Unity model: P(r) = r)
        "PV2_1": 1.0,  # C_1 (coeff of R^1)
        "PV2_2": 0.0,  # C_2 (coeff of R^2)
        "PV2_3": 0.0,  # C_3 (coeff of R^3)
    }
    return header


def test_zpn_precision():
    header = create_zpn_header()
    awcs = AstropyWCS(header)
    twcs = wcs.WCS(header)

    pixels = torch.cartesian_prod(
        torch.linspace(512, 1536, 5), torch.linspace(512, 1536, 5)
    ).to(torch.float64)

    expected_world = awcs.all_pix2world(pixels.numpy(), 0)
    actual_world = twcs.pixel_to_world(pixels, origin=0).numpy()

    # ZPN residuals
    np.testing.assert_allclose(actual_world, expected_world, atol=1e-9)
    print("ZPN Forward Transform Passed!")

    # Inverse
    actual_pix = twcs.world_to_pixel(torch.from_numpy(expected_world), origin=0).numpy()
    np.testing.assert_allclose(actual_pix, pixels.numpy(), atol=1e-5)
    print("ZPN Inverse Transform Passed!")


if __name__ == "__main__":
    test_sip_high_order_precision()
    test_zpn_precision()
