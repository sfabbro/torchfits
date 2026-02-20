import numpy as np
import torch

from torchfits.wcs.core import WCS
from torchfits.wcs.legacy import LegacyPolynomial


def test_legacy_polynomial_parse_and_evaluate() -> None:
    # func_type=3 (simple polynomial), order_x=2, order_y=2, cross_terms=1
    # bounds [-1, 1] keep normalized coordinates equal to x/y.
    poly = LegacyPolynomial("3 2 2 1 -1.0 1.0 -1.0 1.0 1.0 2.0 3.0 4.0")

    assert poly.func_type == 3
    assert poly.order_x == 2
    assert poly.order_y == 2
    assert poly.cross_terms == 1
    assert poly.x_bounds == (-1.0, 1.0)
    assert poly.y_bounds == (-1.0, 1.0)

    x = torch.tensor([0.2], dtype=torch.float64)
    y = torch.tensor([-0.4], dtype=torch.float64)
    val = poly.evaluate(x, y)

    expected = 1.0 + 2.0 * 0.2 + 3.0 * (-0.4) + 4.0 * (0.2 * -0.4)
    np.testing.assert_allclose(val.cpu().numpy(), expected, atol=1e-12)


def _base_tan_header() -> dict[str, float | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 100,
        "NAXIS2": 100,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 50.0,
        "CRPIX2": 50.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.1,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.1,
    }


def test_tnx_zero_polynomial_matches_tan() -> None:
    tan_header = _base_tan_header()

    tnx_header = dict(tan_header)
    tnx_header["CTYPE1"] = "RA---TNX"
    tnx_header["CTYPE2"] = "DEC--TNX"
    tnx_header["WAT1_001"] = 'wtype=tnx lngcor = "3 1 1 0 -5.0 5.0 -5.0 5.0 0.0"'
    tnx_header["WAT2_001"] = 'wtype=tnx latcor = "3 1 1 0 -5.0 5.0 -5.0 5.0 0.0"'

    tan = WCS(tan_header)
    tnx = WCS(tnx_header)

    x = torch.linspace(30.0, 70.0, 8, dtype=torch.float64)
    y = torch.linspace(30.0, 70.0, 8, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="xy")

    ra_tan, dec_tan = tan.pixel_to_world(xx.reshape(-1), yy.reshape(-1), origin=0)
    ra_tnx, dec_tnx = tnx.pixel_to_world(xx.reshape(-1), yy.reshape(-1), origin=0)

    np.testing.assert_allclose(ra_tnx.cpu().numpy(), ra_tan.cpu().numpy(), atol=1e-9)
    np.testing.assert_allclose(dec_tnx.cpu().numpy(), dec_tan.cpu().numpy(), atol=1e-9)


def test_tnx_constant_polynomial_applies_expected_shift_near_center() -> None:
    tnx_header = _base_tan_header()
    tnx_header["CTYPE1"] = "RA---TNX"
    tnx_header["CTYPE2"] = "DEC--TNX"
    tnx_header["WAT1_001"] = 'wtype=tnx lngcor = "3 1 1 0 -5.0 5.0 -5.0 5.0 0.3"'
    tnx_header["WAT2_001"] = 'wtype=tnx latcor = "3 1 1 0 -5.0 5.0 -5.0 5.0 -0.2"'

    tan = WCS(_base_tan_header())
    tnx = WCS(tnx_header)

    x0 = torch.tensor([49.0], dtype=torch.float64)
    y0 = torch.tensor([49.0], dtype=torch.float64)

    ra_tan, dec_tan = tan.pixel_to_world(x0, y0, origin=0)
    ra_tnx, dec_tnx = tnx.pixel_to_world(x0, y0, origin=0)

    # At the projection center, TNX constant terms should appear as near-linear sky shifts.
    np.testing.assert_allclose((ra_tnx - ra_tan).cpu().numpy(), 0.3, atol=2e-2)
    np.testing.assert_allclose((dec_tnx - dec_tan).cpu().numpy(), -0.2, atol=2e-2)
