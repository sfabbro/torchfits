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

def test_parse_wat_keywords_empty():
    from torchfits.wcs.legacy import parse_wat_keywords
    assert parse_wat_keywords({}, prefix="WAT1") == ""

def test_parse_wat_keywords_no_match():
    from torchfits.wcs.legacy import parse_wat_keywords
    header = {"WAT2_001": "something"}
    assert parse_wat_keywords(header, prefix="WAT1") == ""

def test_parse_wat_keywords_normal():
    from torchfits.wcs.legacy import parse_wat_keywords
    header = {
        "WAT1_001": "wtype=tnx ",
        "WAT1_002": "lngcor = \"3 ",
        "WAT1_003": "1 1\"",
    }
    assert parse_wat_keywords(header, prefix="WAT1") == "wtype=tnx lngcor = \"3 1 1\""

def test_parse_wat_keywords_gap():
    from torchfits.wcs.legacy import parse_wat_keywords
    header = {
        "WAT1_001": "part1",
        "WAT1_003": "part3",
    }
    # It stops at the first missing index (002)
    assert parse_wat_keywords(header, prefix="WAT1") == "part1"

def test_parse_wat_keywords_non_string():
    from torchfits.wcs.legacy import parse_wat_keywords
    header = {
        "WAT1_001": 123,
        "WAT1_002": 45.6,
    }
    assert parse_wat_keywords(header, prefix="WAT1") == "12345.6"

def test_parse_wat_keywords_max_limit():
    from torchfits.wcs.legacy import parse_wat_keywords
    header = {f"WAT1_{n:03d}": "a" for n in range(1, 105)}
    # It stops at 99 because range is (1, 100)
    assert len(parse_wat_keywords(header, prefix="WAT1")) == 99

def test_parse_wat_keywords_default_prefix():
    from torchfits.wcs.legacy import parse_wat_keywords
    header = {
        "WAT_001": "default1",
        "WAT_002": "default2",
    }
    assert parse_wat_keywords(header) == "default1default2"


def test_legacy_polynomial_chebyshev_legendre() -> None:
    from torchfits.wcs.legacy import LegacyPolynomial

    # Test Chebyshev (func_type=1), order=3
    # P_0 = 1
    # P_1 = x
    # P_2 = 2x^2 - 1
    # P_3 = 4x^3 - 3x (not used if order=3, order means 0,1,2 terms)
    # Let's use order_x=3, order_y=1, func_type=1
    # bounds [-1, 1], so nx = x
    # Coeffs: c_00, c_10, c_20
    # term = c_00*1*1 + c_10*x*1 + c_20*(2x^2 - 1)*1
    poly_cheb = LegacyPolynomial("1 3 1 0 -1.0 1.0 -1.0 1.0 1.0 2.0 3.0")
    assert poly_cheb.func_type == 1

    x = torch.tensor([0.5], dtype=torch.float64)
    y = torch.tensor([0.0], dtype=torch.float64)
    val_cheb = poly_cheb.evaluate(x, y)

    # expected: 1.0*1 + 2.0*0.5 + 3.0*(2*(0.5**2) - 1) = 1.0 + 1.0 + 3.0*(0.5 - 1) = 2.0 - 1.5 = 0.5
    np.testing.assert_allclose(val_cheb.cpu().numpy(), 0.5, atol=1e-12)

    # Test Legendre (func_type=2), order=3
    # P_0 = 1
    # P_1 = x
    # P_2 = (3x^2 - 1)/2
    # Coeffs: c_00=1.0, c_10=2.0, c_20=3.0
    poly_leg = LegacyPolynomial("2 3 1 0 -1.0 1.0 -1.0 1.0 1.0 2.0 3.0")
    assert poly_leg.func_type == 2

    val_leg = poly_leg.evaluate(x, y)

    # expected: 1.0*1 + 2.0*0.5 + 3.0*(3*(0.5**2) - 1)/2 = 1.0 + 1.0 + 3.0*(0.75 - 1)/2 = 2.0 + 3.0*(-0.25)/2 = 2.0 - 0.375 = 1.625
    np.testing.assert_allclose(val_leg.cpu().numpy(), 1.625, atol=1e-12)

    # Test simple polynomial fallback if bounds don't match, or fallback if ftype=something else
    poly_other = LegacyPolynomial("4 3 1 0 -1.0 1.0 -1.0 1.0 1.0 2.0 3.0")
    # P_2 for simple polynomial is u * P_1 = u^2
    val_other = poly_other.evaluate(x, y)
    # expected: 1.0*1 + 2.0*0.5 + 3.0*(0.5**2) = 1.0 + 1.0 + 3.0*0.25 = 2.75
    np.testing.assert_allclose(val_other.cpu().numpy(), 2.75, atol=1e-12)

    # Test empty param string
    poly_empty = LegacyPolynomial("")
    assert len(poly_empty.coeffs) == 0


def test_extract_tnx_coeffs() -> None:
    from torchfits.wcs.legacy import extract_tnx_coeffs

    wat = 'wtype=tnx lngcor="1 2 3" latcor=4'
    assert extract_tnx_coeffs(wat, "lngcor") == "1 2 3"
    assert extract_tnx_coeffs(wat, "latcor") == "4"
    assert extract_tnx_coeffs(wat, "missing") is None


def test_extract_zpx_params() -> None:
    from torchfits.wcs.legacy import extract_zpx_params

    wat_data = {
        0: "projp1 = 0.5 projp3= 1.5",
        1: "PROJP2 = 2.0",
    }

    params = extract_zpx_params(wat_data)
    assert params == {"PV2_1": 0.5, "PV2_3": 1.5, "PV2_2": 2.0}


def test_project_tnx() -> None:
    from torchfits.wcs.legacy import project_tnx

    # Test None wat_data returns original
    xi = torch.tensor([1.0], dtype=torch.float64)
    eta = torch.tensor([2.0], dtype=torch.float64)

    xi_out, eta_out = project_tnx(xi, eta, None, None)
    torch.testing.assert_close(xi, xi_out)
    torch.testing.assert_close(eta, eta_out)

    # Test with wat_data
    # Use simple polynomial shift: 3 1 1 0 -5.0 5.0 -5.0 5.0 <shift>
    wat_data = {
        1: 'wtype=tnx lngcor="3 1 1 0 -5.0 5.0 -5.0 5.0 0.5"',
        2: 'wtype=tnx latcor="3 1 1 0 -5.0 5.0 -5.0 5.0 -0.5"',
    }

    xi_out, eta_out = project_tnx(xi, eta, None, wat_data)
    torch.testing.assert_close(xi_out, xi + 0.5)
    torch.testing.assert_close(eta_out, eta - 0.5)


def test_project_zpx() -> None:
    from torchfits.wcs.legacy import project_zpx
    from torchfits.wcs.zenithal import project_zenithal

    xi = torch.tensor([0.0], dtype=torch.float64)
    eta = torch.tensor([0.0], dtype=torch.float64)

    # Test None wat_data calls project_zenithal with ZPN
    phi_zpn, theta_zpn = project_zenithal(xi, eta, "ZPN", None)
    phi_zpx, theta_zpx = project_zpx(xi, eta, None, None)

    torch.testing.assert_close(phi_zpn, phi_zpx)
    torch.testing.assert_close(theta_zpn, theta_zpx)

    # Test with wat_data
    # Let's set some projection parameters to ensure they get passed to ZPN
    wat_data = {
        0: "projp1 = 1.0",
        1: 'wtype=zpx lngcor="3 1 1 0 -5.0 5.0 -5.0 5.0 0.0"',
    }

    phi_zpx_params, theta_zpx_params = project_zpx(xi, eta, None, wat_data)

    # Same as calling ZPN with PV2_1 = 1.0 directly
    phi_expected, theta_expected = project_zenithal(xi, eta, "ZPN", {"PV2_1": 1.0})

    torch.testing.assert_close(phi_zpx_params, phi_expected)
    torch.testing.assert_close(theta_zpx_params, theta_expected)
