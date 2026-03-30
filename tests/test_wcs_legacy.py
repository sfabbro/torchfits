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
