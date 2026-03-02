import torch

from torchfits.wcs.core import WCS


def _arc_header() -> dict[str, float | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 120,
        "NAXIS2": 120,
        "CTYPE1": "RA---ARC",
        "CTYPE2": "DEC--ARC",
        "CRPIX1": 60.0,
        "CRPIX2": 60.0,
        "CRVAL1": 0.0,
        "CRVAL2": 90.0,
        "CD1_1": 1.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0,
    }


def _zpx_header_linear() -> dict[str, float | str]:
    header = dict(_arc_header())
    header["CTYPE1"] = "RA---ZPX"
    header["CTYPE2"] = "DEC--ZPX"
    header["WAT1_001"] = "wtype=zpx projp1=1.0"
    header["WAT2_001"] = "wtype=zpx"
    return header


def test_zpx_projp1_linear_forward_is_finite() -> None:
    zpx = WCS(_zpx_header_linear())

    x = torch.linspace(35.0, 85.0, 8, dtype=torch.float64)
    y = torch.linspace(35.0, 85.0, 8, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    px = xx.reshape(-1)
    py = yy.reshape(-1)

    ra_zpx, dec_zpx = zpx.pixel_to_world(px, py, origin=0)
    assert torch.isfinite(ra_zpx).all()
    assert torch.isfinite(dec_zpx).all()
    assert ra_zpx.shape == px.shape
    assert dec_zpx.shape == py.shape


def test_zpx_wat_distortion_changes_solution() -> None:
    base = WCS(_zpx_header_linear())

    distorted_header = _zpx_header_linear()
    distorted_header["WAT1_001"] = (
        'wtype=zpx projp1=1.0 lngcor = "3 1 1 0 -10.0 10.0 -10.0 10.0 0.4"'
    )
    distorted_header["WAT2_001"] = (
        'wtype=zpx latcor = "3 1 1 0 -10.0 10.0 -10.0 10.0 -0.3"'
    )
    distorted = WCS(distorted_header)

    x = torch.tensor([59.0], dtype=torch.float64)
    y = torch.tensor([59.0], dtype=torch.float64)

    ra_base, dec_base = base.pixel_to_world(x, y, origin=0)
    ra_dist, dec_dist = distorted.pixel_to_world(x, y, origin=0)

    assert torch.abs(ra_dist - ra_base) > 1e-3
    assert torch.abs(dec_dist - dec_base) > 1e-3
