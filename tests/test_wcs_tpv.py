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
