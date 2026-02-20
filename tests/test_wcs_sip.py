import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from torchfits.wcs.core import WCS as TorchWCS


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
