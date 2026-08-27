"""Regression tests for null handling in compressed-image (CompImage) HDUs.

B1: the null-pixel probe used to dlsym a CFITSIO symbol that exists in no
upstream release, so undefined pixels in CompImage HDUs silently decoded as
0 instead of NaN. These tests pin the fixed behavior against astropy as an
independent oracle.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torchfits  # noqa: E402

astropy = pytest.importorskip("astropy")
from astropy.io import fits as afits  # noqa: E402


def _write_comp_image_with_null(path, algorithm="RICE_1"):
    """Write a quantized CompImage containing undefined (NaN) pixels."""
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    data[2, 3] = np.nan
    data[5, 6] = np.nan
    hdu = afits.CompImageHDU(data=data, name="NULLIMG")
    # astropy exposes the cfitsio quantization path; NaNs become ZBLANK/nulls
    hdu.writeto(path, overwrite=True)
    return data


@pytest.mark.parametrize("mmap", [True, False])
def test_compressed_float_null_pixels_decode_as_nan(tmp_path, mmap):
    path = str(tmp_path / "nulls.fits.fz")
    original = _write_comp_image_with_null(path)

    got = torchfits.read_tensor(path, hdu=1, mmap=mmap)

    assert got.dtype == torch.float32
    assert got.shape == (8, 8)
    # Null pixels must be NaN, never silent zeros.
    assert bool(torch.isnan(got[2, 3])), f"expected NaN at [2,3], got {got[2, 3]}"
    assert bool(torch.isnan(got[5, 6])), f"expected NaN at [5,6], got {got[5, 6]}"
    # Non-null pixels must survive bit-exactly.
    mask = ~np.isnan(original)
    assert torch.equal(got[torch.from_numpy(mask)], torch.from_numpy(original[mask]))


def test_compressed_null_parity_with_astropy(tmp_path):
    path = str(tmp_path / "parity.fits.fz")
    _write_comp_image_with_null(path)

    got = torchfits.read_tensor(path, hdu=1).numpy()

    ref = afits.getdata(path)
    ref_mask = (
        ~np.ma.getmaskarray(ref)
        if np.ma.isMaskedArray(ref)
        else np.ones_like(ref, bool)
    )
    ref_data = np.ma.getdata(ref) if np.ma.isMaskedArray(ref) else ref
    assert np.isnan(got).sum() >= 2
    np.testing.assert_allclose(
        got[np.array(ref_mask)], ref_data[np.array(ref_mask)], rtol=0, atol=0
    )


def test_uncompressed_float_nan_still_nan(tmp_path):
    """Guard: nulval on native IEEE must not destroy NaN, Inf, or signed zero."""
    path = str(tmp_path / "plain.fits")
    data = np.array(
        [[0.0, -0.0, 1.0, np.nan], [np.inf, -np.inf, 2.0, 3.0]],
        dtype=np.float32,
    )
    afits.PrimaryHDU(data=data).writeto(path)

    for mmap in (True, False):
        got = torchfits.read_tensor(path, hdu=0, mmap=mmap).numpy()
        assert bool(np.isnan(got[0, 3])), f"mmap={mmap}: NaN lost"
        np.testing.assert_array_equal(
            got[~np.isnan(data)],
            data[~np.isnan(data)],
            err_msg=f"mmap={mmap}: Inf or signed zero destroyed",
        )
