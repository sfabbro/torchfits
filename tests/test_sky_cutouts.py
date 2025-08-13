import numpy as np
import torch
import pytest

import torchfits as tf


def _make_wcs_image(path: str, size: int = 512) -> None:
    try:
        from astropy.io import fits as apfits  # type: ignore
    except Exception:  # pragma: no cover - optional dep in some envs
        pytest.skip("astropy required for WCS image creation")
    data = (np.random.rand(size, size).astype(np.float32) * 1000).astype(np.float32)
    hdu = apfits.PrimaryHDU(data)
    hdr = hdu.header
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 200.0
    hdr["CRVAL2"] = 0.0
    hdr["CRPIX1"] = size / 2
    hdr["CRPIX2"] = size / 2
    hdr["CDELT1"] = -0.0002777777778  # ~1 arcsec/pixel
    hdr["CDELT2"] = 0.0002777777778
    hdu.writeto(path, overwrite=True)


def test_read_multi_sky_cutouts_basic(tmp_path):
    p = tmp_path / "wcs.fits"
    _make_wcs_image(str(p), size=512)

    # Build sky points near center
    hdr = tf.get_header(str(p), hdu=0)
    world_center, _ = tf.pixel_to_world([[256, 256]], hdr)
    ra0, dec0 = float(world_center[0][0]), float(world_center[0][1])
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(8):
        dra = (rng.random() - 0.5) * 0.02
        ddec = (rng.random() - 0.5) * 0.02
        pts.append((ra0 + dra, dec0 + ddec))

    radius_arcsec = 30.0
    outs = tf.read_multi_sky_cutouts(str(p), pts, radius_arcsec, hdu=0, device="cpu")
    # Shapes
    assert len(outs) == len(pts)
    sizes = [o.shape[-1] for o in outs if torch.is_tensor(o)]
    assert len(set(sizes)) == 1 and sizes[0] > 0

    # Parity vs slicing from full image using the same WCS transform
    img_tuple = tf.read(str(p), hdu=0)
    img = img_tuple[0] if isinstance(img_tuple, tuple) else img_tuple
    pix_points, _ = tf.world_to_pixel([[ra, dec] for (ra, dec) in pts], hdr)
    # Estimate half window in pixels (assumes 1"/pix from header)
    cdelt2 = float(hdr.get("CDELT2", 1.0))
    pix_per_deg = 1.0 / abs(cdelt2)
    half_hw = int(round((radius_arcsec / 3600.0) * pix_per_deg))
    for i, o in enumerate(outs):
        x, y = float(pix_points[i][0]), float(pix_points[i][1])
        ys = max(0, int(round(y)) - half_hw)
        xs = max(0, int(round(x)) - half_hw)
    ref = img[..., ys : ys + 2 * half_hw, xs : xs + 2 * half_hw]
    assert torch.is_tensor(o) and torch.is_tensor(ref)
    assert o.shape == ref.shape
    assert torch.allclose(o, ref)
