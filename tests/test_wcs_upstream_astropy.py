from __future__ import annotations

import numpy as np
import pytest
import torch

fits = pytest.importorskip("astropy.io.fits")
AstropyWCS = pytest.importorskip("astropy.wcs").WCS
get_pkg_data_contents = pytest.importorskip("astropy.utils.data").get_pkg_data_contents

from torchfits.wcs.core import WCS as TorchWCS  # noqa: E402


FIXTURE_CASES: dict[str, np.ndarray] = {
    "outside_sky.hdr": np.array(
        [[100.0, 500.0], [200.0, 200.0], [1000.0, 1000.0]], dtype=np.float64
    ),
    "zpn-hole.hdr": np.array([[110.0, 110.0], [0.0, 0.0], [256.0, 256.0]]),
    "siponly.hdr": np.array([[0.0, 0.0], [100.0, 200.0], [512.0, 512.0]]),
    "tpvonly.hdr": np.array([[0.0, 0.0], [100.0, -100.0], [200.0, 200.0]]),
}

_NAN_BOUNDARY_XFAILS = {"outside_sky.hdr", "zpn-hole.hdr"}


def _load_wcs_pair(name: str) -> tuple[AstropyWCS, dict[str, object]]:
    raw_header = get_pkg_data_contents(
        f"data/{name}", package="astropy.wcs.tests", encoding="binary"
    )
    awcs = AstropyWCS(raw_header)
    header = fits.Header.fromstring(raw_header, sep="")
    return awcs, dict(header)


def _ra_wrap_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


@pytest.mark.parametrize("fixture_name", sorted(FIXTURE_CASES))
def test_astropy_fixture_pixel_to_world_matches_torchfits(fixture_name: str) -> None:
    if fixture_name in _NAN_BOUNDARY_XFAILS:
        pytest.xfail(
            reason=f"{fixture_name}: torchfits does not propagate NaN for out-of-domain pixels"
        )
    awcs, torch_header = _load_wcs_pair(fixture_name)
    twcs = TorchWCS(torch_header)
    xy = FIXTURE_CASES[fixture_name]

    ra_ast, dec_ast = awcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(xy[:, 0]), torch.from_numpy(xy[:, 1]), origin=0
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()

    np.testing.assert_array_equal(np.isnan(ra_t_np), np.isnan(ra_ast))
    np.testing.assert_array_equal(np.isnan(dec_t_np), np.isnan(dec_ast))

    valid = (
        np.isfinite(ra_ast)
        & np.isfinite(dec_ast)
        & np.isfinite(ra_t_np)
        & np.isfinite(dec_t_np)
    )
    assert np.any(valid), f"Expected finite comparison points for {fixture_name}"

    np.testing.assert_allclose(
        _ra_wrap_delta_deg(ra_t_np[valid], ra_ast[valid]), 0.0, atol=5e-6
    )
    np.testing.assert_allclose(dec_t_np[valid], dec_ast[valid], atol=5e-6)


@pytest.mark.parametrize("fixture_name", sorted(FIXTURE_CASES))
def test_astropy_fixture_world_to_pixel_matches_torchfits(fixture_name: str) -> None:
    if fixture_name in _NAN_BOUNDARY_XFAILS:
        pytest.xfail(
            reason=f"{fixture_name}: torchfits does not propagate NaN for out-of-domain pixels"
        )
    awcs, torch_header = _load_wcs_pair(fixture_name)
    twcs = TorchWCS(torch_header)
    xy = FIXTURE_CASES[fixture_name]

    ra_ast, dec_ast = awcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"Expected finite inverse points for {fixture_name}"

    x_ast, y_ast = awcs.all_world2pix(ra_ast[valid], dec_ast[valid], 0)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_ast[valid]),
        torch.from_numpy(dec_ast[valid]),
        origin=0,
    )

    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=2e-4)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=2e-4)
    np.testing.assert_allclose(x_t.cpu().numpy(), xy[valid, 0], atol=2e-4)
    np.testing.assert_allclose(y_t.cpu().numpy(), xy[valid, 1], atol=2e-4)
