import numpy as np
import torch

from torchfits.wcs.temporal import TemporalWCS


def test_temporal_seconds_roundtrip() -> None:
    twcs = TemporalWCS({"MJDREF": 50000.0, "TIMEUNIT": "s", "TIMESYS": "UTC"})

    t = torch.tensor([0.0, 86400.0, 43200.0], dtype=torch.float64)
    mjd = twcs.to_mjd(t)

    expected = torch.tensor([50000.0, 50001.0, 50000.5], dtype=torch.float64)
    assert torch.allclose(mjd, expected)

    t_back = twcs.from_mjd(mjd)
    assert torch.allclose(t, t_back)


def test_temporal_mjdrefi_mjdreff_support() -> None:
    twcs = TemporalWCS({"MJDREFI": 58000, "MJDREFF": 0.125, "TIMEUNIT": "d"})

    t = torch.tensor([0.0, 2.0], dtype=torch.float64)
    mjd = twcs.to_mjd(t)

    np.testing.assert_allclose(
        mjd.cpu().numpy(), np.array([58000.125, 58002.125]), atol=1e-12
    )


def test_temporal_year_and_century_units() -> None:
    year_wcs = TemporalWCS({"MJDREF": 0.0, "TIMEUNIT": "a"})
    century_wcs = TemporalWCS({"MJDREF": 0.0, "TIMEUNIT": "cy"})

    one_year = torch.tensor([1.0], dtype=torch.float64)
    one_century = torch.tensor([1.0], dtype=torch.float64)

    np.testing.assert_allclose(
        year_wcs.to_mjd(one_year).cpu().numpy(), np.array([365.25]), atol=1e-12
    )
    np.testing.assert_allclose(
        century_wcs.to_mjd(one_century).cpu().numpy(),
        np.array([36525.0]),
        atol=1e-12,
    )


def test_temporal_jd_mjd_conversion_and_iso_output() -> None:
    twcs = TemporalWCS({"MJDREF": 0.0})

    mjd = torch.tensor([0.0, 1.0], dtype=torch.float64)
    jd = twcs.mjd_to_jd(mjd)
    mjd_back = twcs.jd_to_mjd(jd)

    assert torch.allclose(mjd, mjd_back)

    iso = twcs.to_iso8601(torch.tensor([0.0], dtype=torch.float64))
    assert iso[0].startswith("1858-11-17")
