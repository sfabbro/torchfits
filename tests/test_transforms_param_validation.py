"""Degenerate scale/threshold parameters must raise, not silently destroy data.

Round-2 findings R2-030, R2-031 and R2-032, all in ``torchfits.transforms``.
Each one is a *constructor* parameter that inverts or zeroes an arithmetic
identity, so the transform stops rejecting outliers / stops scaling and
rewrites the frame -- with no exception. Every class involved already had an
in-package sibling that refused the same mistake:

    AsymmetricSigmaClip   "n_low and n_high must be > 0"        (clip.py)
    AffineTransform       "AffineTransform scale must be non-zero"  (normalize.py)
    InterquantileScale    "Expected 0.0 <= q_low < q_high <= 1.0"    (normalize.py)

Measured before the fix, on a 10x10 frame of 100 x 10.0 plus outliers
900 / -500 / 12 / 11 (``frame()`` below):

    SigmaClip(n_sigma=3.0)  -> range (10.0, 10.0)   4/100 clipped   ok
    SigmaClip(n_sigma=0.0)  -> range (0.0, 0.0)   100/100 clipped   <<<
    SigmaClip(n_sigma=-3.0) -> range (0.0, 0.0)   100/100 clipped   <<<
    SigmaClip(max_iter=0)   -> output identical to input, no error  <<<

A non-positive threshold inverts ``[mean - n*std, mean + n*std]``, so no pixel
can ever be inside it: every pixel is "clipped", the keep-mask empties, and the
next iteration divides an empty group by ``clamp_min(count, 1)`` -- the whole
frame comes back as the constant 0.0 (NaN under ``fill="nan"``). The output
still *claims* the input's shape and dtype.

    PercentileClipNormalize(1.0, 99.0)    -> range (0.0, 1.0)          ok
    PercentileClipNormalize(99.0, 1.0)    -> range (1.0, 1.0)   <<<
    PercentileClipNormalize(150.0, 200.0) -> RuntimeError: quantile() ...
                                             (raw torch error, not ValueError)

    FITSHeaderScale(bscale=0.0, bzero=5.0) -> forward (5.0, 5.0),
                                             inverse (nan, nan)   <<<

The positive-side boundaries are pinned as hard as the rejections: a tiny but
positive ``n_sigma``, ``max_iter=1``, ``lower_pct == upper_pct`` (a supported
degenerate span -- ``forward`` substitutes 1.0 for the divisor),
``lower_pct=0``/``upper_pct=100`` and a negative BSCALE all remain legal. Each
one is a documented behaviour a too-strict guard would silently break.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torchfits.transforms import (  # noqa: E402
    AsymmetricSigmaClip,
    FITSScaleColumns,
    FITSHeaderScale,
    PercentileClipNormalize,
    SigmaClip,
)


def frame() -> "torch.Tensor":
    """100 identical pixels plus four outliers -- a frame sigma-clip must edit."""
    x = torch.full((10, 10), 10.0)
    x[0, 0], x[0, 1], x[0, 2], x[0, 3] = 900.0, -500.0, 12.0, 11.0
    return x


def ramp() -> "torch.Tensor":
    """A strictly increasing frame, so every quantile is distinct."""
    return torch.arange(100, dtype=torch.float32).reshape(10, 10)


# --------------------------------------------------------------------------
# R2-030: sigma-clip thresholds that no pixel can satisfy
# --------------------------------------------------------------------------


class TestSigmaClipThresholdValidation:
    @pytest.mark.parametrize("n_sigma", [0.0, -0.0, -1e-9, -1.0, -3.0])
    def test_non_positive_n_sigma_raises(self, n_sigma) -> None:
        with pytest.raises(ValueError, match="n_sigma"):
            SigmaClip(n_sigma=n_sigma)

    @pytest.mark.parametrize("n_sigma", [float("nan"), math.nan])
    def test_nan_n_sigma_raises(self, n_sigma) -> None:
        # ``nan > 0`` is False, so the inverted-interval frame wipe applies
        # just as it does for a negative threshold.
        with pytest.raises(ValueError, match="n_sigma"):
            SigmaClip(n_sigma=n_sigma)

    @pytest.mark.parametrize("max_iter", [0, -1, -5])
    def test_max_iter_below_one_raises(self, max_iter) -> None:
        # ``for _ in range(0)`` never runs: no threshold is ever computed and
        # forward() hands the input straight back, reporting every pixel kept.
        with pytest.raises(ValueError, match="max_iter"):
            SigmaClip(max_iter=max_iter)

    def test_non_positive_n_sigma_is_rejected_before_the_fill_check(self) -> None:
        # Both problems, one error: the threshold is the more dangerous one.
        with pytest.raises(ValueError, match="n_sigma"):
            SigmaClip(n_sigma=0.0, fill="bogus")

    @pytest.mark.parametrize("n_sigma", [1e-12, 0.1, 3.0, 100.0])
    def test_positive_n_sigma_still_accepted(self, n_sigma) -> None:
        assert SigmaClip(n_sigma=n_sigma).n_sigma == pytest.approx(n_sigma)

    @pytest.mark.parametrize("max_iter", [1, 2, 50])
    def test_max_iter_of_one_still_clips(self, max_iter) -> None:
        """max_iter=1 is the boundary, not a no-op: the outlier must go."""
        clip = SigmaClip(n_sigma=3.0, max_iter=max_iter, fill="nan")
        out = clip(frame())
        assert math.isnan(float(out[0, 0])), "the 900 outlier must be clipped"
        assert float(out[5, 5]) == 10.0, "the flat background must survive"

    def test_default_constructor_unchanged(self) -> None:
        clip = SigmaClip()
        assert (clip.n_sigma, clip.max_iter, clip.fill) == (3.0, 5, "mean")

    def test_healthy_frame_is_not_replaced(self) -> None:
        """The regression itself: a valid clip keeps every background pixel."""
        clip = SigmaClip(n_sigma=3.0)
        out = clip(frame())
        assert float(out.min()) == 10.0 and float(out.max()) == 10.0
        assert int(clip._last_mask.sum()) == 96


class TestAsymmetricSigmaClipThresholdValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_low": 0.0},
            {"n_low": -1.0},
            {"n_high": 0.0},
            {"n_high": -1.0},
            {"n_low": float("nan")},
            {"n_high": float("nan")},
            {"n_low": float("nan"), "n_high": 3.0},
            {"n_low": 3.0, "n_high": float("nan")},
        ],
    )
    def test_degenerate_thresholds_raise(self, kwargs) -> None:
        # NaN slipped through the original ``n_low <= 0`` guard: every
        # ``x >= median - nan`` is False, so all 100/100 pixels were "clipped"
        # and the frame came back as the median with no error.
        with pytest.raises(ValueError, match="n_low and n_high must be > 0"):
            AsymmetricSigmaClip(**kwargs)

    @pytest.mark.parametrize("n", [1e-12, 0.5, 3.0])
    def test_positive_thresholds_still_accepted(self, n) -> None:
        clip = AsymmetricSigmaClip(n_low=n, n_high=n)
        out = clip(frame())
        assert torch.isfinite(out).all()
        assert int(clip._last_mask.sum()) > 0, "a positive threshold keeps data"

    def test_healthy_frame_keeps_background(self) -> None:
        clip = AsymmetricSigmaClip(n_low=3.0, n_high=3.0)
        out = clip(frame())
        assert int(clip._last_mask.sum()) == 96
        assert float(out[5, 5]) == 10.0


# --------------------------------------------------------------------------
# R2-031: percentile pair that inverts or leaves [0, 100]
# --------------------------------------------------------------------------


class TestPercentileClipNormalizeValidation:
    @pytest.mark.parametrize(
        "lower_pct, upper_pct",
        [(99.0, 1.0), (100.0, 0.0), (50.1, 50.0), (99.9, 0.001)],
    )
    def test_inverted_percentiles_raise(self, lower_pct, upper_pct) -> None:
        # torch.clamp(flux, min=99th pct, max=1st pct) collapses everything
        # onto one quantile: a constant frame that looks like a valid result.
        with pytest.raises(ValueError, match="lower_pct"):
            PercentileClipNormalize(lower_pct=lower_pct, upper_pct=upper_pct)

    @pytest.mark.parametrize(
        "lower_pct, upper_pct",
        [(-50.0, 50.0), (-0.001, 50.0), (10.0, 100.5), (150.0, 200.0)],
    )
    def test_out_of_range_percentiles_raise_value_error(
        self, lower_pct, upper_pct
    ) -> None:
        # Previously: RuntimeError from torch.quantile, leaking the backend.
        with pytest.raises(ValueError, match="lower_pct"):
            PercentileClipNormalize(lower_pct=lower_pct, upper_pct=upper_pct)

    @pytest.mark.parametrize(
        "lower_pct, upper_pct",
        [(float("nan"), 99.0), (1.0, float("nan")), (float("nan"), float("nan"))],
    )
    def test_nan_percentiles_raise(self, lower_pct, upper_pct) -> None:
        with pytest.raises(ValueError, match="lower_pct"):
            PercentileClipNormalize(lower_pct=lower_pct, upper_pct=upper_pct)

    def test_error_message_states_the_ordering_contract(self) -> None:
        with pytest.raises(ValueError) as exc:
            PercentileClipNormalize(lower_pct=99.0, upper_pct=1.0)
        msg = str(exc.value)
        assert "0.0 <= lower_pct <= upper_pct <= 100.0" in msg
        assert "99.0" in msg and "1.0" in msg

    @pytest.mark.parametrize(
        "lower_pct, upper_pct",
        [(0.0, 100.0), (1.0, 99.0), (0.0, 0.0), (100.0, 100.0), (50.0, 50.0)],
    )
    def test_boundary_percentiles_still_accepted(self, lower_pct, upper_pct) -> None:
        """Equal percentiles are a supported degenerate span, not an error.

        forward() substitutes 1.0 for the divisor when upper == lower, so a
        constant frame stays finite. A strict ``<`` here would break that.
        """
        norm = PercentileClipNormalize(lower_pct=lower_pct, upper_pct=upper_pct)
        out = norm(ramp())
        assert torch.isfinite(out).all()
        if lower_pct == upper_pct:
            assert float(out.min()) == 0.0 and float(out.max()) == 0.0
        else:
            assert float(out.min()) == 0.0 and float(out.max()) == 1.0

    def test_in_range_pair_still_normalises(self) -> None:
        out = PercentileClipNormalize(lower_pct=10.0, upper_pct=95.0)(ramp())
        assert float(out.min()) == 0.0
        assert 0.9 < float(out.max()) <= 1.0

    def test_weighted_path_accepts_and_rejects_the_same_pairs(self) -> None:
        payload = {"flux": ramp(), "ivar": torch.ones(10, 10)}
        with pytest.raises(ValueError, match="lower_pct"):
            PercentileClipNormalize(lower_pct=99.0, upper_pct=1.0, weighted=True)
        out = PercentileClipNormalize(lower_pct=5.0, upper_pct=95.0, weighted=True)(
            dict(payload)
        )["flux"]
        assert torch.isfinite(out).all()

    def test_default_constructor_unchanged(self) -> None:
        norm = PercentileClipNormalize()
        assert (norm.lower_pct, norm.upper_pct) == (
            pytest.approx(0.01),
            pytest.approx(0.99),
        )


# --------------------------------------------------------------------------
# R2-032: a zero scale is the one divisor in the package with no floor
# --------------------------------------------------------------------------


class TestFITSHeaderScaleNonZeroScale:
    @pytest.mark.parametrize("bscale", [0.0, -0.0])
    def test_zero_bscale_raises(self, bscale) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            FITSHeaderScale(bscale=bscale, bzero=5.0)

    def test_zero_bscale_from_header_raises(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            FITSHeaderScale.from_header({"BSCALE": 0, "BZERO": 100})

    def test_zero_bscale_from_path_raises(self, tmp_path) -> None:
        afits = pytest.importorskip("astropy.io.fits")
        path = tmp_path / "bscale0.fits"
        hdu = afits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))
        hdu.header["BSCALE"] = 0.0
        hdu.header["BZERO"] = 5.0
        hdu.writeto(path)
        # End to end: the card is readable and numeric, so _read_header_floats
        # hands 0.0 to the constructor, which must refuse it rather than let
        # forward() flatten the image and inverse() return all-NaN.
        with pytest.raises(ValueError, match="non-zero"):
            FITSHeaderScale.from_path(str(path))

    # bscale=1e-6 is deliberately absent: with bzero=3.0 the float32 forward
    # values land within a float32 ULP of each other, so inverse() loses ~0.1
    # absolute on the way back. That is the documented float32 convention, not
    # a guard question.
    @pytest.mark.parametrize("bscale", [0.1, 0.5, 1.0, 2.5, -0.5])
    def test_nonzero_bscale_still_accepted(self, bscale) -> None:
        raw = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        scaler = FITSHeaderScale(bscale=bscale, bzero=3.0)
        out = scaler(raw)
        torch.testing.assert_close(
            out, raw * bscale + 3.0, rtol=1e-6, atol=1e-6, equal_nan=True
        )
        torch.testing.assert_close(
            scaler.inverse(out), raw, rtol=1e-5, atol=1e-5, equal_nan=True
        )

    def test_identity_shortcut_still_short_circuits(self) -> None:
        scaler = FITSHeaderScale()
        assert (scaler.bscale, scaler.bzero) == (1.0, 0.0)
        raw = torch.arange(6, dtype=torch.float32)
        torch.testing.assert_close(scaler(raw), raw, rtol=0, atol=0)

    def test_non_numeric_bscale_still_raises(self) -> None:
        with pytest.raises(ValueError):
            FITSHeaderScale.from_header({"BSCALE": "ABC"})


class TestFITSScaleColumnsNonZeroTscal:
    def test_zero_tscal_raises(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            FITSScaleColumns({"A": (0.0, 0.0)})

    def test_zero_tscal_raises_with_column_name(self) -> None:
        with pytest.raises(ValueError, match="'B'"):
            FITSScaleColumns({"A": (2.0, 1.0), "B": (0.0, 1.0)})

    def test_zero_tscal_from_header_raises(self) -> None:
        header = {
            "XTENSION": "BINTABLE",
            "BITPIX": 8,
            "NAXIS": 2,
            "NAXIS1": 0,
            "NAXIS2": 0,
            "TFIELDS": 1,
            "TTYPE1": "FLUX",
            "TFORM1": "E",
            "TSCAL1": 0.0,
            "TZERO1": 0.0,
        }
        with pytest.raises(ValueError, match="non-zero"):
            FITSScaleColumns.from_header(header)

    @pytest.mark.parametrize("tscal", [1e-6, 0.5, 2.0, -1.5])
    def test_nonzero_tscal_still_accepted(self, tscal) -> None:
        cols = {"FLUX": torch.arange(8, dtype=torch.float32)}
        scaler = FITSScaleColumns({"FLUX": (tscal, 2.0)})
        out = scaler(dict(cols))
        torch.testing.assert_close(
            out["FLUX"], cols["FLUX"].double() * tscal + 2.0, rtol=1e-6, atol=1e-6
        )
        back = scaler.inverse(dict(out))
        torch.testing.assert_close(
            back["FLUX"], cols["FLUX"].double(), rtol=1e-6, atol=1e-6
        )

    def test_unit_scale_is_still_filtered_out(self) -> None:
        # TSCAL=1/TZERO=0 is the FITS default and carries no information; it is
        # dropped from the dict and must not trip the zero-scale guard.
        scaler = FITSScaleColumns({"A": (1.0, 0.0), "B": (2.0, 1.0)})
        assert set(scaler.scales) == {"B"}

    def test_no_scales_is_a_pass_through(self) -> None:
        # No retained scales means no arithmetic at all: the column comes back
        # untouched, in its original dtype (no float64 promotion).
        cols = {"FLUX": torch.arange(4, dtype=torch.float32)}
        scaler = FITSScaleColumns({"FLUX": (1.0, 0.0)})
        out = scaler(dict(cols))
        assert out["FLUX"].dtype is torch.float32
        torch.testing.assert_close(out["FLUX"], cols["FLUX"], rtol=0, atol=0)
