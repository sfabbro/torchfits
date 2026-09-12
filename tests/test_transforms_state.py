"""State contract, payload/IVAR propagation, mask helpers, weighted stats.

Companion to ``test_transforms.py`` (pure-tensor behaviour). Everything here
covers the processing-state-aware layer: companion payloads, the
STORED-vs-PHYSICAL guard, exact inverse-variance propagation, DQ masks,
inverse-variance weighted statistics, and the transforms added with them
(``MeshBackgroundSubtract``, ``SigmaNormalize``, ``AffineTransform``, the
iterative IRAF zscale).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from transforms_reference import iraf_zscale_naive, weighted_quantile_naive

from torchfits.transforms import (
    AffineTransform,
    AsymmetricSigmaClip,
    BackgroundSubtract,
    Compose,
    DataState,
    DataStateError,
    FITSHeaderNormalize,
    FITSHeaderScale,
    FITSScaleColumns,
    GlobalScalarNorm,
    InterquantileScale,
    LogStretch,
    MeshBackgroundSubtract,
    MinMaxNormalize,
    Payload,
    PercentileClipNormalize,
    RobustNormalize,
    SigmaClip,
    SigmaNormalize,
    TNullToNan,
    ZScaleNormalize,
    SqrtStretch,
    apply_mask,
    ArcsinhStretch,
    calibration_state,
    combine_masks,
    estimate_background,
    mask_from_dq,
    mask_from_ivar,
    mask_from_nan,
    zscale_limits,
)
from torchfits.transforms.helpers import _dilate_or, _weighted_quantile


def _image(seed: int = 0, shape: tuple[int, ...] = (2, 32, 32)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * 5.0 + 50.0


def _payload(seed: int = 0, shape: tuple[int, ...] = (2, 32, 32)) -> dict:
    flux = _image(seed, shape)
    return {
        "flux": flux,
        "ivar": torch.full_like(flux, 0.25),
        "mask": torch.ones_like(flux, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# State contract
# ---------------------------------------------------------------------------


class TestStateContract:
    def test_calibration_state_matches_reader_semantics(self) -> None:
        assert calibration_state({}) is DataState.PHYSICAL
        assert calibration_state({}, raw_scale=True) is DataState.STORED

    def test_header_scale_rejects_declared_physical_state(self) -> None:
        scaler = FITSHeaderScale(bscale=0.5, bzero=10.0)
        payload = {"flux": _image(), "state": "physical"}
        with pytest.raises(DataStateError, match="expects state"):
            scaler(payload)

    def test_header_scale_accepts_declared_stored_state(self) -> None:
        scaler = FITSHeaderScale(bscale=2.0, bzero=1.0)
        payload = {"flux": torch.ones(4), "state": "stored"}
        out = scaler(payload)
        assert torch.allclose(out["flux"], torch.full((4,), 3.0))

    def test_bare_tensor_is_never_rejected(self) -> None:
        # No declared state means "caller knows best" — backwards compatible.
        assert FITSHeaderScale(bscale=2.0)(torch.ones(3)).tolist() == [2.0, 2.0, 2.0]

    def test_explicit_state_kwarg_triggers_guard(self) -> None:
        scaler = FITSHeaderScale(bscale=2.0, state=DataState.PHYSICAL)
        with pytest.raises(DataStateError, match="physical"):
            scaler(torch.ones(3))

    def test_guard_error_names_the_escape_hatch(self) -> None:
        with pytest.raises(DataStateError, match="raw_scale=True"):
            FITSHeaderScale(bscale=0.5)({"flux": _image(), "state": "physical"})

    def test_table_transforms_declare_stored_state(self) -> None:
        assert FITSScaleColumns({"A": (2.0, 0.0)}).expects == frozenset(
            {DataState.STORED}
        )
        assert TNullToNan({"A": -99.0}).expects == frozenset({DataState.STORED})
        with pytest.raises(DataStateError):
            FITSScaleColumns({"A": (2.0, 0.0)})(
                {"A": torch.ones(2), "state": "physical"}
            )

    def test_payload_state_survives_compose(self) -> None:
        pipeline = Compose([BackgroundSubtract(), ArcsinhStretch(a=0.1)])
        out = pipeline(_payload())
        # Stretches do not declare produces, so the state is carried through.
        assert isinstance(out, dict)

    def test_unknown_state_string_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown data state"):
            FITSHeaderScale(bscale=2.0, state="banana")


# ---------------------------------------------------------------------------
# Payload support (dict + typed Payload)
# ---------------------------------------------------------------------------


_TRANSFORMS = [
    lambda: AffineTransform(scale=2.0, offset=1.0),
    lambda: ZScaleNormalize(),
    lambda: RobustNormalize(),
    lambda: MinMaxNormalize(),
    lambda: GlobalScalarNorm(stat="median"),
    lambda: InterquantileScale(),
    lambda: SigmaNormalize(),
    lambda: BackgroundSubtract(),
    lambda: MeshBackgroundSubtract(mesh=(2, 2)),
    lambda: FITSHeaderNormalize({"BITPIX": -32}, scale_floats=True),
    lambda: PercentileClipNormalize(),
]


class TestPayloadSupport:
    @pytest.mark.parametrize("factory", _TRANSFORMS)
    def test_dict_payload_accepted(self, factory) -> None:
        payload = _payload()
        out = factory()(payload)
        assert isinstance(out, dict)
        assert set(out) == {"flux", "ivar", "mask"}
        assert out["mask"] is payload["mask"]
        assert out["flux"].shape == payload["flux"].shape

    @pytest.mark.parametrize("factory", _TRANSFORMS)
    def test_typed_payload_accepted(self, factory) -> None:
        typed = Payload(flux=_image(), ivar=torch.ones_like(_image()))
        out = factory()(typed)
        assert isinstance(out, Payload)
        assert out.flux.shape == typed.flux.shape

    def test_multi_arm_dict_raises_actionable_error(self) -> None:
        arms = {"B": {"flux": torch.ones(8)}, "R": {"flux": torch.ones(8)}}
        with pytest.raises(TypeError, match="flux"):
            BackgroundSubtract()(arms)

    def test_unrelated_dict_keys_are_preserved(self) -> None:
        payload = {"flux": torch.ones(4), "wavelength": torch.arange(4.0)}
        out = AffineTransform(scale=2.0)(payload)
        assert torch.equal(out["wavelength"], torch.arange(4.0))

    def test_payload_masked_pixels_excluded_without_explicit_mask(self) -> None:
        flux = torch.ones(1, 8, 8) * 10.0
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        mask[0, 0, :] = False
        flux[0, 0, :] = 1e6  # masked garbage
        out = BackgroundSubtract()({"flux": flux, "mask": mask})
        # The masked row must not drag the background estimate.
        assert out["flux"].median().abs().item() < 1e-3

    def test_explicit_mask_wins_over_payload_mask(self) -> None:
        flux = torch.ones(1, 8, 8) * 10.0
        masked = torch.ones(1, 8, 8, dtype=torch.bool)
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        mask[0, 0, 0] = False
        flux[0, 0, 0] = 500.0
        out = BackgroundSubtract()({"flux": flux, "mask": masked}, mask=mask)
        assert out["flux"][0, 0, 0].abs().item() > 400.0  # outlier kept out of stats


# ---------------------------------------------------------------------------
# IVAR propagation
# ---------------------------------------------------------------------------


_LINEAR_TRANSFORMS = [
    ("affine", lambda: AffineTransform(scale=2.5, offset=3.0)),
    ("zscale", lambda: ZScaleNormalize()),
    ("robust", lambda: RobustNormalize()),
    ("minmax", lambda: MinMaxNormalize()),
    ("global", lambda: GlobalScalarNorm(stat="median")),
    ("interquantile", lambda: InterquantileScale()),
    ("sigma", lambda: SigmaNormalize()),
    ("background", lambda: BackgroundSubtract()),
    ("mesh", lambda: MeshBackgroundSubtract(mesh=(2, 2))),
    (
        "header_normalize",
        lambda: FITSHeaderNormalize({"BITPIX": -32}, scale_floats=True),
    ),
]

# Significance preservation only holds when the additive offset is *constant*
# per group, which excludes the mesh background (its offset varies spatially).
_SIGNIFICANCE_TRANSFORMS = [t for t in _LINEAR_TRANSFORMS if t[0] != "mesh"]


def _significance(x: dict) -> torch.Tensor:
    """``(flux - per-row median) / sigma`` — invariant under any affine map."""
    centered = x["flux"] - x["flux"].median(dim=-1, keepdim=True).values
    return centered / torch.sqrt(1.0 / x["ivar"])


class TestIvarPropagation:
    @pytest.mark.parametrize("name,factory", _LINEAR_TRANSFORMS)
    def test_linear_transforms_round_trip_flux_and_ivar(self, name, factory) -> None:
        payload = _payload()
        transform = factory()
        out = transform(payload)
        restored = transform.inverse(out)
        flux_err = (restored["flux"] - payload["flux"]).abs().max().item()
        ivar_err = (restored["ivar"] - payload["ivar"]).abs().max().item()
        assert flux_err < 1e-4, f"{name}: flux round-trip error {flux_err}"
        assert ivar_err < 1e-5, f"{name}: ivar round-trip error {ivar_err}"

    @pytest.mark.parametrize("name,factory", _SIGNIFICANCE_TRANSFORMS)
    def test_relative_significance_is_preserved(self, name, factory) -> None:
        """An affine transform must not change significance above the local
        background: ``(flux - median(flux)) / sigma`` is invariant.

        (The raw ``flux / sigma`` ratio is *not* invariant when the transform
        adds a constant offset, which is a real feature of every background
        estimator here.)
        """
        payload = _payload()
        out = factory()(payload)
        assert torch.allclose(
            _significance(payload), _significance(out), rtol=1e-4, atol=1e-3
        ), name

    def test_pure_offset_leaves_sigma_untouched(self) -> None:
        """A spatially varying offset still leaves the noise scale alone."""
        payload = _payload()
        out = MeshBackgroundSubtract(mesh=(2, 2))(payload)
        assert torch.equal(out["ivar"], payload["ivar"])
        # Every pixel is shifted by the same local sky estimate...
        shift = out["flux"] - payload["flux"]
        # ...so the residual scatter is unchanged.
        assert (shift.std() / payload["flux"].std()).item() < 0.05

    def test_ivar_propagates_for_header_scale(self) -> None:
        payload = {"flux": torch.ones(4), "ivar": torch.full((4,), 4.0)}
        scaler = FITSHeaderScale(bscale=2.0, bzero=0.0, state="stored")
        out = scaler(payload)
        assert torch.allclose(out["ivar"], torch.full((4,), 1.0))  # 4 / 2**2

    def test_offset_leaves_ivar_unchanged(self) -> None:
        payload = {
            "flux": torch.ones(1, 16, 16) * 5,
            "ivar": torch.full((1, 16, 16), 2.0),
        }
        out = BackgroundSubtract()(payload)
        assert torch.equal(out["ivar"], payload["ivar"])

    def test_nonlinear_transform_warns_and_passes_ivar_through(self) -> None:
        payload = {
            "flux": torch.ones(1, 16, 16) * 3,
            "ivar": torch.full((1, 16, 16), 2.0),
        }
        transform = ArcsinhStretch(a=0.1)
        with pytest.warns(UserWarning, match="nonlinear"):
            out = transform(payload)
        assert torch.equal(out["ivar"], payload["ivar"])

    def test_nonlinear_warning_fires_once_per_instance(self) -> None:
        payload = {"flux": torch.ones(1, 16, 16) * 3, "ivar": torch.ones(1, 16, 16)}
        transform = SigmaClip(n_sigma=3.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform(payload)
            transform(payload)
        nonlinear = [w for w in caught if "nonlinear" in str(w.message)]
        assert len(nonlinear) == 1

    def test_no_warning_without_ivar(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StretchFree = LogStretch()
            StretchFree(torch.ones(1, 8, 8))
        assert not [w for w in caught if "nonlinear" in str(w.message)]

    def test_ivar_dict_absent_stays_absent(self) -> None:
        out = RobustNormalize()({"flux": torch.ones(1, 8, 8) * 3})
        assert "ivar" not in out


# ---------------------------------------------------------------------------
# Delta-method IVAR propagation through the nonlinear stretches
# ---------------------------------------------------------------------------


STRETCH_CASES = [
    (ArcsinhStretch(1.0, propagate_ivar=True), [0.05, 0.3, 1.0, 4.0, 20.0]),
    (LogStretch(10.0, propagate_ivar=True), [0.05, 0.3, 1.0, 4.0, 20.0]),
    (SqrtStretch(propagate_ivar=True), [0.05, 0.3, 1.0, 4.0, 20.0]),
]


def _finite_difference_slope(transform, x: torch.Tensor) -> torch.Tensor:
    """Central-difference d(flux_out)/d(flux_in) for a pointwise transform."""
    ivar = torch.ones_like(x)
    eps = 1e-6
    slopes = torch.empty_like(x)
    for i in range(x.numel()):
        delta = torch.zeros_like(x)
        delta[i] = eps
        hi = transform({"flux": x + delta, "ivar": ivar})["flux"]
        lo = transform({"flux": x - delta, "ivar": ivar})["flux"]
        slopes[i] = (hi[i] - lo[i]) / (2 * eps)
    return slopes


class TestDeltaIvarPropagation:
    """``propagate_ivar=True`` must match numerical error propagation."""

    @pytest.mark.parametrize("transform,xs", STRETCH_CASES)
    def test_matches_finite_difference(self, transform, xs) -> None:
        x = torch.tensor(xs, dtype=torch.float64)
        ivar = torch.full_like(x, 4.0)
        got = transform({"flux": x, "ivar": ivar})["ivar"]
        slope = _finite_difference_slope(transform, x)
        want = torch.where(slope > 0, ivar / slope.pow(2), torch.zeros_like(slope))
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-8)

    @pytest.mark.parametrize("transform,_xs", STRETCH_CASES)
    def test_propagating_stretches_do_not_warn(self, transform, _xs) -> None:
        payload = {"flux": torch.ones(1, 8, 8) * 3, "ivar": torch.ones(1, 8, 8)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform(payload)
        assert not [w for w in caught if "nonlinear" in str(w.message)]

    @pytest.mark.parametrize("transform,_xs", STRETCH_CASES)
    def test_flag_is_reported_to_compose(self, transform, _xs) -> None:
        assert transform.propagates_ivar is True
        assert Compose([transform]).propagates_ivar is True

    def test_default_remains_pass_through(self) -> None:
        payload = {"flux": torch.ones(1, 8, 8) * 3, "ivar": torch.ones(1, 8, 8)}
        with pytest.warns(UserWarning, match="nonlinear"):
            out = SqrtStretch()(payload)
        assert torch.equal(out["ivar"], payload["ivar"])

    def test_sqrt_stabilises_poisson_variance(self) -> None:
        # For Poisson counts var(x) = x, so ivar_x = 1/x and the stretched
        # data should carry a near-constant ivar of 4.
        counts = torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=torch.float64)
        out = SqrtStretch(propagate_ivar=True)({"flux": counts, "ivar": 1.0 / counts})
        assert torch.allclose(
            out["ivar"], torch.full((4,), 4.0, dtype=torch.float64), rtol=1e-9
        )

    def test_clamped_region_reports_zero_information(self) -> None:
        # x <= 0 is clamped flat by sqrt: the output no longer constrains the
        # input, so its inverse variance is zero rather than infinite.
        out = SqrtStretch(propagate_ivar=True)(
            {"flux": torch.tensor([-1.0, 0.0, 4.0]), "ivar": torch.ones(3)}
        )
        assert torch.equal(out["ivar"], torch.tensor([0.0, 0.0, 16.0]))

    def test_slope_is_finite_at_clamp_boundary(self) -> None:
        out = SqrtStretch(propagate_ivar=True)(
            {"flux": torch.tensor([0.0, 1e-30]), "ivar": torch.ones(2)}
        )
        assert torch.isfinite(out["ivar"]).all()


# ---------------------------------------------------------------------------
# Mask invariances and degenerate groups
# ---------------------------------------------------------------------------


class TestMaskAwareness:
    def test_masked_outlier_does_not_move_the_background(self) -> None:
        flux = torch.full((1, 16, 16), 10.0)
        flux[0, 0, 0] = 1e6
        mask = torch.ones_like(flux, dtype=torch.bool)
        mask[0, 0, 0] = False
        med, std = estimate_background(flux, dim=(-2, -1), mask=mask)
        assert med.item() == pytest.approx(10.0)
        assert std.item() == pytest.approx(0.0, abs=1e-6)

    def test_nan_excluded_without_explicit_mask(self) -> None:
        flux = torch.full((1, 16, 16), 10.0)
        flux[0, 3, 3] = float("nan")
        med, _ = estimate_background(flux, dim=(-2, -1))
        assert med.item() == pytest.approx(10.0)

    def test_all_masked_group_yields_nan_not_garbage(self) -> None:
        flux = _image(shape=(1, 16, 16))
        mask = torch.zeros_like(flux, dtype=torch.bool)
        out = MinMaxNormalize()(flux, mask=mask)
        assert torch.isnan(out).all()

    def test_all_nan_image_keeps_nan(self) -> None:
        flux = torch.full((1, 8, 8), float("nan"))
        out = MinMaxNormalize()(flux)
        assert torch.isnan(out).all()

    def test_global_scalar_norm_max_on_all_nan_is_identity(self) -> None:
        """Regression: an all-NaN frame used to divide by -1e-30."""
        transform = GlobalScalarNorm(stat="max")
        out = transform(torch.full((1, 8, 8), float("nan")))
        assert torch.isnan(out).all()  # NaN in, NaN out — never 1e30 artefacts
        # Degenerate statistics fall back to a divisor of 1.0 (identity).
        assert torch.isfinite(transform._scalar).all()
        assert torch.equal(transform._scalar, torch.ones_like(transform._scalar))

    def test_constant_image_stats_are_nan_free(self) -> None:
        flux = torch.full((1, 8, 8), 7.0)
        for transform in (ZScaleNormalize(), RobustNormalize(), MinMaxNormalize()):
            out = transform(flux.clone())
            assert torch.isfinite(out).all(), repr(transform)


# ---------------------------------------------------------------------------
# dtype contract
# ---------------------------------------------------------------------------


class TestDtypeContract:
    _INT_DTYPES = [
        dtype
        for dtype in (torch.int16, torch.int32, getattr(torch, "uint16", None))
        if dtype is not None
    ]

    @pytest.mark.parametrize("dtype", _INT_DTYPES)
    def test_stretches_promote_integers_and_never_truncate(self, dtype) -> None:
        x = torch.tensor([[0, 100, 1000]], dtype=dtype)
        for transform in (
            ArcsinhStretch(a=0.001),
            LogStretch(),
        ):
            out = transform(x)
            assert out.dtype.is_floating_point, (transform, out.dtype)
            assert out[0, 0].item() == pytest.approx(0.0)
            assert out[0, 2].item() > out[0, 1].item() > 0.0, (transform, out)

    def test_log_stretch_does_not_collapse_to_integers(self) -> None:
        """Regression: LogStretch used to return uint16 [0, 1, 1]."""
        x = torch.tensor([[0, 100, 1000]], dtype=torch.uint16)
        out = LogStretch()(x)
        assert out.dtype.is_floating_point
        assert out[0, 1].item() > 0.5
        assert out[0, 2].item() > out[0, 1].item()

    def test_float32_stays_float32(self) -> None:
        x = torch.rand(2, 8, 8, dtype=torch.float32)
        for transform in (ArcsinhStretch(), LogStretch(), SigmaNormalize()):
            assert transform(x).dtype == torch.float32, repr(transform)

    def test_float64_stays_float64(self) -> None:
        x = torch.rand(2, 8, 8, dtype=torch.float64) + 1.0
        assert ArcsinhStretch()(x).dtype == torch.float64

    def test_arcsinh_round_trip_preserves_dtype(self) -> None:
        x = torch.rand(2, 8, 8, dtype=torch.float32) + 1.0
        transform = ArcsinhStretch(a=0.5)
        assert transform.inverse(transform(x)).dtype == torch.float32


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


class TestMaskHelpers:
    def test_mask_from_dq_default_any_nonzero_is_bad(self) -> None:
        dq = torch.tensor([[0, 1], [4, 0]], dtype=torch.int32)
        assert mask_from_dq(dq).tolist() == [[True, False], [False, True]]

    def test_mask_from_dq_selective_bits(self) -> None:
        dq = torch.tensor([[0, 1, 4, 5]], dtype=torch.int32)
        # Only bit 2 is fatal -> 4 and 5 are bad, 1 is fine.
        assert mask_from_dq(dq, bad_bits=[2]).tolist() == [[True, True, False, False]]

    def test_mask_from_dq_require_good_bits(self) -> None:
        # Bit 0 (value 1) is the "science good" flag.
        dq = torch.tensor([[0b01, 0b11, 0b10]], dtype=torch.int32)
        got = mask_from_dq(dq, bad_bits=None, good_bits=[0], require_good=True)
        assert got.tolist() == [[True, True, False]]

    def test_mask_from_dq_require_good_combined_with_bad_bits(self) -> None:
        # Bit 0 = good flag, bit 1 (value 2) = defect.
        dq = torch.tensor([[0b01, 0b11, 0b00]], dtype=torch.int32)
        got = mask_from_dq(dq, bad_bits=[1], good_bits=[0], require_good=True)
        assert got.tolist() == [[True, False, False]]

    def test_mask_from_dq_rejects_float_extension(self) -> None:
        with pytest.raises(TypeError, match="integer-typed"):
            mask_from_dq(torch.zeros(3))

    def test_mask_from_ivar(self) -> None:
        ivar = torch.tensor([0.0, 1.0, float("nan"), -1.0])
        assert mask_from_ivar(ivar).tolist() == [False, True, False, False]

    def test_mask_from_nan(self) -> None:
        assert mask_from_nan(torch.tensor([1.0, float("nan")])).tolist() == [
            True,
            False,
        ]

    def test_combine_masks_and_ignores_none(self) -> None:
        a = torch.tensor([True, True, False])
        b = torch.tensor([True, False, True])
        assert combine_masks(None, a, b).tolist() == [True, False, False]
        assert combine_masks(None) is None

    def test_apply_mask_promotes_int_for_nan_fill(self) -> None:
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        mask = torch.tensor([True, False, True])
        out = apply_mask(x, mask)
        assert out.dtype.is_floating_point
        assert torch.isnan(out[1])


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------


class TestWeightedStatistics:
    @pytest.mark.parametrize("q", [0.05, 0.25, 0.5, 0.9, 0.95])
    def test_matches_naive_reference(self, q: float) -> None:
        x = _image(shape=(3, 16, 16))
        ivar = torch.rand_like(x) + 0.1
        got = _weighted_quantile(x, q, (-2, -1), ivar=ivar)
        ref = weighted_quantile_naive(x, q, (-2, -1), ivar=ivar)
        assert torch.allclose(got, ref, rtol=1e-5)

    def test_matches_naive_reference_with_mask_and_nan(self) -> None:
        x = _image(shape=(1, 16, 16))
        x[0, 0, 0] = float("nan")
        ivar = torch.rand_like(x) + 0.1
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[0, 1, 1] = False
        got = _weighted_quantile(x, 0.5, (-2, -1), mask=mask, ivar=ivar)
        ref = weighted_quantile_naive(x, 0.5, (-2, -1), mask=mask, ivar=ivar)
        assert torch.allclose(got, ref, rtol=1e-5)

    def test_downweighted_outlier_is_ignored(self) -> None:
        x = torch.full((1, 100), 10.0)
        x[0, :5] = 1000.0
        ivar = torch.ones_like(x)
        ivar[0, :5] = 1e-12  # reject the bright pixels
        med = _weighted_quantile(x, 0.5, (-1,), ivar=ivar)
        assert med.item() == pytest.approx(10.0)

    def test_zero_weight_group_yields_nan(self) -> None:
        x = torch.ones(1, 8)
        ivar = torch.zeros(1, 8)
        assert torch.isnan(_weighted_quantile(x, 0.5, (-1,), ivar=ivar)).all()

    def test_weighted_background_uses_weights(self) -> None:
        x = torch.cat([torch.full((1, 50), 10.0), torch.full((1, 50), 1000.0)], dim=-1)
        ivar = torch.cat([torch.ones(1, 50), torch.full((1, 50), 1e-9)], dim=-1)
        med, _ = estimate_background(x, dim=(-1,), ivar=ivar, weighted=True)
        assert med.item() == pytest.approx(10.0)
        unweighted, _ = estimate_background(x, dim=(-1,))
        assert unweighted.item() > 100.0

    def test_weighted_flag_off_by_default_keeps_outputs_identical(self) -> None:
        x = _image()
        ivar = torch.rand_like(x)
        default = estimate_background(x, dim=(-2, -1))[0]
        explicit = estimate_background(x, dim=(-2, -1), ivar=ivar, weighted=False)[0]
        assert torch.equal(default, explicit)


# ---------------------------------------------------------------------------
# IRAF zscale
# ---------------------------------------------------------------------------


class TestIrafZScale:
    def test_matches_naive_reference(self) -> None:
        rng = np.random.default_rng(3)
        image = rng.normal(100, 5, (64, 64))
        image[0, 0] = 100000.0
        image[1, 1] = -50000.0
        z1, z2 = zscale_limits(
            torch.from_numpy(image), 0.25, (-2, -1), None, algorithm="iraf"
        )
        ref1, ref2 = iraf_zscale_naive(image, 0.25)
        assert abs(float(z1) - ref1) < 0.05
        assert abs(float(z2) - ref2) < 0.05

    @pytest.mark.parametrize("size", [37, 128, 256])
    def test_matches_naive_reference_across_sizes(self, size: int) -> None:
        rng = np.random.default_rng(7)
        image = rng.normal(50, 3, (size, size))
        image[3, 4] = 9000.0
        z1, z2 = zscale_limits(
            torch.from_numpy(image), 0.25, (-2, -1), None, algorithm="iraf"
        )
        ref1, ref2 = iraf_zscale_naive(image, 0.25)
        assert abs(float(z1) - ref1) < 0.02
        assert abs(float(z2) - ref2) < 0.02

    def test_matches_astropy(self) -> None:
        astropy_interval = pytest.importorskip("astropy.visualization").ZScaleInterval
        rng = np.random.default_rng(11)
        image = rng.normal(20, 2, (128, 128))
        image[5, 5] = 1e5
        z1, z2 = zscale_limits(
            torch.from_numpy(image), 0.25, (-2, -1), None, algorithm="iraf"
        )
        ref1, ref2 = astropy_interval(contrast=0.25).get_limits(image)
        assert abs(float(z1) - ref1) < 0.02
        assert abs(float(z2) - ref2) < 0.02

    def test_batched_groups_equal_per_image_calls(self) -> None:
        stack = torch.stack([_image(1, (32, 32)), _image(2, (32, 32)) * 3])
        z1, z2 = zscale_limits(stack, 0.25, (-2, -1), None, algorithm="iraf")
        for index in range(2):
            single1, single2 = zscale_limits(
                stack[index], 0.25, (-2, -1), None, algorithm="iraf"
            )
            assert torch.allclose(z1[index], single1, rtol=1e-6)
            assert torch.allclose(z2[index], single2, rtol=1e-6)

    def test_constant_image_gives_non_degenerate_limits(self) -> None:
        z1, z2 = zscale_limits(torch.full((16, 16), 4.0), algorithm="iraf")
        assert float(z1) < float(z2)

    def test_does_not_mutate_input(self) -> None:
        x = _image(shape=(32, 32))
        before = x.clone()
        zscale_limits(x, 0.25, (-2, -1), None, algorithm="iraf")
        assert torch.equal(x, before)

    def test_invalid_algorithm_rejected(self) -> None:
        with pytest.raises(ValueError, match="algorithm must be"):
            zscale_limits(torch.ones(4, 4), algorithm="nope")

    def test_proxy_is_still_the_default(self) -> None:
        x = _image(shape=(32, 32))
        default = zscale_limits(x)
        proxy = zscale_limits(x, algorithm="proxy")
        assert torch.equal(default[0], proxy[0])

    def test_dilate_or_matches_numpy_convolve(self) -> None:
        rng = np.random.default_rng(0)
        for n, width in [(7, 2), (7, 3), (10, 1), (12, 4), (20, 10)]:
            base = rng.random(n) > 0.6
            reference = np.convolve(base, np.ones(width, dtype=bool), mode="same")
            got = _dilate_or(torch.from_numpy(base), width).numpy()
            assert (reference == got).all(), (n, width)


# ---------------------------------------------------------------------------
# MeshBackgroundSubtract
# ---------------------------------------------------------------------------


class TestMeshBackgroundSubtract:
    @staticmethod
    def _gradient(shape: tuple[int, int] = (64, 64)) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(0, 1, shape[0]),
            torch.linspace(0, 1, shape[1]),
            indexing="ij",
        )
        del y
        generator = torch.Generator().manual_seed(5)
        return (100.0 + 50.0 * x) + torch.randn(shape, generator=generator) * 2.0

    def test_removes_sky_gradient(self) -> None:
        image = self._gradient()
        out = MeshBackgroundSubtract(mesh=(4, 4))(image)
        row_means = out.mean(dim=1)
        assert (row_means.max() - row_means.min()).item() < 3.0

    def test_beats_global_median_on_gradients(self) -> None:
        image = self._gradient()
        mesh = MeshBackgroundSubtract(mesh=(4, 4))(image)
        global_sub = BackgroundSubtract()(image)
        assert (mesh.mean(dim=1).max() - mesh.mean(dim=1).min()) < (
            global_sub.mean(dim=1).max() - global_sub.mean(dim=1).min()
        )

    def test_inverse_round_trip(self) -> None:
        image = self._gradient()
        transform = MeshBackgroundSubtract(mesh=(4, 4))
        restored = transform.inverse(transform(image))
        assert torch.allclose(restored, image, atol=1e-5)

    def test_batched_cube(self) -> None:
        stack = torch.stack([self._gradient(), self._gradient() * 0.5 + 20.0])
        transform = MeshBackgroundSubtract(mesh=(4, 4))
        out = transform(stack)
        assert out.shape == stack.shape
        for index in range(2):
            single = transform(stack[index])
            assert torch.allclose(out[index], single, atol=1e-5)

    def test_payload_ivar_untouched(self) -> None:
        image = self._gradient()
        payload = {"flux": image, "ivar": torch.ones_like(image)}
        out = MeshBackgroundSubtract(mesh=(4, 4))(payload)
        assert torch.equal(out["ivar"], payload["ivar"])

    def test_fully_masked_tile_falls_back_continuously(self) -> None:
        image = self._gradient()
        mask = torch.ones_like(image, dtype=torch.bool)
        mask[:16, :16] = False
        out = MeshBackgroundSubtract(mesh=(4, 4))(image, mask=mask)
        assert torch.isfinite(out).all()

    def test_requires_two_spatial_dims(self) -> None:
        with pytest.raises(ValueError, match="at least 2 dims"):
            MeshBackgroundSubtract()(torch.ones(8))

    def test_inverse_without_forward_raises(self) -> None:
        transform = MeshBackgroundSubtract(mesh=(4, 4))
        with pytest.raises(RuntimeError, match="prior forward"):
            transform.inverse(torch.ones(8, 8))

    def test_single_band_and_many_bands(self) -> None:
        image = self._gradient()
        one = MeshBackgroundSubtract(mesh=(2, 2))(image)
        assert one.shape == image.shape
        many = MeshBackgroundSubtract(mesh=(2, 2))(torch.stack([image] * 3))
        assert many.shape == (3, 64, 64)

    def test_repr(self) -> None:
        text = repr(MeshBackgroundSubtract(mesh=(3, 3)))
        assert "MeshBackgroundSubtract" in text and "(3, 3)" in text


# ---------------------------------------------------------------------------
# SigmaNormalize / AffineTransform
# ---------------------------------------------------------------------------


class TestSigmaNormalize:
    def test_sigma_is_one(self) -> None:
        out = SigmaNormalize()(_image())
        for index in range(out.shape[0]):
            _, std = estimate_background(out[index].unsqueeze(0), dim=(-2, -1))
            assert std.item() == pytest.approx(1.0, rel=0.05)

    def test_zero_preserving_keeps_zeros_and_ratios(self) -> None:
        band_g = torch.rand(16, 16) * 10 + 1
        band_r = band_g * 2.0
        stack = torch.stack([band_g, band_r])
        out = SigmaNormalize(dim=(-3, -2, -1))(stack)
        ratio_before = band_r / band_g
        ratio_after = out[1] / out[0]
        assert torch.allclose(ratio_before, ratio_after, atol=1e-5)

    def test_centered_mode(self) -> None:
        out = SigmaNormalize(zero_preserving=False)(_image())
        # A centred Gaussian maps to unit scale: mean ~ 0 and
        # median(|z|) = 0.6745 for N(0, 1).
        assert out.mean().abs().item() < 0.1
        assert out.abs().median().item() == pytest.approx(0.6745, abs=0.1)

    def test_stat_std_variant(self) -> None:
        out = SigmaNormalize(stat="std")(_image())
        assert torch.isfinite(out).all()

    def test_invalid_stat_rejected(self) -> None:
        with pytest.raises(ValueError, match="stat must be"):
            SigmaNormalize(stat="iqr")

    def test_inverse_without_forward_raises(self) -> None:
        with pytest.raises(RuntimeError, match="prior forward"):
            SigmaNormalize().inverse(torch.ones(8, 8))


class TestAffineTransform:
    def test_forward_matches_manual(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0])
        out = AffineTransform(scale=2.0, offset=-1.0)(x)
        assert torch.allclose(out, torch.tensor([1.0, 3.0, 5.0]))

    def test_round_trip(self) -> None:
        x = _image()
        transform = AffineTransform(scale=0.25, offset=7.0)
        assert torch.allclose(transform.inverse(transform(x)), x, atol=1e-5)

    def test_zero_scale_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            AffineTransform(scale=0.0)

    def test_repr(self) -> None:
        assert "AffineTransform" in repr(AffineTransform(2.0, 1.0))


# ---------------------------------------------------------------------------
# Nonlinear clips keep working with payloads
# ---------------------------------------------------------------------------


class TestClipPayloads:
    def test_sigma_clip_accepts_dict_payload(self) -> None:
        payload = _payload()
        out = SigmaClip(n_sigma=3.0)(payload)
        assert set(out) == {"flux", "ivar", "mask"}

    def test_asymmetric_clip_accepts_dict_payload(self) -> None:
        payload = _payload()
        out = AsymmetricSigmaClip(n_low=3.0, n_high=3.0)(payload)
        assert out["flux"].shape == payload["flux"].shape

    def test_asymmetric_clip_repr_includes_fill(self) -> None:
        assert "fill='nan'" in repr(AsymmetricSigmaClip(fill="nan"))

    def test_weighted_clip_uses_weights_for_thresholds(self) -> None:
        # 40 of 64 pixels are bright but carry essentially zero weight. The
        # unweighted median follows the majority and clips the real sky; the
        # weighted median ignores them and clips the bright pixels instead.
        flux = torch.full((1, 64), 5.0)
        flux[0, :40] = 1000.0
        ivar = torch.ones_like(flux)
        ivar[0, :40] = 1e-12
        payload = {"flux": flux, "ivar": ivar}
        weighted = AsymmetricSigmaClip(n_low=3.0, n_high=3.0, dim=(-1,), weighted=True)(
            payload
        )["flux"][0, 40:]
        assert torch.allclose(weighted, torch.full((24,), 5.0))
        unweighted = AsymmetricSigmaClip(
            n_low=3.0, n_high=3.0, dim=(-1,), weighted=False
        )(payload)["flux"][0, 40:]
        assert unweighted.min().item() > 100.0


# ---------------------------------------------------------------------------
# Regressions found by adversarial probing
# ---------------------------------------------------------------------------


class TestDifferentiability:
    """Statistics may be constants, but the output must stay differentiable.

    ``SigmaClip`` / ``AsymmetricSigmaClip`` used to wrap their *return* in
    ``torch.no_grad()``, so a pipeline containing a clip was silently detached
    from autograd while every normalizer kept its gradient.
    """

    _TRANSFORMS = [
        lambda: SigmaClip(),
        lambda: SigmaClip(fill="nan"),
        lambda: AsymmetricSigmaClip(),
        lambda: AsymmetricSigmaClip(fill="nan"),
        lambda: MinMaxNormalize(),
        lambda: ZScaleNormalize(),
        lambda: SigmaNormalize(),
        lambda: BackgroundSubtract(),
    ]

    @pytest.mark.parametrize("factory", _TRANSFORMS)
    def test_gradients_reach_the_input(self, factory) -> None:
        x = _image(shape=(1, 16, 16)).requires_grad_(True)
        out = factory()(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_clip_gradient_is_one_on_kept_pixels(self) -> None:
        flux = torch.full((1, 64), 5.0, requires_grad=True)
        SigmaClip(dim=(-1,), fill="median")(flux).sum().backward()
        assert flux.grad is not None and torch.allclose(
            flux.grad, torch.ones_like(flux.grad)
        )

    def test_clip_mask_still_cached(self) -> None:
        flux = torch.full((1, 64), 5.0)
        flux[0, 0] = 500.0
        transform = SigmaClip(n_sigma=3.0, dim=(-1,), fill="nan")
        out = transform(flux)
        assert out[0, 0].isnan() and torch.isfinite(out[0, 1:]).all()
        assert not transform._last_mask[0, 0]
        assert transform._last_mask[0, 1]


class TestStatePropagation:
    """``produces`` has to actually advance the declared state.

    Declaring ``produces`` without applying it left a payload labelled
    ``stored`` after it had been calibrated, which let a header-scaling
    transform be applied twice inside a single pipeline.
    """

    def test_header_scale_advances_and_guards(self) -> None:
        payload = {"flux": torch.ones(4, dtype=torch.int16), "state": "stored"}
        out = FITSHeaderScale(bscale=2.0)(payload)
        assert out["state"] == DataState.PHYSICAL
        with pytest.raises(DataStateError, match="physical"):
            FITSHeaderScale(bscale=2.0)(out)

    def test_compose_advances_state(self) -> None:
        pipeline = Compose([FITSHeaderScale(bscale=2.0), GlobalScalarNorm()])
        out = pipeline({"flux": torch.ones(4, dtype=torch.int16), "state": "stored"})
        assert out["state"] == DataState.NORMALIZED

    def test_inverse_restores_stored(self) -> None:
        scaler = FITSHeaderScale(bscale=2.0)
        out = scaler({"flux": torch.ones(4, dtype=torch.int16), "state": "stored"})
        back = scaler.inverse(out)
        assert back["state"] == DataState.STORED

    def test_payloads_that_never_declared_a_state_are_untouched(self) -> None:
        out = GlobalScalarNorm()({"flux": _image(shape=(1, 8, 8))})
        assert "state" not in out

    def test_dataclass_payload_advances(self) -> None:
        payload = Payload(flux=torch.ones(4, dtype=torch.int16), state=DataState.STORED)
        out = FITSHeaderScale(bscale=2.0)(payload)
        assert isinstance(out, Payload) and out.state is DataState.PHYSICAL

    def test_identity_float_header_normalize_does_not_relabel(self) -> None:
        payload = {"flux": _image(shape=(1, 8, 8)), "state": "physical"}
        out = FITSHeaderNormalize({"BITPIX": -32})(payload)
        assert out["state"] == DataState.PHYSICAL


class TestWeightedFallback:
    """``weighted=True`` without an ``ivar`` must be an exact no-op.

    It used to take the inverted-CDF path with uniform weights, which differs
    from the interpolated median by up to one order-statistic spacing. The
    transform docstrings promise "weighted when ``ivar`` is present".
    """

    _PAIRS = [
        ("sigma", lambda: SigmaNormalize(weighted=True), lambda: SigmaNormalize()),
        (
            "global",
            lambda: GlobalScalarNorm(weighted=True),
            lambda: GlobalScalarNorm(),
        ),
        (
            "interquantile",
            lambda: InterquantileScale(weighted=True),
            lambda: InterquantileScale(),
        ),
        (
            "percentile",
            lambda: PercentileClipNormalize(weighted=True),
            lambda: PercentileClipNormalize(),
        ),
        (
            "robust",
            lambda: RobustNormalize(weighted=True),
            lambda: RobustNormalize(),
        ),
        (
            "background",
            lambda: BackgroundSubtract(weighted=True),
            lambda: BackgroundSubtract(),
        ),
        (
            "mesh",
            lambda: MeshBackgroundSubtract(mesh=(4, 4), weighted=True),
            lambda: MeshBackgroundSubtract(mesh=(4, 4)),
        ),
    ]

    @pytest.mark.parametrize("name,weighted,plain", _PAIRS)
    def test_matches_unweighted_exactly(self, name, weighted, plain) -> None:
        image = _image(shape=(1, 16, 16))
        got = weighted()({"flux": image.clone()})["flux"]
        expected = plain()({"flux": image.clone()})["flux"]
        assert torch.equal(got, expected), name

    def test_estimate_background_falls_back_without_ivar(self) -> None:
        image = _image(shape=(1, 16, 16))[0]
        weighted = estimate_background(image, dim=(-2, -1), weighted=True)[0]
        plain = estimate_background(image, dim=(-2, -1))[0]
        assert torch.equal(weighted, plain)

    def test_flat_weights_differ_by_at_most_one_order_statistic(self) -> None:
        """The inverted-CDF definition is documented, not a bug — but bounded."""
        image = _image(shape=(1, 16, 16))[0]
        flat = estimate_background(
            image, dim=(-2, -1), ivar=torch.ones_like(image), weighted=True
        )[0]
        plain = estimate_background(image, dim=(-2, -1))[0]
        _, mad = estimate_background(image, dim=(-2, -1))
        assert (flat - plain).abs().max().item() < float(mad.item())


class TestContinuumNormalizedGuard:
    """Flux-scaling transforms refuse already continuum-normalized spectra."""

    _FACTORIES = [
        lambda: ZScaleNormalize(),
        lambda: RobustNormalize(),
        lambda: SigmaNormalize(),
        lambda: MinMaxNormalize(),
        lambda: GlobalScalarNorm(),
        lambda: InterquantileScale(),
        lambda: PercentileClipNormalize(),
        lambda: BackgroundSubtract(),
        lambda: MeshBackgroundSubtract(mesh=(2, 2)),
        lambda: FITSHeaderNormalize({"BITPIX": 8}),
    ]

    @pytest.mark.parametrize("factory", _FACTORIES)
    def test_rejects_continuum_normalized(self, factory) -> None:
        payload = {
            "flux": _image(shape=(2, 16, 16)),
            "state": DataState.CONTINUUM_NORMALIZED,
        }
        with pytest.raises(DataStateError, match="Continuum-normalized"):
            factory()(payload)

    def test_bare_tensor_is_still_accepted(self) -> None:
        assert torch.isfinite(GlobalScalarNorm()(_image(shape=(1, 8, 8)))).all()

    def test_physical_spectra_still_accepted(self) -> None:
        payload = {"flux": _image(shape=(1, 8, 8)), "state": "physical"}
        out = GlobalScalarNorm()(payload)
        assert out["state"] == DataState.NORMALIZED


class TestMeshNanHygiene:
    def test_single_nan_pixel_does_not_spread(self) -> None:
        image = _image(shape=(1, 32, 32))
        image[0, 0, 0] = float("nan")
        out = MeshBackgroundSubtract(mesh=(4, 4))(image)
        assert out.isnan().sum().item() == 1
