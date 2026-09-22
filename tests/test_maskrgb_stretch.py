"""Delta-method IVAR and dtype-contract pins for the nonlinear stretches.

Complements ``tests/test_transforms_state.py`` (which pins the finite-difference
identity and the SqrtStretch Poisson constant) with the slice-level contracts:
clamped regions report ``ivar = 0``, integer inputs promote to float on
``inverse`` exactly like ``forward`` (never wrap around), and the stretches are
monotonic in every accepted dtype.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchfits.transforms import ArcsinhStretch, LogStretch, SqrtStretch


class TestInverseDtypeContract:
    def test_sqrt_inverse_of_integer_input_returns_float(self) -> None:
        # Docs contract (api-transforms.md): every stretch returns float for
        # integer input — forward AND inverse. Squaring in the storage dtype
        # also wraps around silently.
        x = torch.tensor([300, 200], dtype=torch.int16)
        out = SqrtStretch().inverse(x)
        assert out.dtype.is_floating_point
        assert out.tolist() == [90000.0, 40000.0]

    def test_sqrt_inverse_wide_integers_do_not_wrap(self) -> None:
        x = torch.tensor([50000, 50000], dtype=torch.int32)
        out = SqrtStretch().inverse(x)
        assert out.dtype.is_floating_point
        assert out.tolist() == [2.5e9, 2.5e9]

    def test_sqrt_round_trip_integer_input(self) -> None:
        x = torch.tensor([0, 1, 100, 1000], dtype=torch.int32)
        round_trip = SqrtStretch().inverse(SqrtStretch()(x))
        assert round_trip.dtype.is_floating_point
        assert torch.allclose(round_trip.float(), x.float(), rtol=1e-5, atol=1e-4)

    @pytest.mark.parametrize(
        "factory", [SqrtStretch, lambda: LogStretch(a=10.0), lambda: ArcsinhStretch(a=0.5)]
    )
    def test_forward_and_inverse_agree_on_int_promotion(self, factory) -> None:
        x = torch.tensor([[0, 10, 255]], dtype=torch.uint8)
        fwd = factory()(x)
        inv = factory().inverse(x)
        for out in (fwd, inv):
            assert out.dtype.is_floating_point
            assert out[0, 2] > out[0, 1] > 0.0


class TestClampedIvar:
    def test_log_stretch_clamped_region_reports_zero_ivar(self) -> None:
        # x < 0 is clamped flat by log: locally non-injective -> no first-order
        # information -> ivar = 0 (never a spurious finite value).
        out = LogStretch(a=10.0, propagate_ivar=True)(
            {"flux": torch.tensor([-5.0, -0.0, 1.0]), "ivar": torch.ones(3)}
        )
        assert out["ivar"][0].item() == 0.0
        assert out["ivar"][2].item() > 0.0

    def test_log_stretch_kink_at_zero_reports_zero_ivar(self) -> None:
        # At x == 0 the clamped map is non-differentiable (left slope 0,
        # right slope a/ln(1+a)); the pinned convention is the conservative
        # one: the kink carries no first-order information.
        out = LogStretch(a=10.0, propagate_ivar=True)(
            {"flux": torch.tensor([0.0, 1.0]), "ivar": torch.ones(2)}
        )
        assert out["ivar"][0].item() == 0.0

    def test_arcsinh_never_clamps_so_ivar_stays_positive(self) -> None:
        out = ArcsinhStretch(a=1.0, propagate_ivar=True)(
            {"flux": torch.tensor([-50.0, 0.0, 50.0]), "ivar": torch.ones(3)}
        )
        assert (out["ivar"] > 0).all()


class TestDeltaMethodNumerics:
    def test_sqrt_poisson_variance_stabilisation_is_four(self) -> None:
        counts = torch.logspace(0, 6, 25, dtype=torch.float64)
        out = SqrtStretch(propagate_ivar=True)(
            {"flux": counts, "ivar": 1.0 / counts}
        )
        assert torch.allclose(
            out["ivar"], torch.full_like(counts, 4.0), rtol=1e-10
        )

    @pytest.mark.parametrize(
        "factory",
        [SqrtStretch, lambda: LogStretch(a=10.0), lambda: ArcsinhStretch(a=0.5)],
    )
    def test_delta_method_matches_finite_difference_float32(self, factory) -> None:
        t = factory()
        t.propagate_ivar = True
        t.propagates_ivar = True
        x = torch.tensor([0.25, 1.0, 3.5, 12.0], dtype=torch.float32)
        eps = 1e-3
        slope = (t(x + eps) - t(x - eps)) / (2 * eps)
        out = t({"flux": x, "ivar": torch.full_like(x, 4.0)})["ivar"]
        want = 4.0 / slope.pow(2)
        assert torch.allclose(out, want, rtol=1e-2, atol=1e-6)

    @pytest.mark.parametrize("factory", [SqrtStretch, lambda: LogStretch(a=10.0)])
    def test_stretched_values_are_monotonic_in_every_dtype(self, factory) -> None:
        xs = [0, 1, 30, 200]
        for dtype in (torch.uint8, torch.int16, torch.float16, torch.float64):
            out = factory()(torch.tensor(xs, dtype=dtype))
            values = out.tolist()
            assert values == sorted(values), (factory, dtype, values)
            assert values[0] == pytest.approx(0.0, abs=1e-3)

    def test_float16_sqrt_is_not_worse_than_half_ulp_of_float32(self) -> None:
        x = torch.tensor([0.5, 2.0, 60000.0], dtype=torch.float16)
        out16 = SqrtStretch()(x)
        out32 = SqrtStretch()(x.float()).to(torch.float16)
        assert torch.equal(out16, out32)
        assert out16.dtype == torch.float16

    def test_log_stretch_positive_slope_formula(self) -> None:
        # d/dx [log10(1 + a x) / log10(1 + a)] = a / ((1 + a x) ln(1 + a)).
        a = 10.0
        x = torch.tensor([0.7], dtype=torch.float64)
        out = LogStretch(a=a, propagate_ivar=True)(
            {"flux": x, "ivar": torch.ones(1, dtype=torch.float64)}
        )
        slope = a / ((1.0 + a * x) * math.log1p(a))
        assert torch.allclose(out["ivar"], 1.0 / slope.pow(2), rtol=1e-12)
