"""Tests for torchfits.transforms — ML-friendly FITS image preprocessing."""

import pytest
import torch

from transforms_reference import (
    sigma_clip_naive,
)

from torchfits.transforms import (
    ArcsinhStretch,
    AsymmetricSigmaClip,
    BackgroundSubtract,
    Compose,
    FITSHeaderNormalize,
    FITSHeaderScale,
    FITSScaleColumns,
    FITSTransform,
    GlobalScalarNorm,
    LogStretch,
    MinMaxNormalize,
    PercentileClipNormalize,
    RobustNormalize,
    SigmaClip,
    SqrtStretch,
    TNullToNan,
    ZScaleNormalize,
    estimate_background,
    safe_arcsinh,
    safe_log,
    zscale_limits,
)
from torchfits.transforms.helpers import (
    _amin,
    _amax,
    _flatten_dims,
    _median,
    _normalize_dims,
    _quantile,
    _reduce_keepdim,
)


# ---------------------------------------------------------------------------
# Helper: create test tensors with varying shapes and value distributions
# ---------------------------------------------------------------------------


def _make_tensor(
    shape: tuple[int, ...] = (4, 64, 64),
    dtype: "torch.dtype" = torch.float32,
    kind: str = "normal",
) -> torch.Tensor:
    """Create a test tensor with a specific distribution."""
    if kind == "normal":
        x = torch.randn(shape, dtype=dtype) * 10 + 100
    elif kind == "uniform":
        x = torch.rand(shape, dtype=dtype) * 100
    elif kind == "constant":
        x = torch.ones(shape, dtype=dtype) * 5.0
    elif kind == "zeros":
        x = torch.zeros(shape, dtype=dtype)
    elif kind == "mixed_sign":
        x = torch.randn(shape, dtype=dtype) * 50
    elif kind == "high_dr":
        # High dynamic range: few bright pixels on faint background
        x = torch.randn(shape, dtype=dtype) * 5 + 10
        x[..., 0, 0] = 20000  # bright source
        x[..., 1, 1] = 15000
    elif kind == "int16":
        x = torch.randint(-100, 100, shape, dtype=torch.int32).to(dtype)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return x


# ---------------------------------------------------------------------------
# Multi-dim reduction helpers
# ---------------------------------------------------------------------------


class TestNormalizeDims:
    def test_positive_dims(self) -> None:
        assert _normalize_dims(4, (2, 3)) == (2, 3)

    def test_negative_dims(self) -> None:
        assert _normalize_dims(4, (-2, -1)) == (2, 3)

    def test_mixed_dims(self) -> None:
        assert _normalize_dims(4, (0, -1)) == (0, 3)

    def test_removes_duplicates(self) -> None:
        assert _normalize_dims(3, (0, 0, -1)) == (0, 2)

    def test_sorted_output(self) -> None:
        assert _normalize_dims(5, (4, 1, 3)) == (1, 3, 4)


class TestFlattenDims:
    def test_contiguous_dims_at_end(self) -> None:
        x = torch.randn(2, 3, 64, 64)
        flat = _flatten_dims(x, (2, 3))
        assert flat.shape == (2, 3, 64 * 64)

    def test_non_contiguous_dims(self) -> None:
        x = torch.randn(2, 3, 64, 64)
        flat = _flatten_dims(x, (1, 3))  # channels + width
        assert flat.shape == (2, 64, 3 * 64)

    def test_single_dim(self) -> None:
        x = torch.randn(2, 3, 64, 64)
        flat = _flatten_dims(x, (2,))
        # single dim: permute is a no-op, reshape(-1) collapses last dim
        # After permute(*keep, *(2,)): keep=[0,1,3], dims=(2,) → x.permute(0,1,3,2) → (2,3,64,64)
        # reshape(2,3,64,-1) → (2,3,64,64) — but we called flatten on dims=(2,), a single dim
        # So it's just a reshape with -1 on the last dim = original dim 2 size.
        # Actually, with len(dims)==1, we wouldn't call _flatten_dims at all from _reduce_keepdim.
        # But testing directly: it should work.
        assert flat.numel() == x.numel()

    def test_all_dims(self) -> None:
        x = torch.randn(2, 3, 4)
        flat = _flatten_dims(x, (0, 1, 2))
        assert flat.shape == (2 * 3 * 4,)

    def test_preserves_values(self) -> None:
        x = torch.randn(2, 3, 4, 5)
        flat = _flatten_dims(x, (1, 3))
        expected = x.permute(0, 2, 1, 3).reshape(2, 4, 3 * 5)
        assert torch.equal(flat, expected)


class TestReduceKeepdim:
    def test_single_dim_fast_path(self) -> None:
        x = torch.randn(4, 64, 64)
        result = _reduce_keepdim(
            x, (0,), lambda t, d, k: torch.mean(t, dim=d, keepdim=k)
        )
        assert result.shape == (1, 64, 64)
        assert torch.allclose(result[0], x.mean(dim=0))

    def test_multi_dim_flatten_path(self) -> None:
        x = torch.randn(4, 64, 64)
        result = _reduce_keepdim(
            x, (-2, -1), lambda t, d, k: torch.mean(t, dim=d, keepdim=k)
        )
        assert result.shape == (4, 1, 1)
        for i in range(4):
            assert abs(result[i, 0, 0].item() - x[i].mean().item()) < 1e-5

    def test_single_dim_keepdim(self) -> None:
        x = torch.randn(2, 3, 64, 64)
        result = _median(x, (0,))
        assert result.shape == (1, 3, 64, 64)


class TestMedianAminAmaxQuantile:
    def test_median_multi_dim(self) -> None:
        x = torch.randn(4, 64, 64)
        m = _median(x, (-2, -1))
        assert m.shape == (4, 1, 1)

    def test_median_vs_torch(self) -> None:
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        m = _median(x, (-2, -1))
        # torch.median returns the lower median for even element counts
        assert m[0, 0, 0].item() == 2.0  # lower median of [1,2,3,4]
        assert m[1, 0, 0].item() == 6.0  # lower median of [5,6,7,8]

    def test_amin_multi_dim(self) -> None:
        x = torch.randn(4, 64, 64)
        vmin = _amin(x, (-2, -1))
        assert vmin.shape == (4, 1, 1)

    def test_amax_multi_dim(self) -> None:
        x = torch.randn(4, 64, 64)
        vmax = _amax(x, (-2, -1))
        assert vmax.shape == (4, 1, 1)

    def test_quantile_multi_dim(self) -> None:
        x = torch.randn(4, 64, 64)
        q = _quantile(x, 0.5, (-2, -1))
        assert q.shape == (4, 1, 1)

    def test_amin_amax_non_contiguous(self) -> None:
        x = torch.randn(2, 3, 64, 64)
        vmin = _amin(x, (1, -1))  # dims 1 and 3
        assert vmin.shape == (2, 1, 64, 1)


# ---------------------------------------------------------------------------
# Safe math utilities
# ---------------------------------------------------------------------------


class TestSafeMath:
    def test_safe_arcsinh_positive(self) -> None:
        x = torch.tensor([0.1, 1.0, 10.0, 100.0])
        out = safe_arcsinh(x, scale=1.0)
        expected = torch.asinh(x)
        assert torch.allclose(out, expected, rtol=1e-6)

    def test_safe_arcsinh_preserves_dtype(self) -> None:
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out = safe_arcsinh(x)
        assert out.dtype == torch.float32

    def test_safe_log_positive(self) -> None:
        x = torch.tensor([1.0, 10.0, 100.0])
        out = safe_log(x)
        expected = torch.log(x)
        assert torch.allclose(out, expected, rtol=1e-6)

    def test_safe_log_zero_clamped(self) -> None:
        x = torch.tensor([0.0, 1.0])
        out = safe_log(x)
        assert not torch.isinf(out[0])
        assert out[0] > -50  # roughly log(1e-9) in float64

    def test_safe_log_zero_finite(self) -> None:
        x = torch.tensor([0.0])
        out = safe_log(x)
        assert torch.isfinite(out).all()

    def test_safe_log_preserves_dtype(self) -> None:
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out = safe_log(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# estimate_background and zscale_limits
# ---------------------------------------------------------------------------


class TestEstimateBackground:
    def test_normal_distribution(self) -> None:
        x = torch.randn(4, 64, 64) * 5 + 100
        med, std = estimate_background(x)
        assert med.shape == (4, 1, 1)
        assert std.shape == (4, 1, 1)
        # Median should be near 100, std near 5*1.4826
        assert abs(med.mean().item() - 100) < 3
        assert abs(std.mean().item() - 5 * 1.4826) < 3

    def test_constant_image(self) -> None:
        x = torch.ones(4, 64, 64) * 42.0
        med, std = estimate_background(x)
        assert torch.allclose(med, torch.tensor(42.0), atol=1e-5)
        assert torch.all(std < 1e-5)

    def test_mask_excludes_pixels_from_median(self) -> None:
        # 5x5 image: values 1..25, mask out the central 3x3 (values 7..19)
        x = torch.arange(1, 26, dtype=torch.float32).reshape(1, 5, 5)
        mask = torch.ones(1, 5, 5, dtype=torch.bool)
        mask[0, 1:4, 1:4] = False  # exclude central 3x3
        med, _ = estimate_background(x, dim=(-2, -1), mask=mask)
        # Only edge pixels (16 values: 1..6, 20..25, 11..15, 16..19... actually
        # the 16 border values) should contribute to median.
        # Border values: row0 (1-5), row4 (21-25), col0 from rows1-3 (6,11,16),
        # col4 from rows1-3 (10,15,20). Sorted: 1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25,26?
        # Wait, with 5x5 = 25 values, masking 3x3=9 leaves 16. Values 1-25.
        # Border: [1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25]
        # PyTorch's nanmedian returns the lower median for even counts
        # (8th of 16 sorted values = 11).
        assert abs(med.item() - 11.0) < 0.5


class TestZScaleLimits:
    def test_returns_valid_range(self) -> None:
        x = torch.randn(4, 64, 64) * 5 + 100
        z1, z2 = zscale_limits(x)
        assert torch.all(z1 < z2)

    def test_constant_image_fallback(self) -> None:
        x = torch.ones(4, 64, 64) * 42.0
        z1, z2 = zscale_limits(x)
        assert torch.all(z1 < z2)  # fallback adds 1e-6


# ---------------------------------------------------------------------------
# FITSTransform base and Compose
# ---------------------------------------------------------------------------


class TestFITSTransform:
    def test_raises_not_implemented(self) -> None:
        t = FITSTransform()
        with pytest.raises(NotImplementedError):
            t.forward(torch.zeros(3))
        with pytest.raises(NotImplementedError):
            t.inverse(torch.zeros(3))

    def test_call_delegates(self) -> None:
        class Dummy(FITSTransform):
            def forward(
                self, x: "torch.Tensor", mask: "torch.Tensor | None" = None
            ) -> "torch.Tensor":
                return x + 1

            def inverse(
                self, x: "torch.Tensor", mask: "torch.Tensor | None" = None
            ) -> "torch.Tensor":
                return x - 1

        d = Dummy()
        assert d(torch.tensor(1.0)).item() == 2.0
        assert d.inverse(torch.tensor(2.0)).item() == 1.0


class TestCompose:
    def test_forward_chain(self) -> None:
        class AddOne(FITSTransform):
            def forward(
                self, x: "torch.Tensor", mask: "torch.Tensor | None" = None
            ) -> "torch.Tensor":
                return x + 1

            def inverse(
                self, x: "torch.Tensor", mask: "torch.Tensor | None" = None
            ) -> "torch.Tensor":
                return x - 1

        c = Compose([AddOne(), AddOne(), AddOne()])
        assert c(torch.tensor(0.0)).item() == 3.0

    def test_inverse_reverses_chain(self) -> None:
        class MulTwo(FITSTransform):
            def forward(
                self, x: "torch.Tensor", mask: "torch.Tensor | None" = None
            ) -> "torch.Tensor":
                return x * 2

            def inverse(
                self, x: "torch.Tensor", mask: "torch.Tensor | None" = None
            ) -> "torch.Tensor":
                return x / 2

        c = Compose([MulTwo(), MulTwo()])
        x = torch.tensor(5.0)
        fwd = c.forward(x)
        assert fwd.item() == 20.0
        inv = c.inverse(fwd)
        assert torch.allclose(inv, x)

    def test_len_and_getitem(self) -> None:
        class Id(FITSTransform):
            def forward(
                self, x: "torch.Tensor", mask: "torch.Tensor | None" = None
            ) -> "torch.Tensor":
                return x

            def inverse(
                self, x: "torch.Tensor", mask: "torch.Tensor | None" = None
            ) -> "torch.Tensor":
                return x

        c = Compose([Id(), Id(), Id()])
        assert len(c) == 3
        assert c[0] is c.transforms[0]

    def test_repr(self) -> None:
        rep = repr(Compose([ArcsinhStretch(a=0.1)]))
        assert "Compose" in rep
        assert "ArcsinhStretch" in rep


# ---------------------------------------------------------------------------
# Stateless stretch transforms (exact roundtrip)
# ---------------------------------------------------------------------------


class TestArcsinhStretch:
    def test_roundtrip_identity(self) -> None:
        x = _make_tensor((3, 32, 32), kind="uniform") + 1.0
        t = ArcsinhStretch(a=1.0)
        restored = t.inverse(t.forward(x))
        # arcsinh -> sinh through float64 has ~float32-epsilon error
        err = (x - restored).abs().max().item()
        assert err < 5e-5, f"roundtrip error {err} too large"

    def test_output_in_range(self) -> None:
        x = torch.linspace(0, 100, 1000).reshape(10, 100)
        t = ArcsinhStretch(a=1.0)
        out = t.forward(x)
        assert out.min() >= 0
        assert torch.isfinite(out).all()

    def test_different_a_values(self) -> None:
        x = _make_tensor((2, 16, 16), kind="uniform") + 1.0
        for a in [0.01, 0.1, 1.0, 10.0]:
            t = ArcsinhStretch(a=a)
            restored = t.inverse(t.forward(x))
            err = (x - restored).abs().max().item()
            assert err < 1e-4, f"a={a}: roundtrip error {err}"

    def test_preserves_dtype(self) -> None:
        x = torch.randn(4, 32, 32, dtype=torch.float32) * 10
        t = ArcsinhStretch()
        out = t.forward(x)
        assert out.dtype == torch.float32
        inv = t.inverse(out)
        assert inv.dtype == torch.float32

    def test_negative_inputs(self) -> None:
        x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        t = ArcsinhStretch(a=1.0)
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_rejects_non_positive_a(self) -> None:
        """ArcsinhStretch(a=0) and a<0 must raise ValueError (silent NaN otherwise)."""
        with pytest.raises(ValueError, match="must be > 0"):
            ArcsinhStretch(a=0.0)
        with pytest.raises(ValueError, match="must be > 0"):
            ArcsinhStretch(a=-1.0)


class TestLogStretch:
    def test_roundtrip_identity(self) -> None:
        x = torch.linspace(1, 1000, 100)
        t = LogStretch(a=1000.0)
        restored = t.inverse(t.forward(x))
        # log10->pow10 roundtrip through float64 has limited float32 precision
        err = (x - restored).abs().max().item()
        assert err < 2e-3, f"roundtrip error {err}"

    def test_negative_clamped(self) -> None:
        x = torch.tensor([-5.0, 0.0, 5.0])
        t = LogStretch(a=1000.0)
        out = t.forward(x)
        # Negative values should be clamped to 0, producing the same result as x=0
        assert out[0].item() == out[1].item()
        # Positive values produce a larger result
        assert out[2].item() > out[1].item()

    def test_preserves_dtype(self) -> None:
        x = torch.rand(4, 32, 32, dtype=torch.float32) * 100
        t = LogStretch()
        out = t.forward(x)
        assert out.dtype == torch.float32

    def test_rejects_non_positive_a(self) -> None:
        """LogStretch(a=0) and a<0 must raise ValueError (silent NaN otherwise)."""
        with pytest.raises(ValueError, match="must be > 0"):
            LogStretch(a=0.0)
        with pytest.raises(ValueError, match="must be > 0"):
            LogStretch(a=-1.0)


class TestSqrtStretch:
    def test_roundtrip_identity(self) -> None:
        x = torch.linspace(1, 100, 100)
        t = SqrtStretch()
        restored = t.inverse(t.forward(x))
        assert torch.allclose(restored, x, rtol=1e-5)

    def test_negative_clamped(self) -> None:
        x = torch.tensor([-5.0, 0.0, 5.0])
        t = SqrtStretch()
        out = t.forward(x)
        assert out[0] == 0.0
        assert out[1] == 0.0
        assert out[2] > 0.0


# ---------------------------------------------------------------------------
# Stateful normalizer transforms
# ---------------------------------------------------------------------------


class TestZScaleNormalize:
    def test_roundtrip_identity(self) -> None:
        x = _make_tensor((4, 32, 32), kind="normal")
        t = ZScaleNormalize()
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_output_in_01_range(self) -> None:
        x = _make_tensor((4, 32, 32), kind="normal")
        t = ZScaleNormalize()
        out = t.forward(x)
        # Most values should be in [0, 1], but some outliers may be outside
        # due to the contrast-based limits
        assert out.min() >= -0.5
        assert out.max() <= 1.5

    def test_inverse_without_forward_raises(self) -> None:
        t = ZScaleNormalize()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self) -> None:
        x = torch.ones(4, 32, 32) * 42.0
        t = ZScaleNormalize()
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_mask_no_mask_gives_same_result(self) -> None:
        """Mask=None should produce identical result to no mask passed."""
        x = _make_tensor((4, 32, 32), kind="normal")
        t1 = ZScaleNormalize()
        t2 = ZScaleNormalize()
        out1 = t1.forward(x, mask=None)
        out2 = t2.forward(x)
        assert torch.allclose(out1, out2, atol=1e-7)


class TestRobustNormalize:
    def test_roundtrip_identity(self) -> None:
        x = _make_tensor((4, 32, 32), kind="normal")
        t = RobustNormalize()
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_near_zero_median(self) -> None:
        x = _make_tensor((4, 32, 32), kind="normal")
        t = RobustNormalize()
        out = t.forward(x)
        # After robust normalization, the median should be near zero
        for i in range(4):
            assert abs(out[i].median().item()) < 0.1

    def test_inverse_without_forward_raises(self) -> None:
        t = RobustNormalize()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self) -> None:
        x = torch.ones(4, 32, 32) * 42.0
        t = RobustNormalize()
        out = t.forward(x)
        # All values should be ~0 after normalization (image is flat)
        assert out.abs().max() < 1e-5


class TestBackgroundSubtract:
    def test_roundtrip_identity(self) -> None:
        x = _make_tensor((4, 32, 32), kind="normal")
        t = BackgroundSubtract()
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_subtracts_median(self) -> None:
        x = torch.ones(4, 64, 64) * 5.0
        x[0, 0, 0] = 100.0  # outlier
        t = BackgroundSubtract()
        out = t.forward(x)
        # Most values should be near 0 after subtraction
        median_after = out.median().item()
        assert abs(median_after) < 0.5

    def test_inverse_without_forward_raises(self) -> None:
        t = BackgroundSubtract()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))


class TestPercentileClipNormalize:
    def test_roundtrip_identity(self) -> None:
        # PercentileClipNormalize is lossy when percentiles clip values.
        # Use 0/100 to avoid clipping (full range, exact roundtrip).
        x = _make_tensor((4, 32, 32), kind="uniform")
        t = PercentileClipNormalize(lower_pct=0, upper_pct=100)
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_output_range(self) -> None:
        x = _make_tensor((4, 32, 32), kind="normal")
        t = PercentileClipNormalize(lower_pct=0, upper_pct=100)
        out = t.forward(x)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_clips_outliers(self) -> None:
        x = torch.zeros(10, 10)
        x[0, 0] = 1e6  # extreme outlier
        t = PercentileClipNormalize(lower_pct=10, upper_pct=90)
        out = t.forward(x)
        # After clipping, the outlier should be clamped
        assert out[0, 0].item() <= 1.0

    def test_inverse_without_forward_raises(self) -> None:
        t = PercentileClipNormalize()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self) -> None:
        x = torch.ones(4, 32, 32) * 42.0
        t = PercentileClipNormalize(lower_pct=0, upper_pct=100)
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-4)

    def test_per_channel(self) -> None:
        x = torch.stack(
            [
                torch.randn(64, 64) * 10 + 100,
                torch.randn(64, 64) * 2 + 10,
            ]
        )
        t = PercentileClipNormalize(lower_pct=5, upper_pct=95, dim=(-2, -1))
        out = t.forward(x)
        # Each channel should be independently normalized to [0, 1]
        assert out.shape == (2, 64, 64)
        for c in range(2):
            assert out[c].max() <= 1.0 + 1e-6
            assert out[c].min() >= -1e-6


class TestMinMaxNormalize:
    def test_roundtrip_identity(self) -> None:
        x = _make_tensor((4, 32, 32), kind="uniform")
        t = MinMaxNormalize()
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_output_range(self) -> None:
        x = _make_tensor((4, 32, 32), kind="normal")
        t = MinMaxNormalize()
        out = t.forward(x)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_constant_image(self) -> None:
        x = torch.ones(4, 32, 32) * 42.0
        t = MinMaxNormalize()
        out = t.forward(x)
        # Constant image normalizes to all zeros (vmin == 42, vmax ≈ 42 + 1e-6)
        assert out.abs().max() < 1e-5

    def test_inverse_without_forward_raises(self) -> None:
        t = MinMaxNormalize()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))


class TestMaskNanRoundtrip:
    """Verify transforms handle NaN-contaminated data correctly with masks."""

    def test_nan_roundtrip_with_mask(self) -> None:
        """NaN pixels excluded from stats, remain NaN in output, roundtrip works."""
        x = _make_tensor((4, 32, 32), kind="normal")
        # Inject NaN at known positions
        x[0, 0, 0] = float("nan")
        x[2, 15, 15] = float("nan")
        # Build a valid mask that excludes the NaN positions
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[0, 0, 0] = False
        mask[2, 15, 15] = False

        t = MinMaxNormalize()
        out = t.forward(x, mask=mask)
        # NaN pixels should stay NaN in the output
        assert torch.isnan(out[0, 0, 0])
        assert torch.isnan(out[2, 15, 15])
        # Valid pixels should be normalized to [0, 1]
        valid_out = out[~torch.isnan(out)]
        assert valid_out.min() >= -1e-6
        assert valid_out.max() <= 1.0 + 1e-6

        # Roundtrip: NaN positions excluded from min/max so inversion works
        restored = t.inverse(out)
        assert torch.isnan(restored[0, 0, 0])
        assert torch.isnan(restored[2, 15, 15])
        # Non-NaN values should roundtrip
        valid_x = x[mask]
        valid_restored = restored[mask]
        assert torch.allclose(valid_restored, valid_x, atol=1e-4)


# ---------------------------------------------------------------------------
# FITSHeaderScale
# ---------------------------------------------------------------------------


class TestFITSHeaderScale:
    def test_roundtrip_identity(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0])
        t = FITSHeaderScale(bscale=2.0, bzero=10.0)
        physical = t.forward(x)
        expected = torch.tensor([12.0, 14.0, 16.0])
        assert torch.allclose(physical, expected)
        restored = t.inverse(physical)
        assert torch.allclose(restored, x)

    def test_from_header(self) -> None:
        header: dict[str, object] = {"BSCALE": 0.5, "BZERO": 100.0}
        t = FITSHeaderScale.from_header(header)
        assert t.bscale == 0.5
        assert t.bzero == 100.0

    def test_from_header_defaults(self) -> None:
        header: dict[str, object] = {}
        t = FITSHeaderScale.from_header(header)
        assert t.bscale == 1.0
        assert t.bzero == 0.0

    def test_identity_noop(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0])
        t = FITSHeaderScale(bscale=1.0, bzero=0.0)
        out = t.forward(x)
        assert out.data_ptr() == x.data_ptr()  # same tensor, no copy
        inv = t.inverse(x)
        assert inv.data_ptr() == x.data_ptr()

    def test_preserves_int_dtype(self) -> None:
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        t = FITSHeaderScale(bscale=1.0, bzero=0.0)
        out = t.forward(x)
        assert out.dtype == torch.int32
        assert out.data_ptr() == x.data_ptr()

    def test_bscale_only(self) -> None:
        x = torch.tensor([1.0, 2.0])
        t = FITSHeaderScale(bscale=10.0, bzero=0.0)
        physical = t.forward(x)
        assert torch.allclose(physical, torch.tensor([10.0, 20.0]))
        assert torch.allclose(t.inverse(physical), x)

    def test_bzero_only(self) -> None:
        x = torch.tensor([1.0, 2.0])
        t = FITSHeaderScale(bscale=1.0, bzero=100.0)
        physical = t.forward(x)
        assert torch.allclose(physical, torch.tensor([101.0, 102.0]))
        assert torch.allclose(t.inverse(physical), x)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_pixel_image(self) -> None:
        x = torch.tensor([[42.0]])
        transforms = [
            ArcsinhStretch(),
            ZScaleNormalize(dim=(-2, -1)),
            RobustNormalize(dim=(-2, -1)),
            BackgroundSubtract(dim=(-2, -1)),
            PercentileClipNormalize(dim=(-2, -1)),
            MinMaxNormalize(dim=(-2, -1)),
        ]
        for t in transforms:
            out = t.forward(x)
            assert out.shape == (1, 1), f"{t}: shape mismatch"
            restored = t.inverse(out)
            assert torch.allclose(restored, x, atol=1e-4), f"{t}: roundtrip failed"

    def test_1d_vector(self) -> None:
        x = torch.linspace(0, 100, 50)
        t = RobustNormalize(dim=(-1,))
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_3d_cube(self) -> None:
        x = _make_tensor((2, 32, 32), kind="uniform")
        t = ZScaleNormalize(dim=(-2, -1))
        out = t.forward(x)
        assert out.shape == (2, 32, 32)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_batched_4d(self) -> None:
        x = _make_tensor((4, 3, 64, 64), kind="normal")
        t = Compose([BackgroundSubtract(dim=(-2, -1)), ArcsinhStretch(a=0.1)])
        out = t.forward(x)
        assert out.shape == (4, 3, 64, 64)
        restored = t.inverse(out)
        # arcsinh + bg-subtract roundtrip through float64
        err = (x - restored).abs().max().item()
        assert err < 5e-4, f"batch roundtrip error {err} too large"

    def test_zero_std_image(self) -> None:
        x = torch.ones(4, 32, 32) * 10.0
        t = ZScaleNormalize()
        out = t.forward(x)
        # Should not crash and should produce finite output
        assert torch.isfinite(out).all()

    def test_extreme_dynamic_range(self) -> None:
        x = torch.ones(64, 64) * 1e-10
        x[32, 32] = 1e10
        t = ArcsinhStretch(a=1.0)
        out = t.forward(x)
        assert torch.isfinite(out).all()
        restored = t.inverse(out)
        # High DR may lose some precision in float32, but should be close
        rel_err = ((x - restored).abs() / (x.abs() + 1e-30)).max().item()
        assert rel_err < 1e-4  # float64 roundtrip preserves precision well

    def test_int16_tensor(self) -> None:
        x = torch.randint(-100, 100, (4, 32, 32), dtype=torch.int16)
        t = MinMaxNormalize(dim=(-2, -1))
        # Should convert to float internally and work
        out = t.forward(x.float())
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_compose_with_stateful_transforms(self) -> None:
        """Verify Compose chains stateful transforms correctly."""
        x = _make_tensor((4, 32, 32), kind="normal")
        c = Compose(
            [
                BackgroundSubtract(),
                RobustNormalize(),
                ZScaleNormalize(),
            ]
        )
        fwd = c.forward(x)
        assert fwd.shape == x.shape
        inv = c.inverse(fwd)
        err = (x - inv).abs().max().item()
        assert err < 2e-5

    def test_repr_methods(self) -> None:
        """Verify repr is informative and doesn't crash."""
        for cls, args in [
            (ArcsinhStretch, {"a": 0.1}),
            (LogStretch, {"a": 500}),
            (ZScaleNormalize, {"contrast": 0.3}),
            (RobustNormalize, {"dim": (-1,)}),
            (PercentileClipNormalize, {"lower_pct": 5, "upper_pct": 95}),
            (FITSHeaderScale, {"bscale": 2.0, "bzero": 10.0}),
        ]:
            t = cls(**args)  # type: ignore[arg-type]
            r = repr(t)
            assert len(r) > 0
            assert cls.__name__ in r


# ---------------------------------------------------------------------------
# SigmaClip
# ---------------------------------------------------------------------------


class TestIntegerInputDtypes:
    """Stats transforms must accept reader dtypes (UInt16 subsets etc.).

    ``read_subset`` returns BZERO-scaled FITS data as UInt16, and torch has
    no reduction kernels for the uint16/32/64 line; int dtypes must also
    not trip the inf/NaN sentinels.
    """

    def test_minmax_uint16(self) -> None:
        if not hasattr(torch, "uint16"):
            pytest.skip("torch.uint16 unavailable")
        x = torch.tensor([[3, 30000, 7], [9, 1, 20000]], dtype=torch.uint16).unsqueeze(
            0
        )
        out = MinMaxNormalize()(x)
        assert out.dtype == torch.float32
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6
        assert out[0, 0, 1].item() > 0.99

    def test_percentile_int32(self) -> None:
        x = torch.randint(0, 1000, (2, 16, 16), dtype=torch.int32)
        out = PercentileClipNormalize(lower_pct=5.0, upper_pct=95.0)(x)
        assert out.dtype == torch.float32
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_robust_normalize_int64(self) -> None:
        x = torch.randint(0, 1000, (2, 16, 16), dtype=torch.int64)
        out = RobustNormalize()(x)
        assert out.dtype == torch.float64  # int64 keeps precision as f64

    def test_sigma_clip_int32_promotes_float(self) -> None:
        x = torch.full((2, 8, 8), 50, dtype=torch.int32)
        x[0, 0, 0] = 1000
        out = SigmaClip(n_sigma=3.0, max_iter=5)(x)
        assert out.dtype == torch.float32
        assert out[0, 0, 0].item() < 100.0

    def test_median_fill_int32(self) -> None:
        x = torch.full((2, 8, 8), 50, dtype=torch.int32)
        x[0, 0, 0] = 1000
        out = SigmaClip(n_sigma=3.0, max_iter=5, fill="median")(x)
        assert out.dtype == torch.float32
        assert abs(out[0, 0, 0].item() - 50.0) < 1.0

    def test_masked_int_minmax(self) -> None:
        x = torch.tensor([[5, 200, 300]], dtype=torch.int16).unsqueeze(0)
        mask = torch.tensor([[[True, True, False]]])
        out = MinMaxNormalize()(x, mask=mask)
        assert out.dtype == torch.float32
        # Masked pixel excluded from stats: valid range [5, 200] -> [0, 1],
        # the 300 stays outside the normalized range (transform doesn't clip).
        assert abs(out[0, 0, 0].item() - 0.0) < 1e-5
        assert abs(out[0, 0, 1].item() - 1.0) < 1e-5
        assert out[0, 0, 2].item() > 1.0

    def test_estimate_background_uint16(self) -> None:
        if not hasattr(torch, "uint16"):
            pytest.skip("torch.uint16 unavailable")
        x = torch.tensor([[10, 11, 12, 5000]], dtype=torch.uint16).unsqueeze(0)
        from torchfits.transforms.helpers import estimate_background

        med, std = estimate_background(x, dim=(-2, -1))
        assert med.dtype == torch.float32
        assert med[0, 0, 0].item() > 10.0  # median of the background columns


class TestSigmaClip:
    def test_removes_outliers(self) -> None:
        x = torch.ones(10, 10) * 5.0
        x[0, 0] = 100.0  # extreme outlier
        t = SigmaClip(n_sigma=3.0, max_iter=5)
        out = t.forward(x)
        # The outlier should be replaced with the mean (~5)
        assert out[0, 0].item() < 10.0

    def test_no_clipping_on_clean_data(self) -> None:
        x = _make_tensor((4, 32, 32), kind="normal")
        t = SigmaClip(n_sigma=10.0, max_iter=3)
        out = t.forward(x)
        # With n_sigma=10, almost nothing should be clipped — values very close
        assert (out - x).abs().max().item() < 0.5

    def test_inverse_raises(self) -> None:
        t = SigmaClip()
        with pytest.raises(RuntimeError, match="irrecoverable"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self) -> None:
        x = torch.ones(4, 32, 32) * 42.0
        t = SigmaClip()
        out = t.forward(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_per_channel(self) -> None:
        x = torch.randn(4, 32, 32) * 5 + 10
        t = SigmaClip(dim=(-2, -1))
        out = t.forward(x)
        assert out.shape == (4, 32, 32)
        assert torch.isfinite(out).all()

    def test_median_fill(self) -> None:
        x = torch.ones(10, 10) * 5.0
        x[0, 0] = 100.0
        t = SigmaClip(n_sigma=3.0, fill="median")
        out = t.forward(x)
        assert out[0, 0].item() < 10.0

    def test_repr(self) -> None:
        t = SigmaClip(n_sigma=5.0, max_iter=3, fill="median")
        r = repr(t)
        assert "SigmaClip" in r
        assert "5.0" in r


# ---------------------------------------------------------------------------
# FITSHeaderNormalize
# ---------------------------------------------------------------------------


class TestFITSHeaderNormalize:
    def test_int16_scales_to_01(self) -> None:
        header: dict[str, object] = {"BITPIX": 16, "BSCALE": 1.0, "BZERO": 0.0}
        x = torch.tensor([-32768, 0, 32767], dtype=torch.float32)
        t = FITSHeaderNormalize(header)
        out = t.forward(x)
        assert out.min() >= 0
        assert out.max() <= 1.0
        assert out[1].item() == pytest.approx(0.5, abs=0.01)

    def test_int16_with_bzero(self) -> None:
        header: dict[str, object] = {"BITPIX": 16, "BSCALE": 1.0, "BZERO": 32768.0}
        x = torch.tensor([0.0, 32768.0, 65535.0])
        t = FITSHeaderNormalize(header)
        out = t.forward(x)
        assert out.min() >= 0
        assert out.max() <= 1.0
        # Physical range: [-32768, 32767] * 1.0 + 32768 = [0, 65535]
        assert out[1].item() == pytest.approx(0.5, abs=0.01)

    def test_roundtrip_int16(self) -> None:
        header: dict[str, object] = {"BITPIX": 16, "BSCALE": 2.0, "BZERO": 100.0}
        x = torch.tensor([0.0, 500.0, 65534.0])
        t = FITSHeaderNormalize(header)
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-4)

    def test_float32_no_scale(self) -> None:
        header: dict[str, object] = {"BITPIX": -32}
        x = torch.randn(4, 32, 32)
        t = FITSHeaderNormalize(header, scale_floats=False)
        out = t.forward(x)
        assert torch.equal(out, x)

    def test_float32_with_scale(self) -> None:
        header: dict[str, object] = {"BITPIX": -32}
        x = _make_tensor((4, 32, 32), kind="uniform")
        t = FITSHeaderNormalize(header, scale_floats=True)
        out = t.forward(x)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_uint8(self) -> None:
        header: dict[str, object] = {"BITPIX": 8, "BSCALE": 1.0, "BZERO": 0.0}
        x = torch.tensor([0.0, 128.0, 255.0])
        t = FITSHeaderNormalize(header)
        out = t.forward(x)
        assert out.min() >= 0
        assert out.max() <= 1.0
        assert out[1].item() == pytest.approx(0.5, abs=0.01)

    def test_repr(self) -> None:
        t = FITSHeaderNormalize({"BITPIX": -32}, scale_floats=True)
        r = repr(t)
        assert "FITSHeaderNormalize" in r
        assert "bitpix=-32" in r


# ---------------------------------------------------------------------------
# GlobalScalarNorm (P5 — linear, invertible)
# ---------------------------------------------------------------------------


class TestGlobalScalarNorm:
    def test_median_norm_roundtrip(self) -> None:
        x = torch.randn(4, 64, 64) * 10 + 100
        t = GlobalScalarNorm(stat="median")
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_max_norm(self) -> None:
        x = torch.rand(8, 32) * 50
        t = GlobalScalarNorm(stat="max")
        out = t.forward(x)
        assert out.max().item() <= 1.0 + 1e-6
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_mean_norm(self) -> None:
        x = torch.rand(2, 100) + 1.0  # positive values, mean ~1.5
        t = GlobalScalarNorm(stat="mean")
        out = t.forward(x)
        # After dividing by mean, the output mean should be ~1
        assert abs(out.mean().item() - 1.0) < 0.1

    def test_rms_norm(self) -> None:
        x = torch.randn(2, 100)
        t = GlobalScalarNorm(stat="rms")
        out = t.forward(x)
        # RMS of normalized data should be ~1
        rms = torch.sqrt((out**2).mean())
        assert abs(rms.item() - 1.0) < 0.1

    def test_per_image_norm(self) -> None:
        x = torch.randn(4, 32, 32) * 5 + 20
        t = GlobalScalarNorm(stat="median", dim=(-2, -1))
        out = t.forward(x)
        for i in range(4):
            assert abs(out[i].median().item() - 1.0) < 0.1
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_inverse_without_forward_raises(self) -> None:
        t = GlobalScalarNorm()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))

    def test_invalid_stat_raises(self) -> None:
        with pytest.raises(ValueError):
            GlobalScalarNorm(stat="invalid")

    def test_repr(self) -> None:
        r = repr(GlobalScalarNorm(stat="rms", dim=(-1,)))
        assert "GlobalScalarNorm" in r
        assert "rms" in r


# ---------------------------------------------------------------------------
# SigmaClip — zero-alloc buffer vs naive per-iteration-alloc parity tests
# ---------------------------------------------------------------------------


class TestSigmaClipVectorized:
    """Verify the zero-alloc buffer SigmaClip matches naive per-iteration alloc."""

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        n_sigma: float = 3.0,
        max_iter: int = 5,
        dim: tuple[int, ...] = (-2, -1),
        fill: str = "mean",
    ) -> torch.Tensor:
        """Run the real zero-alloc SigmaClip and return clipped output."""
        t = SigmaClip(n_sigma=n_sigma, max_iter=max_iter, dim=dim, fill=fill)
        return t.forward(x)

    def test_removes_single_outlier(self) -> None:
        """Single extreme outlier in a uniform field."""
        torch.manual_seed(42)
        x = torch.ones(10, 10) * 5.0
        x[0, 0] = 100.0
        vec = self._run_vectorized(x, dim=(-2, -1))
        ref = sigma_clip_naive(x, 3.0, 5, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )
        # Outlier should be replaced with ~5
        assert vec[0, 0].item() < 10.0

    def test_clean_data_no_clipping(self) -> None:
        """Clean normal data with wide threshold — no clipping."""
        torch.manual_seed(42)
        x = torch.randn(4, 32, 32) * 5 + 100
        vec = self._run_vectorized(x, n_sigma=10.0, dim=(-2, -1))
        ref = sigma_clip_naive(x, 10.0, 3, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_per_channel(self) -> None:
        """Clipping computed independently per image (dim=-2,-1)."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 16) * 3 + 10
        x[0, 0, 0] = 50.0
        x[2, 0, 0] = -20.0
        vec = self._run_vectorized(x, dim=(-2, -1))
        ref = sigma_clip_naive(x, 3.0, 5, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_median_fill(self) -> None:
        """Fill clipped values with median instead of mean."""
        torch.manual_seed(42)
        x = torch.ones(8, 8) * 5.0
        x[0, 0] = 100.0
        x[1, 1] = -20.0
        vec = self._run_vectorized(x, dim=(-2, -1), fill="median")
        ref = sigma_clip_naive(x, 3.0, 5, (-2, -1), "median")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_wide_threshold_noop(self) -> None:
        """Very wide sigma threshold: no pixels clipped → output == input."""
        torch.manual_seed(42)
        x = torch.randn(3, 32, 32) * 5 + 20
        vec = self._run_vectorized(x, n_sigma=100.0, dim=(-2, -1))
        ref = sigma_clip_naive(x, 100.0, 3, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(vec, x, atol=1e-4)

    def test_global_dim(self) -> None:
        """Global sigma-clip (no dim specified → all pixels considered)."""
        torch.manual_seed(42)
        x = torch.randn(32, 32) * 5 + 10
        x[0, 0] = 100.0
        vec = self._run_vectorized(x, dim=(), fill="mean")
        ref = sigma_clip_naive(x, 3.0, 5, (), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_constant_image(self) -> None:
        """Constant image: std=0, no clipping."""
        x = torch.ones(4, 32, 32) * 42.0
        vec = self._run_vectorized(x, dim=(-2, -1))
        ref = sigma_clip_naive(x, 3.0, 5, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5)
        assert torch.allclose(vec, x, atol=1e-5)


# ---------------------------------------------------------------------------
# AsymmetricSigmaClip (simple one-pass asymmetric outlier rejection)
# ---------------------------------------------------------------------------


class TestAsymmetricSigmaClip:
    def test_clips_positive_outliers(self) -> None:
        x = torch.randn(10, 10) * 2 + 5.0
        x[0, 0] = 100.0  # extreme positive outlier
        t = AsymmetricSigmaClip(n_low=3.0, n_high=3.0)
        out = t.forward(x)
        # Outlier should be replaced with median (~5)
        assert out[0, 0].item() < 10.0

    def test_clips_negative_outliers(self) -> None:
        x = torch.randn(10, 10) * 2 + 5.0
        x[0, 0] = -50.0  # extreme negative outlier
        t = AsymmetricSigmaClip(n_low=3.0, n_high=3.0)
        out = t.forward(x)
        # Outlier should be replaced with median (~5)
        assert out[0, 0].item() > 0.0

    def test_asymmetric_thresholds(self) -> None:
        # Data with real variance so MAD > 0
        x = torch.randn(10, 10) * 2 + 5.0
        x[0, 0] = -8.0  # mild negative outlier
        x[1, 1] = 12.0  # mild positive outlier
        # Aggressive low clipping (n_low=10 -> keep negative), tight high clipping (n_high=1)
        t = AsymmetricSigmaClip(n_low=10.0, n_high=1.0)
        out = t.forward(x)
        # Negative outlier should survive (n_low=10 is very permissive)
        assert out[0, 0].item() < 0.0
        # Positive outlier should be clipped (n_high=1 is very strict)
        assert abs(out[1, 1].item() - 5.0) < 3.0

    def test_no_clipping_on_clean_data(self) -> None:
        x = torch.randn(4, 32, 32) * 5 + 100
        t = AsymmetricSigmaClip(n_low=10.0, n_high=10.0)
        out = t.forward(x)
        # With n_sigma=10, almost nothing should be clipped
        assert (out - x).abs().max().item() < 0.5

    def test_inverse_raises(self) -> None:
        t = AsymmetricSigmaClip()
        with pytest.raises(RuntimeError, match="irrecoverable"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self) -> None:
        x = torch.ones(4, 32, 32) * 42.0
        t = AsymmetricSigmaClip()
        out = t.forward(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_per_channel(self) -> None:
        x = torch.randn(4, 32, 32) * 5 + 10
        t = AsymmetricSigmaClip(dim=(-2, -1))
        out = t.forward(x)
        assert out.shape == (4, 32, 32)
        assert torch.isfinite(out).all()

    def test_invalid_n_raises(self) -> None:
        with pytest.raises(ValueError):
            AsymmetricSigmaClip(n_low=0)
        with pytest.raises(ValueError):
            AsymmetricSigmaClip(n_high=-1)

    def test_repr(self) -> None:
        r = repr(AsymmetricSigmaClip(n_low=5.0, n_high=2.0))
        assert "AsymmetricSigmaClip" in r
        assert "5.0" in r
        assert "2.0" in r

    def test_mask_excludes_pixels_from_background(self) -> None:
        """Mask forwarded to estimate_background changes the background estimate."""
        # 5x5 grid: 13 pixels at 100.0 (majority), 12 pixels at 0.0.
        # Without mask, median = 100.0 because >50% of pixels are 100.0.
        # With mask excluding the 100.0 values, median falls back to 0.0.
        x = torch.zeros(5, 5)
        x[:3, :] = 100.0  # rows 0-2: 15 pixels at 100.0
        x[3, 0] = 100.0  # 1 more = 16 out of 25 → majority

        # Mask: only the 8 pixels at 0.0 (rows 3-4, cols 1-4) are valid
        mask = torch.zeros(5, 5, dtype=torch.bool)
        mask[3:, 1:] = True

        t = AsymmetricSigmaClip(n_low=3.0, n_high=3.0)
        out = t.forward(x.clone(), mask=mask)

        # With mask, median=0.0, so the 100.0 pixels get clipped to ~0.0
        assert out[0, 0].item() < 10.0, (
            f"100.0 pixel not clipped to background: {out[0, 0].item()}"
        )


# ---------------------------------------------------------------------------
# FITSScaleColumns (table column TSCAL/TZERO scaling, invertible)
# ---------------------------------------------------------------------------


class TestFITSScaleColumns:
    def test_roundtrip(self) -> None:
        header: dict[str, object] = {
            "TFIELDS": 2,
            "TTYPE1": "FLUX",
            "TFORM1": "E",
            "TSCAL1": 0.001,
            "TZERO1": 0.0,
            "TTYPE2": "MAG",
            "TFORM2": "E",
            "TSCAL2": 1.0,
            "TZERO2": 25.0,
        }
        flux = torch.tensor([1000.0, 2000.0, 3000.0])
        mag = torch.tensor([10.0, 20.0, 30.0])
        x = {"FLUX": flux, "MAG": mag}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        # FLUX: physical = 0.001 * stored + 0 = [1.0, 2.0, 3.0]
        assert torch.allclose(out["FLUX"], torch.tensor([1.0, 2.0, 3.0]))
        # MAG: physical = 1.0 * stored + 25.0
        assert torch.allclose(out["MAG"], torch.tensor([35.0, 45.0, 55.0]))
        restored = t.inverse(out)
        assert torch.allclose(restored["FLUX"], flux)
        assert torch.allclose(restored["MAG"], mag)

    def test_empty_header(self) -> None:
        header: dict[str, object] = {}
        x = {"A": torch.randn(10)}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        assert torch.equal(out["A"], x["A"])  # no-op

    def test_identity_scales_noop(self) -> None:
        header = {
            "TFIELDS": 1,
            "TTYPE1": "DATA",
            "TFORM1": "E",
            "TSCAL1": 1.0,
            "TZERO1": 0.0,
        }
        x = {"DATA": torch.randn(100)}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        assert torch.equal(out["DATA"], x["DATA"])  # identity scales skipped

    def test_preserves_unrelated_columns(self) -> None:
        header = {
            "TFIELDS": 1,
            "TTYPE1": "FLUX",
            "TFORM1": "E",
            "TSCAL1": 2.0,
            "TZERO1": 10.0,
        }
        extra = torch.randn(5)
        x = {"FLUX": torch.ones(5), "EXTRA": extra}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        assert torch.equal(out["EXTRA"], extra)  # unrelated column untouched
        assert "EXTRA" in out

    def test_preserves_int_dtype(self) -> None:
        header = {
            "TFIELDS": 1,
            "TTYPE1": "N",
            "TFORM1": "J",
            "TSCAL1": 1.0,
            "TZERO1": 0.0,
        }
        x = {"N": torch.tensor([1, 2, 3], dtype=torch.int32)}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        assert out["N"].dtype == torch.int32

    def test_repr(self) -> None:
        header = {
            "TFIELDS": 1,
            "TTYPE1": "F",
            "TFORM1": "E",
            "TSCAL1": 0.5,
            "TZERO1": 100.0,
        }
        t = FITSScaleColumns.from_header(header)
        r = repr(t)
        assert "FITSScaleColumns" in r
        assert "0.5" in r

    def test_forward_with_mask_none(self) -> None:
        """FITSScaleColumns.forward accepts mask=None without error."""
        t = FITSScaleColumns({"FLUX": (0.5, 100.0)})
        data = {"FLUX": torch.tensor([1.0, 2.0, 3.0])}
        out = t.forward(data, mask=None)
        expected = torch.tensor([100.5, 101.0, 101.5])
        assert torch.allclose(out["FLUX"], expected)

    def test_forward_with_mask_tensor(self) -> None:
        """FITSScaleColumns.forward accepts a mask tensor (ignored for pointwise ops)."""
        t = FITSScaleColumns({"FLUX": (2.0, 0.0)})
        data = {"FLUX": torch.tensor([1.0, 2.0, 3.0])}
        mask = torch.ones(3, dtype=torch.bool)
        out = t.forward(data, mask=mask)
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(out["FLUX"], expected)

    def test_inverse_with_mask_none(self) -> None:
        """FITSScaleColumns.inverse also accepts mask=None."""
        t = FITSScaleColumns({"FLUX": (0.5, 100.0)})
        physical = {"FLUX": torch.tensor([100.5, 101.0, 101.5])}
        out = t.inverse(physical, mask=None)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(out["FLUX"], expected)


# ---------------------------------------------------------------------------
# TNullToNan (table column TNULL sentinel → NaN, lossy)
# ---------------------------------------------------------------------------


class TestTNullToNan:
    def test_replaces_sentinel_with_nan(self) -> None:
        header = {"TFIELDS": 1, "TTYPE1": "FLUX", "TFORM1": "J", "TNULL1": -999}
        x = {"FLUX": torch.tensor([1, -999, 3], dtype=torch.int32)}
        t = TNullToNan.from_header(header)
        out = t.forward(x)
        assert out["FLUX"].dtype == torch.float32  # promoted to float
        assert torch.isnan(out["FLUX"][1])
        assert out["FLUX"][0].item() == 1.0
        assert out["FLUX"][2].item() == 3.0

    def test_float_column_no_promotion(self) -> None:
        header = {"TFIELDS": 1, "TTYPE1": "VAL", "TFORM1": "E", "TNULL1": 0.0}
        x = {"VAL": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)}
        t = TNullToNan.from_header(header)
        out = t.forward(x)
        assert out["VAL"].dtype == torch.float32  # already float, no promotion
        assert torch.isnan(out["VAL"][0])

    def test_empty_header(self) -> None:
        header: dict[str, object] = {}
        x = {"A": torch.randn(10)}
        t = TNullToNan.from_header(header)
        out = t.forward(x)
        assert torch.equal(out["A"], x["A"])  # no-op

    def test_nulls_only_on_specified_columns(self) -> None:
        header = {
            "TFIELDS": 2,
            "TTYPE1": "GOOD",
            "TFORM1": "J",
            "TTYPE2": "BAD",
            "TFORM2": "J",
            "TNULL2": -99,
        }
        x = {
            "GOOD": torch.tensor([1, -99, 3], dtype=torch.int32),
            "BAD": torch.tensor([1, -99, 3], dtype=torch.int32),
        }
        t = TNullToNan.from_header(header)
        out = t.forward(x)
        # GOOD has no TNULL, should be untouched
        assert out["GOOD"].dtype == torch.int32
        assert out["GOOD"][1].item() == -99
        # BAD has TNULL=-99, should be converted to NaN
        assert torch.isnan(out["BAD"][1])

    def test_inverse_raises(self) -> None:
        t = TNullToNan.from_header({})
        with pytest.raises(RuntimeError, match="lossy"):
            t.inverse({})

    def test_repr(self) -> None:
        header = {"TFIELDS": 1, "TTYPE1": "X", "TFORM1": "J", "TNULL1": -1}
        t = TNullToNan.from_header(header)
        r = repr(t)
        assert "TNullToNan" in r
        assert "X" in r

    def test_forward_with_mask_none(self) -> None:
        """TNullToNan.forward accepts mask=None without parameter shadowing."""
        t = TNullToNan({"FLUX": -999.0})
        data = {"FLUX": torch.tensor([1.0, -999.0, 3.0])}
        out = t.forward(data, mask=None)
        assert torch.isnan(out["FLUX"][1])
        assert out["FLUX"][0].item() == 1.0
        assert out["FLUX"][2].item() == 3.0

    def test_forward_with_mask_tensor(self) -> None:
        """TNullToNan.forward accepts a mask tensor (ignored for pointwise logic)."""
        t = TNullToNan({"FLUX": -999.0})
        data = {"FLUX": torch.tensor([1.0, -999.0, 3.0])}
        extra_mask = torch.ones(3, dtype=torch.bool)
        out = t.forward(data, mask=extra_mask)
        assert torch.isnan(out["FLUX"][1])

    def test_multiple_columns_with_mask(self) -> None:
        """TNullToNan handles multiple columns with different TNULL values and mask=None."""
        t = TNullToNan({"FLUX": -999.0, "QUAL": 0.0})
        data = {
            "FLUX": torch.tensor([1.0, -999.0, 3.0]),
            "QUAL": torch.tensor([4.0, 0.0, 6.0], dtype=torch.int32),
            "EXTRA": torch.tensor([7.0, 8.0, 9.0]),
        }
        out = t.forward(data, mask=None)
        assert torch.isnan(out["FLUX"][1])
        assert torch.isnan(out["QUAL"][1])
        assert torch.equal(out["EXTRA"], torch.tensor([7.0, 8.0, 9.0]))
        assert out["QUAL"].dtype == torch.float32


# ---------------------------------------------------------------------------
# FITSHeaderScale extended roundtrip tests
# ---------------------------------------------------------------------------


class TestFITSHeaderScaleRoundtrip:
    def test_scaled_image_roundtrip(self) -> None:
        """Simulate a typical int16 image with BSCALE/BZERO."""
        raw = torch.randint(-100, 100, (64, 64), dtype=torch.int16)
        header: dict[str, object] = {"BITPIX": 16, "BSCALE": 0.5, "BZERO": 2000.0}
        t = FITSHeaderScale.from_header(header)
        physical = t.forward(raw.float())
        expected = raw.float() * 0.5 + 2000.0
        assert torch.allclose(physical, expected)
        restored = t.inverse(physical)
        # 0.5 is exactly representable in float32, roundtrip is exact
        assert torch.allclose(restored, raw.float())

    def test_unsigned_uint16_convention(self) -> None:
        """BZERO=32768 is the standard FITS unsigned int16 convention."""
        raw = torch.randint(0, 65535, (32, 32), dtype=torch.int32)
        header: dict[str, object] = {"BITPIX": 16, "BSCALE": 1.0, "BZERO": 32768.0}
        t = FITSHeaderScale.from_header(header)
        physical = t.forward(raw.float())
        expected = raw.float() * 1.0 + 32768.0
        assert torch.allclose(physical, expected)
        restored = t.inverse(physical)
        assert torch.allclose(restored, raw.float())

    def test_bscale_only_convention(self) -> None:
        """BSCALE != 1, BZERO = 0."""
        header: dict[str, object] = {"BITPIX": -32, "BSCALE": 2.0, "BZERO": 0.0}
        orig = torch.rand(16, 16) * 100
        raw = orig.clone()
        t = FITSHeaderScale.from_header(header)
        out = t.forward(raw)
        # 2.0 is exact in float32
        assert torch.allclose(out, orig.mul(2.0))
        assert torch.allclose(t.inverse(out), orig)

    def test_inverse_applied_to_identity(self) -> None:
        """After applying scale, inverse should get back the original."""
        x = torch.tensor([100.0, 200.0, 300.0])
        t = FITSHeaderScale(bscale=0.5, bzero=50.0)
        fwd = t.forward(x)
        inv = t.inverse(fwd)
        assert torch.allclose(inv, x, atol=1e-5)
