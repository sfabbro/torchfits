"""Mask-convention and edge-input pins for :mod:`torchfits.transforms.mask`.

Fits the R2 mask contract: boolean masks mean ``True`` = valid everywhere;
FITS DQ bitfields decode by 0-based bit *position* (value ``2**bit``), any
non-zero DQ is suspect by default; IVAR ``0`` means "no data". Edge inputs
(empty, all-invalid, shape mismatch, wide integers, degenerate knobs) must
either give the documented result or raise a typed error — never silently
mis-decode.
"""

from __future__ import annotations

import pytest
import torch

from torchfits.transforms import (
    apply_mask,
    combine_masks,
    mask_from_dq,
    mask_from_ivar,
    mask_from_nan,
)


class TestDqBitConvention:
    """Bit positions are 0-based, matching FITS DQ flag documentation."""

    def test_bit_positions_are_zero_based_positions(self) -> None:
        # "bit 11: flux low" carries value 2048 == 2**11, like every FITS DQ
        # flag table documents it (HST, DESI, MaNGA).
        dq = torch.tensor([[2048]], dtype=torch.int32)
        assert not mask_from_dq(dq, bad_bits=[11]).item()
        assert mask_from_dq(dq, bad_bits=[10]).item()

    def test_single_position_equals_iterable_of_one(self) -> None:
        dq = torch.tensor([[0, 1, 2, 4]], dtype=torch.int32)
        for bit in (0, 1, 2):
            assert torch.equal(
                mask_from_dq(dq, bad_bits=bit), mask_from_dq(dq, bad_bits=[bit])
            )

    def test_uint32_high_bit_decodes(self) -> None:
        # DQ shipped as unsigned 32-bit with the top flag bit set must decode
        # against bit position 31 without sign confusion.
        dq = torch.tensor([0, 2**31], dtype=torch.int64)  # values as read
        assert mask_from_dq(dq, bad_bits=[31]).tolist() == [True, False]
        assert mask_from_dq(dq, bad_bits=[30]).tolist() == [True, True]

    def test_negative_signed_dq_values_decode_bits(self) -> None:
        # int16 DQ with the sign bit used as a flag reads back negative; two's
        # complement bit tests must still flag exactly the documented bits.
        # -1 == 0xFFFF (every bit set), -2 == 0xFFFE (only bit 0 clear).
        dq = torch.tensor([-1, -2, 0], dtype=torch.int16)
        assert mask_from_dq(dq, bad_bits=[15]).tolist() == [False, False, True]
        assert mask_from_dq(dq, bad_bits=[0]).tolist() == [False, True, True]
        assert (dq.to(torch.int64) == 0).tolist() == [False, False, True]

    def test_top_bit_position_63_supported(self) -> None:
        dq = torch.tensor([-(2**63), 0], dtype=torch.int64)
        assert mask_from_dq(dq, bad_bits=[63]).tolist() == [False, True]


class TestDqTypedErrors:
    def test_position_beyond_63_raises_value_error(self) -> None:
        # 2**64 cannot ride in int64 bit ops; a silent all-valid decode would
        # hide the mistake entirely.
        dq = torch.zeros(3, dtype=torch.int64)
        with pytest.raises(ValueError, match="bit position"):
            mask_from_dq(dq, bad_bits=[64])
        with pytest.raises(ValueError, match="bit position"):
            mask_from_dq(dq, good_bits=[128], require_good=True)

    def test_negative_position_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _ = mask_from_dq(torch.zeros(2, dtype=torch.int32), bad_bits=[-1])

    def test_non_int_position_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="bit position"):
            _ = mask_from_dq(torch.zeros(2, dtype=torch.int32), bad_bits=[1.5])

    def test_float_extension_rejected(self) -> None:
        with pytest.raises(TypeError, match="integer-typed"):
            mask_from_dq(torch.zeros(3))

    def test_complex_extension_rejected_typed(self) -> None:
        # The int64 cast silently discarded the imaginary part (torch warning)
        # and mis-decoded the DQ — name the real rule instead.
        with pytest.raises(TypeError, match="integer-typed"):
            mask_from_dq(torch.zeros(3, dtype=torch.complex64))

    def test_require_good_without_good_bits_is_rejected(self) -> None:
        # Asking for "science-good required" without naming the good bit is a
        # silent no-op today: every pixel passes, the exact opposite of the
        # caller's intent. It must raise instead.
        dq = torch.tensor([[3, 0]], dtype=torch.int32)
        with pytest.raises(ValueError, match="good_bits"):
            mask_from_dq(dq, require_good=True)
        with pytest.raises(ValueError, match="good_bits"):
            mask_from_dq(dq, bad_bits=[1], require_good=True)


class TestMaskEdges:
    def test_empty_inputs(self) -> None:
        empty_i = torch.zeros(0, dtype=torch.int32)
        empty_f = torch.zeros(0)
        assert mask_from_dq(empty_i).shape == (0,)
        assert mask_from_ivar(empty_f).shape == (0,)
        assert mask_from_nan(empty_f).shape == (0,)
        combined = combine_masks(empty_f.to(torch.bool))
        assert combined is not None and combined.shape == (0,)
        assert apply_mask(empty_f, empty_f.to(torch.bool)).shape == (0,)

    def test_all_invalid_and_all_valid(self) -> None:
        dq = torch.full((2, 2), 7, dtype=torch.int32)
        assert not mask_from_dq(dq).any()
        zeros = torch.zeros(2, 2, dtype=torch.int32)
        assert mask_from_dq(zeros).all()
        filled = apply_mask(torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.bool))
        assert torch.isnan(filled).all()
        kept = apply_mask(torch.ones(2, 2), torch.ones(2, 2, dtype=torch.bool))
        assert torch.equal(kept, torch.ones(2, 2))

    def test_combine_masks_broadcasts_and_none_passthrough(self) -> None:
        row = torch.tensor([[True, False, True]])  # (1, 3)
        stack = torch.tensor([[True], [True], [True], [False]])  # (4, 1) -> (4, 3)
        combined = combine_masks(None, row, stack)
        assert combined is not None
        assert combined.shape == (4, 3)
        assert combined[:, 1].logical_not().all()
        assert not combined[3].any()
        assert combine_masks(None, None) is None

    def test_mask_from_ivar_boundaries(self) -> None:
        ivar = torch.tensor([0.0, 1e-30, -1.0, float("nan"), float("inf")])
        # > 0 strictly: zeros ("no data") and negatives are invalid. The
        # default require_finite=True rejects non-finite companions; opting
        # out treats +inf (zero-variance) as maximally valid while NaN stays
        # invalid because NaN comparisons are False in both branches.
        assert mask_from_ivar(ivar).tolist() == [False, True, False, False, False]
        assert mask_from_ivar(ivar, require_finite=False).tolist() == [
            False,
            True,
            False,
            False,
            True,
        ]
        # min_ivar is inclusive-invalid: ivar == min_ivar is not enough.
        assert mask_from_ivar(torch.tensor([0.5, 1.0, 1.5]), min_ivar=1.0).tolist() == [
            False,
            False,
            True,
        ]


class TestApplyMaskPreservesValues:
    def test_wide_integers_are_not_rounded(self) -> None:
        # int32/int64 counts above 2**24 cannot ride in float32: promotion for
        # the fill must not round the *valid* pixels.
        x = torch.tensor([2**24 + 1, 2**31 - 1], dtype=torch.int32)
        out = apply_mask(x, torch.ones(2, dtype=torch.bool))
        assert out.dtype.is_floating_point
        assert out.tolist() == [float(2**24 + 1), float(2**31 - 1)]
        x64 = torch.tensor([2**40 + 1], dtype=torch.int64)
        out64 = apply_mask(x64, torch.ones(1, dtype=torch.bool))
        assert out64.tolist() == [float(2**40 + 1)]

    def test_small_integers_keep_float32(self) -> None:
        x = torch.tensor([1, 2, 3], dtype=torch.int16)
        out = apply_mask(x, torch.tensor([True, False, True]))
        assert out.dtype == torch.float32

    def test_nan_fill_on_float_preserves_dtype(self) -> None:
        x = torch.tensor([1.5, 2.5], dtype=torch.float64)
        out = apply_mask(x, torch.tensor([True, False]))
        assert out.dtype == torch.float64
        assert out[0] == 1.5 and torch.isnan(out[1])
