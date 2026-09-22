# R2 slice C — `transforms/mask.py`, `transforms/rgb.py`, `transforms/stretch.py`

Round R2 (directory `src/torchfits/transforms/`), agent `R2-MaskRGB`, ID prefix `r2c`.
All fixes land in the worktree (no commits). Tests added first, run against unfixed
code, pasted below; then the fix; then the passing run. Test ownership: `tests/test_rgb.py`,
`tests/test_transforms_e2e.py` (extended), `tests/test_maskrgb_masking.py`,
`tests/test_maskrgb_stretch.py` (new). Shared oracle `tests/transforms_reference.py`
untouched (it covers SigmaClip/weighted-quantile/IRAF-zscale only — no mask/RGB/stretch
sections exist for this slice; parity pinned directly against astropy instead).

Targeted runs used throughout: `pixi run pytest tests/<file> -q`.

## Findings

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|---|---|---|---|---|---|---|
| r2c-01 | MAJOR | 1 silent data loss (rounding of *valid* pixels) | `mask.py:apply_mask` | Integer inputs were promoted to **float32 regardless of width**, silently rounding unmasked pixels above 2^24. Repro: `apply_mask(torch.tensor([2**24+1, 2**31-1], dtype=torch.int32), torch.ones(2, dtype=torch.bool))` → `[16777216.0, 2147483648.0]` (both wrong; int32 max even rounds *up*). Before (pytest): `FAILED tests/test_maskrgb_masking.py::TestApplyMaskPreservesValues::test_wide_integers_are_not_rounded - assert [16777216.0, 2147483648.0] == [16777217.0, 2147483647.0]` | Exact promotion: ≥32-bit ints → float64 (float32 for ≤16-bit, and everywhere on MPS which has no float64); docstring states the policy. After: `tests/test_maskrgb_masking.py` 18 passed (incl. `test_wide_integers_are_not_rounded`, `test_small_integers_keep_float32`). Doc update owed at R15: `docs/api-transforms.md:531` "(promotes integers for a NaN fill)" → the exact-promotion policy. | fixed |
| r2c-02 | MAJOR | 5 silent no-op (opposite-of-intent result) | `mask.py:mask_from_dq` | `require_good=True` without `good_bits=` silently did nothing; with `bad_bits=None` it returned **all-True** (every pixel valid — the exact opposite of "require science-good"). Repro: `mask_from_dq(torch.tensor([[3, 0]]), require_good=True)` → `[True, True]`. Before: `FAILED tests/test_maskrgb_masking.py::TestDqTypedErrors::test_require_good_without_good_bits_is_rejected - Failed: DID NOT RAISE ValueError` | Raise `ValueError("require_good=True needs good_bits=…")`. After: passing. Also made the `mask_from_dq` doctest examples self-contained (defined `dq`). Doc note owed at R15: mention the guard on `docs/api-transforms.md:527` row. | fixed |
| r2c-03 | MAJOR | 2 overflow/truncation (silent wrap) | `stretch.py:SqrtStretch.inverse` | Squared integer input **in its storage dtype**: `inverse(int16([300, 200]))` → `int16([24464, -25536])`; `inverse(int32([50000]))` → `-1794967296`. Also returned integer dtype, violating the documented contract ("All stretches promote integer tensors to float32 and return float", `docs/api-transforms.md:139-141`). Before: `FAILED …::test_sqrt_inverse_of_integer_input_returns_float`, `…::test_sqrt_inverse_wide_integers_do_not_wrap`, `…::test_sqrt_round_trip_integer_input - RuntimeError: Float did not match Double`, `…::test_forward_and_inverse_agree_on_int_promotion[SqrtStretch]` | `inverse` now computes `square(_upcast_for_precision(x)).to(_stretch_dtype(x))`; `forward` routed through the same helpers so integer/half-dtype promotion is explicit (float16 output shifts ≤1 ulp — `test_float16_sqrt_is_not_worse_than_half_ulp_of_float32` pins bitwise equality with the float32-then-round path). After: `tests/test_maskrgb_stretch.py` 17 passed. | fixed |
| r2c-04 | MAJOR | 1 silent data loss (type fallthrough) | `rgb.py:lupton_rgb`, `rgb.py:rgb` (`dtype=`) | Integer `dtype=` silently truncated the [0, 1] floats to an **all-zero image**: `lupton_rgb(rand, rand, rand, dtype=torch.uint8).unique()` → `[0]`. Before: `FAILED tests/test_rgb.py::test_integer_dtype_request_raises_not_black_image - Failed: DID NOT RAISE TypeError` | `_require_float_dtype` (module-private) raises `TypeError("dtype must be a floating dtype …")` up-front in both entry points; docstrings updated. After: passing. | fixed |
| r2c-05 | MAJOR | 7 empty-input (empty-result dtype preservation; acceptance item) | `rgb.py:rgb` | The `mixed.numel() == 0` early return **ignored `dtype=`**: `rgb(zeros(0,5)×3, dtype=torch.float16)` → `float64` while non-empty returned `float16`. Before: `FAILED tests/test_rgb.py::test_rgb_dtype_is_exact_on_empty_and_full_images - assert torch.float64 == torch.float16` | Empty path now casts like the full path (`return out if dtype is None else out.to(dtype)`). After: passing. | fixed |
| r2c-06 | MAJOR | UB on write + invalid output | `rgb.py:write_rgb_image` | (a) NaN pixels hit a float→uint8 cast: **undefined behavior** per torch casting semantics (observed value 0 on CPU, not guaranteed). (b) Zero-size input silently wrote a 65-byte **invalid PNG** (IHDR width/height must be ≥ 1). Before: `FAILED tests/test_rgb.py::test_write_rgb_image_rejects_empty_images - Failed: DID NOT RAISE ValueError` | `nan_to_num(…, nan=0.0)` after the clamp (NaN → black, deterministic); `ValueError("cannot write an empty HxW image as PNG")` for H or W == 0. NaN test is a contract pin (UB is unobservable on CPU — code proof: the cast is gone, `bytes(test_rgb.py::test_write_rgb_image_nan_pixels_are_black)` asserts `[0, 0, 0]`). After: both passing. | fixed |
| r2c-07 | MINOR | 8 docs faithfulness (hidden dependency) | `rgb.py:write_rgb_image` | Docstring claims "stdlib only; no Pillow / NumPy import" but the byte export called `scanlines.numpy().tobytes()` — fails at runtime where numpy is not importable (torch does not require it). | `bytes(scanlines.untyped_storage())` — same memcpy cost, no numpy anywhere; claim is now literally true. Covered by the byte-identical round-trip pins (`test_write_rgb_image_roundtrip`, `test_write_rgb_image_produces_valid_png` 1 passed). | fixed |
| r2c-08 | MINOR | 8/1 parity-claim gap | `rgb.py:lupton_rgb` | Missing astropy `Qmax = 1e10` cap (the near-zero-Q floor WAS mirrored, its sibling clamp was not): at `Q > 1e10` output diverges from the claimed astropy `LuptonAsinhStretch` pipeline. Measured: `Q=1e12` max abs diff **0.039** on the parity fixture vs ≤ 5.6e-16 otherwise. Before: `FAILED tests/test_rgb.py::test_lupton_rgb_float_parity_with_astropy - AssertionError: (1000000000000.0, 0.5, 0.03895…)` | Mirror `LuptonAsinhStretch.__init__`: `elif q > _LUPTON_Q_MAX (1e10): q = 1e10`. Float-precision parity now pinned across `(Q, stretch, minimum)` incl. Q=0 (floor) and Q=1e12 (cap) at `atol=1e-12` vs `make_lupton_rgb(..., output_dtype=np.float64)`. After: passing. | fixed |
| r2c-09 | MINOR | 5 error contract | `mask.py:_bit_mask` | Bit positions ≥ 64 raised bare `OverflowError: int too big to convert`; non-int scalars raised misleading `TypeError: 'float' object is not iterable`. Before: `FAILED …::test_position_beyond_63_raises_value_error - OverflowError: int too big to convert`; `FAILED …::test_non_int_position_raises_type_error - Expected regex 'bit position' / Actual message "'float' object is not iterable"` | Typed `ValueError("DQ bit positions must be in 0..63 …")` and `TypeError("DQ bit positions must be ints or an iterable of ints …")`. Position 63 itself works (two's-complement wrap of `2**63` is the correct bit pattern) and is pinned by `test_top_bit_position_63_supported`. After: passing. | fixed |
| r2c-10 | MAJOR | 4 concurrency (shared mutable state) | `base.py:FITSTransform._warn_ivar_not_propagated` (owner **R2-StateCore**; called by all three stretch classes) | `_warn_ivar_not_propagated` wrote `_ivar_warned` into the instance `__dict__` during `forward`/`__call__` — a violation of the round invariant "`__call__` must NOT mutate instance state" for one instance shared across `-J` worker threads. | Fixed in `base.py` by R2-StateCore (module-level `weakref.WeakSet`, once-per-instance semantics preserved) as part of their slice; the defect lived outside my file set and was already fixed in the worktree when my pin ran, so the failing-first/after evidence is owned by R2-StateCore's record. My cross-slice pins: `tests/test_transforms_e2e.py::TestTransformInstanceSafety` (`vars(t)` byte-unchanged across `forward`/`__call__`/`inverse` for all three stretches + 8-thread shared-instance run asserting outputs `torch.equal` to the single-thread result) — passing. | fixed (cross-slice) |
| r2c-11 | MAJOR | 1 silent mis-decode | `mask.py:mask_from_dq` | Complex-typed input passed the float guard and hit `dq.to(torch.int64)`, which **silently discards the imaginary part** (torch `UserWarning`) and mis-decodes the DQ — no error at all. Before-behavior proof (unguarded cast, verbatim): `torch.zeros(3, dtype=torch.complex64).to(torch.int64)` → succeeds with `UserWarning: Casting complex values to real discards the imaginary part`. | Extended the typed guard to `dq.dtype.is_complex` → `TypeError("DQ extensions must be integer-typed …")`. Pinned by `test_complex_extension_rejected_typed`. After: passing. | fixed |

## Evidence excerpts

Failing-first run 1 (`tests/test_maskrgb_masking.py tests/test_maskrgb_stretch.py`, unfixed code):

```
FAILED tests/test_maskrgb_masking.py::TestDqTypedErrors::test_position_beyond_63_raises_value_error - OverflowError: int too big to convert
FAILED tests/test_maskrgb_masking.py::TestDqTypedErrors::test_non_int_position_raises_type_error - AssertionError: Regex pattern did not match.
  Expected regex: 'bit position'
  Actual message: "'float' object is not iterable"
FAILED tests/test_maskrgb_masking.py::TestDqTypedErrors::test_require_good_without_good_bits_is_rejected - Failed: DID NOT RAISE ValueError
FAILED tests/test_maskrgb_masking.py::TestApplyMaskPreservesValues::test_wide_integers_are_not_rounded - assert [16777216.0, 2147483648.0] == [16777217.0, 2147483647.0]
FAILED tests/test_maskrgb_stretch.py::TestInverseDtypeContract::test_sqrt_inverse_of_integer_input_returns_float - assert False
 +  where False = torch.int16.is_floating_point
 +    where torch.int16 = tensor([ 24464, -25536], dtype=torch.int16).dtype
FAILED tests/test_maskrgb_stretch.py::TestInverseDtypeContract::test_sqrt_inverse_wide_integers_do_not_wrap - assert False
 +  where False = torch.int32.is_floating_point
 +    where torch.int32 = tensor([-1794967296, -1794967296], dtype=torch.int32).dtype
FAILED tests/test_maskrgb_stretch.py::TestInverseDtypeContract::test_sqrt_round_trip_integer_input - RuntimeError: Float did not match Double
FAILED tests/test_maskrgb_stretch.py::TestInverseDtypeContract::test_forward_and_inverse_agree_on_int_promotion[SqrtStretch] - assert False
11 failed, 23 passed in 1.35s
```

(3 of the 11 were wrong test expectations of mine — signed-DQ bit pattern, broadcast
shape, mixed-dtype `allclose` — corrected to the proven-correct library behavior before
any code fix; the remaining 8 map to r2c-01/02/03/09.)

Failing-first run 2 (`tests/test_rgb.py tests/test_transforms_e2e.py`, unfixed code):

```
FAILED tests/test_rgb.py::test_rgb_dtype_is_exact_on_empty_and_full_images - assert torch.float64 == torch.float16
 +  where torch.float64 = tensor([], size=(0, 5, 3), dtype=torch.float64).dtype
FAILED tests/test_rgb.py::test_integer_dtype_request_raises_not_black_image - Failed: DID NOT RAISE TypeError
FAILED tests/test_rgb.py::test_lupton_rgb_float_parity_with_astropy - AssertionError: (1000000000000.0, 0.5, np.float64(0.03895153038017746))
FAILED tests/test_rgb.py::test_write_rgb_image_rejects_empty_images - Failed: DID NOT RAISE ValueError
4 failed, 29 passed, 3 warnings in 2.05s
```

Passing runs after the fixes:

```
$ pixi run pytest tests/test_maskrgb_masking.py tests/test_maskrgb_stretch.py tests/test_rgb.py tests/test_transforms_e2e.py -q
68 passed, 3 warnings in 2.02s

$ pixi run pytest tests/test_transforms_state.py \
    tests/test_cli.py::test_lupton_rgb_zero_size_input \
    tests/test_cli.py::test_lupton_rgb_preserves_midtones_with_bright_star \
    tests/test_cli.py::test_lupton_rgb_astropy_parity \
    tests/test_stream_table_and_cache.py::test_write_rgb_image_produces_valid_png -q
233 passed, 6 warnings in 2.43s
```

(Second run = guard of the shared pins this slice touches: the sibling-owned
`test_transforms_state.py` stretch/mask/ivar contract, `test_cli.py`'s Lupton parity and
midtone pins, and the PNG-validity pin.)

## Conventions verified (pinned, no defect)

- **True = valid, end to end.** `mask_from_dq` decodes 0-based bit *positions* per FITS
  DQ flag tables (value `2**bit`: "bit 11" ≡ 2048); `bad_bits=None` = any non-zero is
  suspect; uint32 top bit, negative signed-DQ values (sign bit as flag) and `int64`
  position 63 decode correctly (two's-complement bit ops). `mask_from_ivar`: `ivar >
  min_ivar` strictly (0 = no data invalid), NaN always invalid, `require_finite=True`
  default rejects `+inf`. `combine_masks` broadcasts per torch semantics and is
  all-`None`-safe; `apply_mask(x, None)` is identity. Empty/all-invalid inputs pinned.
- **Delta-method IVAR.** `ivar / (df/dx)^2` verified against central finite differences
  in float32 and float64 for all three stretches (complements the existing
  `test_transforms_state.py::TestDeltaIvarAnalysis` finite-difference pins); SqrtStretch
  Poisson claim verified numerically: `ivar_x = 1/x` over `x = 10^0..10^6` gives
  `ivar_out = 4.0` to rtol 1e-10. Clamped regions (`x < 0` for log/sqrt) report
  `ivar = 0`; the non-differentiable kink at `x == 0` is pinned as `ivar = 0` (matches
  `base.delta_ivar`'s documented "clamped or collapsed → no first-order information"
  convention). `ArcsinhStretch` never clamps — ivar stays positive for all finite input.
- **Lupton parity.** `lupton_rgb` matches astropy's `make_lupton_rgb` /
  `LuptonAsinhStretch` pipeline to ≤ 5.6e-16 (float64 fixture incl. saturated star,
  negative pixels, non-zero minimum) across `(Q, stretch, minimum)` — verified against
  installed astropy source (`LuptonAsinhStretch.__init__`/`__call__`,
  `RGBImageMappingLupton`) and pinned at `atol=1e-12` after r2c-08.
- **Instance safety.** My three files carry no instance mutation on any call path
  (constructors only); the one inherited violation (r2c-10) is fixed in `base.py`.
  `tests/test_transforms_e2e.py::TestTransformInstanceSafety` pins both no-mutation and
  8-thread shared-instance equality (the `-J` worker pattern).

## Observations (no defect; not filed)

- `lupton_rgb` with NaN in one band: the NaN channel propagates NaN and the other
  channels drop to 0 (fac = 0 at non-positive/undefined intensity). Input is
  mathematically undefined there; `rgb()` upstream replaces non-finite with 0-fill
  (`nan_to_num`) and is pinned by `test_rgb_nan_inf_inputs_stay_finite_in_range`.
  astropy's float path yields all-NaN pixels instead — NaN-input parity not claimed.
- `combine_masks`/`apply_mask` silently broadcast (1, C) shapes over (H, C) inputs per
  torch semantics; pinned as intended (`test_combine_masks_broadcasts_and_none_passthrough`).
- `_as_band_stack` interprets a lone 3-D argument as (C, H, W); an (H, W, 3) input with
  H ≤ 7 would be misread as bands. Documented parameter shape ("one (H, W) image, one
  (C, H, W) cube"); ambiguity is inherent — no cheap disambiguation, left as documented.
- `LogStretch`/`SqrtStretch` `inverse` still cannot recover negatives (clamp documented
  in both docstrings and `docs/api-transforms.md:158-159`).

## Deferred

None. (Everything found landed; the r2c-10 fix landed cross-slice in `base.py` by its
owner with shared evidence.)

## Required doc updates (docs land at R15 — not edited here)

1. `docs/api-transforms.md:531` — `apply_mask` row: "(promotes integers for a NaN fill)"
   → exact promotion (float64 for ≥32-bit ints; float32 on MPS; float32 for ≤16-bit).
2. `docs/api-transforms.md:527` — `mask_from_dq` row: note `require_good=True` requires
   `good_bits=` (raises `ValueError` otherwise).
3. `docs/api-transforms.md:528` — `mask_from_ivar(ivar, min_ivar=0.0)` signature column is
   missing the `require_finite=True` kwarg (docs/api-sync).
4. `lupton_rgb`/`rgb` `dtype=` docs: must be a floating dtype, else `TypeError` (new
   guard, r2c-04); `lupton_rgb` Q is clamped to astropy's `[0.1 floor, 1e10 cap]`
   softening (r2c-08) — worth one clause where the Astropy-parity claim appears
   (`docs/examples-transforms.md:53`).
5. `write_rgb_image`: zero-size input now raises `ValueError` (r2c-06) and the
   "stdlib only" claim is now literally true (r2c-07).

## Per-file disposition

| file | depth | finding IDs | status |
|---|---|---|---|
| `src/torchfits/transforms/mask.py` | full (all 5 public fns + `_bit_mask`, docstrings, doctest examples) | r2c-01, r2c-02, r2c-09, r2c-11 | fixed |
| `src/torchfits/transforms/rgb.py` | full (`lupton_rgb`, `rgb`, `_as_band_stack`/`_scalar_stats`/`_equalize_bands`/`_mix_to_rgb`/`_stretch_for_target`/`_srgb_oetf`/`_apply_saturation`, `_SCARLET_MAPS` row sums, `write_rgb_image`, PNG chunker) | r2c-04, r2c-05, r2c-06, r2c-07, r2c-08 | fixed |
| `src/torchfits/transforms/stretch.py` | full (all 3 classes, forward+inverse, ivar delta paths, docstrings) | r2c-03 (r2c-10 cross-slice via `base.py`) | fixed |
