"""Contracts for the int16 quantizer and the auto-mmap policy.

Round 2, unit 2 part 5 -- ``_io_engine/quantize.py`` and
``_io_engine/image_meta.py``.

* A percentile window narrower than the distribution's spread collapses to a
  point; that is not constant data, and taking the constant-data branch
  flattened the whole array onto a single code while reporting
  ``n_clipped == 0``. Reachable from ``write(..., quantize="robust")``.
* ``keep_zero`` flattened every non-positive sample onto code 0 but counted
  only the non-finite ones in ``n_clipped``.
* ``should_use_cold_nommap`` stat'ed the CFITSIO extended-syntax path whole,
  so ``mef.fits[1]`` and ``mef.fits`` resolved opposite policies for the same
  image.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits
from torchfits._io_engine.caches import auto_mmap_cache, cold_nommap_cache
from torchfits._io_engine.caches import image_meta_cache
from torchfits._io_engine.image_meta import resolve_image_mmap, should_use_cold_nommap
from torchfits._io_engine.quantize import (
    BLANK_CODE,
    dequantize_int16,
    quantize_int16_robust,
)

_MIB = 1 << 20


def _sparse_image(
    side: int = 512, n_sources: int = 200, lo: float = 1.0
) -> torch.Tensor:
    """A masked image: a flat background with a handful of bright sources.

    The canonical reason ``quantize=`` exists -- "the value distribution is
    skewed" -- and the shape that triggered the collapse.
    """
    image = torch.zeros(side, side, dtype=torch.float32)
    n = side * side
    image.view(-1)[torch.linspace(0, n - 1, n_sources).long()] = torch.linspace(
        lo, 500.0, n_sources
    )
    return image


def _negative_sparse_image(side: int = 512, n_sources: int = 200) -> torch.Tensor:
    """The same masked shape, with every source below a zero background."""
    image = torch.zeros(side, side, dtype=torch.float32)
    n = side * side
    image.view(-1)[torch.linspace(0, n - 1, n_sources).long()] = torch.linspace(
        -500.0, -1.0, n_sources
    )
    return image


def _clear_policy_caches() -> None:
    image_meta_cache.clear()
    cold_nommap_cache.clear()
    auto_mmap_cache.clear()


def _round_trip(result, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    blank = BLANK_CODE if result.blank_code is not None else None
    return dequantize_int16(result.codes, result.scale, result.zero, blank=blank).to(
        dtype
    )


# --- r2-022: a collapsed percentile window is not constant data -----------


def test_sparse_image_keeps_its_sources():
    """200 sources on a 512x512 zero background must survive the pack."""
    image = _sparse_image()
    result = quantize_int16_robust(image)
    assert result.hi > result.lo, (
        f"percentile window collapsed to lo={result.lo} hi={result.hi}; the "
        "whole image would be flattened onto one code"
    )
    decoded = _round_trip(result)
    lsb = (result.hi - result.lo) / 65531.0
    assert float(decoded.max()) == pytest.approx(float(image.max()), abs=lsb)
    assert float(decoded.min()) == pytest.approx(float(image.min()), abs=lsb)
    assert int((result.codes != result.codes.flatten()[0]).sum()) > 100


@pytest.mark.parametrize("side, n_sources", [(256, 32), (512, 200), (1024, 524)])
def test_sparse_image_keeps_its_sources_at_every_size(side, n_sources):
    """The trigger is the *fraction*, not the array size or the striding."""
    image = _sparse_image(side=side, n_sources=n_sources)
    result = quantize_int16_robust(image)
    assert result.hi > result.lo
    assert float(_round_trip(result).max()) == pytest.approx(
        float(image.max()), abs=(result.hi - result.lo) / 65531.0
    )


def test_sparse_image_of_negative_values_keeps_its_extremes():
    """The reported ``lo`` must be the real minimum, not the background."""
    image = _negative_sparse_image()
    result = quantize_int16_robust(image)
    assert result.lo == pytest.approx(-500.0, rel=1e-6), (
        f"lo={result.lo} ignored the negative sources"
    )
    assert result.hi > result.lo
    assert float(_round_trip(result).min()) == pytest.approx(
        float(image.min()), abs=(result.hi - result.lo) / 65531.0
    )


def test_written_sparse_image_keeps_its_sources(tmp_path):
    """The user-visible path: write(..., quantize='robust') then read back."""
    image = _sparse_image()
    path = str(tmp_path / "sparse.fits")
    torchfits.write(path, image, quantize="robust", overwrite=True)
    # getheader, not hdul[0].header: astropy rewrites BITPIX to the *scaled*
    # type in a live header once .data is touched, so an HDU.header read after
    # the data reports -32 for a file that stores 16.
    assert fits.getheader(path)["BITPIX"] == 16, "quantize= should still pack to int16"
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    assert len(np.unique(data)) > 100, (
        f"the written file holds {len(np.unique(data))} distinct value(s); the "
        "sources were destroyed at write time"
    )
    assert float(data.max()) == pytest.approx(500.0, abs=0.01)
    # Same image without quantize: the loss is the quantizer's, not the writer's.
    plain = str(tmp_path / "plain.fits")
    torchfits.write(plain, image, overwrite=True)
    with fits.open(plain) as hdul:
        assert float(np.asarray(hdul[0].data, dtype=np.float64).max()) == 500.0


def test_constant_data_still_takes_the_degenerate_branch():
    """Non-vacuity: genuinely constant data must still collapse deliberately."""
    result = quantize_int16_robust(torch.full((1000,), 5.0))
    assert result.lo == result.hi == 5.0
    assert result.scale == 1.0 and result.zero == 5.0
    assert bool((result.codes == 0).all())
    decoded = _round_trip(result)
    assert torch.equal(decoded, torch.full((1000,), 5.0))


def test_bounds_are_untouched_above_the_percentile_window():
    """Non-vacuity: the fallback must not run when the window is informative.

    Pinned to ``torch.quantile`` on the full population, so any drift in the
    normal path shows up here rather than hiding behind the collapse fix.
    """
    values = torch.randn(5000, dtype=torch.float64)
    result = quantize_int16_robust(values)
    expected_lo = float(torch.quantile(values, 0.001))
    expected_hi = float(torch.quantile(values, 0.999))
    assert result.lo == pytest.approx(expected_lo, rel=1e-6)
    assert result.hi == pytest.approx(expected_hi, rel=1e-6)
    assert result.hi > result.lo


def test_quantile_boundary_is_unchanged_by_the_fallback():
    """One non-zero above the window keeps the sampled bound, not the maximum.

    263 non-zero pixels in 262144 is the first count that survives the
    0.1/99.9 window; its bound is the 99.9th percentile of the data (0.859),
    not the maximum (500.0). A fallback that ran unconditionally would show
    up here as 500.0.
    """
    image = _sparse_image(side=512, n_sources=263)
    result = quantize_int16_robust(image)
    assert result.hi == pytest.approx(0.859, rel=0.02)
    assert result.hi < 500.0


# --- r2-023: keep_zero must report what it flattened ----------------------


def test_keep_zero_reports_clipped_negatives():
    result = quantize_int16_robust(torch.tensor([-1.0, -2.0, -3.0]), keep_zero=True)
    assert bool((result.codes == 0).all()), "negatives are expected to become 0"
    assert result.n_clipped == 3, "three samples were overwritten, not zero"


def test_keep_zero_counts_negatives_and_blanks_separately():
    values = torch.tensor([-1.0, -2.0, float("nan")])
    result = quantize_int16_robust(values, keep_zero=True)
    assert result.n_clipped == 3, "two negatives plus one non-finite sample"
    assert result.blank_code == BLANK_CODE
    assert int((result.codes == BLANK_CODE).sum()) == 1


def test_keep_zero_reports_no_clips_when_nothing_is_flattened():
    """Non-vacuity: the count must not simply be 'always positive'."""
    result = quantize_int16_robust(torch.tensor([0.0, 0.0, 0.0]), keep_zero=True)
    assert result.n_clipped == 0
    assert result.blank_code is None


def test_keep_zero_reports_only_the_negatives():
    """One added negative must add exactly one clip.

    Expressed as a difference against the same data without the negative: the
    percentile window clips the top of any tiny array, so an absolute count
    would be testing the wrong thing.
    """
    with_negative = quantize_int16_robust(
        torch.tensor([-4.0, 0.0, 1.0, 2.0, 3.0]), keep_zero=True
    )
    without = quantize_int16_robust(torch.tensor([0.0, 1.0, 2.0, 3.0]), keep_zero=True)
    assert with_negative.n_clipped == without.n_clipped + 1


# --- r2-024: the mmap policy is a property of the image, not of the path --


def _write_large_image(path: str) -> str:
    data = np.zeros((2048, 2048), dtype=np.int16)
    fits.HDUList(
        [
            fits.PrimaryHDU(data=np.zeros((8, 8), dtype=np.int16)),
            fits.ImageHDU(data=data, name="SCI"),
        ]
    ).writeto(path, overwrite=True)
    return path


def test_auto_mmap_policy_ignores_a_cfitsio_filter(tmp_path):
    """``mef.fits[1]`` and ``mef.fits`` name the same image, so same policy."""
    path = _write_large_image(str(tmp_path / "mef.fits"))
    assert os.path.getsize(path) >= _MIB
    _clear_policy_caches()
    plain = resolve_image_mmap(path, 1, "auto", 10)
    _clear_policy_caches()
    filtered = resolve_image_mmap(f"{path}[1]", 0, "auto", 10)
    assert plain == filtered, (
        f"same image resolved mmap={plain!r} unfiltered and {filtered!r} with a "
        "CFITSIO HDU filter"
    )


def test_cold_nommap_gate_ignores_a_cfitsio_filter(tmp_path):
    path = _write_large_image(str(tmp_path / "mef.fits"))
    _clear_policy_caches()
    plain = should_use_cold_nommap(path, 1, 10, True)
    _clear_policy_caches()
    filtered = should_use_cold_nommap(f"{path}[1]", 0, 10, True)
    assert plain is filtered
    assert plain is True, "a large int16 image is the case the gate exists for"


def test_small_image_still_prefers_mmap_on_both_spellings(tmp_path):
    """Non-vacuity: the fix must not turn the size gate off entirely."""
    small = str(tmp_path / "small.fits")
    fits.HDUList([fits.PrimaryHDU(data=np.zeros((8, 8), dtype=np.int32))]).writeto(
        small, overwrite=True
    )
    _clear_policy_caches()
    assert resolve_image_mmap(small, 0, "auto", 10) is True
    _clear_policy_caches()
    assert resolve_image_mmap(f"{small}[0]", 0, "auto", 10) is True


def test_pixels_are_identical_under_both_spellings(tmp_path):
    path = _write_large_image(str(tmp_path / "mef.fits"))
    _clear_policy_caches()
    a = torchfits.read_tensor(path, hdu=1, mmap=False)
    _clear_policy_caches()
    b = torchfits.read_tensor(f"{path}[1]", hdu=0, mmap=False)
    assert torch.equal(a, b)
