"""Native-infrastructure regressions that are observable through torchfits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits


@pytest.mark.parametrize(
    "dtype,bits,pad_widths",
    [
        (np.float32, np.uint32, (1, 3)),
        (np.float64, np.uint64, (1, 3, 5, 7)),
    ],
)
def test_mmap_float_byteswap_preserves_ieee_bits_at_odd_offsets(
    tmp_path: Path,
    dtype: type[np.floating],
    bits: type[np.integer],
    pad_widths: tuple[int, ...],
) -> None:
    """Every SIMD-tail length preserves the exact big-endian bit pattern.

    A one-byte leading column deliberately makes the floating-point payload
    unaligned. The cases cover empty/short input, every SSSE3/NEON block size,
    exact block multiples, and non-multiples. The payload pins NaN payloads,
    sign bits, infinities, signed zero, and both ends of the denormal range.
    """
    if bits is np.uint32:
        patterns = np.array(
            [
                0x7FC00001,
                0xFFC54321,
                0x7F800000,
                0xFF800000,
                0x80000000,
                0x00000001,
                0x007FFFFF,
                0x3F800000,
            ],
            dtype=np.uint32,
        )
    else:
        patterns = np.array(
            [
                0x7FF8000000000001,
                0xFFF5432100000001,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x8000000000000000,
                0x0000000000000001,
                0x000FFFFFFFFFFFFF,
                0x3FF0000000000000,
            ],
            dtype=np.uint64,
        )
    max_rows = max(17, len(patterns))

    for pad_width in pad_widths:
        for count in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17):
            values = np.resize(patterns, max_rows).view(dtype)[:count]
            padding = np.zeros(count, dtype=np.uint8)
            path = tmp_path / f"odd_{np.dtype(dtype).itemsize}_{pad_width}_{count}.fits"
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="PAD", format=f"{pad_width}B", array=padding),
                    fits.Column(
                        name="VALUE",
                        format="E" if bits is np.uint32 else "D",
                        array=values,
                    ),
                ]
            ).writeto(path, overwrite=True)

            result = torchfits.table.read_torch(path.as_posix(), hdu=1, mmap=True)[
                "VALUE"
            ]
            expected = values.view(bits)
            actual = result.numpy().view(bits)
            np.testing.assert_array_equal(
                actual,
                expected,
                err_msg=(
                    f"{np.dtype(dtype).name} count={count} pad={pad_width}: "
                    "byteswap changed an IEEE bit pattern"
                ),
            )


@pytest.mark.parametrize("dtype", (np.uint16, np.uint32))
def test_mmap_unsigned_byteswap_preserves_all_offset_values(
    tmp_path: Path, dtype: type[np.integer]
) -> None:
    """Pseudo-unsigned SIMD bodies and tails keep every raw 16/32-bit value."""
    width = np.dtype(dtype).itemsize
    values = np.array([0, 1, 2, 3, 255, 256, 32767, 32768, 65535], dtype=dtype)
    for count in range(1, len(values) + 1):
        path = tmp_path / f"unsigned_{width}_{count}.fits"
        torchfits.write(
            path.as_posix(),
            {"VALUE": torch.from_numpy(values[:count].copy())},
            overwrite=True,
        )
        result = torchfits.table.read_torch(path.as_posix(), hdu=1, mmap=True)["VALUE"]
        np.testing.assert_array_equal(result.numpy(), values[:count])
