"""Bitwise parity: ``read_full_numpy``/``read_full_numpy_cached`` == ``read_tensor``.

The numpy-returning C++ entry points must agree BITWISE with the canonical
tensor read (``torchfits.read_tensor(...).numpy()``): same shape, same dtype,
identical bytes. This pins the house image conventions end to end:

- unsigned packs stay integers (BZERO=32768 -> uint16, BZERO=2^31 -> uint32,
  BZERO=-128 -> int8), BLANK or not (``blank-nulval``: unsigned/signed-byte
  conventions keep integer dtypes even with BLANK);
- BLANK on a non-convention integer image promotes to float32 with NaN at the
  blank pixels (``blank-nulval``);
- native IEEE float/double survive verbatim (Inf / signed zero preserved —
  ``blank-nulval``);
- NAXIS=0 keeps the BITPIX-keyed empty dtype like every sibling reader.

Also pins the image scale accumulation dtype as float32 (A-06) so the 2.0
float64 switch must be deliberate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as astropy_fits

import torchfits
import torchfits._cpp as cpp


def _write_raw_header(path: Path, cards: list[str]) -> None:
    """Minimal handcrafted FITS header (2880-byte blocks, no data unit)."""
    block = "".join(card.ljust(80) for card in cards)
    block += "END".ljust(80)
    pad = (-len(block)) % 2880
    path.write_bytes((block + " " * pad).encode("ascii"))


def _write_raw_image(path: Path, cards: list[str], data: np.ndarray) -> None:
    """Handcrafted FITS image: exact control of header cards and raw values.

    astropy's ``PrimaryHDU(data, header={"BSCALE": ..., "BZERO": ...})``
    silently drops the scaling cards (probe recorded in the R9 findings), so
    genuinely-scaled fixtures must be written byte-wise. Data is stored
    big-endian per the FITS standard.
    """
    block = "".join(card.ljust(80) for card in cards)
    block += "END".ljust(80)
    pad = (-len(block)) % 2880
    blob = (block + " " * pad).encode("ascii")
    raw = data.astype(data.dtype.newbyteorder(">")).tobytes()
    blob += raw + b"\x00" * ((-len(raw)) % 2880)
    path.write_bytes(blob)


@pytest.fixture(scope="module")
def parity_fixtures(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("numpy_parity")
    rng = np.random.default_rng(20260923)
    files: dict[str, str] = {}

    def write(name: str, payload) -> None:
        path = root / f"{name}.fits"
        payload.writeto(path, overwrite=True)
        files[name] = str(path)

    u16 = rng.integers(0, 60000, size=(37, 53)).astype(np.uint16)
    write("uint16_pack", astropy_fits.PrimaryHDU(u16))

    u32 = rng.integers(0, 4_000_000_000, size=(5, 7, 3)).astype(np.uint32)
    write("uint32_pack", astropy_fits.PrimaryHDU(u32))

    write("int8_signed", astropy_fits.PrimaryHDU(rng.integers(-128, 128, (11, 23)).astype(np.int8)))

    write("plain_int16", astropy_fits.PrimaryHDU(rng.integers(0, 3000, (9, 4)).astype(np.int16)))

    scaled = rng.integers(-3000, 3000, size=(12, 10)).astype(np.int16)
    p = root / "scaled_int16.fits"
    _write_raw_image(
        p,
        [
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                   10",
            "NAXIS2  =                   12",
            "BSCALE  =                  0.1",
            "BZERO   =                32768.0",
        ],
        scaled,
    )
    files["scaled_int16"] = str(p)

    blank = rng.integers(0, 3000, (8, 6)).astype(np.int16)
    blank[2, 3] = -32768
    blank[5, 1] = -32768
    write(
        "blank_identity",
        astropy_fits.PrimaryHDU(blank, header=astropy_fits.Header({"BLANK": -32768})),
    )

    # Unsigned convention with BLANK: dtype stays uint16 (blank-nulval) and the
    # blank pixels decode as raw+32768 — i.e. back to their physical values.
    u16_blank = rng.integers(0, 60000, size=(6, 5)).astype(np.uint16)
    u16_blank[1, 1] = 32768  # raw 0 = the BLANK marker
    write(
        "blank_uint16",
        astropy_fits.PrimaryHDU(u16_blank, header=astropy_fits.Header({"BLANK": 0})),
    )

    f32 = rng.normal(size=(7, 5)).astype(np.float32)
    f32[0, 0] = np.inf
    f32[0, 1] = -np.inf
    f32[0, 2] = -0.0
    f32[0, 3] = 0.0
    write("float32_ieee", astropy_fits.PrimaryHDU(f32))

    write("float64", astropy_fits.PrimaryHDU(rng.normal(size=(6, 4)).astype(np.float64)))

    # NAXIS=0 with BITPIX=16: the empty result must keep the BITPIX-keyed
    # dtype (empty int16), like read_tensor and every sibling reader.
    p = root / "naxis0_bitpix16.fits"
    _write_raw_header(
        p,
        [
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    0",
            "EXTEND  =                    T",
        ],
    )
    files["naxis0_bitpix16"] = str(p)

    # Zero-pixel image (NAXIS1=0): shape (5, 0), dtype preserved.
    p = root / "zero_pixel.fits"
    _write_raw_header(
        p,
        [
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                    0",
            "NAXIS2  =                    5",
            "EXTEND  =                    T",
        ],
    )
    files["zero_pixel"] = str(p)

    return files


_EXPECTED_DTYPE = {
    "uint16_pack": np.uint16,
    "uint32_pack": np.uint32,
    "int8_signed": np.int8,
    "plain_int16": np.int16,
    "scaled_int16": np.float32,
    "blank_identity": np.float32,
    "blank_uint16": np.uint16,
    "float32_ieee": np.float32,
    "float64": np.float64,
    "naxis0_bitpix16": np.int16,
    "zero_pixel": np.int16,
}


def _assert_bitwise(got: np.ndarray, expected: np.ndarray, label: str) -> None:
    got = np.asarray(got)
    expected = np.asarray(expected)
    assert got.shape == expected.shape, f"{label}: shape {got.shape} != {expected.shape}"
    assert got.dtype == expected.dtype, f"{label}: dtype {got.dtype} != {expected.dtype}"
    # Byte comparison: exact for NaN payloads and signed zero too.
    got_bytes = np.ascontiguousarray(got).view(np.uint8)
    exp_bytes = np.ascontiguousarray(expected).view(np.uint8)
    np.testing.assert_array_equal(got_bytes, exp_bytes, err_msg=f"{label}: bytes differ")


@pytest.mark.parametrize("name", sorted(_EXPECTED_DTYPE))
@pytest.mark.parametrize("mmap", [False, True])
def test_read_full_numpy_matches_read_tensor(parity_fixtures, name: str, mmap: bool) -> None:
    path = parity_fixtures[name]
    expected = torchfits.read_tensor(path, hdu=0, mmap=mmap).numpy()

    got = cpp.read_full_numpy(path, 0, mmap)
    _assert_bitwise(got, expected, f"read_full_numpy {name} mmap={mmap}")

    got_cached = cpp.read_full_numpy_cached(path, 0, mmap)
    _assert_bitwise(got_cached, expected, f"read_full_numpy_cached {name} mmap={mmap}")

    assert np.asarray(got).dtype == _EXPECTED_DTYPE[name], (
        f"{name}: house dtype {_EXPECTED_DTYPE[name]} != {np.asarray(got).dtype}"
    )


@pytest.mark.parametrize("name", sorted(_EXPECTED_DTYPE))
@pytest.mark.parametrize("mmap", [False, True])
def test_sibling_cpp_readers_agree(parity_fixtures, name: str, mmap: bool) -> None:
    """Adjacent _cpp full-image readers must agree on shape/dtype/values too."""
    path = parity_fixtures[name]
    expected = torchfits.read_tensor(path, hdu=0, mmap=mmap).numpy()

    _assert_bitwise(
        cpp.read_full_nocache(path, 0, mmap).numpy(),
        expected,
        f"read_full_nocache {name} mmap={mmap}",
    )
    _assert_bitwise(
        cpp.read_full_unmapped(path, 0).numpy(),
        expected,
        f"read_full_unmapped {name}",
    )


def test_blank_identity_promotes_to_nan_float32(parity_fixtures) -> None:
    """blank-nulval: BLANK on a non-convention integer image -> float32 NaN."""
    got = np.asarray(cpp.read_full_numpy(parity_fixtures["blank_identity"], 0, True))
    assert got.dtype == np.float32
    assert np.isnan(got[2, 3]) and np.isnan(got[5, 1])
    assert not np.isnan(got).sum() == got.size  # only the blank pixels are NaN


def test_float32_ieee_inf_and_signed_zero_survive(parity_fixtures) -> None:
    """blank-nulval: native IEEE keeps Inf and -0.0 (fnan must not apply)."""
    got = np.asarray(cpp.read_full_numpy(parity_fixtures["float32_ieee"], 0, False))
    assert got[0, 0] == np.inf and got[0, 1] == -np.inf
    assert np.signbit(got[0, 2]) and not np.signbit(got[0, 3])


def test_blank_uint16_keeps_integer_dtype_and_values(parity_fixtures) -> None:
    """blank-nulval: unsigned convention + BLANK keeps uint16 and the physical
    values (raw+32768), including at the BLANK-marked pixel."""
    got = np.asarray(cpp.read_full_numpy(parity_fixtures["blank_uint16"], 0, True))
    assert got.dtype == np.uint16
    assert got[1, 1] == 32768


def test_compressed_float32_ieee_bits_survive(tmp_path) -> None:
    """blank-nulval: native IEEE float storage must read WITHOUT a NaN nulval
    (fnan would destroy Inf / signed zero / subnormals) — including CompImage
    tiles: the nulval decision keys on STORAGE bitpix, not on compression."""
    rng = np.random.default_rng(11)
    f32 = rng.normal(size=(16, 16)).astype(np.float32)
    f32[0, 0] = np.inf
    f32[0, 1] = -np.inf
    f32[0, 2] = -0.0
    f32[0, 3] = 0.0
    f32[0, 4] = np.float32(1e-45)  # subnormal
    path = tmp_path / "compressed_float32.fits"
    astropy_fits.HDUList(
        [
            astropy_fits.PrimaryHDU(),
            astropy_fits.CompImageHDU(f32, compression_type="GZIP_1"),
        ]
    ).writeto(path)

    expected = torchfits.read_tensor(str(path), hdu=1, mmap=True).numpy()
    # Fixture sanity: the stored tiles really carry the special bit patterns.
    assert np.isinf(expected[0, 0]) and expected[0, 0] > 0
    assert np.signbit(expected[0, 2]) and not np.signbit(expected[0, 3])

    _assert_bitwise(
        np.asarray(cpp.read_full_numpy(str(path), 1, True)),
        expected,
        "read_full_numpy compressed float32",
    )
    _assert_bitwise(
        np.asarray(cpp.read_full_unmapped(str(path), 1)),
        expected,
        "read_full_unmapped compressed float32",
    )


def test_image_scale_accumulation_is_float32_pinned(tmp_path):
    """A-06 pin: image scale accumulation is float32 in 1.2.0 on every read
    surface, with the scale actually applied. The 2.0 switch to float64 must
    be deliberate (backlog entry); this test failing means the dtype moved
    without the plan's re-baselining."""
    raw = np.array([[0, 1000, -1000]], dtype=np.int16)
    path = tmp_path / "scale_pin.fits"
    _write_raw_image(
        path,
        [
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                    3",
            "NAXIS2  =                    1",
            "BSCALE  =                  0.1",
            "BZERO   =                32768.0",
        ],
        raw,
    )
    expected = np.array([[32768.0, 32868.0, 32668.0]], dtype=np.float32)

    got = np.asarray(cpp.read_full_numpy(str(path), 0, True))
    assert got.dtype == np.float32 and got.dtype != np.float64
    np.testing.assert_array_equal(got, expected)

    assert torchfits.read_tensor(str(path), hdu=0, mmap=True).dtype == torch_float32()
    np.testing.assert_array_equal(
        torchfits.read_tensor(str(path), hdu=0, mmap=True).numpy(), expected
    )
    got_cached = np.asarray(cpp.read_full_numpy_cached(str(path), 0, False))
    assert got_cached.dtype == np.float32
    np.testing.assert_array_equal(got_cached, expected)
    scaled_cpu = cpp.read_full_scaled_cpu(str(path), 0, True)
    assert scaled_cpu.dtype == torch_float32()
    assert scaled_cpu.dtype != torch_float64()
    np.testing.assert_array_equal(scaled_cpu.numpy(), expected)


def torch_float32():
    import torch

    return torch.float32


def torch_float64():
    import torch

    return torch.float64
