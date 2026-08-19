"""Compression algorithm matrix: every algorithm CFITSIO/torchfits supports,
verified byte-exact against independent oracles (astropy.io.fits and, when
available, the fpack CLI built from the same vendored CFITSIO source).

Scope and guarantees:

* Lossless integer algorithms (RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1 with the
  default zero scale factor, PLIO_1 on its valid inputs, and the
  experimental BZIP2_1) must round-trip bit-identically through
  ``torchfits.write(..., compress=algo)`` / ``torchfits.read(hdu=1)``.
* PLIO_1 is positive-integer-only by CFITSIO design (8/16-bit, or 32-bit up
  to 2**24); invalid payloads must raise, not corrupt.
* HCOMPRESS_1 requires 2D tiles; 1D images must raise.
* Floats are quantized by CFITSIO (default SUBTRACTIVE_DITHER_1 with
  quantize level 4 = 2**4 levels across the data range): the decode error
  is bounded by ``data_range / 8``, and both decoders (torchfits and
  astropy) must agree exactly on the stored values.
* BZIP2_1 is not part of the FITS 4.0 standard: astropy cannot read those
  files (documented limitation), so its oracle is the round-trip itself.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest
import torch

import torchfits

STANDARD_ALGOS = ["RICE_1", "GZIP_1", "GZIP_2", "HCOMPRESS_1", "PLIO_1"]
ALL_ALGOS = STANDARD_ALGOS + ["BZIP2_1"]
LOSSLESS_DTYPES = ["uint8", "int16", "uint16", "int32", "uint32"]

_REFERENCE_FPACK = "/scratch/.tmp-sfabbro/opencode/cfitsio-full/fpack"
_FPACK = os.environ.get("TORCHFITS_FPACK")
if _FPACK is None and os.path.exists(_REFERENCE_FPACK):
    _FPACK = _REFERENCE_FPACK
# Deliberately no PATH fallback: a random system fpack may link an unpatched
# CFITSIO (the pixi-env one SIGABRTs on PLIO_1), and CI must never run it.
requires_fpack = pytest.mark.skipif(not _FPACK, reason="fpack CLI not available")


@pytest.fixture(scope="module")
def bzip2_available(tmp_path_factory):
    """True when the built library includes BZIP2_1 (needs libbz2 at build
    time; plain `pip install` without a conda env disables it)."""
    probe = str(tmp_path_factory.mktemp("bzip2_probe") / "probe.fits")
    data = torch.zeros(8, 8, dtype=torch.int16)
    try:
        torchfits.write(probe, data, overwrite=True, compress="BZIP2_1")
        return True
    except RuntimeError:
        return False


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _make(dtype: str, shape: tuple[int, int], positive: bool = False) -> torch.Tensor:
    """Deterministic random payload for the given dtype; ``positive`` clamps
    to values PLIO_1 accepts."""
    rng = _rng()
    size = shape[0] * shape[1]
    if dtype == "uint8":
        data = rng.integers(0, 256, size=size, dtype=np.uint8)
    elif dtype == "int16":
        lo = 0 if positive else -32768
        data = rng.integers(lo, 32768, size=size, dtype=np.int16)
    elif dtype == "uint16":
        hi = 32768 if positive else 65536
        data = rng.integers(0, hi, size=size, dtype=np.uint16)
    elif dtype == "int32":
        hi = 2**24 if positive else 2**31
        data = rng.integers(0, hi, size=size, dtype=np.int32)
    elif dtype == "uint32":
        data = rng.integers(0, 2**31, size=size, dtype=np.uint32)
    else:
        raise AssertionError(dtype)
    return torch.as_tensor(data.reshape(shape))


def _assert_astropy_matches(path: str, data: torch.Tensor) -> None:
    """astropy (independent implementation) must decode identical values."""
    import astropy
    from astropy.io import fits
    from packaging.version import Version

    # astropy < 7.0 (and no 7.x on py3.10, whose newest is 6.1.7) decodes
    # uint32 (BZERO=2147483648) tile-compressed images with an int64->int32
    # overflow that yields garbage (astropy 7.0.0 fixed it). Skip the oracle
    # only there; the torchfits self-roundtrip above still asserts exactness.
    if data.dtype == torch.uint32 and Version(astropy.__version__) < Version("7.0"):
        return

    astro = np.asarray(fits.getdata(path)).astype(data.numpy().dtype, copy=False)
    np.testing.assert_array_equal(astro, data.numpy())


# ---------------------------------------------------------------------------
# Integer lossless matrix: every standard algorithm, every int dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["RICE_1", "GZIP_1", "GZIP_2", "HCOMPRESS_1"])
@pytest.mark.parametrize("dtype", LOSSLESS_DTYPES)
def test_standard_algo_lossless_matrix_exact(tmp_path, algo, dtype):
    """RICE/GZIP/HCOMPRESS on integer data: bit-identical round-trip,
    astropy oracle, and the ZCMPTYPE header must name the algorithm."""
    data = _make(dtype, (64, 64))
    path = str(tmp_path / f"{algo}_{dtype}.fits")
    torchfits.write(path, data, overwrite=True, compress=algo)

    out = torchfits.read(path, hdu=1)
    assert out.dtype == data.dtype
    assert torch.equal(out, data)

    header = torchfits.read_header(path, hdu=1)
    assert header["ZCMPTYPE"] == algo

    _assert_astropy_matches(path, data)


@pytest.mark.parametrize(
    "dtype,positive",
    [("uint8", True), ("int16", True), ("int32", True)],
)
def test_plio_positive_integer_lossless(tmp_path, dtype, positive):
    """PLIO_1 on its valid integer inputs is lossless and astropy-readable.
    uint16 is intentionally absent: it is pseudo-unsigned (BZERO=32768), so
    the stored pixels are negative and PLIO_1 rejects it."""
    data = _make(dtype, (64, 64), positive=positive)
    path = str(tmp_path / f"plio_{dtype}.fits")
    torchfits.write(path, data, overwrite=True, compress="PLIO_1")

    out = torchfits.read(path, hdu=1)
    assert torch.equal(out, data)

    header = torchfits.read_header(path, hdu=1)
    assert header["ZCMPTYPE"] == "PLIO_1"

    _assert_astropy_matches(path, data)


@pytest.mark.parametrize(
    "dtype,positive",
    [
        ("int16", False),  # negatives are not representable
        ("uint16", False),  # pseudo-unsigned: stored pixels are negative
        ("int32", False),  # values above 2**24 are not representable
    ],
)
def test_plio_invalid_data_rejected(tmp_path, dtype, positive):
    """PLIO_1 payloads outside CFITSIO's format raise instead of corrupting
    the file (regression guard for the PLIO heap-overflow fix)."""
    data = _make(dtype, (64, 64), positive=positive)
    path = str(tmp_path / f"plio_bad_{dtype}.fits")
    with pytest.raises(RuntimeError, match="compressed image"):
        torchfits.write(path, data, overwrite=True, compress="PLIO_1")


def test_plio_float_rejected(tmp_path):
    """PLIO has no float codec (raw float bits fail the 0..2**24 range
    check in imcomp_compress_tile); the write must raise.  If a future
    build ever accepts the payload, the file must still be a valid
    quantized image, not silently mis-encoded data."""
    data = torch.as_tensor(_rng().uniform(0, 100, (64, 64)).astype(np.float32))
    path = str(tmp_path / "plio_f32.fits")
    try:
        torchfits.write(path, data, overwrite=True, compress="PLIO_1")
    except RuntimeError:
        return  # documented behavior: clean rejection
    out = torchfits.read(path, hdu=1)
    span = float(data.numpy().max() - data.numpy().min())
    err = np.abs(out.numpy().astype(np.float64) - data.numpy().astype(np.float64))
    assert err.max() <= span / 8.0


def test_hcompress_1d_rejected(tmp_path):
    """HCOMPRESS needs 2D tiles; a 1D image must raise a clear error."""
    data = torch.as_tensor(_rng().integers(0, 30000, 64).astype(np.int16))
    path = str(tmp_path / "hcomp_1d.fits")
    with pytest.raises(RuntimeError, match="HCOMPRESS.*2D"):
        torchfits.write(path, data, overwrite=True, compress="HCOMPRESS_1")


# ---------------------------------------------------------------------------
# BZIP2_1 (experimental, non-standard): torchfits round-trip only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", LOSSLESS_DTYPES)
def test_bzip2_roundtrip_exact(tmp_path, dtype, bzip2_available):
    """BZIP2_1 round-trips bit-identically (enabled via the vendored
    cfitsio patch); astropy cannot read it — it is outside the FITS
    standard, so the oracle here is the exact round-trip itself. Skipped
    when the library was built without libbz2 (plain pip install)."""
    if not bzip2_available:
        pytest.skip("torchfits built without bzip2 support")
    data = _make(dtype, (64, 64))
    path = str(tmp_path / f"bz2_{dtype}.fits")
    torchfits.write(path, data, overwrite=True, compress="BZIP2_1")

    out = torchfits.read(path, hdu=1)
    assert out.dtype == data.dtype
    assert torch.equal(out, data)

    header = torchfits.read_header(path, hdu=1)
    assert header["ZCMPTYPE"] == "BZIP2_1"

    from astropy.io import fits

    with pytest.raises(ValueError, match="BZIP2_1"):
        fits.getdata(path)


# ---------------------------------------------------------------------------
# Floats: quantized by design, bounded error, decoders agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["RICE_1", "GZIP_1", "GZIP_2", "HCOMPRESS_1"])
def test_float_quantization_error_bound_and_decoder_agreement(tmp_path, algo):
    """CFITSIO quantizes floats with the default dithering (2**4 levels
    across the data range): decode error stays below range/8, and torchfits
    and astropy decode the stored (quantized) values identically."""
    data = torch.as_tensor(_rng().uniform(-1000, 1000, (64, 64)).astype(np.float32))
    span = float(data.numpy().max() - data.numpy().min())
    path = str(tmp_path / f"f32_{algo}.fits")
    torchfits.write(path, data, overwrite=True, compress=algo)

    out = torchfits.read(path, hdu=1)
    err = np.abs(out.numpy().astype(np.float64) - data.numpy().astype(np.float64))
    assert err.max() <= span / 8.0

    from astropy.io import fits

    astro = np.asarray(fits.getdata(path)).astype(np.float32, copy=False)
    np.testing.assert_array_equal(astro, out.numpy())


# ---------------------------------------------------------------------------
# Cutouts on every algorithm (per-tile decode correctness)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ALL_ALGOS)
def test_compressed_cutouts_all_algos_match_uncompressed(
    tmp_path, algo, bzip2_available
):
    """read_subset windows from each compressed algorithm equal the
    uncompressed image slice (exercises per-tile decode of each codec)."""
    if algo == "BZIP2_1" and not bzip2_available:
        pytest.skip("torchfits built without bzip2 support")
    data = _make("int16", (64, 64), positive=(algo == "PLIO_1"))
    plain = str(tmp_path / f"{algo}_plain.fits")
    zipped = str(tmp_path / f"{algo}_zip.fits")
    torchfits.write(plain, data, overwrite=True)
    torchfits.write(zipped, data, overwrite=True, compress=algo)

    windows = [(1, 1, 2, 2), (3, 7, 20, 30), (10, 10, 60, 60), (0, 0, 64, 64)]
    for x1, y1, x2, y2 in windows:
        want = torchfits.read_subset(plain, 0, x1, y1, x2, y2)
        got = torchfits.read_subset(zipped, 1, x1, y1, x2, y2)
        assert torch.equal(got, want), f"{algo} cutout ({x1},{y1},{x2},{y2})"


# ---------------------------------------------------------------------------
# fpack byte-identity (requires the reference CLI built from vendored source)
# ---------------------------------------------------------------------------


def _table_bytes(path: str) -> bytes:
    """Raw bytes of the compressed table (rows + PCOUNT heap): everything
    after the first XTENSION header's END card."""
    raw = open(path, "rb").read()
    off = 0
    while off + 2880 <= len(raw):
        block = raw[off : off + 2880]
        if block[:8] == b"XTENSION":
            return raw[off + 2880 :]
        off += 2880
    raise AssertionError(f"{path}: no XTENSION header found")


@requires_fpack
@pytest.mark.parametrize(
    "algo,flag",
    [
        ("RICE_1", "-r"),
        ("GZIP_1", "-g"),
        ("GZIP_2", "-g2"),
        ("HCOMPRESS_1", "-h"),
        ("PLIO_1", "-p"),
    ],
)
def test_fpack_byte_identity(tmp_path, algo, flag):
    """torchfits output is byte-identical (table + heap) to fpack built from
    the same vendored CFITSIO source. The reference CLI at
    _REFERENCE_FPACK takes precedence over any fpack found on PATH: the
    pixi/system fpack links an unpatched CFITSIO and aborts on PLIO_1.
    fpack must NOT be given -F: that clobber mode writes in place and
    produces no .fz output."""
    rng = _rng()
    data = torch.as_tensor(rng.integers(0, 30000, size=(512, 512)).astype(np.int16))
    torch_path = str(tmp_path / f"{algo}.fits")
    plain = str(tmp_path / "plain.fits")
    torchfits.write(torch_path, data, overwrite=True, compress=algo)
    torchfits.write(plain, data, overwrite=True)

    subprocess.run([_FPACK, flag, plain], check=True, capture_output=True, text=True)
    assert _table_bytes(torch_path) == _table_bytes(str(plain) + ".fz")
