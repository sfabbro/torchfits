"""Cross-library output parity: torchfits must reproduce cfitsio/fitsio/astropy outputs exactly.

Ground truth rule: fixtures written by astropy are read by torchfits, fitsio and
astropy and must be bitwise identical (dtype + shape + values). Files written by
torchfits (plain, quantized, compressed, LONGSTRN, tables with TSCAL/TZERO) are
read back by fitsio — the thin CFITSIO wrapper — and must match exactly.
Float-scaled outputs are compared exact-against-fitsio (identical CFITSIO
BSCALE/BZERO math). CompImage float reads on macOS allow ≤1 dtype eps vs
fitsio/astropy (CFITSIO decompress SIMD/libc noise); all other cases stay
bitwise exact. A seeded fuzz-lite generator sweeps dtypes / shapes /
compression in both directions. This suite gates every performance change:
speedups must be output-invisible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

fitsio = pytest.importorskip("fitsio")
astropy_fits = pytest.importorskip("astropy.io.fits")

import torchfits  # noqa: E402


def _assert_exact(
    got: np.ndarray,
    expected: np.ndarray,
    label: str,
    *,
    atol: float | None = None,
) -> None:
    got = np.asarray(got)
    expected = np.asarray(expected)
    assert got.shape == expected.shape, (
        f"{label}: shape {got.shape} != {expected.shape}"
    )
    byteorder_only = got.dtype == expected.dtype or got.dtype.newbyteorder(
        "="
    ) == expected.dtype.newbyteorder("=")
    assert byteorder_only, f"{label}: dtype {got.dtype} != {expected.dtype}"
    if atol is None:
        np.testing.assert_array_equal(got, expected, err_msg=f"{label}: values differ")
        return
    # atol path: CFITSIO float decompress can differ by ~1 eps across libc/SIMD.
    np.testing.assert_allclose(
        got, expected, rtol=0.0, atol=atol, err_msg=f"{label}: values differ"
    )


def _compressed_float_atol(name: str, sample: np.ndarray) -> float | None:
    """macOS-only slack for CompImage float reads vs fitsio (≤1 float eps)."""
    if sys.platform != "darwin" or not name.startswith("compressed_"):
        return None
    arr = np.asarray(sample)
    if not np.issubdtype(arr.dtype, np.floating):
        return None
    return float(np.finfo(arr.dtype).eps)


def _decode(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return value


@pytest.fixture(scope="session")
def parity_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("parity")


@pytest.fixture(scope="session")
def parity_files(parity_dir) -> dict[str, Path]:
    """Astropy-written reference files (independent ground truth)."""
    files: dict[str, Path] = {}
    rng = np.random.default_rng(20260804)

    def write(name: str, payload, **kwargs) -> None:
        path = parity_dir / f"{name}.fits"
        payload.writeto(path, overwrite=True, **kwargs)
        files[name] = path

    for dtype_name, np_dtype in (
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
        ("int64", np.int64),
        ("float32", np.float32),
        ("float64", np.float64),
    ):
        arr2 = rng.integers(0, 30000, size=(48, 64)).astype(np_dtype)
        if np.issubdtype(np_dtype, np.floating):
            arr2 = rng.normal(size=(48, 64)).astype(np_dtype)
        write(f"plain_{dtype_name}_2d", astropy_fits.PrimaryHDU(arr2))

        cube = rng.integers(0, 30000, size=(6, 24, 32)).astype(np_dtype)
        if np.issubdtype(np_dtype, np.floating):
            cube = rng.normal(size=(6, 24, 32)).astype(np_dtype)
        write(f"plain_{dtype_name}_3d", astropy_fits.PrimaryHDU(cube))

    for dtype_name, np_dtype in (("uint16", np.uint16), ("uint32", np.uint32)):
        arr = rng.integers(0, 60000, size=(48, 64)).astype(np_dtype)
        write(f"unsigned_{dtype_name}_2d", astropy_fits.PrimaryHDU(arr))

    scaled_int = (rng.normal(size=(48, 64)) * 1000 + 32768).astype(np.int16)
    write(
        "scaled_int16_2d",
        astropy_fits.PrimaryHDU(
            scaled_int, header=astropy_fits.Header({"BSCALE": 0.1, "BZERO": 32768})
        ),
    )
    scaled_float = (rng.normal(size=(48, 64)) * 7.0).astype(np.float32)
    write(
        "scaled_float32_2d",
        astropy_fits.PrimaryHDU(
            scaled_float, header=astropy_fits.Header({"BSCALE": 2.0, "BZERO": -1000.0})
        ),
    )

    compressed = rng.normal(size=(128, 128)).astype(np.float32)
    for compression in ("RICE_1", "GZIP_1", "HCOMPRESS_1"):
        write(
            f"compressed_{compression.lower()}",
            astropy_fits.HDUList(
                [
                    astropy_fits.PrimaryHDU(),
                    astropy_fits.CompImageHDU(compressed, compression_type=compression),
                ]
            ),
        )

    vla = np.empty(6, dtype=object)
    for i in range(6):
        vla[i] = rng.normal(size=(i + 1)).astype(np.float64)
    tdata = np.zeros(
        6,
        dtype=[
            ("INDEX", "<i4"),
            ("FLUX", "O"),
            ("NAME", "S4"),
            ("FLAG", "?"),
            ("VAL", "<f8"),
        ],
    )
    tdata["INDEX"] = np.arange(6)
    tdata["FLUX"] = vla
    tdata["NAME"] = np.array([b"a", b"bb", b"ccc", b"d", b"ee", b"f"], dtype="S4")
    tdata["FLAG"] = np.array([True, False, True, False, True, False])
    tdata["VAL"] = rng.normal(size=6)
    write("table_vla", astropy_fits.BinTableHDU(tdata, name="PARITY"))

    return files


# ---------------------------------------------------------------------------
# Read parity: astropy-written files, torchfits == fitsio == astropy (exact).
# ---------------------------------------------------------------------------

_IMAGE_READ_CASES = [
    ("plain_int8_2d", 0),
    ("plain_int16_2d", 0),
    ("plain_int32_2d", 0),
    ("plain_int64_2d", 0),
    ("plain_float32_2d", 0),
    ("plain_float64_2d", 0),
    ("plain_int16_3d", 0),
    ("plain_float32_3d", 0),
    ("unsigned_uint16_2d", 0),
    ("unsigned_uint32_2d", 0),
    ("scaled_int16_2d", 0),
    ("scaled_float32_2d", 0),
    ("compressed_rice_1", 1),
    ("compressed_gzip_1", 1),
    ("compressed_hcompress_1", 1),
]


@pytest.mark.parametrize("name,hdu", _IMAGE_READ_CASES)
@pytest.mark.parametrize("mmap", [False, True])
def test_read_image_parity_exact(parity_files, name: str, hdu: int, mmap: bool) -> None:
    path = str(parity_files[name])
    tf = torchfits.read(path, hdu=hdu, mmap=mmap)
    assert isinstance(tf, torch.Tensor)
    got = np.asarray(tf)
    atol = _compressed_float_atol(name, got)
    fi = fitsio.read(path, ext=hdu)
    _assert_exact(got, fi, f"{name} mmap={mmap} vs fitsio", atol=atol)
    with astropy_fits.open(path) as hdul:
        _assert_exact(got, hdul[hdu].data, f"{name} mmap={mmap} vs astropy", atol=atol)


def test_read_table_parity_scalar_columns(parity_files) -> None:
    path = str(parity_files["table_vla"])
    tf = torchfits.table.read(path, hdu=1)
    rec = fitsio.read(path, ext=1)
    with astropy_fits.open(path) as hdul:
        adata = hdul[1].data
    assert sorted(rec.dtype.names) == sorted(tf.column_names)
    for name in ("INDEX", "VAL", "FLAG"):
        expected_fits = [_decode(v) for v in rec[name].tolist()]
        expected_ast = [_decode(v) for v in adata[name].tolist()]
        assert tf.column(name).to_pylist() == expected_fits, f"{name} vs fitsio"
        assert tf.column(name).to_pylist() == expected_ast, f"{name} vs astropy"


def test_read_table_parity_vla_columns(parity_files) -> None:
    path = str(parity_files["table_vla"])
    tf = torchfits.table.read(path, hdu=1)
    got = [list(v) for v in tf.column("FLUX").to_pylist()]
    with astropy_fits.open(path) as hdul:
        expected_ast = [np.asarray(v).tolist() for v in hdul[1].data["FLUX"]]
    assert got == expected_ast
    # fitsio pads VLA rows to the max repeat (fixed-width read); truncate the
    # padding and compare the true values.
    rec = fitsio.read(path, ext=1)
    for row_got, row_fits in zip(got, rec["FLUX"]):
        assert row_got == np.asarray(row_fits)[: len(row_got)].tolist()


def test_read_table_parity_string_columns(parity_files) -> None:
    path = str(parity_files["table_vla"])
    tf = torchfits.table.read(path, hdu=1)
    rec = fitsio.read(path, ext=1)
    got = [_decode(v) for v in tf.column("NAME").to_pylist()]
    assert got == [_decode(v) for v in rec["NAME"].tolist()]


# ---------------------------------------------------------------------------
# Write parity: torchfits-written files read back by fitsio (raw CFITSIO).
# ---------------------------------------------------------------------------

_WRITE_IMAGE_DTYPES = [
    ("int16", np.int16),
    ("int32", np.int32),
    ("float32", np.float32),
    ("float64", np.float64),
    ("uint16", np.uint16),
    ("uint32", np.uint32),
]


@pytest.mark.parametrize("dtype_name,np_dtype", _WRITE_IMAGE_DTYPES)
def test_write_image_roundtrip_fitsio(parity_dir, dtype_name: str, np_dtype) -> None:
    rng = np.random.default_rng(hash(dtype_name) % 2**32)
    if np.issubdtype(np_dtype, np.floating):
        arr = rng.normal(size=(32, 40)).astype(np_dtype)
    else:
        arr = rng.integers(0, 60000, size=(32, 40)).astype(np_dtype)
    path = parity_dir / f"tw_{dtype_name}.fits"
    torchfits.write(str(path), arr, overwrite=True)
    fi = fitsio.read(str(path), ext=0)
    _assert_exact(fi, arr, f"torchfits-written {dtype_name} vs fitsio")
    with astropy_fits.open(str(path)) as hdul:
        _assert_exact(
            np.asarray(hdul[0].data), arr, f"torchfits-written {dtype_name} vs astropy"
        )


def test_write_quantize_robust_roundtrip_fitsio(parity_dir) -> None:
    rng = np.random.default_rng(11)
    arr = (rng.normal(size=(64, 64)) * 100.0 + 500.0).astype(np.float32)
    path = parity_dir / "tw_quantize_robust.fits"
    torchfits.write(str(path), arr, overwrite=True, quantize="robust")
    header = fitsio.read_header(str(path), ext=0)
    assert "BSCALE" in header and "BZERO" in header
    fi = fitsio.read(str(path), ext=0)
    tf = torchfits.read(str(path), hdu=0)
    _assert_exact(np.asarray(tf), fi, "quantize read-back torchfits vs fitsio")
    scale = float(header["BSCALE"])
    lo, hi = float(header["BZERO"]), float(header["BZERO"]) + scale * 32766.0
    assert np.max(np.abs(fi - arr)) <= (hi - lo) / 2.0 + 1e-6


@pytest.mark.parametrize("compression", ["RICE_1", "GZIP_1", "HCOMPRESS_1"])
def test_write_compressed_roundtrip_fitsio(parity_dir, compression: str) -> None:
    rng = np.random.default_rng(hash(compression) % 2**32)
    arr = rng.normal(size=(64, 64)).astype(np.float32)
    path = parity_dir / f"tw_compressed_{compression.lower()}.fits"
    torchfits.write(str(path), arr, overwrite=True, compress=compression)
    fi = fitsio.read(str(path), ext=1)
    # Decode parity: torchfits and fitsio must read the written file identically.
    tf = torchfits.read(str(path), hdu=1)
    _assert_exact(
        np.asarray(tf), fi, f"torchfits-written {compression} decode vs fitsio"
    )
    # Round-trip parity: GZIP_1 (and integer RICE) is lossless; float RICE and
    # HCOMPRESS use CFITSIO default quantization (lossy, same as astropy and
    # fitsio defaults), so only GZIP_1 must reproduce the input exactly.
    if compression == "GZIP_1":
        _assert_exact(fi, arr, f"torchfits-written {compression} lossless round-trip")


@pytest.mark.parametrize("compression", ["RICE_1", "GZIP_1"])
def test_write_compressed_integer_roundtrip_fitsio(
    parity_dir, compression: str
) -> None:
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 1000, size=(64, 64)).astype(np.int16)
    path = parity_dir / f"tw_compressed_int_{compression.lower()}.fits"
    torchfits.write(str(path), arr, overwrite=True, compress=compression)
    fi = fitsio.read(str(path), ext=1)
    _assert_exact(fi, arr, f"torchfits-written integer {compression} round-trip")


def test_write_longstr_roundtrip_fitsio(parity_dir) -> None:
    long_value = "x" * 300
    path = parity_dir / "tw_longstr.fits"
    torchfits.write(
        str(path),
        np.zeros((4, 4), dtype=np.float32),
        header={"LONGCARD": long_value},
        overwrite=True,
    )
    header = fitsio.read_header(str(path), ext=0)
    assert header["LONGCARD"] == long_value


def test_write_uint64_rejected(parity_dir) -> None:
    path = parity_dir / "tw_uint64.fits"
    with pytest.raises(ValueError):
        torchfits.write(str(path), np.zeros((4,), dtype=np.uint64), overwrite=True)
    with pytest.raises(ValueError):
        torchfits.table.write(
            str(path), {"U": np.zeros((4,), dtype=np.uint64)}, overwrite=True
        )


def test_write_table_quantize_roundtrip_fitsio(parity_dir) -> None:
    rng = np.random.default_rng(7)
    n = 64
    table = {
        "INDEX": np.arange(n, dtype=np.int32),
        "FLUX": (rng.normal(size=n) * 100.0 + 1000.0).astype(np.float64),
        "NAME": np.array([f"obj{i}" for i in range(n)]),
    }
    path = parity_dir / "tw_table_quantize.fits"
    torchfits.table.write(str(path), table, overwrite=True, quantize={"FLUX": "robust"})
    header = fitsio.read_header(str(path), ext=1)
    assert "TSCAL2" in header and "TZERO2" in header
    # fitsio 1.4.2 misreads tables with non-integer TSCAL/TZERO int16 columns
    # (returns constant TZERO, and its whole-table decode trips on the
    # misaligned string column) — a fitsio-side bug reproduced with
    # astropy-written files of identical layout; astropy (CFITSIO) and
    # torchfits agree. Compare the scaled column against astropy; read the
    # unaffected columns via fitsio's column-scoped path.
    rec = fitsio.read(str(path), ext=1, columns=["INDEX", "NAME"])
    np.testing.assert_array_equal(rec["INDEX"], table["INDEX"])
    np.testing.assert_array_equal(np.char.strip(rec["NAME"].astype("U")), table["NAME"])
    tf = torchfits.table.read_torch(str(path), hdu=1, mmap=False)
    with astropy_fits.open(str(path)) as hdul:
        _assert_exact(
            np.asarray(tf["FLUX"]),
            np.asarray(hdul[1].data["FLUX"]),
            "quantized table torchfits vs astropy",
        )
    scale = float(header["TSCAL2"])
    lo, hi = float(header["TZERO2"]), float(header["TZERO2"]) + scale * 32766.0
    ast_flux = astropy_fits.getdata(str(path), ext=1)["FLUX"]
    assert np.max(np.abs(ast_flux - table["FLUX"])) <= (hi - lo) / 2.0 + 1e-9


def test_write_plain_table_roundtrip_fitsio(parity_dir) -> None:
    table = {
        "INDEX": np.arange(16, dtype=np.int32),
        "VAL": np.arange(16, dtype=np.float64) + 0.5,
        "FLAG": np.arange(16) % 2 == 0,
    }
    path = parity_dir / "tw_plain_table.fits"
    torchfits.table.write(str(path), table, overwrite=True)
    rec = fitsio.read(str(path), ext=1)
    np.testing.assert_array_equal(rec["INDEX"], table["INDEX"])
    np.testing.assert_array_equal(rec["VAL"], table["VAL"])
    np.testing.assert_array_equal(rec["FLAG"], table["FLAG"])


# ---------------------------------------------------------------------------
# Fuzz-lite: seeded random dtypes / shapes / compression, both directions.
# ---------------------------------------------------------------------------

FUZZ_SEEDS = [100 + i for i in range(20)]


def _fuzz_array(rng: np.random.Generator, dtype) -> np.ndarray:
    naxis = int(rng.integers(1, 4))
    shape = tuple(int(rng.integers(1, 24)) for _ in range(naxis))
    while int(np.prod(shape)) > 16384:
        shape = shape[:-1] + (max(1, shape[-1] // 2),)
    if np.issubdtype(dtype, np.floating):
        return rng.normal(size=shape).astype(dtype)
    return rng.integers(0, 30000, size=shape).astype(dtype)


@pytest.mark.parametrize("seed", FUZZ_SEEDS)
def test_fuzz_parity_both_directions(parity_dir, seed: int) -> None:
    rng = np.random.default_rng(seed)
    dtype = rng.choice([np.int8, np.int16, np.int32, np.int64, np.float32, np.float64])
    arr = _fuzz_array(rng, dtype)
    if np.issubdtype(dtype, np.floating) and rng.random() < 0.25:
        algo = "RICE_1" if dtype == np.float32 else "GZIP_1"
        astropy_fits.HDUList(
            [
                astropy_fits.PrimaryHDU(),
                astropy_fits.CompImageHDU(arr, compression_type=algo),
            ]
        ).writeto(parity_dir / f"fuzz_ref_{seed}.fits", overwrite=True)
        ext = 1
    else:
        astropy_fits.PrimaryHDU(arr).writeto(
            parity_dir / f"fuzz_ref_{seed}.fits", overwrite=True
        )
        ext = 0
    fi = fitsio.read(str(parity_dir / f"fuzz_ref_{seed}.fits"), ext=ext)
    for mmap in (False, True):
        tf = torchfits.read(
            str(parity_dir / f"fuzz_ref_{seed}.fits"), hdu=ext, mmap=mmap
        )
        _assert_exact(np.asarray(tf), fi, f"fuzz seed={seed} mmap={mmap}")

    torchfits.write(str(parity_dir / f"fuzz_tw_{seed}.fits"), arr, overwrite=True)
    _assert_exact(
        fitsio.read(str(parity_dir / f"fuzz_tw_{seed}.fits"), ext=0),
        arr,
        f"fuzz write seed={seed} vs fitsio",
    )
