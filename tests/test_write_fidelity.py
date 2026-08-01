"""Write/read round-trip fidelity tests.

These tests are deliberately *faithful to the original data*: every test
compares against the exact input values (``torch.equal`` / ``np.array_equal``
for integer and lossless paths — never ``allclose``), and cross-checks with
``astropy.io.fits`` as an independent implementation wherever a FITS
convention is involved (pseudo-unsigned BZERO/TZERO, ASCII tables, BIT
columns, uint64 images).

They guard the classes of silent-corruption bugs previously found in the C++
layer:

* dtype conventions on the chunked / mmap / subset read paths (TUSHORT/TUINT
  element sizes, signed-byte, unsigned offsets),
* lossless compression round-trips and compressed cutouts,
* table column/row selections and the pseudo-unsigned table writer,
* ASCII table routing and string-width handling on append/update,
* BIT/LOGICAL decode in the filtered (gather) path,
* overflow-safe row windows and write payload repeat validation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits

IMAGE_DTYPES = [
    (torch.int8, np.int8),
    (torch.uint8, np.uint8),
    (torch.int16, np.int16),
    (torch.int32, np.int32),
    (torch.int64, np.int64),
    (torch.float32, np.float32),
    (torch.float64, np.float64),
]


def _image_values(shape: tuple[int, int], torch_dtype: torch.dtype) -> torch.Tensor:
    """Boundary-heavy values whose byte patterns are obviously wrong if
    mis-decoded (byteswapped, offset by 128, truncated, etc.)."""
    rng = np.random.default_rng(7)
    size = shape[0] * shape[1]
    if torch_dtype == torch.int8:
        data = np.resize(np.arange(-128, 128, dtype=np.int8), size)
    elif torch_dtype == torch.uint8:
        data = rng.integers(0, 256, size=size, dtype=np.uint8)
    elif torch_dtype == torch.int16:
        data = rng.integers(-32768, 32768, size=size, dtype=np.int16)
    elif torch_dtype == torch.int32:
        data = rng.integers(-(2**31), 2**31, size=size, dtype=np.int32)
    elif torch_dtype == torch.int64:
        vals = np.array(
            [0, 1, -1, 2**40, -(2**40), 2**62, -(2**62), 0x0102030405060708],
            dtype=np.int64,
        )
        data = np.resize(vals, size)
    elif torch_dtype == torch.float32:
        data = rng.uniform(-1000, 1000, size=size).astype(np.float32)
    else:
        data = rng.uniform(-1000, 1000, size=size).astype(np.float64)
    return torch.as_tensor(data.reshape(shape))


# ---------------------------------------------------------------------------
# Images: write -> read -> exact original values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("torch_dtype,np_dtype", IMAGE_DTYPES)
@pytest.mark.parametrize("mmap", [True, False])
def test_image_write_roundtrip_exact(tmp_path, torch_dtype, np_dtype, mmap):
    """torchfits-written images read back bit-faithfully, both read paths."""
    data = _image_values((17, 23), torch_dtype)
    path = str(tmp_path / f"img_{np_dtype.__name__}.fits")
    torchfits.write(path, data, overwrite=True)

    out = torchfits.read(path, mmap=mmap)
    # int8 is written with the BZERO=-128 signed-byte convention and must
    # come back as exact int8 values.
    assert out.dtype == torch_dtype
    assert torch.equal(out, data)

    # Independent oracle: astropy must read the same values from the file.
    from astropy.io import fits

    astro = fits.getdata(path)
    # astropy returns big-endian for FITS-native types; normalize first.
    np.testing.assert_array_equal(
        np.asarray(astro).astype(np_dtype, copy=False), data.numpy()
    )


@pytest.mark.parametrize("torch_dtype", [torch.uint16, torch.uint32])
@pytest.mark.parametrize("mmap", [True, False])
def test_unsigned_image_convention_roundtrip_exact(tmp_path, torch_dtype, mmap):
    """Pseudo-unsigned images (BZERO=32768 / 2**31) round-trip exactly."""
    shape = (11, 13)
    size = shape[0] * shape[1]
    rng = np.random.default_rng(11)
    if torch_dtype == torch.uint16:
        data = rng.integers(0, 65536, size=size, dtype=np.uint16).reshape(shape)
    else:
        data = rng.integers(0, 2**32, size=size, dtype=np.uint32).reshape(shape)
    bits = 16 if torch_dtype == torch.uint16 else 32
    path = str(tmp_path / f"u{bits}.fits")
    torchfits.write(path, torch.as_tensor(data), overwrite=True)

    out = torchfits.read(path, mmap=mmap)
    assert out.dtype == torch_dtype
    assert torch.equal(out, torch.as_tensor(data))

    from astropy.io import fits

    astro = fits.getdata(path)
    assert astro.dtype == data.dtype
    np.testing.assert_array_equal(astro, data)


def test_uint64_image_bzero_2_63_is_detected(tmp_path):
    """LONGLONG BZERO=2**63 images must be detected as scaled (not read as raw
    int64 garbage); they come back as float32 per the documented contract."""
    from astropy.io import fits

    data = np.array([0, 1, 2**20, 2**40, 2**63 - 1], dtype=np.uint64).reshape(1, 5)
    path = str(tmp_path / "u64.fits")
    fits.HDUList([fits.PrimaryHDU(data)]).writeto(path, overwrite=True)

    out = torchfits.read(path)
    # Scaled LONGLONG -> float32 conversion (no exact uint64 dtype exists yet).
    assert out.dtype == torch.float32
    # Regressing to the old bug: unscaled raw int64 values (~ -9.2e18) would
    # fail this by 20+ orders of magnitude.
    np.testing.assert_allclose(
        out.numpy().astype(np.float64), data.astype(np.float64), rtol=1e-6
    )


# ---------------------------------------------------------------------------
# Cutouts: subset reads must equal the full-read slice
# ---------------------------------------------------------------------------

CUTOUT_DTYPES = [
    (torch.uint8, "torch"),
    (torch.int16, "torch"),
    (torch.uint16, "astropy"),
    (torch.uint32, "astropy"),
    (torch.float32, "torch"),
]


@pytest.mark.parametrize("torch_dtype,writer", CUTOUT_DTYPES)
def test_cutout_fidelity_matrix(tmp_path, torch_dtype, writer):
    """Every read_subset window matches the full-read slice exactly."""
    shape = (37, 41)
    if torch_dtype == torch.uint16:
        data = torch.randint(0, 65536, shape, dtype=torch.uint16)
    elif torch_dtype == torch.uint32:
        data = torch.randint(0, 2**31, shape, dtype=torch.uint32)
    else:
        data = _image_values(shape, torch_dtype)
    path = str(tmp_path / f"cut_{torch_dtype}.fits")
    if writer == "astropy":
        from astropy.io import fits

        fits.HDUList([fits.PrimaryHDU(data.numpy())]).writeto(path, overwrite=True)
    else:
        torchfits.write(path, data, overwrite=True)

    full = torchfits.read(path)
    assert torch.equal(full, data)

    # (x1, y1, x2, y2) 0-based half-open windows (matching the documented
    # read_subset convention): single pixel, interior box, 1-px strip, full
    # frame.
    windows = [(1, 1, 2, 2), (5, 7, 15, 17), (7, 5, 8, 6), (0, 0, 41, 37)]
    for x1, y1, x2, y2 in windows:
        want = full[y1:y2, x1:x2]
        got = torchfits.read_subset(path, 0, x1, y1, x2, y2)
        assert got.dtype == full.dtype
        assert torch.equal(got, want), f"read_subset window ({x1},{y1},{x2},{y2})"

        with torchfits.open_subset_reader(path, hdu=0) as reader:
            got2 = reader.read_subset(x1, y1, x2, y2)
        assert torch.equal(got2, want), f"SubsetReader window ({x1},{y1},{x2},{y2})"


@pytest.mark.parametrize("torch_dtype", [torch.uint8, torch.int16, torch.int32])
def test_cutout_compressed_lossless_matches_uncompressed(tmp_path, torch_dtype):
    """Cutouts from losslessly compressed HDUs equal the uncompressed ones."""
    data = _image_values((64, 64), torch_dtype)
    plain = str(tmp_path / "plain.fits")
    zipped = str(tmp_path / "zipped.fits")
    torchfits.write(plain, data, overwrite=True)
    torchfits.write(zipped, data, overwrite=True, compress=True)

    windows = [(1, 1, 2, 2), (3, 7, 20, 30), (10, 10, 60, 60), (0, 0, 64, 64)]
    for x1, y1, x2, y2 in windows:
        want = torchfits.read_subset(plain, 0, x1, y1, x2, y2)
        # Compressed files keep an empty primary; the image lives at HDU 1.
        got = torchfits.read_subset(zipped, 1, x1, y1, x2, y2)
        assert torch.equal(got, want), f"compressed cutout ({x1},{y1},{x2},{y2})"


# ---------------------------------------------------------------------------
# Compression: lossless algorithms must be exact
# ---------------------------------------------------------------------------

# 64-bit integer images are outside the CFITSIO RICE/GZIP data-type support,
# so the lossless set stops at int32.
LOSSY_FREE_DTYPES = [torch.uint8, torch.int16, torch.int32]


@pytest.mark.parametrize("torch_dtype", LOSSY_FREE_DTYPES)
@pytest.mark.parametrize("algo", ["RICE_1", "GZIP_1"])
def test_compressed_lossless_roundtrip_exact(tmp_path, torch_dtype, algo):
    """Integer data through RICE/GZIP comes back bit-identical."""
    data = _image_values((32, 32), torch_dtype)
    path = str(tmp_path / f"c_{algo}_{torch_dtype}.fits")
    torchfits.write(path, data, overwrite=True, compress=algo)

    out = torchfits.read(path, hdu=1)
    assert out.dtype == torch_dtype
    assert torch.equal(out, data)

    from astropy.io import fits

    astro = fits.getdata(path)
    assert astro.dtype == data.numpy().dtype
    np.testing.assert_array_equal(astro, data.numpy())


# ---------------------------------------------------------------------------
# Tables: write -> read -> exact original values
# ---------------------------------------------------------------------------

TABLE_COLUMNS = {
    "I8": np.array([-128, -1, 0, 1, 127], dtype=np.int8),
    "U8": np.array([0, 1, 127, 254, 255], dtype=np.uint8),
    "I16": np.array([-32768, -1, 0, 1, 32767], dtype=np.int16),
    "I32": np.array([-(2**31), -1, 0, 1, 2**31 - 1], dtype=np.int32),
    "I64": np.array([-(2**62), -1, 0, 1, 2**62], dtype=np.int64),
    "U16": np.array([0, 1, 32768, 65534, 65535], dtype=np.uint16),
    "U32": np.array([0, 1, 2**31, 2**32 - 2, 2**32 - 1], dtype=np.uint32),
    "F32": np.array([-0.5, 0.0, 1.5, 1e6, -1e-6], dtype=np.float32),
    "F64": np.array([-0.5, 0.0, 1.5, 1e12, -1e-12], dtype=np.float64),
    "FLAG": np.array([True, False, True, True, False], dtype=np.bool_),
}


def _assert_column_faithful(got: torch.Tensor, want: np.ndarray) -> None:
    """Exact value comparison; int8 is stored as TBYTE and returns as uint8."""
    want_t = torch.as_tensor(want)
    if want.dtype == np.int8:
        assert got.dtype == torch.uint8
        assert torch.equal(got.view(torch.int8), want_t)
    else:
        assert got.dtype == want_t.dtype
        assert torch.equal(got, want_t)


def test_table_write_roundtrip_dtype_matrix(tmp_path):
    """All supported table dtypes round-trip exactly through both read paths."""
    path = str(tmp_path / "table_matrix.fits")
    torchfits.table.write(path, TABLE_COLUMNS, overwrite=True)

    for mmap in (True, False):
        out = torchfits.read(path, hdu=1, mmap=mmap)
        assert isinstance(out, dict)
        for name, want in TABLE_COLUMNS.items():
            _assert_column_faithful(out[name], want)

    # Independent oracle: astropy sees the same values (pseudo-unsigned
    # conventions, logicals, bytes).
    from astropy.io import fits

    with fits.open(path) as hdul:
        for name, want in TABLE_COLUMNS.items():
            actual = np.asarray(hdul[1].data[name])
            if want.dtype == np.int8:
                # int8 is stored as TBYTE; astropy sees the raw unsigned bytes.
                assert actual.dtype == np.uint8
                actual = actual.view(np.int8)
            np.testing.assert_array_equal(actual, want, err_msg=f"astropy col {name}")


def test_table_string_column_roundtrip(tmp_path):
    """String columns (incl. max-width and empty) survive exactly."""
    names = ["alpha", "b", "gamma delta epsilon", "", "zz"]
    path = str(tmp_path / "strings.fits")
    torchfits.table.write(path, {"NAME": names}, overwrite=True)

    arrow = torchfits.table.read(path, hdu=1)
    assert list(arrow.column("NAME").to_pylist()) == names

    from astropy.io import fits

    with fits.open(path) as hdul:
        assert list(hdul[1].data["NAME"]) == names


def test_table_row_windows_and_column_selection(tmp_path):
    """Row windows and column subsets match the full read exactly."""
    nrows = 40
    table = {
        "A": np.arange(nrows, dtype=np.int32),
        "B": np.arange(nrows, dtype=np.float64) * 0.5,
        "C": np.array([i % 2 == 0 for i in range(nrows)], dtype=np.bool_),
    }
    path = str(tmp_path / "windows.fits")
    torchfits.table.write(path, table, overwrite=True)

    full = torchfits.read(path, hdu=1, mmap=True)
    for mmap in (True, False):
        for start, num in [(1, 5), (3, 17), (35, 10), (1, 40), (40, 1)]:
            rows = torchfits.read(path, hdu=1, mmap=mmap, start_row=start, num_rows=num)
            want = {k: v[start - 1 : start - 1 + num] for k, v in full.items()}
            for name in table:
                assert torch.equal(rows[name], want[name]), (
                    f"mmap={mmap} rows {start}:{num} col {name}"
                )

        # Column selection.
        sub = torchfits.read(path, hdu=1, mmap=mmap, columns=["C", "A"])
        assert list(sub.keys()) == ["C", "A"]
        assert torch.equal(sub["A"], full["A"])
        assert torch.equal(sub["C"], full["C"])


def test_table_extreme_row_window_is_clamped(tmp_path):
    """Overflowing start_row+num_rows must clamp to the tail, not fail."""
    table = {"A": np.arange(10, dtype=np.int32)}
    path = str(tmp_path / "clamp.fits")
    torchfits.table.write(path, table, overwrite=True)

    for mmap in (True, False):
        out = torchfits.read(path, hdu=1, mmap=mmap, start_row=8, num_rows=100)
        assert torch.equal(out["A"], torch.arange(7, 10, dtype=torch.int32))


def test_ascii_table_roundtrip_via_all_read_paths(tmp_path):
    """ASCII tables must read exactly through every public path (incl.
    mmap=True, which must route away from the binary mmap layout)."""
    table = {
        "ID": np.array([1, 2, 3], dtype=np.int32),
        "VAL": np.array([0.5, 1.5, 2.5], dtype=np.float64),
        "NAME": ["a", "longish name", "z"],
    }
    path = str(tmp_path / "ascii.fits")
    torchfits.table.write(path, table, table_type="ascii", overwrite=True)

    from astropy.io import fits

    with fits.open(path) as hdul:
        assert str(hdul[1].header.get("XTENSION", "")).upper() == "TABLE"
        astro_id = np.asarray(hdul[1].data["ID"])
        astro_name = list(hdul[1].data["NAME"])

    for mmap in (True, False):
        out = torchfits.read(path, hdu=1, mmap=mmap)
        assert torch.equal(out["ID"].squeeze(-1), torch.as_tensor(astro_id))
        np.testing.assert_array_equal(out["VAL"].squeeze(-1).numpy(), table["VAL"])

    arrow = torchfits.table.read(path, hdu=1)
    assert list(arrow.column("ID").to_pylist()) == table["ID"].tolist()
    assert list(arrow.column("NAME").to_pylist()) == astro_name


# ---------------------------------------------------------------------------
# Mutations: append / update / delete stay faithful to the data
# ---------------------------------------------------------------------------


def test_mutation_sequence_fidelity(tmp_path):
    """append + update-window + delete leaves rows exactly as astropy reads."""
    table = {
        "ID": np.array([1, 2, 3], dtype=np.int32),
        "VAL": np.array([0.1, 0.2, 0.3], dtype=np.float64),
    }
    path = str(tmp_path / "mut.fits")
    torchfits.table.write(path, table, overwrite=True)

    torchfits.table.append_rows(
        path,
        {"ID": np.array([4, 5], dtype=np.int32), "VAL": np.array([0.4, 0.5])},
        hdu=1,
    )
    torchfits.table.update_rows(path, {"VAL": np.array([9.9, 8.8])}, slice(0, 2), hdu=1)
    torchfits.table.delete_rows(path, 2, hdu=1)

    from astropy.io import fits

    with fits.open(path) as hdul:
        astro_id = np.asarray(hdul[1].data["ID"]).astype(np.int32)
        astro_val = np.asarray(hdul[1].data["VAL"]).astype(np.float64)
    # delete_rows uses 0-based row indices: row 2 is the third row (ID=3).
    assert np.array_equal(astro_id, np.array([1, 2, 4, 5], dtype=np.int32))
    assert np.allclose(astro_val, np.array([9.9, 8.8, 0.4, 0.5]))

    for mmap in (True, False):
        out = torchfits.read(path, hdu=1, mmap=mmap)
        assert torch.equal(out["ID"].squeeze(-1), torch.as_tensor(astro_id))
        assert torch.equal(out["VAL"].squeeze(-1), torch.as_tensor(astro_val))


def test_ascii_string_width_handling_matches_astropy(tmp_path):
    """Wide strings on append/update behave exactly like astropy (truncate
    to the ASCII field width)."""
    table = {"NAME": ["abcdef"]}  # width-6 ASCII column
    path = str(tmp_path / "ascii_width.fits")
    torchfits.table.write(path, table, table_type="ascii", overwrite=True)

    torchfits.table.append_rows(path, {"NAME": ["x" * 12]}, hdu=1)
    torchfits.table.update_rows(path, {"NAME": ["y" * 10]}, slice(0, 1), hdu=1)

    from astropy.io import fits

    with fits.open(path) as hdul:
        astro = list(hdul[1].data["NAME"])
    assert astro[0] == "y" * 6  # truncated to field width
    assert astro[1] == "x" * 6

    arrow = torchfits.table.read(path, hdu=1)
    assert list(arrow.column("NAME").to_pylist()) == astro


def test_cpp_append_rows_truncates_wide_strings_to_field_width(tmp_path):
    """The C++ writer must truncate to the ASCII field width itself (the
    Python layer pre-truncates, so this guards the raw binding)."""
    table = {"NAME": ["abcdef"]}  # width-6 ASCII column
    path = str(tmp_path / "ascii_raw.fits")
    torchfits.table.write(path, table, table_type="ascii", overwrite=True)

    torchfits._C.append_fits_table_rows(path, 1, {"NAME": ["z" * 12]})

    from astropy.io import fits

    with fits.open(path) as hdul:
        assert list(hdul[1].data["NAME"]) == ["abcdef", "z" * 6]


def test_append_rows_repeat_mismatch_is_rejected(tmp_path):
    """2D payload width != column repeat must raise instead of interleaving
    cells and corrupting every following row."""
    table = {"A": np.zeros(3, dtype=np.int32)}
    path = str(tmp_path / "repeat.fits")
    torchfits.table.write(path, table, overwrite=True)

    with pytest.raises(RuntimeError, match="repeat mismatch"):
        torchfits.table.append_rows(
            path, {"A": np.zeros((2, 2), dtype=np.int32)}, hdu=1
        )
    with pytest.raises(RuntimeError, match="repeat mismatch"):
        torchfits.table.update_rows(
            path, {"A": np.zeros((2, 2), dtype=np.int32)}, slice(0, 2), hdu=1
        )

    # File untouched: still exactly the original rows.
    out = torchfits.read(path, hdu=1)
    assert torch.equal(out["A"].squeeze(-1), torch.zeros(3, dtype=torch.int32))


# ---------------------------------------------------------------------------
# Filtered (gather) path: BIT/LOGICAL decode fidelity
# ---------------------------------------------------------------------------


def _write_bit_table(path: str) -> None:
    """Binary table with BIT (8X), LOGICAL (L), STRING and numeric columns."""
    from astropy.io import fits

    n = 9
    bits = np.array([i % 255 for i in range(n)], dtype=np.uint8)
    flags = np.array([i % 2 == 0 for i in range(n)], dtype=np.bool_)
    nums = np.arange(n, dtype=np.int32)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="BITS", format="8X", array=bits.reshape(n, 1)),
            fits.Column(name="FLAG", format="L", array=flags),
            fits.Column(
                name="NAME",
                format="4A",
                array=["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii"],
            ),
            fits.Column(name="NUM", format="J", array=nums),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def test_filtered_gather_bit_and_logical_decode_matches_mask(tmp_path):
    """Gathered BIT/LOGICAL columns from the filtered path equal the full-read
    columns masked by the numeric predicate (MSB-first bit unpacking)."""
    import torchfits._C as cpp
    from torchfits._table.read import _compile_where_to_simple_predicates

    path = str(tmp_path / "bit_gather.fits")
    _write_bit_table(path)
    full = torchfits.read(path, hdu=1, mmap=False)
    keep = full["NUM"] > 3
    want_bits = full["BITS"][keep]
    want_flag = full["FLAG"][keep]

    predicates = _compile_where_to_simple_predicates("NUM > 3")
    assert predicates is not None
    got = cpp.read_fits_table_filtered(
        path, 1, ["BITS", "FLAG", "NUM"], list(predicates)
    )
    assert torch.equal(got["BITS"], want_bits)
    assert torch.equal(got["FLAG"], want_flag)
    assert torch.equal(got["NUM"], full["NUM"][keep])


def test_filtered_rejects_unsupported_column_types(tmp_path):
    """Filtering on LOGICAL/STRING columns raises instead of silently
    matching nothing (garbage in, garbage out)."""
    import torchfits._C as cpp

    path = str(tmp_path / "bit_gather.fits")
    _write_bit_table(path)
    with pytest.raises(RuntimeError, match="Unsupported filter value type"):
        cpp.read_fits_table_filtered(path, 1, ["NUM"], [("FLAG", "==", True)])
    with pytest.raises(RuntimeError, match="Unsupported filter value type"):
        cpp.read_fits_table_filtered(path, 1, ["NUM"], [("NAME", "==", "aa")])
    # Numeric filters on the same table still work.
    out = cpp.read_fits_table_filtered(path, 1, ["NUM"], [("NUM", ">=", 5)])
    assert torch.equal(out["NUM"], torch.arange(5, 9, dtype=torch.int32))


def test_where_on_numeric_column_matches_full_read_mask(tmp_path):
    """Public where= returns exactly the rows the torch mask selects."""
    table = {
        "ID": np.arange(20, dtype=np.int32),
        "V": np.linspace(0, 1, 20).astype(np.float64),
    }
    path = str(tmp_path / "where.fits")
    torchfits.table.write(path, table, overwrite=True)

    got = torchfits.table.read_torch(path, hdu=1, columns=["ID", "V"], where="ID >= 10")
    full = torchfits.read(path, hdu=1)
    mask = full["ID"] >= 10
    assert torch.equal(got["ID"], full["ID"][mask])
    assert torch.equal(got["V"], full["V"][mask])


# ---------------------------------------------------------------------------
# Chunked read path (large images)
# ---------------------------------------------------------------------------


def test_chunked_read_large_uint16_exact(tmp_path):
    """Images above the 128 MB chunking threshold keep element-size stepping
    (TUSHORT) exact — regression guard for chunked-read corruption."""
    from astropy.io import fits

    # 8192x8192 uint16 = 134 MB > 128 MB chunk threshold (64M-px chunks).
    shape = (8192, 8192)
    rng = np.random.default_rng(5)
    data = rng.integers(0, 65536, size=shape[0] * shape[1], dtype=np.uint16).reshape(
        shape
    )
    path = str(tmp_path / "big_u16.fits")
    fits.HDUList([fits.PrimaryHDU(data)]).writeto(path, overwrite=True)

    out = torchfits.read(path)
    assert out.dtype == torch.uint16
    assert torch.equal(out, torch.as_tensor(data))
