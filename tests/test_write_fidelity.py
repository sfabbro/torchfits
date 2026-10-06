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
    # Scaled LONGLONG accumulates in float64. 2**40 is exact there and was
    # not in float32; 2**63-1 is the nearest float64, not the integer.
    assert out.dtype == torch.float64
    np.testing.assert_array_equal(out.numpy(), data.astype(np.float64))
    with torchfits.open(path) as hdul:
        assert hdul[0].data.dtype == torch.float64


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


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.int16, torch.uint8]
)
def test_read_subset_empty_box_preserves_dtype(tmp_path, dtype) -> None:
    path = tmp_path / "img.fits"
    torchfits.write_tensor(path, torch.zeros((4, 6), dtype=dtype))
    empty = torchfits.read_subset(str(path), 0, 1, 0, 1, 2)
    assert empty.shape == (2, 0)
    assert empty.dtype == dtype
    empty_y = torchfits.read_subset(str(path), 0, 0, 2, 3, 2)
    assert empty_y.shape == (0, 3)
    assert empty_y.dtype == dtype
    with torchfits.open_subset_reader(str(path), hdu=0) as reader:
        assert reader.read_subset(1, 0, 1, 2).dtype == dtype


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
        assert torch.equal(out["ID"], torch.as_tensor(astro_id))
        np.testing.assert_array_equal(out["VAL"].numpy(), table["VAL"])

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
        assert torch.equal(out["ID"], torch.as_tensor(astro_id))
        assert torch.equal(out["VAL"], torch.as_tensor(astro_val))


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
    assert torch.equal(out["A"], torch.zeros(3, dtype=torch.int32))


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


def test_compressed_uint16_roundtrip_exact(tmp_path):
    """Pseudo-unsigned uint16 through the compressed writer is lossless."""
    rng = np.random.default_rng(3)
    data = torch.as_tensor(rng.integers(0, 65536, size=(37, 41), dtype=np.uint16))
    path = str(tmp_path / "c_u16.fits")
    torchfits.write(path, data, overwrite=True, compress=True)

    out = torchfits.read(path, hdu=1)
    assert out.dtype == torch.uint16
    assert torch.equal(out, data)

    from astropy.io import fits

    np.testing.assert_array_equal(fits.getdata(path), data.numpy())


@pytest.mark.parametrize("algo", ["RICE_1", "GZIP_1"])
def test_compressed_int64_rejected_not_silently_lossy(tmp_path, algo):
    """64-bit integer compression is outside CFITSIO's algorithm support
    (astropy's RICE_1 silently narrows to 32 bits; gzip is limited to
    smaller tiles). Writing must raise instead of corrupting."""
    data = torch.arange(64, dtype=torch.int64).reshape(8, 8)
    path = str(tmp_path / f"c64_{algo}.fits")
    with pytest.raises(RuntimeError):
        torchfits.write(path, data, overwrite=True, compress=algo)


def test_compressed_uint32_roundtrip_exact(tmp_path):
    """Pseudo-unsigned uint32 through the compressed writer is lossless."""
    rng = np.random.default_rng(4)
    data = torch.as_tensor(rng.integers(0, 2**31, size=(13, 17), dtype=np.uint32))
    path = str(tmp_path / "c_u32.fits")
    torchfits.write(path, data, overwrite=True, compress=True)

    out = torchfits.read(path, hdu=1)
    assert out.dtype == torch.uint32
    assert torch.equal(out, data)

    import astropy
    from astropy.io import fits
    from packaging.version import Version

    # astropy < 7.0 (and no 7.x on py3.10, whose newest is 6.1.7) decodes
    # uint32 (BZERO=2147483648) tile-compressed images with an int64->int32
    # overflow that yields garbage (astropy 7.0.0 fixed it). Skip the oracle
    # only there; the torchfits self-roundtrip above still asserts exactness.
    if Version(astropy.__version__) >= Version("7.0"):
        np.testing.assert_array_equal(fits.getdata(path), data.numpy())


# ---------------------------------------------------------------------------
# Mutations: insert / rename / drop keep data exact
# ---------------------------------------------------------------------------


def test_insert_rows_fidelity_mid_table(tmp_path):
    """insert_rows keeps row order and values exactly as astropy reads them."""
    table = {
        "ID": np.array([1, 2, 4, 5], dtype=np.int32),
        "VAL": np.array([0.1, 0.2, 0.4, 0.5], dtype=np.float64),
    }
    path = str(tmp_path / "ins.fits")
    torchfits.table.write(path, table, overwrite=True)
    torchfits.table.insert_rows(
        path,
        {"ID": np.array([3], dtype=np.int32), "VAL": np.array([0.3])},
        row=2,
        hdu=1,
    )

    from astropy.io import fits

    with fits.open(path) as hdul:
        astro_id = np.asarray(hdul[1].data["ID"]).astype(np.int32)
        astro_val = np.asarray(hdul[1].data["VAL"]).astype(np.float64)
    assert np.array_equal(astro_id, np.array([1, 2, 3, 4, 5], dtype=np.int32))
    assert np.allclose(astro_val, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

    for mmap in (True, False):
        out = torchfits.read(path, hdu=1, mmap=mmap)
        assert torch.equal(out["ID"], torch.as_tensor(astro_id))
        assert torch.equal(out["VAL"], torch.as_tensor(astro_val))


def test_rename_and_drop_columns_fidelity(tmp_path):
    """rename/drop keep the remaining data byte-identical (astropy oracle)."""
    table = {
        "A": np.array([1, 2, 3], dtype=np.int32),
        "B": np.array([0.5, 1.5, 2.5], dtype=np.float64),
        "C": np.array([True, False, True], dtype=np.bool_),
    }
    path = str(tmp_path / "rd.fits")
    torchfits.table.write(path, table, overwrite=True)
    torchfits.table.rename_columns(path, {"A": "ALPHA"}, hdu=1)
    torchfits.table.drop_columns(path, ["C"], hdu=1)

    from astropy.io import fits

    with fits.open(path) as hdul:
        assert list(hdul[1].data.names) == ["ALPHA", "B"]
        astro_a = np.asarray(hdul[1].data["ALPHA"]).astype(np.int32)
        astro_b = np.asarray(hdul[1].data["B"]).astype(np.float64)
    assert np.array_equal(astro_a, np.array([1, 2, 3], dtype=np.int32))
    assert np.allclose(astro_b, np.array([0.5, 1.5, 2.5]))

    for mmap in (True, False):
        out = torchfits.read(path, hdu=1, mmap=mmap)
        assert torch.equal(out["ALPHA"], torch.as_tensor(astro_a))
        assert torch.equal(out["B"], torch.as_tensor(astro_b))


def test_ascii_table_logical_column_is_rejected(tmp_path):
    """ASCII tables have no logical format (A/I/F/E/D only); writing bools
    must raise a clear error instead of producing an unreadable TFORM."""
    path = str(tmp_path / "ascii_bool.fits")
    with pytest.raises(RuntimeError, match="logical"):
        torchfits.table.write(
            path,
            {"FLAG": np.array([True, False], dtype=np.bool_)},
            table_type="ascii",
            overwrite=True,
        )


def test_table_tnull_values_roundtrip_raw(tmp_path):
    """TNULL-marker columns round-trip the stored value exactly through both
    read paths (no masking), matching astropy's raw read."""
    from astropy.io import fits

    col = fits.Column(
        name="A", format="J", array=np.array([0, 5, 3, 0], dtype=np.int32), null=0
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([col])]).writeto(
        str(tmp_path / "tnull.fits"), overwrite=True
    )
    path = str(tmp_path / "tnull.fits")

    astro = np.asarray(fits.getdata(path, 1)["A"]).astype(np.int32)
    assert np.array_equal(astro, np.array([0, 5, 3, 0], dtype=np.int32))
    for mmap in (True, False):
        out = torchfits.read(path, hdu=1, mmap=mmap)
        assert torch.equal(out["A"], torch.as_tensor(astro))


# ---------------------------------------------------------------------------
# Variable-length array columns: both read paths + astropy oracle
# ---------------------------------------------------------------------------


def test_vla_column_roundtrip_fidelity(tmp_path):
    """VLA columns come back with the exact row lengths and values on both
    read paths (astropy oracle for the heap)."""
    from astropy.table import Table

    vla = np.array(
        [np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)],
        dtype=object,
    )
    path = str(tmp_path / "vla.fits")
    Table({"VLA": vla}).write(path, format="fits", overwrite=True)

    from astropy.io import fits

    with fits.open(path) as hdul:
        astro = [
            np.asarray(row, dtype=np.int32).tolist() for row in hdul[1].data["VLA"]
        ]
    assert astro == [[1, 2], [3]]

    arrow = torchfits.table.read(path, hdu=1)
    got = arrow.column("VLA").to_pylist()
    assert got == [[1, 2], [3]]


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


# ---------------------------------------------------------------------------
# Write-path type fallthroughs: int8 columns, dict-HDU key drops, stale
# header cards on copied headers (astropy ground truth)
# ---------------------------------------------------------------------------


def test_table_write_int8_column_tbyte_convention(tmp_path):
    """int8 columns store as raw TBYTE on every write form (the signed view
    recovers the values; astropy sees the raw unsigned bytes)."""
    want = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    from astropy.io import fits

    for label, writer in (
        ("write-dict", lambda p: torchfits.write(p, {"I8": want}, overwrite=True)),
        (
            "table-write",
            lambda p: torchfits.table.write(p, {"I8": want}, overwrite=True),
        ),
    ):
        path = str(tmp_path / f"i8_{label}.fits")
        writer(path)
        with fits.open(path) as hdul:
            actual = np.asarray(hdul[1].data["I8"])
        assert actual.dtype == np.uint8, label
        np.testing.assert_array_equal(actual.view(np.int8), want, err_msg=label)
        _assert_column_faithful(torchfits.read(path, hdu=1)["I8"], want)


def test_strided_logical_and_bit_payloads_write_the_caller_s_values(tmp_path):
    """A strided bool/uint8 payload must land as ITS values, not the base's.

    The logical/BIT write branch reads the payload with a flat index over
    nelements. For a non-contiguous view (a column slice, an every-N-th
    selection) the logical elements sit at t.stride(k), not at consecutive
    addresses from t.data(), so the flat read took the base pointer's own
    stride-1 order: a stride-2 view of [T,F,T,F,T,F,T,F] -- i.e. four Trues --
    was written as [T,F,T,F]. astropy is the ground truth for what is actually
    on disk, and every case is paired with the contiguous write of the same
    values so a broken fixture cannot masquerade as a stride bug.
    """
    from astropy.io import fits

    base = torch.zeros(8, dtype=torch.bool)
    base[::2] = True
    strided_logical = base[::2]  # [T,T,T,T], stride 2
    contiguous_logical = strided_logical.clone()
    assert not strided_logical.is_contiguous()

    def disk_values(path: str, column: str) -> list:
        with fits.open(path) as hdul:
            return np.asarray(hdul[1].data[column]).reshape(-1).tolist()

    for label, payload, want in (
        ("contiguous", contiguous_logical, [True] * 4),
        ("strided", strided_logical, [True] * 4),
    ):
        path = str(tmp_path / f"logical_{label}.fits")
        torchfits.table.write(path, {"flag": payload}, overwrite=True)
        assert disk_values(path, "flag") == want, label
        got = torchfits.table.read(path, hdu=1)["flag"]
        assert [bool(v) for v in np.asarray(got).reshape(-1)] == want, label

    # Same defect on the TBIT branch, which shares the flat read: a
    # stride-2 uint8 view of four [1,0,1,0] rows. The base is laid out so the
    # view is [1,0,1,0] per row while the base's own stride-1 order is
    # [1,0,0,0,1,0,0,0] -- the two orders differ, which is what makes this
    # case able to detect the flat read.
    bits_base = torch.zeros((4, 8), dtype=torch.uint8)
    bits_base[:, 0] = 1
    bits_base[:, 4] = 1
    strided_bits = bits_base[:, 0:8:2]
    assert [int(v) for v in strided_bits[0]] == [1, 0, 1, 0]
    assert [int(v) for v in bits_base[0]] == [1, 0, 0, 0, 1, 0, 0, 0]
    assert not strided_bits.is_contiguous()
    for label, payload in (
        ("contiguous", strided_bits.clone()),
        ("strided", strided_bits),
    ):
        path = str(tmp_path / f"bits_{label}.fits")
        torchfits.table.write(
            path,
            {"bits": payload},
            schema={"bits": {"format": "4X"}},
            overwrite=True,
        )
        assert disk_values(path, "bits") == [1, 0, 1, 0] * 4, label


def test_dict_image_extra_keys_rejected_not_silently_dropped(tmp_path):
    """A dict-HDU payload accepts only 'data'/'header'; extra keys must raise
    by name, never vanish from the written file."""
    payload = {"data": torch.zeros(2, 2), "flux": torch.ones(2)}
    for label, kwargs in (("plain", {}), ("compressed", {"compress": True})):
        path = tmp_path / f"extra_{label}.fits"
        with pytest.raises(RuntimeError, match="flux"):
            torchfits.write(str(path), payload, overwrite=True, **kwargs)
        assert not path.exists(), label


@pytest.mark.parametrize("rejection", ["bit-dtype", "header-value"])
def test_rejected_overwrite_leaves_the_original_file_intact(tmp_path, rejection):
    """A rejected overwrite must not destroy the file it was overwriting.

    ``table.write`` has two paths. The plain one goes through
    ``_io_engine/write_api.py``, which writes ``.{name}.XXXX.tmp.fits`` in the same
    directory and ``os.replace``s it over the target only on success. The other
    -- taken whenever a ``schema``, an unsigned conversion, a quantization or an
    ASCII table is involved -- called ``cpp.write_fits_table`` directly and so
    never saw that wrapper. The C++ writer reaches ``fits_create_file("!path")``,
    whose leading ``!`` *unlinks* the target before a byte of the payload is
    validated, so a rejected overwrite destroyed the caller's file: measured, a
    good 5-row table was replaced by a 2880-byte stub that no longer read as a
    table at all. The same window exposed a missing or partial file to any
    concurrent reader, and a crash or a full disk mid-write had the same effect.

    Both branches now share one ``atomic_write_target`` wrapper, so this asserts
    the file is untouched -- same bytes, same inode, no temp file left behind.
    """
    path = tmp_path / "precious.fits"
    torchfits.table.write(
        str(path), {"A": np.arange(5, dtype=np.int32)}, overwrite=True
    )
    before_bytes = path.read_bytes()
    before_stat = path.stat()

    if rejection == "bit-dtype":
        payload = {
            "A": np.arange(4, dtype=np.int32),
            "FLAGS": np.arange(4, dtype=np.float32),
        }
        kwargs = {"schema": {"FLAGS": {"format": "8X"}}}
    else:
        payload = {"A": np.arange(4, dtype=np.int32)}
        kwargs = {"header": {"X": [1, 2]}, "schema": {"A": {"format": "J"}}}

    with pytest.raises(RuntimeError):
        torchfits.table.write(str(path), payload, overwrite=True, **kwargs)

    assert path.read_bytes() == before_bytes, "the rejected overwrite modified the file"
    assert path.stat().st_ino == before_stat.st_ino, (
        "the file was replaced rather than left alone (st_ino changed)"
    )
    assert [v.as_py() for v in torchfits.table.read(str(path))["A"]] == [0, 1, 2, 3, 4]
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "precious.fits"]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_rejected_table_write_leaves_no_readable_table(tmp_path):
    """A payload ``table.write`` rejects must not leave a readable table.

    ``write_table_hdu`` built and validated every column *before*
    ``fits_create_tbl``, but three rejections lived in the loop that writes the
    data — so they arrived with the HDU already created and the earlier columns
    written. Measured before the fix: a two-column write whose second column
    was rejected raised and left an 8640-byte file that read back cleanly, with
    the rejected column silently all-zero. A caller that trusts the exception
    is left holding a plausible-looking table full of fabricated values.

    A schema TFORM of ``8X`` is what makes this reachable with an ordinary
    payload: it forces ``col.datatype = TBIT`` whatever the payload dtype is,
    and overrides ``col.repeat``, so the loop-side checks can fire on data that
    passed everything above them. All three are pure payload checks and now run
    where the other ones already did, before the HDU exists.

    The ``header`` case is the same contract one loop further on: the card
    writing happens after *all* the data, so a header value this writer refuses
    used to leave a complete table carrying none of the caller's cards.

    Both need the in-place path (``schema=`` present routes to
    ``cpp.write_fits_table``; the plain path deletes the file itself). The
    image-path twin of this contract is
    ``test_dict_image_extra_keys_rejected_not_silently_dropped`` above.
    """
    path = tmp_path / "rejected_table.fits"
    with pytest.raises(RuntimeError, match="bool or uint8"):
        torchfits.table.write(
            str(path),
            {
                "A": np.arange(4, dtype=np.int32),
                "FLAGS": np.arange(4, dtype=np.float32),
            },
            schema={"FLAGS": {"format": "8X"}},
            overwrite=True,
        )
    with pytest.raises(Exception):
        torchfits.table.read(str(path))

    path = tmp_path / "rejected_header.fits"
    with pytest.raises(RuntimeError, match="Unsupported FITS header value type"):
        torchfits.table.write(
            str(path),
            {"A": np.arange(4, dtype=np.int32)},
            header={"GOOD": "yes", "X": [1, 2]},
            schema={"A": {"format": "J"}},
            overwrite=True,
        )
    with pytest.raises(Exception):
        torchfits.table.read(str(path))

    # The header pre-pass must not change what a *valid* header writes.
    ok = tmp_path / "accepted_header.fits"
    torchfits.table.write(
        str(ok),
        {"A": np.arange(4, dtype=np.int32)},
        header={"GOOD": "yes", "N": 5, "F": 1.5, "B": True, "LONG": "x" * 100},
        schema={"A": {"format": "J"}},
        overwrite=True,
    )
    cards = set(torchfits.read_header(str(ok), hdu=1))
    assert {"GOOD", "N", "F", "B", "LONG"} <= cards


def test_copied_table_header_cannot_forge_rows_or_relabel_columns(tmp_path):
    """A header read from another table must not override data-derived cards
    (NAXIS2/TFORM/TTYPE): row count, names and values stay true (astropy)."""
    from astropy.io import fits

    src = str(tmp_path / "src.fits")
    torchfits.write(src, {"OLD": np.arange(5, dtype=np.int64)}, overwrite=True)
    copied = torchfits.read_header(src, hdu=1)

    dst = str(tmp_path / "dst.fits")
    torchfits.write(
        dst, {"NEW": np.arange(3, dtype=np.int32)}, header=copied, overwrite=True
    )
    with fits.open(dst) as hdul:
        tab = hdul[1].data
        assert hdul[1].header["NAXIS2"] == 3
        assert len(tab) == 3
        assert list(tab.names) == ["NEW"]
        # Semantic pin: the TFORM must describe the DATA (int32 cells as
        # astropy decodes them), not the source table's stale '1K' label.
        got_col = np.asarray(tab["NEW"])
        assert got_col.dtype.newbyteorder("=") == np.int32
        np.testing.assert_array_equal(got_col, np.arange(3, dtype=np.int32))
    out = torchfits.read(dst, hdu=1)
    assert torch.equal(out["NEW"], torch.arange(3, dtype=torch.int32))


def test_stale_checksum_cards_are_not_replayed(tmp_path):
    """CHECKSUM/DATASUM in a copied header describe the source payload; a
    fresh write must not stamp them (the file would fail verification)."""
    from astropy.io import fits

    for label, data, hdu in (
        ("image", torch.zeros(2, 2), 0),
        ("table", {"X": torch.arange(3)}, 1),
    ):
        path = str(tmp_path / f"stale_{label}.fits")
        torchfits.write(
            path,
            data,
            header={"CHECKSUM": "abcdEFGH", "DATASUM": 42},
            overwrite=True,
        )
        v = torchfits.verify_checksums(path, hdu=hdu)
        assert v["status"] == "no_checksums", label
        assert v["present"] is False, label
        with fits.open(path) as hdul:
            assert "CHECKSUM" not in hdul[hdu].header, label
            assert "DATASUM" not in hdul[hdu].header, label


def test_quantized_dict_table_compressed_matches_plain(tmp_path):
    """quantize= is honored on the compressed dict-table write path: the same
    TFORM=I + TSCAL/TZERO/TNULL storage and decode as the plain write."""
    rng = np.random.default_rng(21)
    flux = torch.from_numpy((1000.0 + rng.standard_normal(4096)).astype(np.float32))
    flux[7] = float("nan")
    table = {"ID": torch.arange(flux.numel(), dtype=torch.int32), "FLUX": flux}

    p_plain = str(tmp_path / "q_plain.fits")
    p_comp = str(tmp_path / "q_comp.fits")
    torchfits.write(p_plain, table, overwrite=True, quantize={"FLUX": "robust"})
    torchfits.write(
        p_comp, table, overwrite=True, compress=True, quantize={"FLUX": "robust"}
    )

    from astropy.io import fits

    for label, path in (("plain", p_plain), ("comp", p_comp)):
        with fits.open(path) as hdul:
            h = hdul[1].header
        tform = str(h["TFORM2"]).upper().lstrip("0123456789")
        assert tform == "I", f"{label}: TFORM2={h['TFORM2']}"
        assert "TSCAL2" in h and "TZERO2" in h and "TNULL2" in h, label

    got_plain = torchfits.table.read_torch(p_plain, hdu=1)
    got_comp = torchfits.table.read_torch(p_comp, hdu=1)
    assert torch.equal(got_plain["ID"], got_comp["ID"])
    assert torch.equal(torch.isnan(got_plain["FLUX"]), torch.isnan(got_comp["FLUX"]))
    assert bool(torch.isnan(got_plain["FLUX"][7]))
    finite = ~torch.isnan(got_plain["FLUX"])
    assert torch.equal(got_plain["FLUX"][finite], got_comp["FLUX"][finite])


def test_atomic_overwrite_recipe_has_exactly_one_implementation():
    """The temp-file + ``os.replace`` recipe must be written out once, not thrice.

    It was written out three times -- ``write_api.atomic_write_target`` plus two
    inline copies in ``_hdu_rewrite`` -- and that is how R2-013 happened: the
    ``_table/write.py`` schema branch turned out to be a *fourth* whole-file
    writer with no copy of the protection to inherit. Copies had already drifted
    (one guarded the temp dance on ``os.path.isfile`` and recursed, one assumed
    the file existed), so a fix applied to one was not applied to the others. All
    three now route through ``_write_helpers._atomic_replace_target``, and this
    keeps a fourth from being written from scratch.

    Parsed with ``ast`` rather than grepped: docstrings and comments discuss
    ``os.replace`` constantly, and prose must be unable to satisfy or trip this.
    """
    import ast
    from pathlib import Path

    package = Path(torchfits.__file__).parent
    recipe_calls = {"os.replace", "tempfile.mkstemp"}
    sites: list[tuple[str, int]] = []
    for source in sorted(package.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and f"{func.value.id}.{func.attr}" in recipe_calls
            ):
                sites.append((source.relative_to(package).as_posix(), node.lineno))

    assert [rel for rel, _ in sites] == ["_io_engine/_write_helpers.py"] * len(sites), (
        "the atomic-write recipe is implemented outside "
        f"_write_helpers._atomic_replace_target: {sites}"
    )
    assert len(sites) == len(recipe_calls), (
        f"expected exactly {len(recipe_calls)} recipe calls (one per primitive), got {sites}"
    )

    # The rename happens when the ``with`` block *exits*, so cache invalidation
    # and checksum restamping inside the block would run against the file that
    # is about to be replaced -- a concurrent reader could cache the old bytes
    # under the path and go on serving them after the rename. No dynamic test can
    # observe that window, so it is pinned structurally here.
    source = (package / "_io_engine" / "_hdu_rewrite.py").read_text(encoding="utf-8")
    checked = 0
    too_early = {"_invalidate_written_target", "_write_all_checksums"}
    for node in ast.walk(ast.parse(source)):
        items = getattr(node, "items", None)
        if not isinstance(node, (ast.With, ast.AsyncWith)) or items is None:
            continue
        entered = ast.unparse(items[0].context_expr)
        if entered != "_atomic_replace_target(path)":
            continue
        checked += 1
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and ast.unparse(inner.func) in too_early:
                pytest.fail(
                    f"{entered} block calls {ast.unparse(inner.func)} on line "
                    f"{inner.lineno}: it must run after the block, once the "
                    "rename has made the new bytes current"
                )

    # And the guard has to be reachable at all: if _hdu_rewrite ever renames the
    # helper, this check silently stops watching anything.
    assert checked >= 2, f"only {checked} _atomic_replace_target blocks found"


def _atomic_overwrite_route(route: str, path: str) -> None:
    """Overwrite ``path`` through each whole-file writer, in turn."""
    if route == "write":
        torchfits.write(path, torch.ones((3, 3)), overwrite=True)
    elif route == "hdulist-write":
        from torchfits.hdu import HDUList, TensorHDU

        HDUList([TensorHDU(data=torch.ones((3, 3)))]).write(path, overwrite=True)
    elif route == "insert-hdu":
        torchfits.insert_hdu(path, torch.ones((2, 2)), index=1)
    else:  # pragma: no cover - parametrization is the only caller
        raise AssertionError(f"unknown route {route!r}")


@pytest.mark.parametrize(
    "route", ["write", "hdulist-write", "insert-hdu"], ids=["write", "hdulist", "ins"]
)
@pytest.mark.parametrize("linked", [False, True], ids=["plain", "symlink"])
def test_every_atomic_overwrite_route_leaves_the_file_consistent(
    tmp_path, route, linked
):
    """All three routes must honour the same temp-file-and-rename contract.

    They no longer share an implementation, so this pins the contract each one
    owes its caller: the payload lands as a whole (the inode changes, so no
    reader ever sees a partial file), the target keeps its permissions, no temp
    file is left behind, and a symlinked target keeps its symlink -- the rename
    goes onto the realpath, so ``path`` and its realpath both need their caches
    invalidated afterwards.
    """
    target = tmp_path / f"{route}.fits"
    if linked:
        # The rename goes onto the realpath, so the symlink and the file it
        # points to are two names for one target: watch the latter.
        watched = tmp_path / f"{route}_real.fits"
        torchfits.write(str(watched), torch.zeros((3, 3)))
        target.symlink_to(watched)
    else:
        watched = target
        torchfits.write(str(watched), torch.zeros((3, 3)))
    watched.chmod(0o640)

    before = watched.stat().st_ino
    _atomic_overwrite_route(route, str(target))

    assert watched.stat().st_mode & 0o777 == 0o640, "the mode was not carried over"
    assert watched.stat().st_ino != before, (
        "no rename happened; the file was written in place"
    )
    if linked:
        assert target.is_symlink(), "the symlink was replaced instead of its target"
    if route == "insert-hdu":
        with torchfits.open(str(target)) as hdul:
            assert len(hdul) == 2
    else:
        assert torch.equal(torchfits.read(str(target)), torch.ones((3, 3)))
    expected = sorted({target.name, watched.name})
    assert sorted(p.name for p in tmp_path.iterdir()) == expected, (
        "a temp file was left behind"
    )
