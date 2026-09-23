"""WHERE row-set equivalence matrix (r5a).

mmap on/off x read / read_torch / scan x TNULL / unsigned / out-of-range
literals: every strategy must select identical row sets
(``where-prefer-mask``, ``tnull-read-torch``).
"""

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits

import torchfits.table as table


@pytest.fixture(scope="module")
def matrix_fits(tmp_path_factory):
    """One binary table exercising TNULL, unsigned and scaled conventions.

    Row index is the ``ID`` column so every strategy can be compared on the
    selected row set alone.
    """
    path = str(tmp_path_factory.mktemp("matrix") / "matrix.fits")
    n = 12
    ids = np.arange(n, dtype=np.int32)
    # Physical uint16 values (stored as int16 with TZERO=32768).
    v16 = np.array(
        [0, 1, 32768, 40000, 65535, 100, 2, 3, 50000, 4, 5, 6], dtype=np.int64
    )
    stored16 = (v16 - 32768).astype(np.int16)
    # Raw integer TNULL sentinel at rows 2 and 7 (no TSCAL -> sentinel kept).
    nulj = np.array(
        [0, 5, -9999, 10, -3, 20, 30, -9999, 40, -1, 50, 60], dtype=np.int32
    )
    f32 = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0, -0.5, 2.0, 3.0, 4.0, 5.0, 0.5, 0.5],
        dtype=np.float32,
    )
    sclr_stored = np.arange(n, dtype=np.int16)  # physical = 0.5*i + 100

    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ID", format="J", array=ids),
            fits.Column(name="V16", format="I", array=stored16),
            fits.Column(name="NULJ", format="J", array=nulj, null=-9999),
            fits.Column(name="F32", format="E", array=f32),
            fits.Column(name="SCLR", format="I", array=sclr_stored),
        ]
    )
    hdu.header["TZERO2"] = 32768
    hdu.header["TSCAL5"] = 0.5
    hdu.header["TZERO5"] = 100.0
    hdu.writeto(path)
    return path


def _read_ids(path, where, **kw):
    return sorted(table.read(path, where=where, **kw)["ID"].to_pylist())


def _scan_ids(path, where, **kw):
    batches = list(table.scan(path, where=where, **kw))
    if not batches:
        return []
    return sorted(pa.Table.from_batches(batches)["ID"].to_pylist())


def _read_torch_ids(path, where, **kw):
    data = table.read_torch(path, columns=["ID"], where=where, **kw)
    return sorted(data["ID"].tolist())


# (predicate, expected row set) — the ground truth every strategy must match.
SIMPLE_CASES = [
    ("ID >= 2 AND ID < 8", [2, 3, 4, 5, 6, 7]),
    ("V16 > 40000", [4, 8]),
    ("V16 > 70000", []),  # out-of-range literal for uint16 storage
    ("V16 == 0", [0]),
    ("NULJ < 0", [4, 9]),  # TNULL sentinel rows 2/7 never match
    ("NULJ == -9999", []),  # sentinel literal never matches either
    ("NULJ > -5000", [0, 1, 3, 4, 5, 6, 8, 9, 10, 11]),
    ("F32 >= 0.5", [2, 3, 4, 6, 7, 8, 9, 10, 11]),
    ("SCLR > 102.0", [5, 6, 7, 8, 9, 10, 11]),  # scaled column physical values
    ("ID BETWEEN 3 AND 6", [3, 4, 5, 6]),
]

FULL_DIALECT_CASES = SIMPLE_CASES + [
    ("NULJ IN (-3, -1)", [4, 9]),
    ("NOT ID == 2", [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ("NULJ IS NULL", [2, 7]),
    ("NULJ IS NOT NULL", [0, 1, 3, 4, 5, 6, 8, 9, 10, 11]),
]


@pytest.mark.parametrize("pred,expected", FULL_DIALECT_CASES)
def test_matrix_read_strategies_agree(matrix_fits, pred, expected):
    """read()/scan() row sets are identical across engine and mmap settings."""
    assert _read_ids(matrix_fits, pred, mmap=True) == expected
    assert _read_ids(matrix_fits, pred, mmap=False) == expected
    assert _read_ids(matrix_fits, pred, backend="cpp") == expected
    assert _read_ids(matrix_fits, pred, backend="torch") == expected
    assert _scan_ids(matrix_fits, pred, mmap=True) == expected
    assert _scan_ids(matrix_fits, pred, mmap=False) == expected


@pytest.mark.parametrize("pred,expected", SIMPLE_CASES)
def test_matrix_read_torch_agrees(matrix_fits, pred, expected):
    """read_torch(where=) simple dialect selects the same rows as table.read."""
    assert _read_torch_ids(matrix_fits, pred) == expected
    assert _read_torch_ids(matrix_fits, pred, mmap=False) == expected


@pytest.mark.parametrize(
    "pred", ["NULJ IN (-3, -1)", "NOT ID == 2", "NULJ IS NULL", "NULJ IS NOT NULL"]
)
def test_matrix_read_torch_rejects_full_dialect(matrix_fits, pred):
    """Documented contract: read_torch(where=) raises ValueError on OR / IN /
    NOT / IS NULL spellings."""
    with pytest.raises(ValueError):
        _read_torch_ids(matrix_fits, pred)


def test_matrix_projection_and_predicate_columns(matrix_fits):
    """WHERE may reference columns outside the projection in every strategy."""
    expected = [4, 8]
    for mmap in (True, False):
        t = table.read(matrix_fits, columns=["ID"], where="V16 > 40000", mmap=mmap)
        assert t.column_names == ["ID"]
        assert sorted(t["ID"].to_pylist()) == expected
    batches = list(table.scan(matrix_fits, columns=["ID"], where="V16 > 40000"))
    assert sorted(pa.Table.from_batches(batches)["ID"].to_pylist()) == expected
    data = table.read_torch(matrix_fits, columns=["ID"], where="V16 > 40000")
    assert sorted(data["ID"].tolist()) == expected


def test_matrix_empty_projection_is_all_strategies(matrix_fits):
    """columns=[] projects like columns=None in every strategy (r5a-09)."""
    for mmap in (True, False):
        t = table.read(matrix_fits, columns=[], where="ID > 2", mmap=mmap)
        assert t.column_names == ["ID", "V16", "NULJ", "F32", "SCLR"]
        assert t.num_rows == 9
    batches = list(table.scan(matrix_fits, columns=[], where="ID > 2"))
    assert pa.Table.from_batches(batches).column_names == [
        "ID",
        "V16",
        "NULJ",
        "F32",
        "SCLR",
    ]
    assert table.read(matrix_fits, columns=[]).column_names == [
        "ID",
        "V16",
        "NULJ",
        "F32",
        "SCLR",
    ]
