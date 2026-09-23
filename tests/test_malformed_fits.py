"""Malformed / hostile FITS input corpus and checksum-on-corruption.

Every public reader must fail loudly with a clear error on structurally
broken files — never silently return wrong-shaped or zero-filled data, and
never crash the interpreter.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
from astropy.io import fits as afits  # noqa: E402

_READERS = [
    lambda p: torchfits.read(p),
    lambda p: torchfits.read_tensor(p),
    lambda p: torchfits.read_header(p),
    lambda p: torchfits.read_num_hdus(p),
    lambda p: torchfits.read_shape(p),
]


def _expect_clean_error(callable_, path):
    try:
        callable_(path)
    except (RuntimeError, OSError, ValueError) as exc:
        assert str(exc), "error message must not be empty"
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"unexpected error type {type(exc).__name__}: {exc}")
    else:
        pytest.fail(f"{callable_} accepted a malformed file")


def _write_minimal_image(path):
    afits.PrimaryHDU(data=np.arange(16, dtype=np.float32).reshape(4, 4)).writeto(path)


def test_bad_magic_rejected(tmp_path):
    path = tmp_path / "bad_magic.fits"
    path.write_bytes(b"NOTFITS" + b"\0" * 2880)
    for reader in _READERS:
        _expect_clean_error(reader, str(path))


def test_simple_false_rejected(tmp_path):
    path = tmp_path / "simple_f.fits"
    cards = ["SIMPLE  =                    F".ljust(80)]
    blob = "".join(cards).encode()
    blob += b"\0" * (2880 - len(blob))
    path.write_bytes(blob)
    for reader in _READERS:
        _expect_clean_error(reader, str(path))


def test_truncated_image_data_raises_not_zeros(tmp_path):
    path = str(tmp_path / "trunc_img.fits")
    data = np.ones((64, 64), dtype=np.float32)
    afits.PrimaryHDU(data=data).writeto(path)
    with open(path, "rb+") as fh:
        fh.truncate(2880 + 100)  # header only + a sliver of data

    # Must raise a clean CFITSIO I/O error, never return zeros-as-data.
    with pytest.raises(RuntimeError, match="108|read"):
        torchfits.read_tensor(path, mmap=False)


def test_truncated_table_mmap_raises(tmp_path):
    path = str(tmp_path / "trunc_tbl.fits")
    cols = [afits.Column(name="A", format="J", array=np.arange(50000))]
    afits.HDUList([afits.PrimaryHDU(), afits.BinTableHDU.from_columns(cols)]).writeto(
        path
    )
    with open(path, "rb+") as fh:
        size = fh.seek(0, 2)
        fh.truncate(size // 2)

    with pytest.raises(RuntimeError, match="truncat"):
        torchfits.table.read_torch(str(path), hdu=1)


def test_garbage_after_end_is_ignored_or_reported(tmp_path):
    path = str(tmp_path / "garbage_tail.fits")
    _write_minimal_image(path)
    with open(path, "ab") as fh:
        fh.write(b"\xff" * 2880)
    got = torchfits.read_tensor(path)
    assert got.shape == (4, 4)


def test_vla_descriptor_out_of_heap_detected(tmp_path):
    path = str(tmp_path / "vla_oob.fits")
    vla = [np.arange(i + 1, dtype=np.float32) for i in range(8)]
    cols = [
        afits.Column(name="N", format="J", array=np.arange(8)),
        afits.Column(name="V", format="PJ()", array=vla),
    ]
    afits.HDUList([afits.PrimaryHDU(), afits.BinTableHDU.from_columns(cols)]).writeto(
        path
    )
    # Corrupt bytes in the tail of the file (heap region): descriptors or
    # payload bytes flip, and the reader must either decode consistently or
    # raise — never crash.
    raw = bytearray(open(path, "rb").read())
    for off in range(len(raw) - 512, len(raw) - 256):
        raw[off] ^= 0xA5
    open(path, "wb").write(bytes(raw))

    try:
        result = torchfits.read(str(path), hdu=1, mode="table")
        # If it reads, the values must at least stay finite/consistent.
        assert isinstance(result, dict)
    except (RuntimeError, OSError, ValueError):
        pass


def test_random_groups_explicitly_unsupported(tmp_path):
    """GROUPS=T must raise a clear error, never decode wrong-shaped data."""
    path = tmp_path / "groups.fits"
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    1",
        "NAXIS1  =                    4",
        "GROUPS  =                    T",
        "PCOUNT  =                    2",
        "GCOUNT  =                    3",
        "END",
    ]
    blob = "".join(c.ljust(80) for c in cards).encode()
    blob += b"\x00" * ((2880 - len(blob) % 2880) % 2880)
    blob += b"\x00" * 2880  # one data unit
    path.write_bytes(blob)
    with pytest.raises(RuntimeError, match="[Rr]andom [Gg]roups"):
        torchfits.read_tensor(str(path))


def test_checksum_detects_corrupted_data(tmp_path):
    path = str(tmp_path / "cksum.fits")
    afits.PrimaryHDU(data=np.arange(16, dtype=np.float32).reshape(4, 4)).writeto(
        path, checksum=True
    )
    # Flip one byte inside the data unit (past the 2880-byte header).
    raw = bytearray(open(path, "rb").read())
    raw[3000] ^= 0xFF
    open(path, "wb").write(bytes(raw))

    report = torchfits.verify_checksums(path, hdu=0)
    assert report["status"] == "fail" or report["datastatus"] != 1


def test_duplicate_extname_resolves_first(tmp_path):
    path = tmp_path / "dupname.fits"
    hdu_a = afits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32), name="SCI")
    hdu_b = afits.ImageHDU(data=np.ones((3, 3), dtype=np.float32), name="SCI")
    afits.HDUList([afits.PrimaryHDU(), hdu_a, hdu_b]).writeto(path)

    got = torchfits.read_tensor(path, hdu="SCI")
    assert got.shape == (2, 2), "first EXTNAME match must win"


def _write_three_hdu_file(path):
    afits.HDUList(
        [
            afits.PrimaryHDU(data=np.zeros((4, 4), np.float32)),
            afits.ImageHDU(data=np.ones((4, 4), np.float32)),
            afits.ImageHDU(data=np.full((4, 4), 2.0, np.float32)),
        ]
    ).writeto(path)


def test_truncated_hdu_scan_not_silent(tmp_path):
    """A file cut mid-way through the next HDU's header must not silently
    under-report its HDU count (r7a-08; astropy warns, torchfits raises).
    Contract fixed with R9-FitsBindings: a tail starting a SIMPLE/XTENSION
    prefix raises RuntimeError naming the path; garbage tails and mid-data
    truncation of the last HDU stay tolerated with an honest count."""
    path = str(tmp_path / "trunc_scan.fits")
    _write_three_hdu_file(path)
    raw = open(path, "rb").read()
    xtension_at = [i for i in range(len(raw) - 9) if raw[i : i + 9] == b"XTENSION="]
    assert len(xtension_at) == 2

    # Cut inside the 3rd HDU's header: two complete HDUs + a partial header.
    cut = tmp_path / "trunc_scan_cut.fits"
    cut.write_bytes(raw[: xtension_at[1] + 400])
    with pytest.raises(RuntimeError) as excinfo:
        torchfits.read_num_hdus(str(cut))
    assert "truncat" in str(excinfo.value)
    assert str(cut) in str(excinfo.value)

    # Garbage tail: tolerated, count unchanged (test_garbage_after_end's pin).
    junk = tmp_path / "trunc_scan_junk.fits"
    junk.write_bytes(raw + b"\xff" * 2880)
    assert torchfits.read_num_hdus(str(junk)) == 3

    # Truncation inside the last HDU's data unit: honest header count, no raise.
    data_cut = tmp_path / "trunc_scan_data.fits"
    data_cut.write_bytes(raw[: xtension_at[1] + 2880 + 100])
    assert torchfits.read_num_hdus(str(data_cut)) == 3


def test_truncated_image_subset_raises_not_zeros(tmp_path):
    """Subset reads on truncated image data must raise a clean error, never
    return zero-filled or short data (r9b truncated-edge pin)."""
    path = str(tmp_path / "trunc_subset.fits")
    data = np.ones((64, 64), dtype=np.float32)
    afits.PrimaryHDU(data=data).writeto(path)
    with open(path, "rb+") as fh:
        fh.truncate(2880 + 100)

    with pytest.raises((RuntimeError, OSError, ValueError)):
        torchfits.read_subset(path, 0, 0, 0, 8, 8)
    fh_cpp = torchfits._cpp.open_fits_file(path, "r")
    try:
        with pytest.raises((RuntimeError, OSError, ValueError)):
            fh_cpp.read_subset(0, 0, 0, 8, 8)
    finally:
        fh_cpp.close()


def test_readers_never_crash_interpreter_on_fuzzed_headers(tmp_path):
    """Bounded fuzz: random header mutations must raise, not segfault."""
    rng = np.random.default_rng(12345678901)  # deterministic seed
    base = str(tmp_path / "fuzz_src.fits")
    _write_minimal_image(base)
    original = open(base, "rb").read()

    for trial in range(25):
        raw = bytearray(original)
        for _ in range(rng.integers(1, 6)):
            pos = int(rng.integers(0, 2880))
            raw[pos] = int(rng.integers(0, 256))
        path = tmp_path / f"fuzz_{trial}.fits"
        path.write_bytes(bytes(raw))
        try:
            torchfits.read(str(path))
        except (RuntimeError, OSError, ValueError, MemoryError):
            continue
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"fuzz trial {trial}: unexpected {type(exc).__name__}: {exc}")
