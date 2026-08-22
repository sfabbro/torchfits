"""Wave-5 deep-review regressions: LONGSTRN/CONTINUE (F1/G1), uint64 writes
(F2), float TSCAL/TZERO scaling (F3), mmap-fallback for scaled tables (F4),
mmap GIL release, per-batch reopen elimination, NIOBUF guards, and the
removed scale-on-device dead helpers."""

from __future__ import annotations

import itertools
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

import torchfits


# --- F1/G1: LONGSTRN '&' and bare CONTINUE header values -------------------


def test_long_string_roundtrip_writes_continue_and_reads_back(tmp_path):
    """A >68-char string value must be written as LONGSTRN CONTINUE cards and
    read back in full (previously truncated at 68 chars)."""
    long_val = "x" * 120
    p = tmp_path / "longstr.fits"
    torchfits.write(p, torch.zeros(2, 2), header={"MYLONG": long_val})

    raw = p.read_bytes()
    assert b"CONTINUE" in raw, "G1: longstr write must emit CONTINUE cards"

    hdr = torchfits.read_header(str(p), hdu=0)
    assert hdr["MYLONG"] == long_val


def test_astropy_longstrn_file_reads_back_in_full(tmp_path):
    """Astropy-written LONGSTRN (&'-terminated) files parse to full values."""
    from astropy.io import fits

    long_val = "abc" * 40  # 120 chars, forces LONGSTRN in astropy
    p = tmp_path / "astropy_long.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32))
    hdu.header["MYLONG"] = long_val
    hdu.writeto(p, overwrite=True)

    hdr = torchfits.read_header(str(p), hdu=0)
    assert hdr["MYLONG"] == long_val


def test_trailing_ampersand_without_continue_is_literal(tmp_path):
    """A quoted value ending in '&' with no CONTINUE card keeps the '&'."""
    from torchfits.header_parser import FastHeaderParser

    card1 = "K1      = 'abc&'                  / comment".ljust(80)
    hdr = FastHeaderParser.parse_header_string(card1 + "END".ljust(80))
    assert hdr["K1"] == "abc&"


def test_plain_continue_card_appends_segment(tmp_path):
    """Bare CONTINUE cards (no '&' marker) still assemble into the value."""
    from torchfits.header_parser import FastHeaderParser

    card1 = "S1      = 'abc'                    / first".ljust(80)
    card2 = "CONTINUE  'def'                   / second".ljust(80)
    hdr = FastHeaderParser.parse_header_string(card1 + card2 + "END".ljust(80))
    assert hdr["S1"] == "abcdef"


def test_longstrn_chain_with_comments_on_every_card(tmp_path):
    """CONTINUE chains keep working when every card carries a comment.

    Regression: the ``&'`` chain markers were detected on the raw card
    text including the comment, so comment-bearing chains kept literal
    '&' characters mid-value.
    """
    from torchfits.header_parser import fast_parse_header, fast_parse_header_cards

    header = (
        "LONGSTR = 'This is a long string that needs &' / first".ljust(80)
        + "CONTINUE  'the continuation of the string.&' / second".ljust(80)
        + "CONTINUE  'final part.' / third".ljust(80)
        + "END".ljust(80)
    )
    want = "This is a long string that needs the continuation of the string.final part."
    cards = {k: v for k, v, _c in fast_parse_header_cards(header)}
    assert cards["LONGSTR"] == want
    assert fast_parse_header(header)["LONGSTR"] == want

    # A broken chain (no CONTINUE follows) restores the literal '&'.
    broken = "K1      = 'abc&'                  / note".ljust(80) + "END".ljust(80)
    cards_broken = {k: v for k, v, _c in fast_parse_header_cards(broken)}
    assert cards_broken["K1"] == "abc&"


def test_hierarch_cards_parse_to_typed_long_keys(tmp_path):
    """ESO HIERARCH cards expose ``KEY WORD NAME`` -> typed value."""
    from torchfits.header_parser import fast_parse_header

    header = (
        "HIERARCH ESO TEL AMBI TEMP = 12.5 / ambient temp".ljust(80)
        + 'HIERARCH ESO DET CHIP1 ID = "RED"'.ljust(80)
        + "END".ljust(80)
    )
    hdr = fast_parse_header(header)
    assert hdr["ESO TEL AMBI TEMP"] == 12.5
    assert hdr["ESO DET CHIP1 ID"] == "RED"


# --- F2: uint64 writes are rejected with guidance --------------------------


def test_uint64_image_write_rejected(tmp_path):
    t = torch.zeros(2, 2, dtype=torch.uint64)
    with pytest.raises(ValueError, match="uint64"):
        torchfits.write(tmp_path / "u64img.fits", t)


def test_uint64_table_column_write_rejected(tmp_path):
    with pytest.raises(ValueError, match="uint64"):
        torchfits.write(
            tmp_path / "u64col.fits",
            {"col": torch.zeros(3, dtype=torch.uint64)},
        )


# --- F3/F4: float TSCAL/TZERO scaling + mmap fallback ----------------------


def _write_scaled_table(tmp_path, storage_dtype, storage_values):
    """Write a binary table whose first column carries TSCAL1/TZERO1."""
    from astropy.io import fits

    p = tmp_path / f"scaled_{storage_dtype}.fits"
    col = fits.Column(
        name="val",
        format="D" if storage_dtype == np.float64 else "J",
        array=np.asarray(storage_values, dtype=storage_dtype),
    )
    hdu = fits.BinTableHDU.from_columns([col])
    hdu.header["TSCAL1"] = 2.0
    hdu.header["TZERO1"] = 500.0
    hdu.writeto(p, overwrite=True)
    return p


@pytest.mark.parametrize("storage_dtype", [np.float64, np.int32])
def test_scaled_columns_return_physical_values_buffered_and_mmap(
    tmp_path, storage_dtype
):
    """TSCAL/TZERO must apply to FLOAT/DOUBLE columns too (F3), and the mmap
    row path must transparently fall back to buffered reads (F4)."""
    from torchfits.table import read_torch

    p = _write_scaled_table(tmp_path, storage_dtype, [250, 251, 252, 253])
    expected = [1000.0, 1002.0, 1004.0, 1006.0]

    buffered = read_torch(str(p), hdu=1, mmap=False)
    assert buffered["val"].tolist() == expected

    mmap_ok = read_torch(str(p), hdu=1, mmap=True)
    assert mmap_ok["val"].tolist() == expected


def test_scaled_where_selects_physical_values(tmp_path):
    """WHERE filters run against physical (scaled) values on the unified path."""
    from torchfits.table import read_torch

    p = _write_scaled_table(tmp_path, np.float64, [150, 151, 152, 253])
    # physical: [800, 802, 804, 1006]

    got = read_torch(str(p), hdu=1, mmap=True, where="val > 900")
    assert got["val"].tolist() == [1006.0]


# --- GIL: mmap batch reads release the GIL during the C++ read ---------------


def test_concurrent_mmap_full_reads_are_consistent(tmp_path):
    """Concurrent mmap full reads from several threads stay correct (the mmap
    read path must not hold the GIL while blocking on CFITSIO)."""
    import threading

    from torchfits.table import read_torch

    p = tmp_path / "gil.fits"
    torchfits.write(p, {"c1": torch.zeros(16, dtype=torch.float64)})
    errors = []

    def worker():
        try:
            for _ in range(10):
                out = read_torch(str(p), hdu=1, mmap=True)
                assert out["c1"].numel() == 16
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# --- Per-batch reopen elimination -------------------------------------------


def test_scan_reuses_one_mmap_reader_across_batches(tmp_path):
    """Scan with mmap=True must open the reader once and read every batch from
    it (previously re-opened + re-parsed the file per batch)."""
    import torchfits._C as cpp
    from torchfits.table import scan

    p = tmp_path / "batches.fits"
    torchfits.write(p, {"v": torch.arange(256, dtype=torch.float64)})

    counts = {"opens": 0, "reader_rows": 0, "legacy_rows": 0}
    real_open = cpp.open_fits_mmap_reader
    real_reader_rows = cpp.read_fits_table_rows_mmap_from_reader
    real_legacy = cpp.read_fits_table_rows

    def counted_open(*a, **k):
        counts["opens"] += 1
        return real_open(*a, **k)

    def counted_reader(*a, **k):
        counts["reader_rows"] += 1
        return real_reader_rows(*a, **k)

    def counted_legacy(*a, **k):
        counts["legacy_rows"] += 1
        return real_legacy(*a, **k)

    with (
        mock.patch.object(cpp, "open_fits_mmap_reader", counted_open),
        mock.patch.object(cpp, "read_fits_table_rows_mmap_from_reader", counted_reader),
        mock.patch.object(cpp, "read_fits_table_rows", counted_legacy),
    ):
        it = scan(str(p), hdu=1, batch_size=16, mmap=True)
        batches = list(itertools.islice(it, 8))

    assert counts["opens"] == 1, f"expected 1 open, got {counts['opens']}"
    assert counts["reader_rows"] == 8
    assert counts["legacy_rows"] == 0
    assert [b.num_rows for b in batches] == [16] * 8


# --- NIOBUF/MINDIRECT tuning guards -----------------------------------------


def test_vendored_cfitsio_has_niobuf_mindirect_guards():
    """The vendored CFITSIO headers must carry the #ifndef guards that let
    -DTORCHFITS_NIOBUF / -DTORCHFITS_MINDIRECT override the defaults."""
    fitsio_h = Path("extern/cfitsio/fitsio.h")
    fitsio2_h = Path("extern/cfitsio/fitsio2.h")
    if not fitsio_h.exists():
        pytest.skip("vendored CFITSIO not materialized")
    assert "#ifndef NIOBUF" in fitsio_h.read_text()
    assert "#ifndef MINDIRECT" in fitsio2_h.read_text()


# --- Scale-on-device dead helpers removed -----------------------------------


def test_scale_on_device_dead_helpers_removed():
    """The never-called _apply_scale_on_device/_apply_unsigned_offset helpers
    were removed; the live path is exercised by test_scale_on_device.py."""
    from torchfits._io_engine import _read_pipeline as rp

    assert not hasattr(rp, "_apply_scale_on_device")
    assert not hasattr(rp, "_apply_unsigned_offset")
