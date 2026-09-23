"""LONGSTRN '&'+CONTINUE reassembly parity (r4c-15 wrapper) and the
``HDUList.fromfile`` root fix (r4b-13).

``torchfits.open`` rejoined chains since r4c-15 via the ``open_hdulist``
wrapper, but ``HDUList.fromfile`` — the constructor every rewrite
(``write``/``insert_hdu``/``replace_hdu``/``delete_hdu``) reads headers
through — kept the ``&`` marker and detached CONTINUE cards. These tests
exercise ``fromfile`` DIRECTLY (no wrapper) and pin reassembly semantics
against the ``read_header`` oracle, including hostile chain shapes.
"""

from __future__ import annotations

import numpy as np
import torch
from astropy.io import fits

import torchfits


def _write_header(path, pairs) -> str:
    hdul = fits.HDUList([fits.PrimaryHDU()])
    for key, value in pairs:
        hdul[0].header[key] = value
    hdul.writeto(str(path), overwrite=True)
    return str(path)


def _write_raw_cards(path, header_cards: str) -> str:
    """Write a header-only primary HDU from raw 80-char cards.

    Raw cards are the only way to express hostile chain shapes (per-card
    comments on CONTINUE cards, orphan CONTINUE, empty segments) that the
    writers refuse to produce.
    """
    data = (
        "SIMPLE  =                    T / conforms to FITS standard".ljust(80)
        + "BITPIX  =                   -32".ljust(80)
        + "NAXIS   =                    0".ljust(80)
        + header_cards
        + "END".ljust(80)
    ).ljust(2880)
    path.write_bytes(data.encode("ascii"))
    return str(path)


def _fromfile_header(path):
    hdul = torchfits.HDUList.fromfile(path)  # direct: no open_hdulist wrapper
    try:
        return hdul[0].header
    finally:
        hdul.close()


def test_read_fast_header_false_rejoins_longstr(tmp_path):
    """fast_header=False used to return the raw CFITSIO triples, so a value
    longer than 68 characters kept the '&' marker and a detached CONTINUE."""
    value = "A" * 80
    path = tmp_path / "slow_header.fits"
    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    primary = fits.PrimaryHDU(image)
    primary.header["LONGSTR"] = value
    primary.writeto(path, overwrite=True)

    _tensor, header = torchfits.read(
        path, return_header=True, fast_header=False, use_cache=False
    )
    assert header["LONGSTR"] == value
    assert all(card.key != "CONTINUE" for card in header.cards)


def test_open_header_reassembles_longstr_chain(tmp_path):
    """torchfits.open must agree with read_header on LONGSTRN values."""
    value = "A" * 80
    path = _write_header(tmp_path / "longstr.fits", [("LONGSTR", value)])

    assert torchfits.read_header(path, hdu=0)["LONGSTR"] == value
    header = torchfits.open(path)[0].header
    assert header["LONGSTR"] == value
    assert all(card.key != "CONTINUE" for card in header.cards)


def test_open_header_reassembles_multi_continue_chain(tmp_path):
    value = "B" * 200
    path = _write_header(tmp_path / "longstr3.fits", [("LONGSTR", value)])
    assert torchfits.read_header(path, hdu=0)["LONGSTR"] == value
    assert torchfits.open(path)[0].header["LONGSTR"] == value


def test_open_header_chain_preserves_inner_ampersands(tmp_path):
    """Only the chain marker '&' is notation; content '&' survives joining."""
    value = "abc&" + "D" * 70
    path = _write_header(tmp_path / "ampmid.fits", [("AMPEND", value)])
    assert torchfits.read_header(path, hdu=0)["AMPEND"] == value
    assert torchfits.open(path)[0].header["AMPEND"] == value


def test_open_header_keeps_literal_trailing_ampersand_without_continue(tmp_path):
    """A '&' with no CONTINUE card is content and must be restored verbatim."""
    value = "xy&"
    path = _write_header(tmp_path / "ampend.fits", [("AMPEND", value)])
    assert torchfits.read_header(path, hdu=0)["AMPEND"] == value
    assert torchfits.open(path)[0].header["AMPEND"] == value


def test_open_header_normal_cards_unchanged(tmp_path):
    """Non-chain cards must pass through the wrapper byte-identically.

    open() headers expose raw card values as strings by pinned design
    (tests/test_hdu_file_ops.py); typed values live on the read_header route.
    """
    data = np.zeros((2, 3), dtype=np.float32)
    path = str(tmp_path / "plain.fits")
    fits.PrimaryHDU(data).writeto(path, overwrite=True)
    direct = torchfits.HDUList.fromfile(path)[0].header
    wrapped = torchfits.open(path)[0].header
    assert dict(wrapped) == dict(direct)
    assert [c.key for c in wrapped.cards] == [c.key for c in direct.cards]
    assert torchfits.read_header(path, hdu=0)["BITPIX"] == -32


# ---------------------------------------------------------------------------
# r4b-13 root fix: HDUList.fromfile (direct callers, no wrapper) must rejoin
# ---------------------------------------------------------------------------


def test_fromfile_direct_joins_longstr_chain(tmp_path):
    """A >68-char value must read back whole through fromfile itself."""
    value = "A" * 80
    path = _write_header(tmp_path / "longstr.fits", [("LONGSTR", value)])

    header = _fromfile_header(path)
    assert header["LONGSTR"] == value
    assert all(card.key != "CONTINUE" for card in header.cards)
    # Byte-identical with the read_header oracle and with open().
    assert header["LONGSTR"].encode("ascii") == torchfits.read_header(path, hdu=0)[
        "LONGSTR"
    ].encode("ascii")
    assert torchfits.open(path)[0].header["LONGSTR"] == value


def test_fromfile_direct_joins_chain_with_per_card_comments(tmp_path):
    """Quoted-comment edge case: comments on every chain card must not hide
    the '&' markers; the joined value keeps the first card's comment."""
    expected = (
        "This is a long string that needs the continuation of the string.final part."
    )
    path = _write_raw_cards(
        tmp_path / "chain_comments.fits",
        "LONGSTR = 'This is a long string that needs &' / first".ljust(80)
        + "CONTINUE  'the continuation of the string.&' / second".ljust(80)
        + "CONTINUE  'final part.' / third".ljust(80),
    )

    header = _fromfile_header(path)
    assert header["LONGSTR"] == expected
    assert header.card("LONGSTR").comment == "first"
    assert all(card.key != "CONTINUE" for card in header.cards)
    assert torchfits.read_header(path, hdu=0)["LONGSTR"] == expected
    assert torchfits.open(path)[0].header["LONGSTR"] == expected


def test_fromfile_direct_joins_multi_continue_empty_segment_and_final_amp(
    tmp_path,
):
    """Astropy's comment-on-trailing-CONTINUE split, an empty mid-chain
    segment, and a final segment whose trailing '&' is content."""
    value2 = "B" * 200
    p_long = _write_header(
        tmp_path / "astropy_long.fits",
        [("LONGSTR", (value2, "the first-card comment"))],
    )
    p_empty = _write_raw_cards(
        tmp_path / "empty_segment.fits",
        "CHUNKY  = 'abcdef&'".ljust(80)
        + "CONTINUE  ''".ljust(80)
        + "CONTINUE  'gh&'".ljust(80)
        + "CONTINUE  'ij'".ljust(80),
    )
    p_amp = _write_raw_cards(
        tmp_path / "final_amp.fits",
        "FINAMP  = 'start&'".ljust(80) + "CONTINUE  'end-with-amp&'".ljust(80),
    )

    assert _fromfile_header(p_long)["LONGSTR"] == value2
    assert _fromfile_header(p_empty)["CHUNKY"] == "abcdefghij"
    assert _fromfile_header(p_amp)["FINAMP"] == "startend-with-amp&"
    for path, key, expected in (
        (p_long, "LONGSTR", value2),
        (p_empty, "CHUNKY", "abcdefghij"),
        (p_amp, "FINAMP", "startend-with-amp&"),
    ):
        assert torchfits.read_header(path, hdu=0)[key] == expected, path
        assert torchfits.open(path)[0].header[key] == expected, path


def test_fromfile_direct_keeps_literal_ampersand_and_broken_chain(tmp_path):
    """A '&' with no CONTINUE after it is content; an interrupted chain keeps
    its '&' verbatim."""
    p_lit = _write_raw_cards(
        tmp_path / "amp_literal.fits",
        "AMPEND  = 'xy&'                  / note".ljust(80),
    )
    p_broken = _write_raw_cards(
        tmp_path / "broken_chain.fits",
        "BROKEN  = 'abc&'".ljust(80) + "OTHER   = 5".ljust(80),
    )

    assert _fromfile_header(p_lit)["AMPEND"] == "xy&"
    assert _fromfile_header(p_broken)["BROKEN"] == "abc&"
    assert torchfits.read_header(p_lit, hdu=0)["AMPEND"] == "xy&"
    assert torchfits.read_header(p_broken, hdu=0)["BROKEN"] == "abc&"


def test_fromfile_direct_joins_escaped_quote_segments(tmp_path):
    """A chain segment with ''-escaped quotes reassembles to the exact value."""
    value = "it's " + "x" * 150 + " done"
    path = _write_header(tmp_path / "escaped_quotes.fits", [("QUOTED", value)])

    assert _fromfile_header(path)["QUOTED"] == value
    assert torchfits.read_header(path, hdu=0)["QUOTED"] == value
    assert torchfits.open(path)[0].header["QUOTED"] == value


def test_fromfile_direct_bare_continue_appends(tmp_path):
    """A marker-less CONTINUE card still appends to the preceding string card
    (read_header semantics)."""
    path = _write_raw_cards(
        tmp_path / "bare_continue.fits",
        "S1      = 'abc'                    / first".ljust(80)
        + "CONTINUE  'def'                   / second".ljust(80),
    )

    header = _fromfile_header(path)
    assert header["S1"] == "abcdef"
    assert all(card.key != "CONTINUE" for card in header.cards)
    assert torchfits.read_header(path, hdu=0)["S1"] == "abcdef"


def test_fromfile_direct_orphan_continue_stays_verbatim(tmp_path):
    """A malformed CONTINUE after non-string-typed cards must stay visible as
    its own card (read_header parity), never fused into an unrelated keyword."""
    path = _write_raw_cards(
        tmp_path / "orphan_continue.fits",
        "CONTINUE  'orphan segment'".ljust(80),
    )

    header = _fromfile_header(path)
    assert str(header["NAXIS"]) == "0"  # never '0orphan segment'
    continue_cards = [c for c in header.cards if c.key == "CONTINUE"]
    assert len(continue_cards) == 1
    oracle = torchfits.read_header(path, hdu=0)
    assert continue_cards[0].value == oracle["CONTINUE"] == "orphan segment"


def test_fromfile_longstr_survives_rewrite_chain(tmp_path):
    """r4b-13 repro: >68-char values must survive an HDUList
    write/insert/replace/delete rewrite chain byte-identically when the HDUList
    came straight from fromfile."""
    for length in (80, 200):
        value = "C" * length
        src = _write_header(
            tmp_path / f"src{length}.fits",
            [("LONGSTR", value), ("PLAIN", "keep")],
        )
        hdul = torchfits.HDUList.fromfile(src)
        try:
            hdul.write(str(tmp_path / f"stage{length}.fits"), True)
        finally:
            hdul.close()
        stage = str(tmp_path / f"stage{length}.fits")

        torchfits.insert_hdu(stage, torch.zeros(2, 2), 1)
        torchfits.replace_hdu(stage, 1, torch.ones(2, 2))
        torchfits.delete_hdu(stage, 1)

        via_astropy = fits.getheader(stage)
        assert via_astropy["LONGSTR"] == value
        assert via_astropy["LONGSTR"].encode("ascii") == value.encode("ascii")
        assert via_astropy["PLAIN"] == "keep"
        assert torchfits.read_header(stage, hdu=0)["LONGSTR"] == value
        assert torchfits.open(stage)[0].header["LONGSTR"] == value
