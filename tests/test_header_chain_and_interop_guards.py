"""Guards for the unit-8 findings: header chain assembly and interop policies.

R2-046 LONGSTRN '&'+CONTINUE chains whose first segment merely *looks* numeric
        or complex were joined to the PRECEDING keyword: the chain's own value
        came back truncated with the '&' marker lost, and the previous
        keyword absorbed the segments. ``_is_string_typed`` was asked about a
        card but handed the already-unquoted value.
R2-047 to_arrow / to_polars accepted a misspelled ``vla_policy`` whenever the
        table had no VLA column to reach the check; to_pandas always refused.
"""

import pytest
import torch

from torchfits._hdu.card import _is_string_typed, _reassemble_longstr_cards
from torchfits.header_parser import fast_parse_header_cards

CHAIN_VALUES = {
    # name -> value written by astropy, which splits anything over 68 chars
    # into a LONGSTRN '&'+CONTINUE chain.
    "NUMSTR": "1234567890" * 10,
    "TSTR": "T" * 100,
    "FSTR": "F" * 100,
    "CPLX": "(1.5,2.5)" * 20,
    "CPLXNUM": "(1" + ",2.5)" * 30,
    "MIXED": "ab" * 60,
    "DOTTED": "1.5.2.5" * 20,
}


def _write_longstrn_file(path):
    afits = pytest.importorskip("astropy.io.fits")
    import numpy as np

    hdu = afits.PrimaryHDU(data=np.arange(16, dtype=np.float32).reshape(4, 4))
    for key, value in CHAIN_VALUES.items():
        hdu.header[key] = value
    hdu.writeto(str(path), overwrite=True)
    return str(path)


def _expected_pairs():
    """(segment, has_ampersand) triples as the cpp layer hands them over."""
    pairs = []
    for key, value in CHAIN_VALUES.items():
        rest = value
        while rest:
            head, rest = rest[:68], rest[68:]
            marker = "&" if rest else ""
            pairs.append((key, head + marker, bool(rest)))
            if not rest:
                break
    return pairs


# --------------------------------------------------------------------------
# R2-046: a chain is joined to itself, whatever its content looks like
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,value", sorted(CHAIN_VALUES.items()))
def test_open_and_read_header_agree_on_every_chain_shape(tmp_path, key, value):
    """Both public APIs must return the file's own string, character for character."""
    import torchfits

    path = _write_longstrn_file(tmp_path / f"{key}.fits")
    torchfits.clear_file_cache()

    via_read = torchfits.read_header(path, 0).get(key)
    with torchfits.open(path) as hdul:
        via_open = hdul[0].header.get(key)

    assert via_read == value
    assert via_open == value
    assert via_open == via_read


@pytest.mark.parametrize("key,value", sorted(CHAIN_VALUES.items()))
def test_a_chain_never_leaks_into_the_previous_keyword(tmp_path, key, value):
    """The failure mode: the chain's segments land on an unrelated keyword."""
    import torchfits

    path = _write_longstrn_file(tmp_path / f"{key}.fits")
    torchfits.clear_file_cache()
    with torchfits.open(path) as hdul:
        header = hdul[0].header
        assert header.get(key) == value
        for other, other_value in CHAIN_VALUES.items():
            if other != key:
                assert header.get(other) == other_value
        # and no card is left holding a chain marker or a stray CONTINUE
        assert not [c for c in header.cards if c.key == "CONTINUE"]
        assert not [
            k for k, v in header.items() if isinstance(v, str) and v.endswith("&")
        ]


@pytest.mark.parametrize("key,value", sorted(CHAIN_VALUES.items()))
def test_the_unmarked_tail_of_a_chain_keeps_no_ampersand(tmp_path, key, value):
    """The '&' is notation; it must never survive into the stored value."""
    import torchfits

    path = _write_longstrn_file(tmp_path / f"{key}.fits")
    torchfits.clear_file_cache()
    with torchfits.open(path) as hdul:
        assert "&" not in (hdul[0].header.get(key) or "")


def test_reassembly_fuses_a_chain_onto_its_own_card():
    """The unit under test, on the cpp shape (CONTINUE field in the comment)."""
    cards = [
        ("PLAIN", "a plain value", ""),
        ("PREV", "the previous keyword", ""),
        ("CPLX", "(1.5,2.5)&", ""),
        ("CONTINUE", "", " '(1.5,2.5)&'"),
        ("CONTINUE", "", " '(1.5)'"),
    ]
    out = _reassemble_longstr_cards(cards)
    by_key = {(c.key, c.value) for c in out}
    assert ("PREV", "the previous keyword") in by_key
    assert ("CPLX", "(1.5,2.5)(1.5,2.5)(1.5)") in by_key
    assert [c.key for c in out] == ["PLAIN", "PREV", "CPLX"]


def test_a_single_ampersand_still_works_for_a_plain_chain():
    cards = [
        ("KEY", "first segment&", ""),
        ("CONTINUE", "", " ' second segment'"),
    ]
    out = _reassemble_longstr_cards(cards)
    assert [(c.key, c.value) for c in out] == [("KEY", "first segment second segment")]


def test_an_uncontinued_ampersand_stays_content():
    """No CONTINUE in the sequence: the input comes back untouched."""
    cards = [("KEY", "dangling&", "")]
    out = _reassemble_longstr_cards(cards)
    assert list(out) == [("KEY", "dangling&", "")]


def test_an_orphan_continue_after_a_number_stays_visible():
    """The fail-safe the fix must not weaken: no fusion into a numeric card."""
    cards = [
        ("NUM", 12345, ""),
        ("CONTINUE", "", " ' segment'"),
    ]
    out = _reassemble_longstr_cards(cards)
    assert [(c.key, c.value) for c in out] == [
        ("NUM", 12345),
        ("CONTINUE", " segment"),
    ]


def test_string_typed_is_still_false_for_an_unmarked_numeric_value():
    """The remaining role of the predicate, unchanged by R2-046."""
    assert _is_string_typed("NUM", "12345") is False
    assert _is_string_typed("NUM", "T") is False
    assert _is_string_typed("NUM", "F") is False
    assert _is_string_typed("NUM", "(1.5,2.5)") is False
    assert _is_string_typed("NUM", "  ") is False
    # Forced-string keywords keep their documented exception
    assert _is_string_typed("EXTNAME", "12") is True
    assert _is_string_typed("HISTORY", "some text") is True


def test_the_two_reassembly_implementations_agree(tmp_path):
    """header_parser's lax join and the card module must reach one answer."""
    import torchfits

    path = _write_longstrn_file(tmp_path / "both.fits")
    torchfits.clear_file_cache()
    raw = torchfits._core_api.read_header_string(path, 0)
    lax = {k: v for k, v, _c in fast_parse_header_cards(raw)}
    handle = torchfits._C.open_fits_file(path, "r")
    try:
        triples = torchfits._C.read_header(handle, 0)
    finally:
        handle.close()
    strict = {c.key: c.value for c in _reassemble_longstr_cards(triples)}
    for key, value in CHAIN_VALUES.items():
        assert lax[key] == strict[key] == value


# --------------------------------------------------------------------------
# R2-047: vla_policy is checked before the data is walked
# --------------------------------------------------------------------------


def _plain():
    return {"x": torch.arange(3.0)}


def _with_vla():
    return {
        "x": torch.arange(3.0),
        "v": [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
    }


@pytest.mark.parametrize("policy", ["typo", "LIST", "list ", "objec", "", "none"])
def test_to_arrow_refuses_an_unknown_policy_without_a_vla_column(policy):
    from torchfits.interop import to_arrow

    with pytest.raises(ValueError, match="vla_policy must be 'list' or 'drop'"):
        to_arrow(_plain(), vla_policy=policy)


@pytest.mark.parametrize("policy", ["typo", "LIST", ""])
def test_to_polars_refuses_an_unknown_policy_too(policy):
    pytest.importorskip("polars")
    from torchfits.interop import to_polars

    with pytest.raises(ValueError, match="vla_policy must be 'list' or 'drop'"):
        to_polars(_plain(), vla_policy=policy)


@pytest.mark.parametrize("policy", ["list", "drop"])
def test_the_two_legal_policies_still_work_without_a_vla_column(policy):
    from torchfits.interop import to_arrow

    assert to_arrow(_plain(), vla_policy=policy).column_names == ["x"]


def test_the_two_legal_policies_still_work_with_a_vla_column():
    from torchfits.interop import to_arrow

    assert to_arrow(_with_vla(), vla_policy="list").column_names == ["x", "v"]
    assert to_arrow(_with_vla(), vla_policy="drop").column_names == ["x"]


@pytest.mark.parametrize("policy", ["object", "drop"])
def test_to_pandas_keeps_its_own_policy_spelling(policy):
    pytest.importorskip("pandas")
    from torchfits.interop import to_pandas

    frame = to_pandas(_with_vla(), vla_policy=policy)
    assert list(frame.columns) == (["x", "v"] if policy == "object" else ["x"])
