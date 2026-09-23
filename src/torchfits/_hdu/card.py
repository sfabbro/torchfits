"""FITS header card — a single (key, value, comment) record."""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence, Union


class Card(NamedTuple):
    key: str
    value: Any = None
    comment: str = ""

    @property
    def keyword(self) -> str:
        return self.key


def _is_string_typed(key: str, value: str) -> bool:
    """Whether ``FastHeaderParser._parse_card`` would type ``value`` as ``str``.

    Cpp card triples carry untyped value strings (quoting already stripped),
    so CONTINUE target selection re-derives the typed-string test from
    ``fast_parse_header_cards``: numeric literals, ``T``/``F``, complex
    literals and blank values are not string-typed, and a CONTINUE card after
    them is malformed (kept verbatim) instead of fused into an unrelated
    keyword. Quoted values that merely *look* numeric/boolean stay ambiguous
    in this direction and are kept apart — fail-safe, never fused.
    """
    from ..header_parser import FastHeaderParser, _parse_fits_number

    text = value.strip()
    if not text:
        return False
    if key in FastHeaderParser._STRING_KEYWORDS:
        return True
    if text in ("T", "F") or text.startswith("("):
        return False
    return _parse_fits_number(text) is None


def _reassemble_longstr_cards(
    cards: Sequence[Union["Card", tuple[Any, ...]]],
) -> Union[list[Card], Sequence[Any]]:
    """Join LONGSTRN ``&``+CONTINUE chains in an ordered card sequence.

    ``cpp.open_and_read_headers`` / ``cpp.read_header`` surface parsed
    ``(key, value, comment)`` triples where a CONTINUE card's raw quoted
    field rides in its comment slot and the ``&`` chain markers survive in
    the values. Mirrors ``fast_parse_header_cards`` (the ``read_header``
    oracle): a string value ending in ``&`` is chain notation and joins the
    CONTINUE card(s) that follow, while the ``&`` is restored verbatim when
    no CONTINUE card continues it. Bare CONTINUE cards append to the
    preceding string-typed card; a CONTINUE card after anything else is
    malformed and stays visible as its own card.

    Accepts ``Card`` records or plain tuples; returns a rebuilt
    ``list[Card]``, or the input unchanged when no CONTINUE card is present
    (the overwhelmingly common case).
    """
    from ..header_parser import FastHeaderParser

    seq = list(cards)
    if not any(str(card[0]) == "CONTINUE" for card in seq):
        return cards

    out: list[Card] = []
    target: int | None = None  # index of the last string-typed card
    marker: int | None = None  # index of a card with a pending trailing '&'

    def restore_marker() -> None:
        nonlocal marker
        if marker is not None:
            card = out[marker]
            out[marker] = Card(card.key, card.value + "&", card.comment)
            marker = None

    for card in seq:
        key, value, comment = card
        key = str(key)
        comment = "" if comment is None else str(comment)
        if key != "CONTINUE":
            # Any non-CONTINUE card ends the chain: the '&' is content again.
            restore_marker()
            out.append(Card(key, value, comment))
            if isinstance(value, str):
                if value.endswith("&"):
                    out[-1] = Card(key, value[:-1], comment)
                    marker = len(out) - 1
                if _is_string_typed(key, value):
                    target = len(out) - 1
            continue

        # Cpp shape keeps the raw quoted field in the comment slot and the
        # parsed (empty) value in the value slot; an already-normalized card
        # carries its segment in the value instead.
        from_comment = not (isinstance(value, str) and value.strip())
        field = comment if from_comment else value
        if target is None or not isinstance(out[target].value, str):
            # Orphan CONTINUE (malformed): keep it visible as its own card.
            if from_comment:
                cpos = FastHeaderParser._find_comment_separator(field)
                if cpos != -1:
                    segment = field[:cpos].strip()
                    seg_comment = field[cpos + 1 :].strip()
                else:
                    segment = field.strip()
                    seg_comment = ""
                seg_value = FastHeaderParser._parse_string_value(segment)
                out.append(Card(key, seg_value, seg_comment))
            else:
                out.append(Card(key, value, comment))
            continue
        cpos = FastHeaderParser._find_comment_separator(field)
        segment = (field[:cpos] if cpos != -1 else field).strip()
        if not segment:
            # Blank segment: contributes nothing and closes the marker chain.
            marker = None
            continue
        has_marker = segment.startswith("'") and segment.endswith("&'")
        seg_value = FastHeaderParser._parse_string_value(segment)
        if has_marker and isinstance(seg_value, str) and seg_value.endswith("&"):
            # This segment continues the chain: its own marker is notation,
            # not content.
            seg_value = seg_value[:-1]
        head = out[target]
        out[target] = Card(head.key, head.value + seg_value, head.comment)
        marker = target if has_marker else None

    restore_marker()
    return out
