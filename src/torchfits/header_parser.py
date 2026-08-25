"""
Fast Python-side header parser for FITS headers.

This module provides optimized parsing of FITS header strings returned by the
bulk C++ read_header_to_string() function, avoiding Python/C++ round trips.
"""

import re
from typing import Any, Dict, Optional

# Used for mypy: header values can be str, int, float, bool, complex, or None


class FastHeaderParser:
    """
    High-performance FITS header parser.

    Parses the raw header string returned by C++ fits_hdr2str() into a
    Python dictionary, minimizing overhead and providing comprehensive
    FITS keyword handling.
    """

    # Pre-compiled regex patterns for maximum performance
    _KEYWORD_PATTERN = re.compile(r"^(.{8})(=)\s*(.{70})$|^(.{8})\s*(.{72})$")

    # FITS value type patterns
    _STRING_PATTERN = re.compile(r"'([^']*(?:''[^']*)*)'")
    _COMPLEX_PATTERN = re.compile(
        r"^\(\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)$"
    )

    # Reserved FITS keywords that should always be strings
    _STRING_KEYWORDS = {
        "EXTNAME",
        "EXTTYPE",
        "COMMENT",
        "HISTORY",
        "CONTINUE",
        "CTYPE1",
        "CTYPE2",
        "CTYPE3",
        "CTYPE4",
        "CUNIT1",
        "CUNIT2",
        "CUNIT3",
        "CUNIT4",
        "OBJECT",
        "TELESCOP",
        "INSTRUME",
        "OBSERVER",
        "DATE-OBS",
        "DATE",
        "ORIGIN",
    }

    @classmethod
    def parse_header_string(cls, header_string: str) -> Dict[str, Any]:
        """
        Parse a FITS header string into a dictionary.

        Args:
            header_string: Raw header string from fits_hdr2str()

        Returns:
            Dictionary of header keywords and values

        Raises:
            ValueError: If header string is malformed
        """
        if not header_string:
            return {}

        header: Dict[str, Any] = {}
        _find_comment_separator = cls._find_comment_separator
        _parse_string_value = cls._parse_string_value
        string_keywords = cls._STRING_KEYWORDS
        complex_pattern = cls._COMPLEX_PATTERN
        startswith = str.startswith
        isspace = str.isspace

        # The keyword whose string value the next CONTINUE card extends.
        # FITS CONTINUE cards continue the string value of the preceding
        # string-valued card in the logical record.
        last_string_keyword: Optional[str] = None

        # LONGSTRN '&' continuation: a quoted value whose field ends with '&'
        # is continued by the following CONTINUE card(s); the '&' is dropped
        # and restored only if no CONTINUE card actually follows.
        pending_ampersand_keyword: Optional[str] = None

        # Pre-calculate string lengths and slices outside the loop
        str_len = len(header_string)

        # Iterate directly over the string by 80-character chunks
        # instead of building an intermediate list of cards
        for i in range(0, str_len, 80):
            card = header_string[i : i + 80]
            keyword = None

            # Any card that is not a CONTINUE card ends a LONGSTRN '&' chain:
            # the '&' then belongs to the value itself.
            if (
                not startswith(card, "CONTINUE")
                and pending_ampersand_keyword is not None
            ):
                prev_val = header.get(pending_ampersand_keyword)
                if isinstance(prev_val, str):
                    header[pending_ampersand_keyword] = prev_val + "&"
                pending_ampersand_keyword = None

            # Bolt optimization: stop parsing immediately at first END card.
            # FITS headers are padded with 2880-byte blocks of spaces. Breaking
            # early avoids thousands of redundant regex/string checks on empty padding.
            if startswith(card, "END     "):
                break

            if not card or isspace(card):
                continue

            if len(card) < 80:
                card = card.ljust(80)

            # Most FITS cards have an '=' at index 8.
            if card[8] == "=":
                keyword = card[:8].rstrip()
                value_comment = card[9:].strip()

                # Find comment separator fast check
                idx = value_comment.find("/")
                if idx == -1:
                    value_str = value_comment
                    comment = None
                elif "'" not in value_comment[:idx]:
                    value_str = value_comment[:idx].strip()
                    comment = value_comment[idx + 1 :].strip()
                else:
                    # Fallback to precise check
                    comment_start = _find_comment_separator(value_comment)
                    if comment_start != -1:
                        value_str = value_comment[:comment_start].strip()
                        comment = value_comment[comment_start + 1 :].strip()
                    else:
                        value_str = value_comment
                        comment = None

                value: Any = None
                value_ends_ampersand = False
                if value_str:
                    first_char = value_str[0]
                    if first_char == "'":
                        value = _parse_string_value(value_str)
                        if value_str.endswith("&'"):
                            # LONGSTRN marker: strip the '&'; a CONTINUE card
                            # must follow (restored if it does not).
                            value = value[:-1]
                            value_ends_ampersand = True
                    elif keyword in string_keywords:
                        value = value_str
                    elif first_char in "+-0123456789.":
                        parsed = _parse_fits_number(value_str)
                        if parsed is not None:
                            value = parsed

                    if value is None:
                        if value_str == "T":
                            value = True
                        elif value_str == "F":
                            value = False
                        elif first_char == "(":
                            complex_match = complex_pattern.match(value_str)
                            if complex_match:
                                real_part = float(complex_match.group(1))
                                imag_part = float(complex_match.group(2))
                                value = complex(real_part, imag_part)

                        if value is None:
                            value = value_str

                if keyword:
                    header[keyword] = value
                    if isinstance(value, str):
                        last_string_keyword = keyword
                        # A quoted value ending in '&' expects a CONTINUE card
                        # next; a non-string value cannot start a chain.
                        pending_ampersand_keyword = (
                            keyword if value_ends_ampersand else None
                        )
                    else:
                        pending_ampersand_keyword = None
                    if comment:
                        header[f"{keyword}_COMMENT"] = comment
            elif startswith(card, ("COMMENT ", "HISTORY ", "CONTINUE")):
                keyword = card[:8].rstrip()
                if keyword == "CONTINUE" and last_string_keyword is not None:
                    # CONTINUE: append the segment to the preceding string
                    # value. Per the FITS standard, trailing blanks of each
                    # segment are dropped before concatenation.
                    segment_raw = card[8:]
                    comment_start = _find_comment_separator(segment_raw)
                    if comment_start != -1:
                        segment = segment_raw[:comment_start].strip()
                    else:
                        segment = segment_raw.strip()
                    if segment:
                        has_marker = segment.startswith("'") and segment.endswith("&'")
                        seg_value = (
                            _parse_string_value(segment)
                            if segment[0] == "'"
                            else segment
                        )
                        if (
                            has_marker
                            and isinstance(seg_value, str)
                            and seg_value.endswith("&")
                        ):
                            # The marker means "more coming": it is notation,
                            # not content (a comment after the quote must not
                            # hide it — detection above uses the stripped
                            # value field).
                            seg_value = seg_value[:-1]
                        prev = header.get(last_string_keyword)
                        if isinstance(prev, str):
                            header[last_string_keyword] = prev + seg_value
                        # Each non-final LONGSTRN segment itself ends in '&',
                        # so the chain stays open only while that marker is
                        # present on the just-consumed card.
                        pending_ampersand_keyword = (
                            last_string_keyword if has_marker else None
                        )
                elif keyword:
                    header[keyword] = card[8:].strip()
                    if keyword in string_keywords:
                        last_string_keyword = keyword
            else:
                # No equals sign - might be a comment-only keyword
                keyword = card[:8].rstrip()
                if keyword == "HIERARCH" and "=" in card[8:]:
                    # ESO convention: expose the full long keyword with a
                    # typed value (matches _parse_hierarch_card).
                    rest = card[8:]
                    eq = rest.find("=")
                    hier_key = rest[:eq].strip()
                    value_comment = rest[eq + 1 :].strip()
                    idx2 = value_comment.find("/")
                    if idx2 == -1 or ("'" in value_comment[:idx2]):
                        cstart = _find_comment_separator(value_comment)
                        vstr = (
                            value_comment[:cstart].strip()
                            if cstart != -1
                            else value_comment
                        )
                    else:
                        # Dict parser stores values only; the comment text
                        # after '/' is dropped here.
                        vstr = value_comment[:idx2].strip()
                    value2: Any = None
                    if vstr:
                        fc2 = vstr[0]
                        if fc2 == "'":
                            value2 = _parse_string_value(vstr)
                        elif fc2 == '"':
                            value2 = vstr.strip('"')
                        elif fc2 in "+-0123456789.":
                            parsed2 = _parse_fits_number(vstr)
                            if parsed2 is not None:
                                value2 = parsed2
                        if value2 is None:
                            if vstr == "T":
                                value2 = True
                            elif vstr == "F":
                                value2 = False
                            else:
                                value2 = vstr
                    if hier_key:
                        header[hier_key] = value2 if value2 is not None else vstr
                        if isinstance(value2, str):
                            last_string_keyword = hier_key
                elif keyword:
                    header[keyword] = card[8:].strip()

        return header

    @classmethod
    def _parse_card(cls, card: str) -> tuple[Optional[str], Any, Optional[str]]:
        """
        Parse a single 80-character FITS card.

        Returns:
            (keyword, value, comment) tuple
        """
        if len(card) != 80:
            card = card.ljust(80)

        # Skip empty cards
        if card.isspace() or not card:
            return None, None, None

        # Handle comment-only cards (COMMENT, HISTORY, etc.)
        if card.startswith(("COMMENT ", "HISTORY ", "CONTINUE")):
            keyword = card[:8].strip()
            if keyword == "CONTINUE":
                # Parse the continuation segment: split off any comment,
                # unquote, and strip trailing blanks like a string value.
                segment_raw = card[8:]
                comment_start = cls._find_comment_separator(segment_raw)
                if comment_start != -1:
                    segment = segment_raw[:comment_start].strip()
                    comment = segment_raw[comment_start + 1 :].strip()
                else:
                    segment = segment_raw.strip()
                    comment = None
                if segment and segment[0] == "'":
                    segment_value = cls._parse_string_value(segment)
                else:
                    segment_value = segment
                return keyword, segment_value, comment
            value: Any = card[8:].strip()
            return keyword, value, None

        # Look for equals sign at position 8
        if len(card) > 8 and card[8] == "=":
            keyword = card[:8].strip()
            value_comment = card[9:].strip()

            # Find comment separator
            comment_start = cls._find_comment_separator(value_comment)
            if comment_start != -1:
                value_str = value_comment[:comment_start].strip()
                comment = value_comment[comment_start + 1 :].strip()
            else:
                value_str = value_comment
                comment = None

            # Parse the value (inlined from _parse_value for performance)
            value = None
            if value_str:
                pv = value_str.strip()
                if pv:
                    fc = pv[0]
                    if fc == "'":
                        value = cls._parse_string_value(pv)
                    elif keyword in cls._STRING_KEYWORDS:
                        value = pv
                    elif fc in "+-0123456789.":
                        try:
                            if "." in pv or "e" in pv or "E" in pv:
                                value = float(pv)
                            else:
                                value = int(pv)
                        except ValueError:
                            pass
                    if value is None:
                        if pv == "T":
                            value = True
                        elif pv == "F":
                            value = False
                        elif fc == "(":
                            complex_match = cls._COMPLEX_PATTERN.match(pv)
                            if complex_match:
                                value = complex(
                                    float(complex_match.group(1)),
                                    float(complex_match.group(2)),
                                )
                        if value is None:
                            value = pv
            return keyword, value, comment
        else:
            # No equals sign - might be a comment-only keyword
            keyword = card[:8].strip()
            if keyword == "HIERARCH":
                # ESO convention: ``HIERARCH ESO TEL AMBI TEMP = 12.5 / c``
                # exposes the full dotted name as the keyword with a typed
                # value, instead of a raw "HIERARCH" -> text mapping.
                return cls._parse_hierarch_card(card)
            if keyword:
                return keyword, card[8:].strip(), None

        return None, None, None

    @classmethod
    def _parse_hierarch_card(cls, card: str) -> tuple[str, Any, Optional[str]]:
        """Parse an ESO HIERARCH card into ``(full_keyword, value, comment)``."""
        rest = card[8:]
        eq = rest.find("=")
        if eq <= 0:
            return "HIERARCH", rest.strip(), None
        long_key = rest[:eq].strip() or "HIERARCH"
        value_comment = rest[eq + 1 :]
        # Reuse the standard value parser through a synthetic normal card.
        probe = (long_key[:8].ljust(8) + "=" + value_comment).ljust(80)
        _kw, value, comment = cls._parse_card(probe)
        return long_key, value, comment

    @classmethod
    def _find_comment_separator(cls, value_comment: str) -> int:
        """
        Find the position of the comment separator ('/').

        Handles quoted strings properly to avoid false positives.
        """
        idx = value_comment.find("/")
        if idx == -1:
            return -1

        quote_idx = value_comment.find("'")
        if quote_idx == -1 or quote_idx > idx:
            return idx

        # Fast path: check if the first slash comes after the LAST quote in the string
        r_quote_idx = value_comment.rfind("'")
        if idx > r_quote_idx:
            return idx

        in_quotes = False
        i = 0
        n = len(value_comment)
        while i < n:
            char = value_comment[i]
            if char == "'":
                if in_quotes and i + 1 < n and value_comment[i + 1] == "'":
                    # Escaped quote inside string
                    i += 2
                    continue
                else:
                    # Toggle quote state
                    in_quotes = not in_quotes
            elif char == "/" and not in_quotes:
                return i
            i += 1
        return -1

    @classmethod
    def _parse_string_value(cls, quoted_str: str) -> str:
        """
        Parse a quoted FITS string value.

        Handles escaped quotes and proper string termination.
        """
        if not quoted_str.startswith("'"):
            return quoted_str

        # Check if there are internal quotes. If not, simple slice is fastest.
        # The string ends with the first unmatched quote.
        # For a string without escaped quotes (''), it's just stripping first/last quote.
        end_idx = quoted_str.find("'", 1)
        if end_idx == -1:
            return quoted_str[1:]

        if "''" not in quoted_str:
            return quoted_str[1:end_idx].rstrip()

        # We search for the first unescaped quote AFTER the opening quote.
        # We use [1:] to protect the opening quote at index 0 from being replaced.
        end_idx = quoted_str[1:].replace("''", "  ").find("'")
        if end_idx == -1:
            return quoted_str[1:].replace("''", "'").rstrip()
        # Add 1 back to end_idx because we searched in quoted_str[1:]
        return quoted_str[1 : end_idx + 1].replace("''", "'").rstrip()


def _parse_fits_number(text: str) -> "float | int | None":
    """Parse a FITS numeric literal; return None when it is not one.

    Accepts Fortran-style D exponents (``1D5``), which legacy headers use and
    Python's float() rejects. Rejects Python-only spellings that are not FITS
    numbers (underscore separators such as ``1_0``).
    """
    cleaned = text.strip()
    if "_" in cleaned or not cleaned:
        return None
    try:
        if "." in cleaned or "e" in cleaned or "E" in cleaned:
            return float(cleaned.replace("D", "E").replace("d", "e"))
        return int(cleaned)
    except ValueError:
        try:
            return float(cleaned.replace("D", "E").replace("d", "e"))
        except ValueError:
            return None


def fast_parse_header(header_string: str) -> Dict[str, Any]:
    """
    Convenience function for fast header parsing.

    Args:
        header_string: Raw header string from C++ fits_hdr2str()

    Returns:
        Dictionary of header keywords and values
    """
    return FastHeaderParser.parse_header_string(header_string)


def fast_parse_header_cards(
    header_string: str,
) -> list[tuple[str, Any, str]]:
    """Parse a FITS header string into a list of (keyword, value, comment) tuples.

    Args:
        header_string: Raw header string from C++ fits_hdr2str()

    Returns:
        List of ``(keyword, value, comment)`` tuples. ``comment`` is ``""``
        when no comment separator was found on the card.
    """
    cards: list[tuple[str, Any, str]] = []
    str_len = len(header_string)
    # Index into cards of the last string-valued card; LONGSTRN '&' chains
    # and bare CONTINUE cards extend its value.
    last_value_index = -1
    # Index of a card whose quoted value ended in '&' (LONGSTRN marker). The
    # marker is dropped and restored only if no CONTINUE card follows.
    pending_ampersand = -1
    for i in range(0, str_len, 80):
        card = header_string[i : i + 80]
        if card.startswith("END     "):
            if pending_ampersand != -1:
                pk, pv, pc = cards[pending_ampersand]
                cards[pending_ampersand] = (pk, pv + "&", pc)
            break
        if not card or card.isspace():
            continue
        if len(card) < 80:
            card = card.ljust(80)
        kw, val, comment = FastHeaderParser._parse_card(card)
        if kw is None:
            continue

        if kw == "CONTINUE":
            if last_value_index != -1 and isinstance(cards[last_value_index][1], str):
                pk, pv, pc = cards[last_value_index]
                # Chain state must be decided on the VALUE field only: a
                # trailing comment (``CONTINUE 'more &' / note``) must not
                # hide the LONGSTRN ``&'`` marker.
                raw_field = card[8:]
                cpos = FastHeaderParser._find_comment_separator(raw_field)
                seg_field = (raw_field[:cpos] if cpos != -1 else raw_field).strip()
                has_marker = seg_field.startswith("'") and seg_field.endswith("&'")
                seg_val = val if isinstance(val, str) else str(val)
                if has_marker and seg_val.endswith("&"):
                    # This segment continues the chain: its own marker is
                    # notation, not content.
                    seg_val = seg_val[:-1]
                cards[last_value_index] = (pk, pv + seg_val, pc)
                pending_ampersand = last_value_index if has_marker else -1
            else:
                # Orphan CONTINUE card (malformed): keep it visible as-is.
                cards.append((kw, val, "" if comment is None else str(comment)))
            continue

        if pending_ampersand != -1:
            # The '&' was not continued: it is part of the value.
            pk, pv, pc = cards[pending_ampersand]
            cards[pending_ampersand] = (pk, pv + "&", pc)
            pending_ampersand = -1
        cards.append((kw, val, "" if comment is None else str(comment)))
        if isinstance(val, str):
            last_value_index = len(cards) - 1
        if isinstance(val, str) and len(card) > 8 and card[8] == "=":
            # LONGSTRN marker detection on the VALUE field only: a comment
            # after the closing quote must not hide the ``&'`` marker.
            raw_field = card[9:]
            cpos = FastHeaderParser._find_comment_separator(raw_field)
            value_field = (raw_field[:cpos] if cpos != -1 else raw_field).strip()
            if value_field.startswith("'") and value_field.endswith("&'"):
                # Drop the '&' from the stored value; a CONTINUE card must
                # follow (restored if it does not).
                cards[last_value_index] = (
                    kw,
                    val[:-1],
                    "" if comment is None else str(comment),
                )
                pending_ampersand = last_value_index
    return cards
