"""
Fast Python-side header parser for FITS headers.

This module provides optimized parsing of FITS header strings returned by the
bulk C++ read_header_to_string() function, avoiding Python/C++ round trips.
"""

import re
from typing import Any, Dict


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
        _parse_value = cls._parse_value
        _find_comment_separator = cls._find_comment_separator

        # Pre-calculate string lengths and slices outside the loop
        str_len = len(header_string)

        # Iterate directly over the string by 80-character chunks
        # instead of building an intermediate list of cards
        for i in range(0, str_len, 80):
            card = header_string[i : i + 80]

            # Bolt optimization: stop parsing immediately at first END card.
            # FITS headers are padded with 2880-byte blocks of spaces. Breaking
            # early avoids thousands of redundant regex/string checks on empty padding.
            if card.startswith("END     "):
                break

            if not card or card.isspace():
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

                value = _parse_value(value_str, keyword)
                if keyword:
                    header[keyword] = value
                    if comment:
                        header[f"{keyword}_COMMENT"] = comment
            elif card.startswith(("COMMENT ", "HISTORY ", "CONTINUE")):
                keyword = card[:8].rstrip()
                if keyword:
                    header[keyword] = card[8:].strip()
            else:
                # No equals sign - might be a comment-only keyword
                if keyword:
                    header[keyword] = card[8:].strip()

        return header

    @classmethod
    def _parse_card(cls, card: str) -> tuple:
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
            value = card[8:].strip()
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

            # Parse the value
            value = cls._parse_value(value_str, keyword)
            return keyword, value, comment
        else:
            # No equals sign - might be a comment-only keyword
            keyword = card[:8].strip()
            if keyword:
                return keyword, card[8:].strip(), None

        return None, None, None

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
    def _parse_value(cls, value_str: str, keyword: str) -> Any:
        """
        Parse a FITS value string into appropriate Python type.

        Args:
            value_str: Value portion of FITS card
            keyword: Keyword name (for context-sensitive parsing)

        Returns:
            Parsed value in appropriate Python type
        """
        if not value_str:
            return None

        value_str = value_str.strip()
        if not value_str:
            return None

        first_char = value_str[0]

        # 1. String values (quoted)
        if first_char == "'":
            return cls._parse_string_value(value_str)

        # Force string parsing for certain keywords
        if keyword in cls._STRING_KEYWORDS:
            return value_str

        # 2. Fast path for numbers without regex
        if first_char in "+-0123456789.":
            try:
                if "." in value_str or "e" in value_str or "E" in value_str:
                    return float(value_str)
                return int(value_str)
            except ValueError:
                pass

        # 3. Logical values
        if value_str == "T":
            return True
        if value_str == "F":
            return False

        # 4. Complex numbers
        if first_char == "(":
            complex_match = cls._COMPLEX_PATTERN.match(value_str)
            if complex_match:
                real_part = float(complex_match.group(1))
                imag_part = float(complex_match.group(2))
                return complex(real_part, imag_part)

        # 5. Default to string
        return value_str

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
            return quoted_str[1:end_idx]

        end_idx = 1
        while True:
            end_idx = quoted_str.find("'", end_idx)
            if end_idx == -1:
                return quoted_str[1:].replace("''", "'")

            if end_idx + 1 < len(quoted_str) and quoted_str[end_idx + 1] == "'":
                end_idx += 2
            else:
                return quoted_str[1:end_idx].replace("''", "'")

    @classmethod
    def parse_with_performance_tracking(cls, header_string: str) -> tuple:
        """
        Parse header with performance metrics.

        Returns:
            (header_dict, metrics_dict) tuple
        """
        import time

        start_time = time.perf_counter()
        header = cls.parse_header_string(header_string)
        end_time = time.perf_counter()

        metrics = {
            "parse_time_ms": (end_time - start_time) * 1000,
            "header_size_bytes": len(header_string),
            "num_keywords": len(header),
            "throughput_kb_per_ms": (
                len(header_string) / 1024 / ((end_time - start_time) * 1000)
                if end_time > start_time
                else 0
            ),
        }

        return header, metrics


def fast_parse_header(header_string: str) -> Dict[str, Any]:
    """
    Convenience function for fast header parsing.

    Args:
        header_string: Raw header string from C++ fits_hdr2str()

    Returns:
        Dictionary of header keywords and values
    """
    return FastHeaderParser.parse_header_string(header_string)


def benchmark_header_parsing(
    header_string: str, num_iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark header parsing performance.

    Args:
        header_string: Header string to parse
        num_iterations: Number of parsing iterations

    Returns:
        Performance metrics dictionary
    """
    import time

    # Warmup
    FastHeaderParser.parse_header_string(header_string)

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        FastHeaderParser.parse_header_string(header_string)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    throughput_headers_per_sec = num_iterations / total_time

    return {
        "avg_parse_time_ms": avg_time_ms,
        "throughput_headers_per_sec": throughput_headers_per_sec,
        "total_time_s": total_time,
        "header_size_bytes": len(header_string),
        "num_iterations": num_iterations,
    }
