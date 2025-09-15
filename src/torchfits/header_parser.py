"""
Fast Python-side header parser for FITS headers.

This module provides optimized parsing of FITS header strings returned by the
bulk C++ read_header_to_string() function, avoiding Python/C++ round trips.
"""

import re
from typing import Dict, Optional, Union, Any


class FastHeaderParser:
    """
    High-performance FITS header parser.
    
    Parses the raw header string returned by C++ fits_hdr2str() into a 
    Python dictionary, minimizing overhead and providing comprehensive
    FITS keyword handling.
    """
    
    # Pre-compiled regex patterns for maximum performance
    _KEYWORD_PATTERN = re.compile(
        r'^(.{8})(=)\s*(.{70})$|^(.{8})\s*(.{72})$'
    )
    
    # FITS value type patterns
    _STRING_PATTERN = re.compile(r"'([^']*(?:''[^']*)*)'")
    _INTEGER_PATTERN = re.compile(r'^[+-]?\d+$')
    _FLOAT_PATTERN = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')
    _LOGICAL_PATTERN = re.compile(r'^[TF]$')
    _COMPLEX_PATTERN = re.compile(r'^\(\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)$')
    
    # Reserved FITS keywords that should always be strings
    _STRING_KEYWORDS = {
        'EXTNAME', 'EXTTYPE', 'COMMENT', 'HISTORY', 'CONTINUE',
        'CTYPE1', 'CTYPE2', 'CTYPE3', 'CTYPE4', 'CUNIT1', 'CUNIT2', 'CUNIT3', 'CUNIT4',
        'OBJECT', 'TELESCOP', 'INSTRUME', 'OBSERVER', 'DATE-OBS', 'DATE', 'ORIGIN'
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
            
        header = {}
        cards = cls._split_into_cards(header_string)
        
        for card in cards:
            if not card.strip():
                continue
                
            keyword, value, comment = cls._parse_card(card)
            if keyword:
                header[keyword] = value
                
                # Store comment separately if present
                if comment and comment.strip():
                    header[f'{keyword}_COMMENT'] = comment.strip()
        
        return header
    
    @classmethod
    def _split_into_cards(cls, header_string: str) -> list:
        """Split header string into 80-character FITS cards."""
        cards = []
        for i in range(0, len(header_string), 80):
            card = header_string[i:i+80]
            if len(card) < 80:
                card = card.ljust(80)  # Pad short cards
            cards.append(card)
        return cards
    
    @classmethod
    def _parse_card(cls, card: str) -> tuple:
        """
        Parse a single 80-character FITS card.
        
        Returns:
            (keyword, value, comment) tuple
        """
        if len(card) != 80:
            card = card.ljust(80)
            
        # Skip END cards and empty cards
        if card.startswith('END     ') or card.strip() == '':
            return None, None, None
            
        # Handle comment-only cards (COMMENT, HISTORY, etc.)
        if card.startswith(('COMMENT ', 'HISTORY ', 'CONTINUE')):
            keyword = card[:8].strip()
            value = card[8:].strip()
            return keyword, value, None
            
        # Look for equals sign at position 8
        if len(card) > 8 and card[8] == '=':
            keyword = card[:8].strip()
            value_comment = card[9:].strip()
            
            # Find comment separator
            comment_start = cls._find_comment_separator(value_comment)
            if comment_start != -1:
                value_str = value_comment[:comment_start].strip()
                comment = value_comment[comment_start+1:].strip()
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
        in_quotes = False
        i = 0
        while i < len(value_comment):
            char = value_comment[i]
            if char == "'":
                if in_quotes and i + 1 < len(value_comment) and value_comment[i + 1] == "'":
                    # Escaped quote inside string
                    i += 2
                    continue
                else:
                    # Toggle quote state
                    in_quotes = not in_quotes
            elif char == '/' and not in_quotes:
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
        
        # Force string parsing for certain keywords
        if keyword in cls._STRING_KEYWORDS:
            if value_str.startswith("'") and value_str.endswith("'"):
                return cls._parse_string_value(value_str)
            else:
                return value_str
        
        # Try different value types in order of specificity
        
        # 1. String values (quoted)
        if value_str.startswith("'"):
            return cls._parse_string_value(value_str)
        
        # 2. Logical values
        logical_match = cls._LOGICAL_PATTERN.match(value_str)
        if logical_match:
            return value_str == 'T'
        
        # 3. Complex numbers
        complex_match = cls._COMPLEX_PATTERN.match(value_str)
        if complex_match:
            real_part = float(complex_match.group(1))
            imag_part = float(complex_match.group(2))
            return complex(real_part, imag_part)
        
        # 4. Integer values
        if cls._INTEGER_PATTERN.match(value_str):
            try:
                return int(value_str)
            except ValueError:
                pass
        
        # 5. Float values
        if cls._FLOAT_PATTERN.match(value_str):
            try:
                return float(value_str)
            except ValueError:
                pass
        
        # 6. Default to string
        return value_str
    
    @classmethod
    def _parse_string_value(cls, quoted_str: str) -> str:
        """
        Parse a quoted FITS string value.
        
        Handles escaped quotes and proper string termination.
        """
        if not quoted_str.startswith("'"):
            return quoted_str
            
        # Find the closing quote, handling escaped quotes
        content = ""
        i = 1  # Skip opening quote
        while i < len(quoted_str):
            char = quoted_str[i]
            if char == "'":
                if i + 1 < len(quoted_str) and quoted_str[i + 1] == "'":
                    # Escaped quote
                    content += "'"
                    i += 2
                else:
                    # End of string
                    break
            else:
                content += char
                i += 1
                
        return content
    
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
            'parse_time_ms': (end_time - start_time) * 1000,
            'header_size_bytes': len(header_string),
            'num_keywords': len(header),
            'throughput_kb_per_ms': len(header_string) / 1024 / ((end_time - start_time) * 1000) if end_time > start_time else 0
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


def benchmark_header_parsing(header_string: str, num_iterations: int = 100) -> Dict[str, float]:
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
        'avg_parse_time_ms': avg_time_ms,
        'throughput_headers_per_sec': throughput_headers_per_sec,
        'total_time_s': total_time,
        'header_size_bytes': len(header_string),
        'num_iterations': num_iterations
    }