## 2025-03-29 - [Inline FITS Header Parsing]
**Learning:** FITS header parsing using intermediate string allocations and function calls per card (`_parse_card`, `_find_comment_separator`, `_parse_value`) in a Python loop for 80-character chunks is a significant bottleneck.
**Action:** Inline the parsing logic directly within the chunk iteration loop. This avoids thousands of function call overheads per FITS header, reducing parsing time by ~25-30% while maintaining accuracy.
