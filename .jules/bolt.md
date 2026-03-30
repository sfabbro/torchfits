## 2024-05-24 - FITS Header Parsing Optimization
**Learning:** FITS headers are padded with spaces to fill 2880-byte blocks, meaning after the `END` keyword, there can be thousands of bytes of empty space. Parsing these empty characters introduces significant overhead for large FITS files.
**Action:** When reading FITS header blocks, always break parsing loops early as soon as the `END     ` card is encountered to skip processing trailing padding.

## 2025-03-05 - FITS Header Parsing Optimization
**Learning:** In a performance-sensitive parser loop like reading FITS headers (which can have thousands of 80-character cards), Python function call overhead and class attribute lookups (e.g., `cls._STRING_KEYWORDS`) can be a significant bottleneck. Inlining the parsing logic and aliasing class attributes to local variables provided a ~20% speedup.
**Action:** When working on tight loops or parsing functions, look for opportunities to inline small helper functions and locally alias frequently accessed class attributes to reduce lookup overhead.

## 2025-03-29 - [Inline FITS Header Parsing]
**Learning:** FITS header parsing using intermediate string allocations and function calls per card (`_parse_card`, `_find_comment_separator`, `_parse_value`) in a Python loop for 80-character chunks is a significant bottleneck.
**Action:** Inline the parsing logic directly within the chunk iteration loop. This avoids thousands of function call overheads per FITS header, reducing parsing time by ~25-30% while maintaining accuracy.
