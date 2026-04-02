## Bolt's Journal

## 2024-03-24 - Fast Path Parsing FITS Headers
**Learning:** `FastHeaderParser.parse_header_string` in `src/torchfits/header_parser.py` was a performance bottleneck because FITS headers are parsed card-by-card, resulting in thousands of function calls to `_parse_card` per file. Additionally, `_parse_where_expression` was repeatedly parsing identical queries (e.g. `MAG_G < 20.0`) during chunked reads.
**Action:** Inlined the `_parse_card` logic directly into the 80-character chunk loop and avoided `str.strip()` overhead by using `rstrip()` where appropriate. Added a fast-path for `/` comment separator detection. Added an `@lru_cache` to `_parse_where_expression` to avoid redundant AST parsing during streaming/chunked table reads.

## 2024-05-14 - Fast EXTNAME lookup in HDUList
**Learning:** `HDUList.__getitem__` iterated over all HDUs to find a matching EXTNAME, which is O(N) and slow for files with many extensions, or when accessed repeatedly.
**Action:** Adding a cached dictionary mapping `EXTNAME` to HDU index makes lookup O(1) and ~40x faster for repeated lookups with many HDUs.

## 2024-05-24 - FITS Header Parsing Optimization
**Learning:** FITS headers are padded with spaces to fill 2880-byte blocks, meaning after the `END` keyword, there can be thousands of bytes of empty space. Parsing these empty characters introduces significant overhead for large FITS files.
**Action:** When reading FITS header blocks, always break parsing loops early as soon as the `END     ` card is encountered to skip processing trailing padding.

## 2025-03-05 - FITS Header Parsing Optimization
**Learning:** In a performance-sensitive parser loop like reading FITS headers (which can have thousands of 80-character cards), Python function call overhead and class attribute lookups (e.g., `cls._STRING_KEYWORDS`) can be a significant bottleneck. Inlining the parsing logic and aliasing class attributes to local variables provided a ~20% speedup.
**Action:** When working on tight loops or parsing functions, look for opportunities to inline small helper functions and locally alias frequently accessed class attributes to reduce lookup overhead.

## 2025-03-29 - [Inline FITS Header Parsing]
**Learning:** FITS header parsing using intermediate string allocations and function calls per card (`_parse_card`, `_find_comment_separator`, `_parse_value`) in a Python loop for 80-character chunks is a significant bottleneck.
**Action:** Inlined the parsing logic directly within the chunk iteration loop. This avoids thousands of function call overheads per FITS header, reducing parsing time by ~25-30% while maintaining accuracy.

## 2025-03-30 - Fast String Parsing for FITS Headers
**Learning:** Python string methods like `find`, `rfind`, and `in` operators are significantly faster than iterating character-by-character or taking string slices, since string slicing allocates a new string in memory. FITS headers often contain simple string values without escaped quotes (`''`) or slashes inside strings, meaning we can use fast string methods to parse them.
**Action:** When parsing well-formatted text data like FITS headers in Python, prioritize fast paths using built-in string methods (`find`, `rfind`, `in`) before falling back to manual character iteration. This is especially true for strings, where slicing creates new objects in memory.

## 2026-04-02 - Batched Quantile Calculations
**Learning:** Calculating multiple quantiles sequentially (e.g., calling `torch.quantile(tensor, 0.05)` and then `torch.quantile(tensor, 0.95)`) is a major performance bottleneck because PyTorch performs a full or partial sort of the tensor for each call. Batching these into a single call with a tensor of probabilities (e.g., `torch.quantile(tensor, torch.tensor([0.05, 0.95]))`) avoids redundant sorting and can reduce calculation time by ~50%.
**Action:** Always batch quantile calculations when multiple quantiles are needed from the same tensor to prevent unnecessary repeated sorting overhead.
