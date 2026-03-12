## 2026-03-04 - [Python logic optimization in I/O path]
**Learning:** Found redundant conditionals and duplicate code in the Python I/O wrapper (`torchfits/io.py`) that handles tensor scaling based on the target device. Replacing nested/repeated conditions with a single clean conditional branch reduces Python interpreter overhead.
**Action:** When working in performance-critical Python wrappers around C++ extensions, minimize branching, repeated `if` statements, and redundant variable lookups. Use optimized single conditionals and ternary operators for variable assignment where possible.

## 2026-03-10 - [String parsing optimization in Python]
**Learning:** Python-level `while` loops iterating character-by-character over strings (e.g., FITS header parsing in `header_parser.py`) are extremely slow. They can often be replaced by built-in C-optimized string methods like `find()`, `count()`, and `replace()`. For example, tracking `in_quotes` state in a loop can be replaced by checking if `string.count("'") % 2 == 0`.
**Action:** When working on performance-critical string parsing in Python, avoid character-by-character loops. Always prefer built-in methods like `find`, `count`, `split`, and `replace`.

## $(date +%Y-%m-%d) - [FITS Header Parsing Optimization]
**Learning:** Python function call overhead and string operations in tight loops (like parsing an 80-character card for FITS headers) can dominate performance.
**Action:** Inline operations like `strip()` with faster alternatives like `isspace()`. Reduce string slicing by utilizing `find()` boundaries like `find("'", 0, idx) == -1`.
