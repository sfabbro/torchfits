## 2026-03-04 - [Python logic optimization in I/O path]
**Learning:** Found redundant conditionals and duplicate code in the Python I/O wrapper (`torchfits/io.py`) that handles tensor scaling based on the target device. Replacing nested/repeated conditions with a single clean conditional branch reduces Python interpreter overhead.
**Action:** When working in performance-critical Python wrappers around C++ extensions, minimize branching, repeated `if` statements, and redundant variable lookups. Use optimized single conditionals and ternary operators for variable assignment where possible.
