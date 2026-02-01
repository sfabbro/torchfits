## 2025-05-15 - C++ ctype.h Undefined Behavior
**Vulnerability:** Using `std::isalnum` or `std::toupper` with signed `char` can lead to Undefined Behavior if the char is negative (high-bit/extended ASCII).
**Learning:** Functions in `<cctype>` expect `unsigned char` or `EOF`. On systems where `char` is signed, extended ASCII characters (e.g., from UTF-8 sequences) are negative integers, which can index out of bounds in lookup tables used by these functions.
**Prevention:** Always `static_cast<unsigned char>(c)` before passing `char` to `<cctype>` functions.

## 2025-05-21 - CFITSIO Command Injection via Overwrite Flag
**Vulnerability:** `cfitsio` command injection using pipe syntax (`|`) was blocked, but the block could be bypassed using the overwrite flag (`!`) which `torchfits` prepends when `overwrite=True`.
**Learning:** Security checks must account for all ways inputs are modified before reaching the vulnerable sink. Blocking `|` at the start is insufficient if the library prepends characters (like `!`) that the underlying library (cfitsio) interprets as flags before the pipe.
**Prevention:** Normalize or inspect the "effective" filename that the underlying library will process, or strip known flags before applying the security check.
