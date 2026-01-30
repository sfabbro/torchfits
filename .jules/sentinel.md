## 2025-05-15 - C++ ctype.h Undefined Behavior
**Vulnerability:** Using `std::isalnum` or `std::toupper` with signed `char` can lead to Undefined Behavior if the char is negative (high-bit/extended ASCII).
**Learning:** Functions in `<cctype>` expect `unsigned char` or `EOF`. On systems where `char` is signed, extended ASCII characters (e.g., from UTF-8 sequences) are negative integers, which can index out of bounds in lookup tables used by these functions.
**Prevention:** Always `static_cast<unsigned char>(c)` before passing `char` to `<cctype>` functions.

## 2025-05-15 - CFITSIO Command Injection
**Vulnerability:** CFITSIO treats filenames starting or ending with `|` as shell commands, allowing arbitrary command execution if filenames are not sanitized.
**Learning:** Libraries often support "extended syntax" for convenience (like pipes, URLs, memory files) that can be security hazards when exposed to untrusted input.
**Prevention:** Validate filenames against dangerous characters/patterns before passing to library functions. For CFITSIO, block `|` at start/end.
