## 2025-05-15 - C++ ctype.h Undefined Behavior
**Vulnerability:** Using `std::isalnum` or `std::toupper` with signed `char` can lead to Undefined Behavior if the char is negative (high-bit/extended ASCII).
**Learning:** Functions in `<cctype>` expect `unsigned char` or `EOF`. On systems where `char` is signed, extended ASCII characters (e.g., from UTF-8 sequences) are negative integers, which can index out of bounds in lookup tables used by these functions.
**Prevention:** Always `static_cast<unsigned char>(c)` before passing `char` to `<cctype>` functions.

## 2025-01-27 - cfitsio Command Injection Mitigation
**Vulnerability:** `cfitsio` library supports extended filename syntax where filenames starting or ending with `|` are interpreted as shell commands. Several entry points in `torchfits` C++ bindings passed user-provided filenames directly to `fits_open_file` or `fits_create_file` without validation, exposing the application to command injection.
**Learning:** Libraries like `cfitsio` often have "magic" filename handling that can be dangerous when exposed to untrusted input. Standard file I/O functions (like `open`, `mmap`) are safe, but library-specific openers need careful validation.
**Prevention:** Always sanitize or validate filenames before passing them to `cfitsio` functions. A centralized validation function `validate_fits_filename` was implemented to handle `|` checks and `!` overwrite prefix logic consistently across all C++ entry points.
