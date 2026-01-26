## 2025-05-15 - C++ ctype.h Undefined Behavior
**Vulnerability:** Using `std::isalnum` or `std::toupper` with signed `char` can lead to Undefined Behavior if the char is negative (high-bit/extended ASCII).
**Learning:** Functions in `<cctype>` expect `unsigned char` or `EOF`. On systems where `char` is signed, extended ASCII characters (e.g., from UTF-8 sequences) are negative integers, which can index out of bounds in lookup tables used by these functions.
**Prevention:** Always `static_cast<unsigned char>(c)` before passing `char` to `<cctype>` functions.

## 2025-05-15 - CFITSIO Command Injection via Filenames
**Vulnerability:** `cfitsio` library interprets filenames starting or ending with `|` as shell commands to execute and pipe. This allows command injection if user-controlled filenames are passed directly to `fits_open_file`.
**Learning:** Even "read-only" file opening functions in C/C++ libraries (like `fits_open_file` with `READONLY` mode) can be vulnerable to command injection if the library supports extended syntax like pipes. Always sanitize filenames before passing them to such libraries.
**Prevention:** Validate filenames against a whitelist or blacklist of dangerous prefixes/suffixes (like `|`) before calling the library function.
