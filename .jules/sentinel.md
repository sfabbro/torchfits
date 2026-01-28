## 2025-05-15 - C++ ctype.h Undefined Behavior
**Vulnerability:** Using `std::isalnum` or `std::toupper` with signed `char` can lead to Undefined Behavior if the char is negative (high-bit/extended ASCII).
**Learning:** Functions in `<cctype>` expect `unsigned char` or `EOF`. On systems where `char` is signed, extended ASCII characters (e.g., from UTF-8 sequences) are negative integers, which can index out of bounds in lookup tables used by these functions.
**Prevention:** Always `static_cast<unsigned char>(c)` before passing `char` to `<cctype>` functions.

## 2025-05-23 - cfitsio Command Injection
**Vulnerability:** The `cfitsio` library interprets filenames starting or ending with `|` as shell commands to execute. This feature allows command injection if user-provided filenames are passed directly to `fits_open_file` without sanitization.
**Learning:** High-performance C libraries often include "convenience" features like pipe execution or URL fetching that become security vulnerabilities when exposed to untrusted input. Native checks (like  constructor) can be bypassed if other entry points () are added without reusing the validation logic.
**Prevention:** Centralize filename validation logic in a shared utility function and enforce its usage across all C++ entry points that interact with the underlying library.

## 2025-05-23 - cfitsio Command Injection
**Vulnerability:** The `cfitsio` library interprets filenames starting or ending with `|` as shell commands to execute. This feature allows command injection if user-provided filenames are passed directly to `fits_open_file` without sanitization.
**Learning:** High-performance C libraries often include "convenience" features like pipe execution or URL fetching that become security vulnerabilities when exposed to untrusted input. Native checks (like `FITSFileV2` constructor) can be bypassed if other entry points (`read_image_fast`) are added without reusing the validation logic.
**Prevention:** Centralize filename validation logic in a shared utility function and enforce its usage across all C++ entry points that interact with the underlying library.
