## 2025-05-15 - C++ ctype.h Undefined Behavior
**Vulnerability:** Using `std::isalnum` or `std::toupper` with signed `char` can lead to Undefined Behavior if the char is negative (high-bit/extended ASCII).
**Learning:** Functions in `<cctype>` expect `unsigned char` or `EOF`. On systems where `char` is signed, extended ASCII characters (e.g., from UTF-8 sequences) are negative integers, which can index out of bounds in lookup tables used by these functions.
**Prevention:** Always `static_cast<unsigned char>(c)` before passing `char` to `<cctype>` functions.

## 2025-05-21 - Unbounded Thread Creation (DoS)
**Vulnerability:** The `read_images_batch` function created a new thread for every input file without limit. A large input list could exhaust system threads or file descriptors, leading to Denial of Service.
**Learning:** When parallelizing I/O operations based on user input (e.g., list of files), always impose an upper limit on concurrency. `std::thread` does not manage a pool; it spawns a system thread.
**Prevention:** Use a thread pool or batching mechanism to limit concurrent threads to `std::thread::hardware_concurrency()` or a safe fixed limit.
