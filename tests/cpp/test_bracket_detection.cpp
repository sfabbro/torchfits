// Standalone self-check for has_cfitsio_extended_filename_syntax.
// No test framework needed: header-only, plain checks.
//
// Failures are counted and reported by a `check()` helper rather than written
// as `assert(...)`. Under -DNDEBUG every assert() compiles to nothing, so an
// assert-based self-check prints "all checks passed" and exits 0 having
// verified nothing at all; the count is a real runtime value and cannot be
// compiled away. tests/test_security.py compiles and runs this file, so the
// "all checks passed" line and the exit status are the contract it asserts on.
//
// Run:
//   clang++ -std=c++17 -I src/torchfits/cpp_src tests/cpp/test_bracket_detection.cpp \
//       -o /tmp/test_bracket_detection && /tmp/test_bracket_detection
#include <cstdio>

#include "security.h"

using torchfits::has_cfitsio_extended_filename_syntax;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "  ok  " : " FAIL ", what);
    if (!ok) ++failures;
}

}  // namespace

int main() {
    // CFITSIO extended filename syntax: bracket section terminates the path.
    check(has_cfitsio_extended_filename_syntax("file.fits[1]"), "file.fits[1]");
    check(has_cfitsio_extended_filename_syntax("file.fits[1:10,1:10]"),
          "file.fits[1:10,1:10]");
    check(has_cfitsio_extended_filename_syntax("/data/obs/file.fits[1]"),
          "absolute path with a trailing section");
    check(has_cfitsio_extended_filename_syntax("file.fits[1][1:10,1:10]"),
          "two stacked sections");

    // False positives from a naive find('[') != npos check: literal '[' in a
    // directory component, not a trailing CFITSIO section.
    check(!has_cfitsio_extended_filename_syntax("/home/user/[data]/file.fits"),
          "'[' inside a directory component is not a section");
    check(!has_cfitsio_extended_filename_syntax("/home/user/[data]/file.fits]"),
          "unbalanced bracket in a directory component is not a section");

    // No brackets at all.
    check(!has_cfitsio_extended_filename_syntax("/data/obs/file.fits"),
          "plain path");
    check(!has_cfitsio_extended_filename_syntax(""), "empty path");

    if (failures == 0) {
        std::printf("test_bracket_detection: all checks passed\n");
        return 0;
    }
    std::printf("test_bracket_detection: FAILURES: %d\n", failures);
    return 1;
}
