// Standalone self-check for has_cfitsio_extended_filename_syntax.
// No test framework needed: header-only, plain asserts.
//
// Run:
//   clang++ -std=c++17 -I src/torchfits/cpp_src tests/cpp/test_bracket_detection.cpp \
//       -o /tmp/test_bracket_detection && /tmp/test_bracket_detection
#include <cassert>
#include <iostream>

#include "security.h"

using torchfits::has_cfitsio_extended_filename_syntax;

int main() {
    // CFITSIO extended filename syntax: bracket section terminates the path.
    assert(has_cfitsio_extended_filename_syntax("file.fits[1]"));
    assert(has_cfitsio_extended_filename_syntax("file.fits[1:10,1:10]"));
    assert(has_cfitsio_extended_filename_syntax("/data/obs/file.fits[1]"));
    assert(has_cfitsio_extended_filename_syntax("file.fits[1][1:10,1:10]"));

    // False positives from a naive find('[') != npos check: literal '[' in a
    // directory component, not a trailing CFITSIO section.
    assert(!has_cfitsio_extended_filename_syntax("/home/user/[data]/file.fits"));
    assert(!has_cfitsio_extended_filename_syntax("/home/user/[data]/file.fits]"));

    // No brackets at all.
    assert(!has_cfitsio_extended_filename_syntax("/data/obs/file.fits"));
    assert(!has_cfitsio_extended_filename_syntax(""));

    std::cout << "test_bracket_detection: all checks passed\n";
    return 0;
}
