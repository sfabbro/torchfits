// Standalone self-check: one FitsReader shared by several threads.
//
// Regression for a real defect. CFITSIO keeps one mutable current-HDU cursor per
// fitsfile handle. FitsReader's accessors used to lock only around the move
// (fits_movabs_hdu) and then call the fits_* query that depends on that cursor
// with the lock released, so two threads sharing a reader could interleave into
// a query about another HDU, a spurious "Could not read image dimensions", or
// a segfault. Measured on a 37-HDU Rice-compressed MegaCam MEF with 4 threads:
// wrong answers, spurious exceptions, and a crash. After holding the lock across
// the whole move-then-read: 80,000 iterations, zero of each.
//
// The answer for each HDU is read serially first, so a probe bug cannot be
// mistaken for a race, and every threaded call is wrapped so a spurious throw is
// counted rather than terminating the process. The shape is compared in full --
// every axis, not just the width -- because a cursor race that returned the
// right NAXIS1 with another HDU's NAXIS2 would otherwise read as correct.
//
// Requirements (checked before any threads start) are reported and end the run
// with a non-zero status rather than as `assert`, so a -DNDEBUG build cannot
// skip them and then report success.
//
// Run it with the shared runner, which finds the built library and CFITSIO
// and writes a suitable multi-extension MEF for you:
//   tests/cpp/run_all.sh
//
// The build needs the torch-free core library and CFITSIO's headers/archive,
// and neither is installed into the pixi environment -- they live in the
// per-build scratch tree (.pixi/bld/torchfits/<hash>/bld), whose hash changes on
// every rebuild. That is why this file used to carry a literal
// "<build>/libtorchfits_core.dylib" placeholder that had to be hand-substituted
// before the command could be used at all.
//
// Standalone equivalent, if you have already located those two artifacts:
//   c++ -std=c++17 -O2 -I src/torchfits/cpp_src -I extern/cfitsio \
//       tests/cpp/test_fitsreader_threads.cpp <core-lib> <cfitsio-archive> \
//       -lpthread -o /tmp/probe
//   /tmp/probe <mef.fits>        # any FITS file with >= 2 usable HDUs
#include <atomic>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "core/metadata_api.h"

using torchfits::core::FitsReader;

namespace {

// Every axis of the shape, comma-separated, so a reader that returned the right
// width with the wrong height is caught.
std::string shape_str(const std::vector<long>& shape) {
    std::string out;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out += "x";
        out += std::to_string(shape[i]);
    }
    return out.empty() ? "<empty>" : out;
}

// A precondition: report it and end the run, rather than assert() and hope.
bool require(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "  ok  " : " FAIL ", what);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <multi-extension.fits>\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];

    FitsReader reader(path);
    const int num_hdus = reader.num_hdus();
    std::printf("num_hdus=%d\n", num_hdus);
    if (!require(num_hdus >= 2, "the file has at least two HDUs")) return 1;

    // Serial baseline. HDUs that raise are excluded from the threaded pass and
    // reported, so an unsupported HDU cannot masquerade as a race.
    std::vector<std::string> want_type(static_cast<size_t>(num_hdus));
    std::vector<std::string> want_shape(static_cast<size_t>(num_hdus));
    std::vector<int> usable;
    for (int hdu = 0; hdu < num_hdus; ++hdu) {
        try {
            want_type[static_cast<size_t>(hdu)] = reader.hdu_type(hdu);
            want_shape[static_cast<size_t>(hdu)] = shape_str(reader.shape(hdu));
            usable.push_back(hdu);
        } catch (const std::exception& exc) {
            std::printf("  hdu %d raises on shape(): %s\n", hdu, exc.what());
        }
    }
    std::printf("usable hdus: %zu\n", usable.size());
    if (!require(usable.size() >= 2, "at least two HDUs answer serially")) return 1;

    std::atomic<long> wrong_type{0};
    std::atomic<long> wrong_shape{0};
    std::atomic<long> spurious{0};
    std::atomic<long> iters{0};

    auto worker = [&](int hdu) {
        for (int i = 0; i < 20000; ++i) {
            try {
                if (reader.hdu_type(hdu) != want_type[static_cast<size_t>(hdu)]) {
                    wrong_type.fetch_add(1);
                }
                if (shape_str(reader.shape(hdu)) != want_shape[static_cast<size_t>(hdu)]) {
                    wrong_shape.fetch_add(1);
                }
            } catch (const std::exception&) {
                // A throw here is a defect: the same call succeeded serially.
                spurious.fetch_add(1);
            }
            iters.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < usable.size() && i < 4; ++i) {
        threads.emplace_back(worker, usable[i]);
    }
    for (auto& thread : threads) thread.join();

    std::printf("iters=%ld wrong_hdu_type=%ld wrong_shape=%ld spurious_exceptions=%ld\n",
                iters.load(), wrong_type.load(), wrong_shape.load(), spurious.load());

    if (wrong_type.load() || wrong_shape.load() || spurious.load()) {
        std::fprintf(stderr,
                     "test_fitsreader_threads: FAILED -- a shared reader returned another "
                     "HDU's answer, or threw spuriously\n");
        return 1;
    }
    std::cout << "test_fitsreader_threads: all checks passed\n";
    return 0;
}
