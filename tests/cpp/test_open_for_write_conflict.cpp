// Standalone self-check: the CFITSIO contract that open_fits_for_write relies on.
//
// `open_fits_for_write` (fits_rw.h) opens a file READWRITE and, if that fails
// with status 104, evicts any cached TableReader and retries once. That retry
// exists because CFITSIO refuses to reopen a file READWRITE while a handle for
// it is still registered as READONLY: fits_already_open() in the vendored
// extern/cfitsio/cfileio.c returns FILE_NOT_OPENED.
//
// This probe pins both halves of that contract, because the number is load-
// bearing and nothing else in the suite could notice it changing:
//
//   1. With a READONLY handle held, opening READWRITE returns FILE_NOT_OPENED.
//      Without this the retry never fires and read-then-mutate breaks.
//   2. FILE_NOT_OPENED is CFITSIO's *generic* open-failure status, not a
//      conflict-specific one -- a missing directory returns it too. The retry
//      therefore also fires on ordinary failures, which is a known limitation
//      rather than a bug, and the comment says so.
//
// The historical reason this probe exists: the comment above the retry named
// "fits_already_open" as if it were the status code. It is a CFITSIO *function*.
// Looking 104 up in fitsio.h gives FILE_NOT_OPENED -- "could not open the named
// file" -- which points away from the cache-conflict meaning and made the retry
// read as bogus to a reviewer. Pinning the real behaviour is what stops that
// from being "simplified" away.
//
// The file under test is created by the caller and passed as argv[1]; it must
// be a real FITS file with a readable primary HDU, because CFITSIO reports
// UNKNOWN_REC (252) rather than FILE_NOT_OPENED for an empty one and the first
// assertion would then pass for the wrong reason.
//
// Run it with the shared runner, which finds the built library and CFITSIO and
// writes a suitable FITS file for you:
//   tests/cpp/run_all.sh
//
// The build links the shared core, which force-loads the whole CFITSIO archive
// so every fits_* symbol and its zlib/bzip2/curl dependencies resolve. That
// library is not installed into the pixi environment -- it lives in the
// per-build tree, whose path changes on every rebuild -- which is why this
// file used to carry a literal "<build>/libtorchfits_core.dylib" placeholder
// that had to be hand-substituted before the command could be used.
//
// Standalone equivalent, if you have already located the library:
//   c++ -std=c++17 -O2 -I extern/cfitsio \
//       tests/cpp/test_open_for_write_conflict.cpp <core-lib> -o /tmp/probe
//   /tmp/probe <file.fits>       # any FITS file with a readable primary HDU
#include <cstdio>
#include <cstring>
#include <string>

#include <fitsio.h>
#include <longnam.h>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "  ok  " : " FAIL ", what);
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <file.fits>\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];

    // --- 1. A held READONLY handle blocks a READWRITE open. ---------------
    fitsfile* reader = nullptr;
    int status = 0;
    fits_open_file(&reader, path.c_str(), 0 /* READONLY */, &status);
    if (status != 0) {
        std::fprintf(stderr, "could not open %s READONLY (status %d)\n",
                     path.c_str(), status);
        return 2;
    }
    check(true, "opened the file READONLY (status 0)");

    fitsfile* writer = nullptr;
    status = 0;
    fits_open_file(&writer, path.c_str(), 1 /* READWRITE */, &status);
    check(status == FILE_NOT_OPENED,
          "READWRITE while READONLY is held returns FILE_NOT_OPENED (104)");
    if (status != FILE_NOT_OPENED) {
        std::printf("       got status %d, expected %d\n", status,
                    FILE_NOT_OPENED);
    }
    if (writer != nullptr) {
        // Should not happen, but do not leak it if CFITSIO ever hands one back.
        fits_close_file(writer, &status);
    }

    // --- 2. Closing the READONLY handle makes READWRITE succeed. ---------
    // This is what the retry relies on to be able to fix the conflict.
    status = 0;
    fits_close_file(reader, &status);
    check(status == 0, "closed the READONLY handle (status 0)");

    status = 0;
    writer = nullptr;
    fits_open_file(&writer, path.c_str(), 1 /* READWRITE */, &status);
    check(status == 0, "READWRITE now succeeds once nothing is held");
    if (writer != nullptr) {
        status = 0;
        fits_close_file(writer, &status);
    }

    // --- 3. FILE_NOT_OPENED is generic, not conflict-specific. ------------
    // Documented limitation: the retry in open_fits_for_write cannot tell the
    // two apart, so it also fires on ordinary failures. This assertion is here
    // so that limitation stays recorded if CFITSIO ever starts distinguishing
    // them -- the fix then is to narrow the retry's condition.
    status = 0;
    fitsfile* never = nullptr;
    const std::string missing =
        path + ".definitely-not-a-directory-xyz/nope.fits";
    fits_open_file(&never, missing.c_str(), 1 /* READWRITE */, &status);
    check(status == FILE_NOT_OPENED,
          "a missing directory also returns FILE_NOT_OPENED (why the retry "
          "is broad)");
    if (never != nullptr) {
        fits_close_file(never, &status);
    }

    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
