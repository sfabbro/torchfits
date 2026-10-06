// Standalone self-check: a failed raw-fd open must not be latched in the cache.
//
// get_shared_raw_fd() memoises the descriptor in SharedReadMeta so the next
// read skips an open() syscall. It used to memoise the *failure* too: the
// holder was created unconditionally and an fd of -1 was as good a cache
// entry as any, so one open() that failed left that path on the slow
// CFITSIO path for the rest of the process -- with nothing to clear it except
// a stat identity change (new inode, size or mtime), which a permission fix
// does not produce.
//
// Measured before the fix: after the first failing open the second call
// returned fd = -1 forever, even with the file readable again.
//
// The failure is induced with chmod rather than by exhausting descriptors, so
// the check is deterministic and needs no RLIMIT juggling: stat() still
// succeeds on an unreadable file, which is exactly the state that made the
// stale entry invisible to the invalidation.
//
// No `assert`, for the reason tests/cpp/README.md gives: under -DNDEBUG an
// assert-based check compiles to nothing and exits 0 having verified nothing.
// Failures are counted and returned.
//
// Build (needs the torch-free core library + CFITSIO -- see run_all.sh):
//   c++ -std=c++17 -I src/torchfits/cpp_src -I <cfitsio/include> \
//       tests/cpp/test_raw_fd_retry.cpp -L<cfitsio/lib> -lcfitsio \
//       <libtorchfits_core> -Wl,-rpath,<dirs> -o /tmp/probe
// Run: /tmp/probe <a readable FITS file>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "core/fits_core.h"

namespace {

// Copy the fixture somewhere this check owns: the chmod below would break the
// runner's fixture for every check after this one.
bool copy_file(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in) return false;
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << in.rdbuf();
    return out.good();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <readable fits file>\n", argv[0]);
        return 2;
    }
    if (geteuid() == 0) {
        printf("  skip  running as root: chmod cannot make open() fail\n");
        return 0;
    }

    char tmpl[] = "/tmp/torchfits-rawfd-XXXXXX";
    const int fd = mkstemp(tmpl);
    if (fd == -1) {
        fprintf(stderr, "mkstemp failed\n");
        return 2;
    }
    ::close(fd);
    const std::string path(tmpl);
    ::unlink(path.c_str());  // mkstemp's file is not a FITS file; copy to a fresh name

    std::string tmp = path + ".fits";
    if (!copy_file(argv[1], tmp)) {
        fprintf(stderr, "could not copy the fixture to %s\n", tmp.c_str());
        ::unlink(tmp.c_str());
        return 2;
    }
    if (::chmod(tmp.c_str(), S_IRUSR | S_IWUSR) != 0) {
        fprintf(stderr, "chmod failed\n");
        ::unlink(tmp.c_str());
        return 2;
    }

    int failures = 0;
    // Prime the meta so the stat identity (inode, size, mtime) is recorded
    // while the file is still readable -- chmod leaves all three unchanged.
    auto meta = torchfits::detail::get_shared_meta_for_path(tmp);

    ::chmod(tmp.c_str(), 0);
    auto denied = torchfits::detail::get_shared_raw_fd(meta, tmp);
    const bool denied_as_expected = !denied || denied->fd == -1;
    printf("  %s  unreadable file yields no usable fd\n",
           denied_as_expected ? "ok  " : "FAIL");
    if (!denied_as_expected) failures++;

    ::chmod(tmp.c_str(), S_IRUSR | S_IWUSR);
    auto granted = torchfits::detail::get_shared_raw_fd(meta, tmp);
    const bool retried = granted && granted->fd != -1;
    printf("  %s  the next read re-opens it (fd = %d)\n", retried ? "ok  " : "FAIL",
           granted ? granted->fd : -1);
    if (!retried) {
        printf("        a failed open was cached: %s returns the same -1 holder\n"
               "        forever, because the stat identity chmod leaves alone\n"
               "        (inode, size, mtime) never changes.\n",
               "get_shared_raw_fd");
        failures++;
    }

    ::unlink(tmp.c_str());
    if (failures == 0) printf("all checks passed\n");
    return failures == 0 ? 0 : 1;
}
