// Standalone self-check: core::parallel_for must survive fork().
//
// The pool is created lazily and lives for the life of the process. A forked
// child inherits the *pointer* to it, but not the threads it names, so a batch
// queued in the child is never worked on and the caller blocks on its
// condition variable forever. Measured before the fix: with TORCHFITS_NUM_THREADS=4
// the child produced no output and was still spinning at SIGALRM (10s).
//
// It is a hang rather than a crash, which is the worst shape for this library:
// a multiprocessing worker that wedges takes the user's job with it and leaves
// no trace. The project already works around it at its one fork site by asking
// for the "spawn" start method; this guards the library half.
//
// No `assert`, for the reason tests/cpp/README.md gives: under -DNDEBUG an
// assert-based check compiles to nothing and exits 0 having verified nothing.
// Failures are counted and returned.
//
// Build (links core/parallel.cpp directly, no project library needed):
//   c++ -std=c++17 -O1 -I src/torchfits/cpp_src \
//       tests/cpp/test_fork_after_pool.cpp src/torchfits/cpp_src/core/parallel.cpp \
//       -o /tmp/probe -lpthread
// Run: /tmp/probe
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>

#include "core/parallel.h"

namespace {

// Enough chunks to actually offload and start workers; small ranges stay
// inline and would never touch the pool.
constexpr std::int64_t kTotal = 400000;
constexpr std::int64_t kGrain = 1000;
constexpr unsigned kChildAlarmSeconds = 10;

void burn(std::int64_t begin, std::int64_t end) {
    volatile std::int64_t sink = 0;
    for (std::int64_t i = begin; i < end; ++i) sink += i;
}

}  // namespace

int main() {
    // Let any lazy runtime initialisation settle on this thread first. macOS
    // aborts in the child when fork() races another thread's ObjC-init, and
    // that is platform noise rather than the behaviour under test.
    usleep(300000);

    // Warm the pool in the parent. With one hardware thread there are no
    // workers, every range runs inline, and the hazard cannot arise; that is
    // not a false pass, it is a configuration where the bug is unreachable.
    torchfits::core::parallel_for(
        0, kTotal, kGrain, [](std::int64_t b, std::int64_t e) { burn(b, e); });

    const pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "fork failed\n");
        return 1;
    }
    if (pid == 0) {
        alarm(kChildAlarmSeconds);
        torchfits::core::parallel_for(
            0, kTotal, kGrain, [](std::int64_t b, std::int64_t e) { burn(b, e); });
        _exit(0);  // reached only if the child did not hang
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(kChildAlarmSeconds + 5);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (waitpid(pid, &status, WNOHANG) == pid) {
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                printf("  ok    the forked child's parallel_for returned\n");
                return 0;
            }
            if (WIFSIGNALED(status) && WTERMSIG(status) == SIGALRM) {
                printf(
                    "  FAIL  the child hung in parallel_for and its %us alarm fired\n",
                    kChildAlarmSeconds);
                printf(
                    "        the pool it inherited names threads that do not exist\n"
                    "        in the child; the fix is the pthread_atfork handler in\n"
                    "        core/parallel.cpp that drops the pool pointer.\n");
                return 1;
            }
            printf("  FAIL  child exited abnormally (status %d)\n", status);
            return 1;
        }
        usleep(50000);
    }

    printf("  FAIL  the child neither finished nor died -- deadlock\n");
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    return 1;
}
