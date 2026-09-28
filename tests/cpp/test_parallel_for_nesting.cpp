// Standalone self-check for core::parallel_for nesting and error propagation.
// No test framework needed: plain checks and a bounded main.
//
// Failures are counted by a `check()` helper rather than written as
// `assert(...)`. Under -DNDEBUG every assert() compiles to nothing, so this
// file used to report "all checks passed" and exit 0 with nothing verified --
// measured against a run_chunks reduced to a no-op, where the assert build
// aborted (SIGABRT) and the -DNDEBUG build passed at both 1 and 4 threads. A
// count is a runtime value and cannot be compiled away.
//
// The nesting case is a deadlock regression. A parallel_for issued from inside
// a chunk that a pool worker is executing must not submit work and wait for it:
// the workers that would take it are themselves inside user code, so the wait
// can never be satisfied. Measured deadlock at every pool size >= 2 before the
// worker-inline branch in run_chunks. The test therefore runs the nested case
// at several TORCHFITS_NUM_THREADS values via argv.
//
// A regression in that branch does not fail this check, it *hangs* it -- that
// is the failure shape. Run it through tests/cpp/run_all.sh, which bounds every
// check with a timeout, or wrap a manual run in `timeout 60 ...`.
//
// Run:
//   clang++ -std=c++17 -I src/torchfits/cpp_src \
//       tests/cpp/test_parallel_for_nesting.cpp src/torchfits/cpp_src/core/parallel.cpp \
//       -lpthread -o /tmp/test_parallel_for && /tmp/test_parallel_for 4
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/parallel.h"

using torchfits::core::parallel_for;
using torchfits::core::thread_count;

namespace {

// Outer range, split into 8 chunks of 512; every chunk body runs the whole
// nested range of 64. Both are deliberate: 8 is enough chunks that a real pool
// has several workers to hand them to, and 64/8 keeps the nested call itself
// multi-chunk so it would queue work and wait if the worker-inline branch were
// absent.
constexpr std::int64_t kOuterRange = 4096;
constexpr std::int64_t kOuterGrain = 512;
constexpr std::int64_t kNestedRange = 64;
constexpr std::int64_t kNestedGrain = 8;
constexpr int kOuterChunks =
    static_cast<int>((kOuterRange + kOuterGrain - 1) / kOuterGrain);

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf(" FAIL  %s\n", what);
        ++failures;
    }
}

// The chunks handed to the body must tile [begin, end) exactly once each: a
// gap is work that never ran, and a range that does not start where the
// previous one ended is either an overlap or out-of-range work. The vector is
// sorted first so this holds at any thread count -- at 1 thread it is one
// chunk spanning the whole range, at N it is N ranges tiling it.
void check_tiles_exactly(
    const char* what,
    std::vector<std::pair<std::int64_t, std::int64_t>> chunks,
    std::int64_t begin,
    std::int64_t end
) {
    std::sort(chunks.begin(), chunks.end());
    std::int64_t cursor = begin;
    for (const auto& chunk : chunks) {
        if (chunk.first != cursor) return check(false, what);
        cursor = chunk.second;
    }
    check(cursor == end, what);
}

// A body invoked from a worker that itself calls parallel_for. Completing this
// is the whole point: it deadlocked at every pool size >= 2 before the fix.
void check_worker_side_nesting_completes() {
    std::atomic<int> nested_bodies{0};
    std::atomic<int> nested_items{0};
    std::mutex mutex;
    std::vector<std::pair<std::int64_t, std::int64_t>> outer_chunks;

    parallel_for(0, kOuterRange, kOuterGrain, [&](std::int64_t lo, std::int64_t hi) {
        nested_bodies.fetch_add(1);
        {
            std::lock_guard<std::mutex> lock(mutex);
            outer_chunks.emplace_back(lo, hi);
        }
        parallel_for(0, kNestedRange, kNestedGrain, [&](std::int64_t lo2, std::int64_t hi2) {
            nested_items.fetch_add(static_cast<int>(hi2 - lo2));
        });
    });

    // Every nested call must cover its range exactly, so the two counters are
    // pinned against each other: `nested_bodies` counts outer bodies run and
    // `nested_items` counts nested units visited, which can only agree if each
    // nested call visited all kNestedRange units. This is what notices a nested
    // call that drops or repeats a chunk (measured: a worker-inline branch with
    // a doubled stride gives 288 units against 512 expected).
    check(nested_bodies.load() >= 1, "at least one outer body ran");
    check(
        nested_items.load() == nested_bodies.load() * static_cast<int>(kNestedRange),
        "every nested call covered its whole range exactly once"
    );
    // ... and the outer chunks themselves must tile the range, which is a
    // separate property: a skipped or repeated outer chunk moves both counters
    // together and leaves the identity above intact.
    check_tiles_exactly(
        "outer chunks tile [0, 4096) exactly once",
        std::move(outer_chunks),
        0,
        kOuterRange
    );
}

// A nested call must still honour the throw contract rather than swallowing.
void check_nested_exception_propagates() {
    bool threw = false;
    try {
        parallel_for(0, kOuterRange, kOuterGrain, [&](std::int64_t, std::int64_t) {
            parallel_for(0, kNestedRange, kNestedGrain, [&](std::int64_t, std::int64_t) {
                throw std::runtime_error("nested body failed");
            });
        });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "a throw from a nested body reaches the caller");
}

// Every unit of a range must be visited exactly once, at a grain size that
// splits it, so a chunk-boundary off-by-one cannot hide.
void check_chunks_partition_the_range_exactly() {
    std::atomic<int> visited{0};
    parallel_for(0, 4096, 8, [&](std::int64_t lo, std::int64_t hi) {
        visited.fetch_add(static_cast<int>(hi - lo));
    });
    check(visited.load() == 4096, "4096 units of the range were visited");
}

// A top-level throw must still surface on the calling thread, and every chunk
// must be accounted for before the rethrow so no task outlives the stack the
// ctx pointer refers to.
void check_top_level_exception_propagates() {
    std::atomic<int> visited{0};
    bool threw = false;
    try {
        parallel_for(0, 4096, 8, [&](std::int64_t lo, std::int64_t hi) {
            visited.fetch_add(static_cast<int>(hi - lo));
            throw std::runtime_error("top-level body failed");
        });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "a throw from a top-level body reaches the caller");
    // All 4096 units accounted for: the caller waits for the whole batch
    // before rethrowing, so no worker is still writing through ctx.
    check(visited.load() == 4096, "every chunk finished before the rethrow");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        const int requested = std::atoi(argv[1]);
        if (requested > 0) setenv("TORCHFITS_NUM_THREADS", argv[1], 1);
    }

    const int threads = thread_count();
    std::printf("thread_count=%d\n", threads);
    check(threads >= 1, "thread_count() is at least 1");

    // Single-threaded: every call stays on the calling thread.
    check_worker_side_nesting_completes();
    check_nested_exception_propagates();
    check_top_level_exception_propagates();

    // The same properties with a real pool, where the deadlock used to be.
    if (threads > 1) {
        check_worker_side_nesting_completes();
        check_nested_exception_propagates();
        check_top_level_exception_propagates();
    }

    // Ranges that are empty or below one grain must be no-ops, not errors.
    std::atomic<int> touched{0};
    parallel_for(5, 5, 1, [&](std::int64_t, std::int64_t) { touched.fetch_add(1); });
    parallel_for(0, 3, 1 << 20, [&](std::int64_t, std::int64_t) {
        touched.fetch_add(1);
    });
    check(touched.load() == 1, "an empty range and a sub-grain range are no-ops");

    check_chunks_partition_the_range_exactly();

    if (failures == 0) {
        std::printf(
            "test_parallel_for_nesting: all checks passed "
            "(%d outer chunks of %lld, threads=%d)\n",
            kOuterChunks,
            static_cast<long long>(kOuterGrain),
            threads
        );
        return 0;
    }
    std::printf("test_parallel_for_nesting: FAILURES: %d\n", failures);
    return 1;
}
