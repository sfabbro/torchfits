#pragma once

// Torch-free replacement for at::parallel_for.
//
// libtorchfits_core must never link libtorch, so it cannot use ATen's thread
// pool.  The API here mirrors the at::parallel_for contract the codebase
// relies on -- inclusive/exclusive range [begin, end), a grain size in work
// units, and a body invoked as body(begin, end) that must be safe to run
// concurrently -- so call sites can be moved between the two libraries
// unchanged.
//
// Thread count: TORCHFITS_NUM_THREADS when set (same spelling ATen honours),
// otherwise the hardware concurrency, clamped to [1, 64].  A single caller
// thread plus the pool workers means at most `thread_count()` threads are ever
// running user code, so nesting a core parallel_for inside an at::parallel_for
// cannot oversubscribe the machine.
//
// Nesting a core parallel_for inside another one is safe but does not fan out
// a second time: a call made from inside a pool worker runs its chunks inline
// on that worker.  Submitting from a worker and waiting for the results is a
// measured deadlock at every pool size >= 2, because the workers that would
// take the work are themselves inside user code.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "core_export.h"

namespace torchfits {
namespace core {

// Effective worker count (>= 1). Resolved once per process.
TORCHFITS_CORE_API int thread_count();

namespace detail {

// Type-erased chunk runner. The template wrapper below adapts a lambda to this
// signature without a std::function allocation on the hot path.
using ParallelThunk = void (*)(void* ctx, std::int64_t begin, std::int64_t end);

TORCHFITS_CORE_API void run_chunks(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain_size,
    ParallelThunk thunk,
    void* ctx
);

}  // namespace detail

// Split [begin, end) into ceil((end - begin) / grain_size) chunks and run them
// across the pool. Work smaller than one grain runs inline on the calling
// thread: spawning helpers for a 4 KiB range costs far more than the range.
template <typename F>
inline void parallel_for(std::int64_t begin, std::int64_t end, std::int64_t grain_size, F&& body) {
    using Body = typename std::remove_reference<F>::type;
    auto thunk = [](void* ctx, std::int64_t b, std::int64_t e) {
        (*static_cast<Body*>(ctx))(b, e);
    };
    detail::run_chunks(begin, end, grain_size, thunk, &body);
}

}  // namespace core
}  // namespace torchfits
