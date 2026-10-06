#include "core/parallel.h"

#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace torchfits {
namespace core {
namespace {

// One parallel_for call's worth of work. Every finished chunk decrements
// `remaining`; the calling thread blocks until it reaches zero and then
// rethrows the first captured exception on its own stack, so callers keep the
// "throws out of parallel_for" contract at::parallel_for provides.
struct Batch {
    std::mutex mutex;
    std::condition_variable cv;
    int remaining = 0;
    std::exception_ptr error;
};

struct Task {
    detail::ParallelThunk thunk = nullptr;
    void* ctx = nullptr;
    std::int64_t begin = 0;
    std::int64_t end = 0;
    std::shared_ptr<Batch> batch;
};

// True while this thread is executing user code on behalf of the pool.
//
// A worker must never submit a parallel_for and then wait for it. The workers
// that would pick the work up may themselves all be inside user code, so the
// wait can never be satisfied: measured deadlock at every pool size >= 2. A
// nested parallel_for is therefore run inline on the worker that asked for it.
thread_local bool t_is_pool_worker = false;

// Fixed-size worker pool. Created on first use and deliberately never
// destroyed: workers touch process-lifetime state, and a static destructor
// joining them would run after the CFITSIO/nanobind teardown that a late
// parallel_for could otherwise still reach.
class Pool {
public:
    explicit Pool(int worker_count) {
        workers_.reserve(static_cast<size_t>(worker_count));
        for (int i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    int worker_count() const { return static_cast<int>(workers_.size()); }

    void submit(const std::vector<Task>& tasks) {
        if (tasks.empty()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.insert(queue_.end(), tasks.begin(), tasks.end());
        }
        work_cv_.notify_all();
    }

private:
    static void finish(const Task& task, const std::exception_ptr& err) {
        std::shared_ptr<Batch> batch = task.batch;
        if (!batch) return;
        {
            std::lock_guard<std::mutex> lock(batch->mutex);
            if (err && !batch->error) batch->error = err;
            --batch->remaining;
        }
        batch->cv.notify_all();
    }

    void worker_loop() {
        for (;;) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_cv_.wait(lock, [this] { return !queue_.empty(); });
                task = queue_.front();
                queue_.pop_front();
            }
            std::exception_ptr err;
            t_is_pool_worker = true;
            try {
                task.thunk(task.ctx, task.begin, task.end);
            } catch (...) {
                err = std::current_exception();
            }
            t_is_pool_worker = false;
            finish(task, err);
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::deque<Task> queue_;
};

// Resolve TORCHFITS_NUM_THREADS once. ATen reads the same variable, so a
// process that pins ATen also pins the core pool.
int resolve_thread_count() {
    int threads = 0;
    if (const char* raw = std::getenv("TORCHFITS_NUM_THREADS")) {
        try {
            threads = std::stoi(std::string(raw));
        } catch (const std::exception&) {
            threads = 0;
        }
    }
    if (threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        threads = hw > 0 ? static_cast<int>(hw) : 1;
    }
    if (threads > 64) threads = 64;
    if (threads < 1) threads = 1;
    return threads;
}

// The pool pointer lives at namespace scope rather than inside a function-local
// static so the fork handler below can drop it.
Pool* g_pool = nullptr;

// fork() hands the child a copy of this address space: the std::thread objects
// the pool holds name threads that do not exist in the child, so a batch queued
// there is never worked on and the caller waits on its condition variable
// forever. Measured: a child that had not forked before the pool was warm hung
// until SIGALRM (10s) with no output. `tests/cpp/test_fork_after_pool.cpp`
// guards it. Assigning the pointer is async-signal-safe, which is all an
// atfork child handler may do.
void drop_pool_in_forked_child() {
    g_pool = nullptr;
}

Pool& pool() {
    static const bool registered = [] {
        pthread_atfork(nullptr, nullptr, drop_pool_in_forked_child);
        return true;
    }();
    (void)registered;
    if (!g_pool) {
        g_pool = new Pool(resolve_thread_count() - 1);
    }
    return *g_pool;
}

}  // namespace

int thread_count() {
    static const int count = resolve_thread_count();
    return count;
}

namespace detail {

void run_chunks(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain_size,
    ParallelThunk thunk,
    void* ctx
) {
    if (end <= begin) return;
    if (grain_size < 1) grain_size = 1;

    // Nested inside a parallel_for: run the chunks serially here rather than
    // queueing them for workers we would then have to wait on. The work still
    // gets done and the throw contract is preserved; it just does not fan out
    // again, which is the correct trade against a guaranteed deadlock.
    if (t_is_pool_worker) {
        std::exception_ptr nested_error;
        for (std::int64_t lo = begin; lo < end; lo += grain_size) {
            const std::int64_t hi = (lo + grain_size < end) ? lo + grain_size : end;
            try {
                thunk(ctx, lo, hi);
            } catch (...) {
                if (!nested_error) nested_error = std::current_exception();
            }
        }
        if (nested_error) std::rethrow_exception(nested_error);
        return;
    }

    Pool& workers = pool();
    const std::int64_t total = end - begin;
    const std::int64_t chunks = (total + grain_size - 1) / grain_size;

    // One chunk, or no workers to hand it to: stay on the calling thread.
    // Metadata scans are routinely smaller than one grain, and waking the
    // pool for those would cost more than the work itself.
    if (chunks < 2 || workers.worker_count() == 0) {
        thunk(ctx, begin, end);
        return;
    }

    // The caller keeps the last chunk so a parallel_for nested on the *calling*
    // thread cannot deadlock waiting on workers the outer call owns. Worker-side
    // nesting never reaches here: it is handled by the t_is_pool_worker branch
    // above, which never waits.
    const std::int64_t offloaded = chunks - 1;
    auto batch = std::make_shared<Batch>();
    batch->remaining = static_cast<int>(offloaded);

    std::vector<Task> tasks;
    tasks.reserve(static_cast<size_t>(offloaded));
    for (std::int64_t i = 0; i < offloaded; ++i) {
        const std::int64_t lo = begin + i * grain_size;
        Task task;
        task.thunk = thunk;
        task.ctx = ctx;
        task.begin = lo;
        task.end = (lo + grain_size < end) ? lo + grain_size : end;
        task.batch = batch;
        tasks.push_back(task);
    }
    workers.submit(tasks);

    const std::int64_t last_begin = begin + offloaded * grain_size;
    std::exception_ptr inline_error;
    try {
        thunk(ctx, last_begin, end);
    } catch (...) {
        inline_error = std::current_exception();
    }

    std::exception_ptr worker_error;
    {
        std::unique_lock<std::mutex> lock(batch->mutex);
        batch->cv.wait(lock, [&batch] { return batch->remaining == 0; });
        worker_error = batch->error;
    }
    if (inline_error) {
        std::rethrow_exception(inline_error);
    }
    if (worker_error) {
        std::rethrow_exception(worker_error);
    }
}

}  // namespace detail
}  // namespace core
}  // namespace torchfits
