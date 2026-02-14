#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <unordered_map>
#include <thread>
#include <array>
#include <cmath>
#include <chrono>
#include <memory>
#include <mutex>
#include <atomic>
#include <limits>
#include <cerrno>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>
#include <ATen/Parallel.h>
#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#endif
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include "torchfits_torch.h"
#include <fitsio.h>

#include "hardware.h"

namespace {
using fits_is_compressed_with_nulls_fn = int (*)(fitsfile*);

static inline uint16_t _bswap16(uint16_t x) { return __builtin_bswap16(x); }
static inline uint32_t _bswap32(uint32_t x) { return __builtin_bswap32(x); }
static inline uint64_t _bswap64(uint64_t x) { return __builtin_bswap64(x); }

inline void _xor_sign_bit_u8(uint8_t* p, size_t nbytes) {
    if (!p || nbytes == 0) {
        return;
    }
    static const size_t kParallelMinBytes = []() -> size_t {
        constexpr int64_t kDefault = 1 << 18;  // 256 KiB
        int64_t parsed = kDefault;
        if (const char* v = std::getenv("TORCHFITS_XOR_PARALLEL_MIN_BYTES")) {
            try {
                parsed = std::stoll(std::string(v));
            } catch (...) {
                parsed = kDefault;
            }
        }
        if (parsed <= 0) {
            parsed = 1;
        }
        return static_cast<size_t>(parsed);
    }();

    auto xor_block = [](uint8_t* ptr, size_t len) {
        if (!ptr || len == 0) {
            return;
        }

        constexpr uint64_t kMask64 = 0x8080808080808080ULL;
        size_t i = 0;

        // Align to 8-byte boundary first.
        while (i < len && ((reinterpret_cast<uintptr_t>(ptr + i) & 7u) != 0u)) {
            ptr[i] ^= 0x80;
            ++i;
        }

        uint64_t* p64 = reinterpret_cast<uint64_t*>(ptr + i);
        const size_t n64 = (len - i) / sizeof(uint64_t);
        for (size_t j = 0; j < n64; ++j) {
            p64[j] ^= kMask64;
        }
        i += n64 * sizeof(uint64_t);

        while (i < len) {
            ptr[i] ^= 0x80;
            ++i;
        }
    };

    if (nbytes < kParallelMinBytes) {
        xor_block(p, nbytes);
        return;
    }
    at::parallel_for(0, static_cast<int64_t>(nbytes), 1 << 20, [&](int64_t begin, int64_t end) {
        xor_block(p + begin, static_cast<size_t>(end - begin));
    });
}

constexpr bool _host_is_little_endian() {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
    return __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
#else
    // Conservative default; most modern targets are little-endian.
    return true;
#endif
}

struct SharedReadMeta {
    uint64_t uid = 0;
    std::unordered_map<int, std::tuple<int, int, std::array<LONGLONG, 9>>> image_info_cache;
    std::unordered_map<int, bool> compressed_cache;
    std::unordered_map<int, bool> compressed_parallel_cache;
    std::unordered_map<int, bool> compressed_nulls_cache;
    std::unordered_map<int, std::tuple<bool, bool, double, double>> scale_cache;
    bool has_stat = false;
    off_t size = 0;
    int64_t mtime_ns = 0;
    ino_t inode = 0;
    int64_t last_stat_check_ns = 0;
    int raw_fd = -1;
    std::mutex mutex;

    ~SharedReadMeta() {
        if (raw_fd != -1) {
            ::close(raw_fd);
            raw_fd = -1;
        }
    }
};

std::mutex g_shared_meta_mutex;
std::unordered_map<std::string, std::shared_ptr<SharedReadMeta>> g_shared_meta;
std::atomic<uint64_t> g_shared_meta_uid{1};

inline bool env_flag_default_true(const char* name) {
    const char* v = std::getenv(name);
    if (!v) {
        return true;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(s == "0" || s == "false" || s == "off" || s == "no");
}

inline int64_t env_nonnegative_int(const char* name, int64_t default_value) {
    const char* v = std::getenv(name);
    if (!v) {
        return default_value;
    }
    try {
        int64_t parsed = std::stoll(std::string(v));
        return parsed < 0 ? 0 : parsed;
    } catch (...) {
        return default_value;
    }
}

inline int64_t monotonic_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline int64_t mtime_ns_from_stat(const struct stat& st) {
#if defined(__APPLE__)
    return (static_cast<int64_t>(st.st_mtimespec.tv_sec) * 1000000000LL) +
           static_cast<int64_t>(st.st_mtimespec.tv_nsec);
#else
    return (static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000LL) +
           static_cast<int64_t>(st.st_mtim.tv_nsec);
#endif
}

// Validate shared metadata against filesystem state. This adds a stat() on each open
// of a given path (even if the FITS handle itself is cached). Keep enabled by default
// for external-overwrite safety; allow opt-out via TORCHFITS_SHARED_META_VALIDATE=0.
const bool kValidateSharedMeta = []() {
    return env_flag_default_true("TORCHFITS_SHARED_META_VALIDATE");
}();

// Validate shared metadata at most once per interval by default to avoid
// per-read stat() overhead on tiny hot loops. Set interval to 0 for strict mode.
const int64_t kSharedMetaValidateIntervalNs = []() {
    // Balance stale-file detection with hot-path latency. A longer default interval
    // reduces repeated stat() overhead in tight read loops.
    constexpr int64_t kDefaultMs = 1000;
    return env_nonnegative_int("TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS", kDefaultMs) *
           1000000LL;
}();

std::shared_ptr<SharedReadMeta> get_shared_meta_for_path(const std::string& filename) {
    bool can_stat = kValidateSharedMeta && filename.find('[') == std::string::npos;
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    auto it = g_shared_meta.find(filename);
    std::shared_ptr<SharedReadMeta> meta;
    if (it == g_shared_meta.end()) {
        meta = std::make_shared<SharedReadMeta>();
        meta->uid = g_shared_meta_uid.fetch_add(1, std::memory_order_relaxed);
        g_shared_meta.emplace(filename, meta);
    } else {
        meta = it->second;
    }

    if (!can_stat) {
        return meta;
    }

    const int64_t now_ns = monotonic_now_ns();
    if (kSharedMetaValidateIntervalNs > 0 && meta->last_stat_check_ns != 0 &&
        (now_ns - meta->last_stat_check_ns) < kSharedMetaValidateIntervalNs) {
        return meta;
    }
    meta->last_stat_check_ns = now_ns;

    struct stat st {};
    bool has_stat = stat(filename.c_str(), &st) == 0;
    if (has_stat) {
        int64_t cur_mtime_ns = mtime_ns_from_stat(st);
        if (!meta->has_stat || meta->size != st.st_size ||
            meta->mtime_ns != cur_mtime_ns || meta->inode != st.st_ino) {
            std::lock_guard<std::mutex> meta_lock(meta->mutex);
            if (meta->raw_fd != -1) {
                ::close(meta->raw_fd);
                meta->raw_fd = -1;
            }
            meta->image_info_cache.clear();
            meta->compressed_cache.clear();
            meta->compressed_parallel_cache.clear();
            meta->compressed_nulls_cache.clear();
            meta->scale_cache.clear();
            meta->has_stat = true;
            meta->size = st.st_size;
            meta->mtime_ns = cur_mtime_ns;
            meta->inode = st.st_ino;
        }
    }
    return meta;
}

int open_readonly_fd(const std::string& filename) {
#ifdef O_CLOEXEC
    int fd = ::open(filename.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd != -1) {
        return fd;
    }
#endif
    return ::open(filename.c_str(), O_RDONLY);
}

int get_shared_raw_fd(const std::shared_ptr<SharedReadMeta>& meta, const std::string& filename) {
    if (!meta || filename.find('[') != std::string::npos) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(meta->mutex);
    if (meta->raw_fd != -1) {
        return meta->raw_fd;
    }
    meta->raw_fd = open_readonly_fd(filename);
    return meta->raw_fd;
}

bool read_region_via_fd(int fd, off_t offset, void* dst_void, size_t nbytes) {
    if (fd == -1 || !dst_void || nbytes == 0) {
        return false;
    }
    uint8_t* dst = static_cast<uint8_t*>(dst_void);
    size_t remaining = nbytes;
    off_t off = offset;
    while (remaining > 0) {
        ssize_t got = ::pread(fd, dst, remaining, off);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (got == 0) {
            break;
        }
        dst += static_cast<size_t>(got);
        off += static_cast<off_t>(got);
        remaining -= static_cast<size_t>(got);
    }
    if (remaining == 0) {
        return true;
    }

    struct stat sb {};
    if (fstat(fd, &sb) != 0) {
        return false;
    }
    if (offset < 0) {
        return false;
    }
    if (static_cast<size_t>(sb.st_size) < static_cast<size_t>(offset) + nbytes) {
        return false;
    }

    void* map_ptr = mmap(nullptr, static_cast<size_t>(sb.st_size), PROT_READ, MAP_SHARED, fd, 0);
    if (map_ptr == MAP_FAILED) {
        return false;
    }
    const uint8_t* src = static_cast<const uint8_t*>(map_ptr) + offset;
    std::memcpy(dst_void, src, nbytes);
    munmap(map_ptr, static_cast<size_t>(sb.st_size));
    return true;
}

void invalidate_shared_meta_for_path(const std::string& filename) {
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    g_shared_meta.erase(filename);
}

void clear_shared_meta_cache() {
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    g_shared_meta.clear();
}

bool has_compressed_nulls(fitsfile* fptr) {
#if defined(__APPLE__) || defined(__linux__)
    static fits_is_compressed_with_nulls_fn fn = []() -> fits_is_compressed_with_nulls_fn {
        void* sym = dlsym(RTLD_DEFAULT, "fits_is_compressed_with_nulls");
        if (!sym) {
            return nullptr;
        }
        return reinterpret_cast<fits_is_compressed_with_nulls_fn>(sym);
    }();
    if (fn) {
        return fn(fptr) != 0;
    }
#endif
    return false;
}
}  // namespace
#include "cache.cpp"

namespace nb = nanobind;

namespace torchfits {

void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header, nb::object schema_obj, bool is_ascii);
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header);

// Clears per-path read metadata (image info/compression/scale caches). This is
// called on writes so subsequent reads don't reuse stale cached info.
void invalidate_shared_meta(const std::string& filename) {
    invalidate_shared_meta_for_path(filename);
}

void clear_shared_read_meta_cache() {
    clear_shared_meta_cache();
}

namespace {
inline bool host_is_little_endian() {
    const uint16_t x = 1;
    return *reinterpret_cast<const uint8_t*>(&x) == 1;
}

inline uint16_t bswap_16(uint16_t v) { return __builtin_bswap16(v); }
inline uint32_t bswap_32(uint32_t v) { return __builtin_bswap32(v); }
inline uint64_t bswap_64(uint64_t v) { return __builtin_bswap64(v); }

template <typename T>
inline T load_bswap(const void* src);

template <>
inline uint16_t load_bswap<uint16_t>(const void* src) {
    uint16_t v;
    std::memcpy(&v, src, sizeof(v));
    return bswap_16(v);
}

template <>
inline uint32_t load_bswap<uint32_t>(const void* src) {
    uint32_t v;
    std::memcpy(&v, src, sizeof(v));
    return bswap_32(v);
}

template <>
inline uint64_t load_bswap<uint64_t>(const void* src) {
    uint64_t v;
    std::memcpy(&v, src, sizeof(v));
    return bswap_64(v);
}

// Opt-in-by-default, tightly gated parallel read path for large tile-compressed
// images. This keeps normal uncompressed/read paths untouched.
inline bool compressed_parallel_enabled() {
    return env_flag_default_true("TORCHFITS_COMPRESSED_PARALLEL");
}

inline int64_t compressed_parallel_min_pixels() {
    constexpr int64_t kDefault = 1024LL * 1024LL;
    return env_nonnegative_int("TORCHFITS_COMPRESSED_PARALLEL_MIN_PIXELS", kDefault);
}

inline int64_t compressed_parallel_min_rows_per_thread() {
    constexpr int64_t kDefault = 256;
    int64_t v = env_nonnegative_int("TORCHFITS_COMPRESSED_PARALLEL_MIN_ROWS_PER_THREAD", kDefault);
    return v > 0 ? v : 1;
}

inline int64_t compressed_parallel_max_threads() {
    constexpr int64_t kDefault = 2;
    int64_t v = env_nonnegative_int("TORCHFITS_COMPRESSED_PARALLEL_MAX_THREADS", kDefault);
    return v > 0 ? v : 1;
}

inline bool compressed_parallel_hcompress_enabled() {
    return env_flag_default_true("TORCHFITS_COMPRESSED_PARALLEL_HCOMPRESS");
}

inline size_t datatype_elem_size(int datatype) {
    switch (datatype) {
        case TBYTE:
        case TSBYTE:
            return sizeof(uint8_t);
        case TSHORT:
            return sizeof(uint16_t);
        case TINT:
            return sizeof(uint32_t);
        case TLONGLONG:
            return sizeof(uint64_t);
        case TFLOAT:
            return sizeof(float);
        case TDOUBLE:
            return sizeof(double);
        default:
            return 0;
    }
}

inline bool try_read_compressed_rows_parallel(
    fitsfile* fptr,
    const std::string& path,
    int target_hdu,
    int naxis,
    const std::array<LONGLONG, 9>& naxes_ll,
    LONGLONG nelements,
    int datatype,
    bool allow_float,
    void* dst
) {
    if (!compressed_parallel_enabled() || !fptr || !dst) {
        return false;
    }
    if (path.find('[') != std::string::npos) {
        return false;
    }
    if (naxis != 2) {
        return false;
    }
    if (nelements < compressed_parallel_min_pixels()) {
        return false;
    }
    const size_t elem_size = datatype_elem_size(datatype);
    if (elem_size == 0) {
        return false;
    }
    if ((datatype == TFLOAT || datatype == TDOUBLE) && !allow_float) {
        return false;
    }

    const LONGLONG width_ll = naxes_ll[0];
    const LONGLONG rows_ll = naxes_ll[1];
    if (width_ll <= 0 || rows_ll <= 1) {
        return false;
    }
    if (width_ll > static_cast<LONGLONG>(std::numeric_limits<long>::max()) ||
        rows_ll > static_cast<LONGLONG>(std::numeric_limits<long>::max())) {
        return false;
    }

    long tile_dims[2] = {0, 0};
    int status = 0;
    fits_get_tile_dim(fptr, 2, tile_dims, &status);
    if (status != 0) {
        return false;
    }
    if (tile_dims[0] <= 0 || tile_dims[1] <= 0) {
        return false;
    }
    const LONGLONG tile_h_ll = static_cast<LONGLONG>(tile_dims[1]);
    if (tile_h_ll <= 0) {
        return false;
    }
    const LONGLONG tile_rows = (rows_ll + tile_h_ll - 1) / tile_h_ll;
    if (tile_rows <= 1) {
        return false;
    }

    const int64_t hw_threads = std::max<int64_t>(
        1,
        static_cast<int64_t>(std::thread::hardware_concurrency())
    );
    const int64_t max_threads = std::min<int64_t>(
        std::max<int64_t>(1, compressed_parallel_max_threads()),
        hw_threads
    );
    const int64_t min_rows_per_thread = compressed_parallel_min_rows_per_thread();
    const int64_t min_tile_rows_per_thread =
        std::max<int64_t>(1, (min_rows_per_thread + tile_h_ll - 1) / tile_h_ll);
    const int64_t by_tile_rows = tile_rows / min_tile_rows_per_thread;
    const int64_t nthreads = std::min<int64_t>(max_threads, by_tile_rows);
    if (nthreads < 2) {
        return false;
    }

    auto* dst_u8 = static_cast<uint8_t*>(dst);
    std::atomic<int> first_status{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(nthreads));
    std::vector<fitsfile*> local_handles(static_cast<size_t>(nthreads), nullptr);

    for (int64_t t = 0; t < nthreads; ++t) {
        fitsfile* local = nullptr;
        int st = 0;
        ffreopen(fptr, &local, &st);
        if (st != 0 || !local) {
            for (fitsfile* opened : local_handles) {
                if (opened) {
                    int close_status = 0;
                    fits_close_file(opened, &close_status);
                }
            }
            return false;
        }
        fits_movabs_hdu(local, target_hdu, nullptr, &st);
        if (st != 0) {
            int close_status = 0;
            fits_close_file(local, &close_status);
            for (fitsfile* opened : local_handles) {
                if (opened) {
                    close_status = 0;
                    fits_close_file(opened, &close_status);
                }
            }
            return false;
        }
        local_handles[static_cast<size_t>(t)] = local;
    }

    for (int64_t t = 0; t < nthreads; ++t) {
        const int64_t tile_row_begin = (tile_rows * t) / nthreads;
        const int64_t tile_row_end = (tile_rows * (t + 1)) / nthreads;
        const int64_t row_begin = std::min<int64_t>(rows_ll, tile_row_begin * tile_h_ll);
        const int64_t row_end = std::min<int64_t>(rows_ll, tile_row_end * tile_h_ll);
        if (row_end <= row_begin) {
            fitsfile* local = local_handles[static_cast<size_t>(t)];
            if (local) {
                int close_status = 0;
                fits_close_file(local, &close_status);
                local_handles[static_cast<size_t>(t)] = nullptr;
            }
            continue;
        }

        workers.emplace_back([=, &first_status]() {
            fitsfile* local = local_handles[static_cast<size_t>(t)];
            if (!local) {
                return;
            }

            if (first_status.load(std::memory_order_relaxed) != 0) {
                return;
            }

            int st = 0;
            std::array<long, 2> fpixel{
                1L,
                static_cast<long>(row_begin + 1),
            };
            std::array<long, 2> lpixel{
                static_cast<long>(width_ll),
                static_cast<long>(row_end),
            };
            std::array<long, 2> inc{1L, 1L};
            int anynul = 0;
            size_t elem_offset = static_cast<size_t>(row_begin) * static_cast<size_t>(width_ll);
            void* chunk_ptr = static_cast<void*>(dst_u8 + (elem_offset * elem_size));

            fits_read_subset(
                local,
                datatype,
                fpixel.data(),
                lpixel.data(),
                inc.data(),
                nullptr,
                chunk_ptr,
                &anynul,
                &st
            );
            if (st != 0) {
                int expected = 0;
                first_status.compare_exchange_strong(expected, st);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    for (fitsfile* local : local_handles) {
        if (!local) {
            continue;
        }
        int close_status = 0;
        fits_close_file(local, &close_status);
    }

    return first_status.load(std::memory_order_relaxed) == 0;
}

}  // namespace
// Helper to sanitize FITS strings (keep only printable ASCII)
std::string sanitize_fits_string(const std::string& input) {
    std::string output = input;
    // Remove non-printable characters
    output.erase(std::remove_if(output.begin(), output.end(), [](unsigned char c) {
        return c < 32 || c > 126;
    }), output.end());
    return output;
}

// Helper to validate/sanitize FITS keyword/column names
// FITS standard: uppercase, digits, underscore, hyphen.
std::string sanitize_fits_key(const std::string& input) {
    std::string output;
    output.reserve(input.length());
    for (char c : input) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
            output.push_back(std::toupper(static_cast<unsigned char>(c))); // Standard keys are uppercase
        }
    }
    if (output.empty()) return "UNKNOWN";
    return output;
}
std::vector<torch::Tensor> read_images_batch(const std::vector<std::string>& paths, int hdu_num);
std::vector<torch::Tensor> read_hdus_batch(const std::string& path, const std::vector<int>& hdus, bool use_mmap);
torch::Tensor read_hdus_sequence_last(const std::string& path, const std::vector<int>& hdus, bool use_mmap);

class FITSFile {
public:
    FITSFile(const char* filename, int mode) : filename_(filename), mode_(mode) {
        // Security check: Prevent command injection via cfitsio pipe syntax
        if (!filename_.empty()) {
            size_t first = filename_.find_first_not_of(" \t");
            size_t last = filename_.find_last_not_of(" \t");

            if (first != std::string::npos) {
                if (filename_[first] == '|' || filename_[last] == '|') {
                     throw std::runtime_error("Security Error: Filenames starting or ending with '|' are not allowed to prevent command execution.");
                }
            }
        }

        int status = 0;
        if (mode == 0) {
            fptr_ = torchfits::get_or_open_cached(filename_);
            use_cache_ = true;
            if (!fptr_) {
                status = 1;
            }
        } else {
            fits_create_file(&fptr_, filename, &status);
            use_cache_ = false;
        }

        if (status != 0) {
            throw std::runtime_error("Could not open FITS file: " + filename_);
        }
        cached_ = false;
        if (mode == 0) {
            shared_meta_ = get_shared_meta_for_path(filename_);
        }

        // For regular filenames (no CFITSIO extended syntax), always use primary-HDU based
        // indexing and avoid extra CFITSIO calls in the constructor. Cached handles may
        // be left on a non-primary HDU; we handle that lazily in ensure_hdu().
        const bool has_extension = filename_.find('[') != std::string::npos;
        if (!has_extension) {
            start_hdu_ = 1;
            current_hdu_ = -1;  // force first ensure_hdu() to move
        } else {
            // Store the initial HDU number (important for extended filename syntax/virtual files)
            fits_get_hdu_num(fptr_, &start_hdu_);
            current_hdu_ = start_hdu_;
        }
    }
    
    ~FITSFile() {
        close();
    }
    
    void close() {
        close_raw_fd();
        if (fptr_) {
            if (use_cache_) {
                torchfits::release_cached(filename_);
            } else {
                int status = 0;
                fits_close_file(fptr_, &status);
            }
            fptr_ = nullptr;
        }
    }

    fitsfile* get_fptr() const { return fptr_; }

    void ensure_hdu(int hdu_num, int* status) {
        if (!fptr_) {
            throw std::runtime_error("FITSFile is closed");
        }
        int target_hdu = hdu_num + start_hdu_;
        if (current_hdu_ != target_hdu) {
            fits_movabs_hdu(fptr_, target_hdu, nullptr, status);
            if (*status == 0) {
                current_hdu_ = target_hdu;
            }
        }
    }

    struct ScaleInfo {
        bool scaled = false;
        bool trusted = true;
        double bscale = 1.0;
        double bzero = 0.0;
    };

    const ScaleInfo& get_scale_info(int hdu_num, int bitpix) {
        auto it = scale_cache_.find(hdu_num);
        if (it != scale_cache_.end()) {
            return it->second;
        }
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            auto sit = shared_meta_->scale_cache.find(hdu_num);
            if (sit != shared_meta_->scale_cache.end()) {
                auto [scaled, trusted, bscale, bzero] = sit->second;
                ScaleInfo shared_info;
                shared_info.scaled = scaled;
                shared_info.trusted = trusted;
                shared_info.bscale = bscale;
                shared_info.bzero = bzero;
                auto inserted = scale_cache_.emplace(hdu_num, shared_info);
                return inserted.first->second;
            }
        }
        ScaleInfo info;
        if (bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG) {
            auto inserted = scale_cache_.emplace(hdu_num, info);
            if (shared_meta_) {
                std::lock_guard<std::mutex> lock(shared_meta_->mutex);
                shared_meta_->scale_cache[hdu_num] = std::make_tuple(
                    info.scaled, info.trusted, info.bscale, info.bzero
                );
            }
            return inserted.first->second;
        }

        int status = 0;
        double bscale = 1.0;
        double bzero = 0.0;

        status = 0;
        fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &status);
        if (status == 0) {
            info.bscale = bscale;
            if (bscale != 1.0) {
                info.scaled = true;
            }
        } else if (status != KEY_NO_EXIST) {
            info.scaled = true;
            info.trusted = false;
        }

        status = 0;
        fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &status);
        if (status == 0) {
            info.bzero = bzero;
            if (bzero != 0.0) {
                info.scaled = true;
            }
        } else if (status != KEY_NO_EXIST) {
            info.scaled = true;
            info.trusted = false;
        }

        auto inserted = scale_cache_.emplace(hdu_num, info);
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            shared_meta_->scale_cache[hdu_num] = std::make_tuple(
                info.scaled, info.trusted, info.bscale, info.bzero
            );
        }
        return inserted.first->second;
    }

    ScaleInfo get_scale_info_for_hdu(int hdu_num) {
        const auto& info = get_image_info(hdu_num);
        int bitpix = std::get<0>(info);
        return get_scale_info(hdu_num, bitpix);
    }

    bool is_compressed_image_cached(int hdu_num) {
        auto it = compressed_cache_.find(hdu_num);
        if (it != compressed_cache_.end()) {
            return it->second;
        }
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            auto sit = shared_meta_->compressed_cache.find(hdu_num);
            if (sit != shared_meta_->compressed_cache.end()) {
                compressed_cache_[hdu_num] = sit->second;
                return sit->second;
            }
        }
        int status = 0;
        int is_compressed = fits_is_compressed_image(fptr_, &status);
        bool result = (status == 0 && is_compressed);
        compressed_cache_[hdu_num] = result;
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            shared_meta_->compressed_cache[hdu_num] = result;
        }
        return result;
    }

    bool has_compressed_nulls_cached(int hdu_num) {
        auto it = compressed_nulls_cache_.find(hdu_num);
        if (it != compressed_nulls_cache_.end()) {
            return it->second;
        }
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            auto sit = shared_meta_->compressed_nulls_cache.find(hdu_num);
            if (sit != shared_meta_->compressed_nulls_cache.end()) {
                compressed_nulls_cache_[hdu_num] = sit->second;
                return sit->second;
            }
        }
        bool result = has_compressed_nulls(fptr_);
        compressed_nulls_cache_[hdu_num] = result;
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            shared_meta_->compressed_nulls_cache[hdu_num] = result;
        }
        return result;
    }

    bool is_parallel_compressed_codec_cached(int hdu_num) {
        auto it = compressed_parallel_cache_.find(hdu_num);
        if (it != compressed_parallel_cache_.end()) {
            return it->second;
        }
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            auto sit = shared_meta_->compressed_parallel_cache.find(hdu_num);
            if (sit != shared_meta_->compressed_parallel_cache.end()) {
                compressed_parallel_cache_[hdu_num] = sit->second;
                return sit->second;
            }
        }

        bool result = false;
        char zcmptype[FLEN_VALUE];
        std::memset(zcmptype, 0, sizeof(zcmptype));
        int status = 0;
        fits_read_key(fptr_, TSTRING, "ZCMPTYPE", zcmptype, nullptr, &status);
        if (status == 0) {
            std::string zcmp(zcmptype);
            std::transform(
                zcmp.begin(), zcmp.end(), zcmp.begin(),
                [](unsigned char c) { return static_cast<char>(std::toupper(c)); }
            );
            if (zcmp.find("RICE") != std::string::npos) {
                result = true;
            } else if (compressed_parallel_hcompress_enabled() &&
                       zcmp.find("HCOMPRESS") != std::string::npos) {
                result = true;
            }
        } else {
            // Missing key (or read failure) => treat as unsupported codec.
            status = 0;
        }

        compressed_parallel_cache_[hdu_num] = result;
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            shared_meta_->compressed_parallel_cache[hdu_num] = result;
        }
        return result;
    }

    struct BScaleGuard {
        fitsfile* fptr = nullptr;
        double bscale = 1.0;
        double bzero = 0.0;
        bool active = false;
        ~BScaleGuard() {
            if (!active || !fptr) return;
            int status = 0;
            fits_set_bscale(fptr, bscale, bzero, &status);
        }
    };

    const std::tuple<int, int, std::array<LONGLONG, 9>>& get_image_info(int hdu_num) {
        auto it = image_info_cache_.find(hdu_num);
        if (it != image_info_cache_.end()) {
            return it->second;
        }
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            auto sit = shared_meta_->image_info_cache.find(hdu_num);
            if (sit != shared_meta_->image_info_cache.end()) {
                auto inserted = image_info_cache_.emplace(hdu_num, sit->second);
                return inserted.first->second;
            }
        }
        int status = 0;
        int bitpix = 0;
        int naxis = 0;
        std::array<LONGLONG, 9> naxes_ll{};
        naxes_ll.fill(0);
        fits_get_img_paramll(fptr_, 9, &bitpix, &naxis, naxes_ll.data(), &status);
        if (status != 0) {
            throw std::runtime_error("Could not read image parameters");
        }
        auto inserted = image_info_cache_.emplace(hdu_num, std::make_tuple(bitpix, naxis, naxes_ll));
        if (shared_meta_) {
            std::lock_guard<std::mutex> lock(shared_meta_->mutex);
            shared_meta_->image_info_cache[hdu_num] = inserted.first->second;
        }
        return inserted.first->second;
    }

    torch::Tensor read_image(int hdu_num, bool use_mmap = true) {
        int status = 0;
        
        ensure_hdu(hdu_num, &status);
        if (status != 0) {
            throw std::runtime_error("Could not move to HDU");
        }

        const bool want_mmap = use_mmap;
        
        int bitpix = 0;
        int naxis = 0;
        std::array<LONGLONG, 9> naxes_ll{};
        {
            const auto& info = get_image_info(hdu_num);
            bitpix = std::get<0>(info);
            naxis = std::get<1>(info);
            naxes_ll = std::get<2>(info);
        }

        // Fast return for empty images (e.g., empty primary HDU in MEF files)
        if (naxis == 0) {
            torch::ScalarType dtype;
            switch (bitpix) {
                case BYTE_IMG:   dtype = torch::kUInt8; break;
                case SHORT_IMG:  dtype = torch::kInt16; break;
                case LONG_IMG:   dtype = torch::kInt32; break;
                case FLOAT_IMG:  dtype = torch::kFloat32; break;
                case DOUBLE_IMG: dtype = torch::kFloat64; break;
                default:         dtype = torch::kUInt8; break;
            }
            return torch::empty({0}, torch::TensorOptions().dtype(dtype));
        }

        const auto& scale_info = get_scale_info(hdu_num, bitpix);
        bool scaled = scale_info.scaled;

        LONGLONG nelements = 0;
        if (naxis > 0) {
            nelements = 1;
        for (int i = 0; i < naxis; ++i) {
            nelements *= naxes_ll[i];
        }
        }
        
        int64_t torch_shape[9];
        for (int i = 0; i < naxis; ++i) {
            torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]); // Reverse for C-contiguous
        }

        torch::ScalarType dtype;
        int datatype;

        bool compressed = is_compressed_image_cached(hdu_num);

        if (scaled) {
            // Scaled images usually return float32, but signed-byte images are a
            // common special case: BITPIX=8 stored as uint8 with BZERO=-128.
            // For these, return int8 directly to avoid slow float scaling.
            if (bitpix == BYTE_IMG && scale_info.bscale == 1.0 && scale_info.bzero == -128.0) {
                dtype = at::kChar;  // int8
                datatype = TSBYTE;
            } else {
                // If scaled, read as float32 (standard for images)
                dtype = torch::kFloat32;
                datatype = TFLOAT;
            }
	        } else {
	            switch (bitpix) {
	                case BYTE_IMG:   dtype = torch::kUInt8; datatype = TBYTE; break;
	                case SHORT_IMG:  dtype = torch::kInt16; datatype = TSHORT; break;
	                case LONG_IMG:   dtype = torch::kInt32; datatype = TINT; break;
	                case LONGLONG_IMG: dtype = torch::kInt64; datatype = TLONGLONG; break;
	                case FLOAT_IMG:  dtype = torch::kFloat32; datatype = TFLOAT; break;
	                case DOUBLE_IMG: dtype = torch::kFloat64; datatype = TDOUBLE; break;
	                default: throw std::runtime_error("Unsupported BITPIX");
	            }
	        }

        auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));

        if (compressed && !scaled && is_parallel_compressed_codec_cached(hdu_num)) {
            bool allow_float_parallel = true;
            if (datatype == TFLOAT || datatype == TDOUBLE) {
                // Keep float path conservative: only parallelize when compressed-null
                // handling is not needed.
                allow_float_parallel = !has_compressed_nulls_cached(hdu_num);
            }
            if (try_read_compressed_rows_parallel(
                    fptr_,
                    filename_,
                    hdu_num + start_hdu_,
                    naxis,
                    naxes_ll,
                    nelements,
                    datatype,
                    allow_float_parallel,
                    tensor.data_ptr())) {
                return tensor;
            }
        }

        // Fast uncompressed raw path for BYTE images: pread()/mmap + memcpy.
        // Also handle common signed-byte scaling (BITPIX=8 with BZERO=-128) by
        // xor'ing the sign bit in-place after reading raw bytes.
        const bool signed_byte_scaled =
            scaled && bitpix == BYTE_IMG && scale_info.bscale == 1.0 && scale_info.bzero == -128.0;
        if (want_mmap && !compressed && bitpix == BYTE_IMG && (!scaled || signed_byte_scaled) &&
            filename_.find('[') == std::string::npos) {
            // Extended filename syntax can refer to virtual files/HDUs; fall back to CFITSIO.
            LONGLONG headstart = 0, data_offset = 0, dataend = 0;
            status = 0;
            fits_get_hduaddrll(fptr_, &headstart, &data_offset, &dataend, &status);
            if (status == 0 && data_offset > 0) {
                const size_t nbytes = static_cast<size_t>(nelements);
                if (nbytes > 0) {
                    const size_t end_off = static_cast<size_t>(data_offset) + nbytes;
                    if (ensure_raw_fd(end_off)) {
                        // Prefer pread() over mmap+memcpy to avoid per-call mmap/munmap overhead.
                        uint8_t* dst = static_cast<uint8_t*>(tensor.data_ptr());
                        size_t remaining = nbytes;
                        off_t off = static_cast<off_t>(data_offset);
                        bool ok = true;
                        while (remaining > 0) {
                            ssize_t got = ::pread(raw_fd_, dst, remaining, off);
                            if (got < 0) {
                                if (errno == EINTR) {
                                    continue;
                                }
                                ok = false;
                                break;
                            }
                            if (got == 0) {
                                ok = false;
                                break;
                            }
                            dst += static_cast<size_t>(got);
                            off += static_cast<off_t>(got);
                            remaining -= static_cast<size_t>(got);
                        }
                        if (!ok) {
                            void* map_ptr = mmap(
                                nullptr,
                                static_cast<size_t>(raw_file_size_),
                                PROT_READ,
                                MAP_SHARED,
                                raw_fd_,
                                0
                            );
                            if (map_ptr != MAP_FAILED) {
                                const uint8_t* src = static_cast<const uint8_t*>(map_ptr) + data_offset;
                                std::memcpy(tensor.data_ptr(), src, nbytes);
                                munmap(map_ptr, static_cast<size_t>(raw_file_size_));
                                ok = true;
                            }
                        }
                        if (ok) {
                            if (signed_byte_scaled) {
                                _xor_sign_bit_u8(static_cast<uint8_t*>(tensor.data_ptr()), nbytes);
                            }
                            return tensor;
                        }
                    }
                }
            } else {
                status = 0;
            }
        }

        // Fast uncompressed raw path for multi-byte native dtypes:
        // read raw bytes directly and byteswap in-place (FITS is big-endian).
        if (want_mmap && !compressed && !scaled && filename_.find('[') == std::string::npos) {
            size_t elem_size = 0;
            switch (bitpix) {
                case SHORT_IMG: elem_size = sizeof(uint16_t); break;
                case LONG_IMG: elem_size = sizeof(uint32_t); break;
                case LONGLONG_IMG: elem_size = sizeof(uint64_t); break;
                case FLOAT_IMG: elem_size = sizeof(uint32_t); break;
                case DOUBLE_IMG: elem_size = sizeof(uint64_t); break;
                default: elem_size = 0; break;
            }
            if (elem_size > 0) {
                LONGLONG headstart = 0, data_offset = 0, dataend = 0;
                status = 0;
                fits_get_hduaddrll(fptr_, &headstart, &data_offset, &dataend, &status);
                if (status == 0 && data_offset > 0) {
                    const size_t nbytes = static_cast<size_t>(nelements) * elem_size;
                    if (nbytes > 0) {
                        const size_t end_off = static_cast<size_t>(data_offset) + nbytes;
                        if (ensure_raw_fd(end_off)) {
                            uint8_t* dst = static_cast<uint8_t*>(tensor.data_ptr());
                            size_t remaining = nbytes;
                            off_t off = static_cast<off_t>(data_offset);
                            bool ok = true;
                            while (remaining > 0) {
                                ssize_t got = ::pread(raw_fd_, dst, remaining, off);
                                if (got < 0) {
                                    if (errno == EINTR) {
                                        continue;
                                    }
                                    ok = false;
                                    break;
                                }
                                if (got == 0) {
                                    ok = false;
                                    break;
                                }
                                dst += static_cast<size_t>(got);
                                off += static_cast<off_t>(got);
                                remaining -= static_cast<size_t>(got);
                            }
                            if (!ok) {
                                void* map_ptr = mmap(
                                    nullptr,
                                    static_cast<size_t>(raw_file_size_),
                                    PROT_READ,
                                    MAP_SHARED,
                                    raw_fd_,
                                    0
                                );
                                if (map_ptr != MAP_FAILED) {
                                    const uint8_t* src = static_cast<const uint8_t*>(map_ptr) + data_offset;
                                    std::memcpy(tensor.data_ptr(), src, nbytes);
                                    munmap(map_ptr, static_cast<size_t>(raw_file_size_));
                                    ok = true;
                                }
                            }
                            if (ok) {
                                if (host_is_little_endian()) {
                                    if (elem_size == sizeof(uint16_t)) {
                                        auto* p = static_cast<uint16_t*>(tensor.data_ptr());
                                        at::parallel_for(0, static_cast<int64_t>(nelements), 1 << 16, [&](int64_t begin, int64_t end) {
                                            for (int64_t i = begin; i < end; ++i) {
                                                p[i] = bswap_16(p[i]);
                                            }
                                        });
                                    } else if (elem_size == sizeof(uint32_t)) {
                                        auto* p = static_cast<uint32_t*>(tensor.data_ptr());
                                        at::parallel_for(0, static_cast<int64_t>(nelements), 1 << 16, [&](int64_t begin, int64_t end) {
                                            for (int64_t i = begin; i < end; ++i) {
                                                p[i] = bswap_32(p[i]);
                                            }
                                        });
                                    } else if (elem_size == sizeof(uint64_t)) {
                                        auto* p = static_cast<uint64_t*>(tensor.data_ptr());
                                        at::parallel_for(0, static_cast<int64_t>(nelements), 1 << 16, [&](int64_t begin, int64_t end) {
                                            for (int64_t i = begin; i < end; ++i) {
                                                p[i] = bswap_64(p[i]);
                                            }
                                        });
                                    }
                                }
                                return tensor;
                            }
                        }
                    }
                } else {
                    status = 0;
                }
            }
        }

        // Note: CFITSIO also exposes a tile cache API for compressed images in
        // some builds, but our vendored CFITSIO doesn't provide a stable symbol
        // for it. We instead tune CFITSIO buffering via compile-time knobs in
        // our CMake config.

        int anynul = 0;
        float fnullval = NAN;
        double dnullval = NAN;
        void* nullval_ptr = nullptr;
        if ((datatype == TFLOAT || datatype == TDOUBLE) && compressed) {
            if (has_compressed_nulls_cached(hdu_num)) {
                if (datatype == TFLOAT) {
                    nullval_ptr = &fnullval;
                } else {
                    nullval_ptr = &dnullval;
                }
            }
        }

        fits_read_img(
            fptr_,
            datatype,
            1,
            nelements,
            nullval_ptr,
            tensor.data_ptr(),
            &anynul,
            &status
        );

        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error reading image data: status=" + std::to_string(status) + " msg=" + std::string(err_text));
        }

        return tensor;
    }

    torch::Tensor read_image_raw(int hdu_num, bool use_mmap = true) {
        int status = 0;
        
        ensure_hdu(hdu_num, &status);
        if (status != 0) {
            throw std::runtime_error("Could not move to HDU");
        }
        const bool want_mmap = use_mmap;

        int bitpix = 0;
        int naxis = 0;
        std::array<LONGLONG, 9> naxes_ll{};
        {
            const auto& info = get_image_info(hdu_num);
            bitpix = std::get<0>(info);
            naxis = std::get<1>(info);
            naxes_ll = std::get<2>(info);
        }

        // Fast return for empty images (e.g., empty primary HDU in MEF files)
        if (naxis == 0) {
            torch::ScalarType dtype;
            switch (bitpix) {
                case BYTE_IMG:   dtype = torch::kUInt8; break;
                case SHORT_IMG:  dtype = torch::kInt16; break;
                case LONG_IMG:   dtype = torch::kInt32; break;
                case FLOAT_IMG:  dtype = torch::kFloat32; break;
                case DOUBLE_IMG: dtype = torch::kFloat64; break;
                default:         dtype = torch::kUInt8; break;
            }
            return torch::empty({0}, torch::TensorOptions().dtype(dtype));
        }

        torch::ScalarType dtype;
        int datatype;
        switch (bitpix) {
            case BYTE_IMG:   dtype = torch::kUInt8; datatype = TBYTE; break;
            case SHORT_IMG:  dtype = torch::kInt16; datatype = TSHORT; break;
            case LONG_IMG:   dtype = torch::kInt32; datatype = TINT; break;
            case FLOAT_IMG:  dtype = torch::kFloat32; datatype = TFLOAT; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; datatype = TDOUBLE; break;
            default: throw std::runtime_error("Unsupported BITPIX");
        }

        LONGLONG nelements = 0;
        if (naxis > 0) {
            nelements = 1;
            for (int i = 0; i < naxis; ++i) {
                nelements *= naxes_ll[i];
            }
        }

        int64_t torch_shape[9];
        for (int i = 0; i < naxis; ++i) {
            torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);
        }

        auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));

        // Fast uncompressed image raw path: mmap + direct decode.
        if (want_mmap && !is_compressed_image_cached(hdu_num) && bitpix == BYTE_IMG) {
            if (filename_.find('[') == std::string::npos) {
                LONGLONG headstart = 0, data_offset = 0, dataend = 0;
                status = 0;
                fits_get_hduaddrll(fptr_, &headstart, &data_offset, &dataend, &status);
                if (status == 0 && data_offset > 0) {
                    const size_t elem_size = 1;
                    if (elem_size > 0) {
                        const size_t nbytes = static_cast<size_t>(nelements) * elem_size;
                        const size_t end_off = static_cast<size_t>(data_offset) + nbytes;
                        if (ensure_raw_fd(end_off)) {
                            uint8_t* dst = static_cast<uint8_t*>(tensor.data_ptr());
                            size_t remaining = nbytes;
                            off_t off = static_cast<off_t>(data_offset);
                            bool ok = true;
                            while (remaining > 0) {
                                ssize_t got = ::pread(raw_fd_, dst, remaining, off);
                                if (got < 0) {
                                    if (errno == EINTR) {
                                        continue;
                                    }
                                    ok = false;
                                    break;
                                }
                                if (got == 0) {
                                    ok = false;
                                    break;
                                }
                                dst += static_cast<size_t>(got);
                                off += static_cast<off_t>(got);
                                remaining -= static_cast<size_t>(got);
                            }
                            if (ok) {
                                return tensor;
                            }

                            void* map_ptr = mmap(
                                nullptr,
                                static_cast<size_t>(raw_file_size_),
                                PROT_READ,
                                MAP_SHARED,
                                raw_fd_,
                                0
                            );
                            if (map_ptr != MAP_FAILED) {
                                const uint8_t* src = static_cast<const uint8_t*>(map_ptr) + data_offset;
                                std::memcpy(tensor.data_ptr(), src, nbytes);
                                munmap(map_ptr, static_cast<size_t>(raw_file_size_));
                                return tensor;
                            }
                        }
                    }
                } else {
                    status = 0;
                }
            }
        }

        // Disable CFITSIO scaling for raw reads and restore after.
        FITSFile::BScaleGuard guard;
        guard.fptr = fptr_;
        {
            int key_status = 0;
            double bscale = 1.0;
            double bzero = 0.0;

            key_status = 0;
            fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &key_status);
            if (key_status != 0 && key_status != KEY_NO_EXIST) {
                bscale = 1.0;
            }

            key_status = 0;
            fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &key_status);
            if (key_status != 0 && key_status != KEY_NO_EXIST) {
                bzero = 0.0;
            }

            guard.bscale = bscale;
            guard.bzero = bzero;
        }

        status = 0;
        fits_set_bscale(fptr_, 1.0, 0.0, &status);
        if (status == 0) {
            guard.active = true;
        } else {
            status = 0;
        }

        int anynul = 0;
        float fnullval = NAN;
        double dnullval = NAN;
        void* nullval_ptr = nullptr;
        if (datatype == TFLOAT || datatype == TDOUBLE) {
            if (is_compressed_image_cached(hdu_num) && has_compressed_nulls_cached(hdu_num)) {
                if (datatype == TFLOAT) {
                    nullval_ptr = &fnullval;
                } else {
                    nullval_ptr = &dnullval;
                }
            }
        }

        fits_read_img(
            fptr_,
            datatype,
            1,
            nelements,
            nullval_ptr,
            tensor.data_ptr(),
            &anynul,
            &status
        );

        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error reading image data: status=" + std::to_string(status) + " msg=" + std::string(err_text));
        }

        return tensor;
    }

    bool write_image(nb::ndarray<> tensor, int hdu_num, double bscale, double bzero) {
        int status = 0;
        
        int naxis = tensor.ndim();
        std::vector<long> naxes(naxis);
        for (int i = 0; i < naxis; ++i) {
            naxes[i] = tensor.shape(i);
        }
        
        // FITS expects C-contiguous (row-major) order for dimensions?
        // Actually, FITS is Fortran-order (column-major) conceptually, but C libraries usually handle it.
        // cfitsio expects naxes[0] to be the fastest varying dimension (width).
        // PyTorch/NumPy are C-contiguous: shape is (height, width).
        // So naxes[0] is height, naxes[1] is width.
        // We need to reverse the shape for cfitsio if we want it to match standard FITS interpretation?
        // Or does cfitsio handle C arrays correctly?
        // Let's stick to what we had: reverse the shape.
        std::reverse(naxes.begin(), naxes.end());
        
        int bitpix = FLOAT_IMG;
        int datatype = TFLOAT;
        nb::dlpack::dtype dt = tensor.dtype();
        
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) { bitpix = BYTE_IMG; datatype = TBYTE; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) { bitpix = SHORT_IMG; datatype = TSHORT; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) { bitpix = LONG_IMG; datatype = TINT; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) { bitpix = FLOAT_IMG; datatype = TFLOAT; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) { bitpix = DOUBLE_IMG; datatype = TDOUBLE; }
        else throw std::runtime_error("Unsupported tensor dtype");
        
        long nelements = 1;
        for (long dim : naxes) nelements *= dim;
        
        if (hdu_num == 0) {
            // Create Primary HDU
            fits_create_img(fptr_, bitpix, naxis, naxes.data(), &status);
        } else {
            // Create new HDU
            fits_create_img(fptr_, bitpix, naxis, naxes.data(), &status);
        }
        
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error creating image: status=" + std::to_string(status) + " msg=" + std::string(err_text));
        }
        
        // Write data
        // nb::ndarray::data() returns void*
        fits_write_img(fptr_, datatype, 1, nelements, tensor.data(), &status);
        
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error writing image: status=" + std::to_string(status) + " msg=" + std::string(err_text));
        }
        return true;
    }

    // ...

    std::vector<std::tuple<std::string, std::string, std::string>> get_header(int hdu_num) {
        int status = 0;
        ensure_hdu(hdu_num, &status);
        if (status != 0) throw std::runtime_error("Could not move to HDU");
        
        int nkeys = 0;
        int morekeys = 0;
        fits_get_hdrspace(fptr_, &nkeys, &morekeys, &status);
        
        std::vector<std::tuple<std::string, std::string, std::string>> header;
        header.reserve(nkeys);
        
        char keyname[FLEN_KEYWORD];
        char value[FLEN_VALUE];
        char comment[FLEN_COMMENT];
        int length;
        
        for (int i = 1; i <= nkeys; i++) {
            fits_read_keyn(fptr_, i, keyname, value, comment, &status);
            if (status == 0) {
                std::string key_str(keyname);
                std::string val_str(value);
                std::string com_str(comment);
                
                // Sanitize string (keep only ASCII printable)
                val_str.erase(std::remove_if(val_str.begin(), val_str.end(), [](unsigned char c) {
                    return c < 32 || c > 126;
                }), val_str.end());

                // Parse string values: remove quotes and trim
                if (val_str.length() >= 2 && val_str.front() == '\'') {
                    // Find the last quote (ignoring trailing comments if any, but fits_read_keyn separates comment)
                    // value contains the value string. For strings it is 'TEXT'.
                    size_t last_quote = val_str.rfind('\'');
                    if (last_quote != std::string::npos && last_quote > 0) {
                        val_str = val_str.substr(1, last_quote - 1);
                        // Trim trailing spaces
                        size_t last_char = val_str.find_last_not_of(' ');
                        if (last_char != std::string::npos) {
                            val_str = val_str.substr(0, last_char + 1);
                        } else {
                            val_str = "";
                        }
                        // Handle escaped quotes '' -> '
                        size_t pos = 0;
                        while ((pos = val_str.find("''", pos)) != std::string::npos) {
                            val_str.replace(pos, 2, "'");
                            pos += 1;
                        }
                    }
                }
                
                // For HISTORY and COMMENT, value is often empty and content is in comment?
                // Or fits_read_keyn puts the text in comment?
                // Let's check if key is HISTORY or COMMENT
                if (key_str == "HISTORY" || key_str == "COMMENT") {
                     // For these, the "value" is the comment string.
                     // But fits_read_keyn might put it in comment arg?
                     // Actually, for HISTORY, there is no value field. The text starts at col 9.
                     // fits_read_keyn docs say: "returns the comment string".
                     // It seems for HISTORY, value is empty string, and comment contains the text.
                     // But we want the text as the "value" in our tuple?
                     // Astropy treats it as a list of values.
                     if (val_str.empty() && !com_str.empty()) {
                         val_str = com_str;
                         com_str = "";
                     }
                }
                
                header.emplace_back(key_str, val_str, com_str);
            } else {
                status = 0; // Ignore error for single key
            }
        }
        return header;
    }

    std::vector<long> get_shape(int hdu_num) {
        int status = 0;
        ensure_hdu(hdu_num, &status);
        
        int naxis = 0;
        fits_get_img_dim(fptr_, &naxis, &status);
        std::vector<long> naxes(naxis);
        fits_get_img_size(fptr_, naxis, naxes.data(), &status);
        
        // Return in numpy/torch order (C-contiguous)
        std::reverse(naxes.begin(), naxes.end());
        return naxes;
    }

    int get_dtype(int hdu_num) {
        int status = 0;
        ensure_hdu(hdu_num, &status);
        
        int bitpix = 0;
        fits_get_img_type(fptr_, &bitpix, &status);
        return bitpix;
    }

    torch::Tensor read_subset(int hdu_num, long x1, long y1, long x2, long y2) {
        // Subset reading for 2D images (x, y) with exclusive x2/y2 bounds
        int status = 0;
        ensure_hdu(hdu_num, &status);
        if (status != 0) {
            throw std::runtime_error("Could not move to HDU");
        }

        int bitpix = 0;
        int naxis = 0;
        long naxes[9] = {0};
        {
            const auto& info = get_image_info(hdu_num);
            bitpix = std::get<0>(info);
            naxis = std::get<1>(info);
            const auto& naxes_ll = std::get<2>(info);
            for (int i = 0; i < 9; ++i) {
                naxes[i] = static_cast<long>(naxes_ll[i]);
            }
        }
        if (naxis < 2) {
            throw std::runtime_error("Subset reading requires at least 2D image");
        }

        long max_x = naxes[0];
        long max_y = naxes[1];

        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 > max_x) x2 = max_x;
        if (y2 > max_y) y2 = max_y;

        if (x2 <= x1 || y2 <= y1) {
            return torch::empty({0, 0}, torch::TensorOptions().dtype(torch::kFloat32));
        }

        const auto& scale_info = get_scale_info(hdu_num, bitpix);
        bool scaled = scale_info.scaled;

        torch::ScalarType dtype;
        int datatype;
        if (scaled) {
            dtype = torch::kFloat32;
            datatype = TFLOAT;
        } else {
            switch (bitpix) {
                case BYTE_IMG:   dtype = torch::kUInt8; datatype = TBYTE; break;
                case SHORT_IMG:  dtype = torch::kInt16; datatype = TSHORT; break;
                case LONG_IMG:   dtype = torch::kInt32; datatype = TINT; break;
                case LONGLONG_IMG: dtype = torch::kInt64; datatype = TLONGLONG; break;
                case FLOAT_IMG:  dtype = torch::kFloat32; datatype = TFLOAT; break;
                case DOUBLE_IMG: dtype = torch::kFloat64; datatype = TDOUBLE; break;
                default: throw std::runtime_error("Unsupported BITPIX");
            }
        }

        long width = x2 - x1;
        long height = y2 - y1;
        std::vector<int64_t> shape = {height, width}; // Torch order (y, x)
        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));

        long fpixel[2] = {x1 + 1, y1 + 1}; // FITS is 1-based
        long lpixel[2] = {x2, y2};         // exclusive bounds -> inclusive in FITS
        long inc[2] = {1, 1};
        int anynul = 0;

        fits_read_subset(
            fptr_, datatype, fpixel, lpixel, inc, nullptr, tensor.data_ptr(), &anynul, &status
        );

        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error reading subset: status=" + std::to_string(status) + " msg=" + std::string(err_text));
        }

        return tensor;
    }

    std::unordered_map<std::string, double> compute_stats(int hdu_num) {
        return {};
    }

    int get_num_hdus() {
        int status = 0;
        int nhdus = 0;
        fits_get_num_hdus(fptr_, &nhdus, &status);
        return nhdus;
    }

    std::string get_hdu_type(int hdu_num) {
        int status = 0;
        ensure_hdu(hdu_num, &status);
        int hdutype = 0;
        fits_get_hdu_type(fptr_, &hdutype, &status);
        if (hdutype == IMAGE_HDU) return "IMAGE";
        if (hdutype == ASCII_TBL) return "ASCII_TABLE";
        if (hdutype == BINARY_TBL) return "BINARY_TABLE";
        return "UNKNOWN";
    }



    bool write_hdus(nb::list hdus, bool overwrite) {
        int hdu_count = 0;
        
        for (auto handle : hdus) {
            nb::object hdu_obj = nb::cast<nb::object>(handle);
            
            // Check for TableHDU (prefer raw data to preserve strings/VLA)
            if (nb::hasattr(hdu_obj, "_raw_data")) {
                nb::dict data_dict = nb::cast<nb::dict>(hdu_obj.attr("_raw_data"));
                nb::dict header_dict;
                if (nb::hasattr(hdu_obj, "header")) {
                     header_dict = nb::cast<nb::dict>(hdu_obj.attr("header"));
                }
                write_table_hdu(fptr_, data_dict, header_dict, nb::none(), false);
                hdu_count++;
                continue;
            }
            if (nb::hasattr(hdu_obj, "feat_dict")) {
                nb::dict data_dict = nb::cast<nb::dict>(hdu_obj.attr("feat_dict"));
                nb::dict header_dict;
                if (nb::hasattr(hdu_obj, "header")) {
                     header_dict = nb::cast<nb::dict>(hdu_obj.attr("header"));
                }
                write_table_hdu(fptr_, data_dict, header_dict, nb::none(), false);
                hdu_count++;
                continue;
            }
            
            // Assume TensorHDU or Image
            nb::object data_obj;
            bool has_data = false;
            
            if (nb::hasattr(hdu_obj, "to_tensor")) {
                 // Use to_tensor() to get data
                 try {
                     data_obj = hdu_obj.attr("to_tensor")();
                     has_data = true;
                 } catch (...) {}
            }
            
            if (!has_data && nb::hasattr(hdu_obj, "data")) {
                data_obj = hdu_obj.attr("data");
                has_data = true;
            } else if (!has_data && nb::isinstance<nb::dict>(hdu_obj)) {
                nb::dict d = nb::cast<nb::dict>(hdu_obj);
                if (d.contains("data")) {
                    data_obj = d["data"];
                    has_data = true;
                }
            }
            
            if (has_data) {
                try {
                    nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(data_obj);
                    write_image(tensor, hdu_count, 1.0, 0.0);
                } catch (...) {
                    // If data is not a tensor (e.g. None or empty), write empty image
                    int status = 0;
                    long naxes[1] = {0};
                    fits_create_img(fptr_, BYTE_IMG, 0, naxes, &status);
                }
            } else {
                int status = 0;
                long naxes[1] = {0};
                fits_create_img(fptr_, BYTE_IMG, 0, naxes, &status);
            }
            
            // Write header
            nb::object header_obj;
            if (nb::hasattr(hdu_obj, "header")) {
                header_obj = hdu_obj.attr("header");
            } else if (nb::isinstance<nb::dict>(hdu_obj)) {
                nb::dict d = nb::cast<nb::dict>(hdu_obj);
                if (d.contains("header")) {
                    header_obj = d["header"];
                }
            }
            
            if (header_obj.is_valid()) {
                try {
                    nb::dict header = nb::cast<nb::dict>(header_obj);
                    for (auto item : header) {
                        std::string key = nb::cast<std::string>(item.first);
                        key = sanitize_fits_key(key);

                        // Never overwrite the structural keywords that CFITSIO
                        // sets up when creating image/table HDUs.
                        std::string key_upper = key;
                        std::transform(key_upper.begin(), key_upper.end(), key_upper.begin(),
                                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                        if (key_upper == "END" ||
                            key_upper == "SIMPLE" ||
                            key_upper == "XTENSION" ||
                            key_upper == "BITPIX" ||
                            key_upper == "NAXIS" ||
                            key_upper == "EXTEND" ||
                            key_upper == "PCOUNT" ||
                            key_upper == "GCOUNT" ||
                            key_upper == "TFIELDS" ||
                            key_upper == "THEAP" ||
                            key_upper.rfind("NAXIS", 0) == 0) {
                            continue;
                        }
                        
                        try {
                            int key_status = 0;
                            if (nb::isinstance<nb::str>(item.second)) {
                                std::string val = nb::cast<std::string>(item.second);
                                val = sanitize_fits_string(val);
                                fits_update_key(fptr_, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &key_status);
                            } else if (nb::isinstance<int>(item.second)) {
                                int val = nb::cast<int>(item.second);
                                fits_update_key(fptr_, TINT, key.c_str(), &val, nullptr, &key_status);
                            } else if (nb::isinstance<float>(item.second)) {
                                float val = nb::cast<float>(item.second);
                                fits_update_key(fptr_, TFLOAT, key.c_str(), &val, nullptr, &key_status);
                            } else if (nb::isinstance<double>(item.second)) {
                                double val = nb::cast<double>(item.second);
                                fits_update_key(fptr_, TDOUBLE, key.c_str(), &val, nullptr, &key_status);
                            } else if (nb::isinstance<bool>(item.second)) {
                                int val = nb::cast<bool>(item.second) ? 1 : 0;
                                fits_update_key(fptr_, TLOGICAL, key.c_str(), &val, nullptr, &key_status);
                            }
                        } catch (...) {}
                    }
                } catch (...) {}
            }
            
            hdu_count++;
        }
        return true;
    }

    bool write_hdus_compressed_images(nb::list hdus, int compression_type) {
        // Write an empty primary HDU, then store each image as a tile-compressed
        // image extension using CFITSIO's transparent compression interface.
        int status = 0;
        long naxes0[1] = {0};
        fits_create_img(fptr_, BYTE_IMG, 0, naxes0, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to create primary HDU for compressed file");
        }

        auto write_header_dict = [&](nb::object hdu_obj) {
            nb::object header_obj;
            if (nb::hasattr(hdu_obj, "header")) {
                header_obj = hdu_obj.attr("header");
            } else if (nb::isinstance<nb::dict>(hdu_obj)) {
                nb::dict d = nb::cast<nb::dict>(hdu_obj);
                if (d.contains("header")) {
                    header_obj = d["header"];
                }
            }
            if (!header_obj.is_valid()) {
                return;
            }
            nb::dict header = nb::cast<nb::dict>(header_obj);
            for (auto item : header) {
                std::string key = nb::cast<std::string>(item.first);
                key = sanitize_fits_key(key);

                std::string key_upper = key;
                std::transform(key_upper.begin(), key_upper.end(), key_upper.begin(),
                               [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                if (key_upper == "END" ||
                    key_upper == "SIMPLE" ||
                    key_upper == "XTENSION" ||
                    key_upper == "BITPIX" ||
                    key_upper == "NAXIS" ||
                    key_upper == "EXTEND" ||
                    key_upper == "PCOUNT" ||
                    key_upper == "GCOUNT" ||
                    key_upper == "TFIELDS" ||
                    key_upper == "THEAP" ||
                    key_upper == "DATASUM" ||
                    key_upper == "CHECKSUM" ||
                    key_upper == "ZIMAGE" ||
                    key_upper == "ZCMPTYPE" ||
                    key_upper == "ZBITPIX" ||
                    key_upper == "ZNAXIS" ||
                    key_upper == "ZPCOUNT" ||
                    key_upper == "ZGCOUNT" ||
                    key_upper == "ZHECKSUM" ||
                    key_upper == "ZDATASUM" ||
                    key_upper.rfind("NAXIS", 0) == 0) {
                    continue;
                }
                if (key_upper.rfind("ZNAXIS", 0) == 0 ||
                    key_upper.rfind("ZTILE", 0) == 0 ||
                    key_upper.rfind("ZNAME", 0) == 0 ||
                    key_upper.rfind("ZVAL", 0) == 0) {
                    continue;
                }

                int key_status = 0;
                try {
                    if (nb::isinstance<nb::str>(item.second)) {
                        std::string val = nb::cast<std::string>(item.second);
                        val = sanitize_fits_string(val);
                        fits_update_key(fptr_, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &key_status);
                    } else if (nb::isinstance<int>(item.second)) {
                        int val = nb::cast<int>(item.second);
                        fits_update_key(fptr_, TINT, key.c_str(), &val, nullptr, &key_status);
                    } else if (nb::isinstance<float>(item.second)) {
                        float val = nb::cast<float>(item.second);
                        fits_update_key(fptr_, TFLOAT, key.c_str(), &val, nullptr, &key_status);
                    } else if (nb::isinstance<double>(item.second)) {
                        double val = nb::cast<double>(item.second);
                        fits_update_key(fptr_, TDOUBLE, key.c_str(), &val, nullptr, &key_status);
                    } else if (nb::isinstance<bool>(item.second)) {
                        int val = nb::cast<bool>(item.second) ? 1 : 0;
                        fits_update_key(fptr_, TLOGICAL, key.c_str(), &val, nullptr, &key_status);
                    }
                } catch (...) {
                    // ignore per-key failures (unsupported types)
                }
            }
        };

        for (auto handle : hdus) {
            nb::object hdu_obj = nb::cast<nb::object>(handle);

            // Table HDUs are written as regular FITS table extensions.
            if (nb::hasattr(hdu_obj, "_raw_data")) {
                nb::dict data_dict = nb::cast<nb::dict>(hdu_obj.attr("_raw_data"));
                nb::dict header_dict;
                if (nb::hasattr(hdu_obj, "header")) {
                    header_dict = nb::cast<nb::dict>(hdu_obj.attr("header"));
                }
                write_table_hdu(fptr_, data_dict, header_dict, nb::none(), false);
                continue;
            }
            if (nb::hasattr(hdu_obj, "feat_dict")) {
                nb::dict data_dict = nb::cast<nb::dict>(hdu_obj.attr("feat_dict"));
                nb::dict header_dict;
                if (nb::hasattr(hdu_obj, "header")) {
                    header_dict = nb::cast<nb::dict>(hdu_obj.attr("header"));
                }
                write_table_hdu(fptr_, data_dict, header_dict, nb::none(), false);
                continue;
            }

            nb::object data_obj;
            bool has_data = false;
            if (nb::hasattr(hdu_obj, "to_tensor")) {
                try {
                    data_obj = hdu_obj.attr("to_tensor")();
                    has_data = true;
                } catch (...) {}
            }
            if (!has_data && nb::hasattr(hdu_obj, "data")) {
                data_obj = hdu_obj.attr("data");
                has_data = true;
            } else if (!has_data && nb::isinstance<nb::dict>(hdu_obj)) {
                nb::dict d = nb::cast<nb::dict>(hdu_obj);
                if (d.contains("data")) {
                    data_obj = d["data"];
                    has_data = true;
                }
            }
            if (!has_data) {
                throw std::runtime_error("Compressed writing requires image data");
            }

            nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(data_obj);

            // Compute FITS naxes (reverse of ndarray shape, to match existing writer).
            int naxis = tensor.ndim();
            if (naxis <= 0) {
                throw std::runtime_error("Unsupported image ndim for compressed write");
            }
            std::vector<long> naxes(naxis);
            for (int i = 0; i < naxis; ++i) {
                naxes[i] = tensor.shape(i);
            }
            std::reverse(naxes.begin(), naxes.end());

            // Set compression parameters before creating the image header.
            status = 0;
            fits_set_compression_type(fptr_, compression_type, &status);
            if (status != 0) {
                throw std::runtime_error("Failed to set compression type");
            }
            std::vector<long> tilesize(naxis, 1);
            tilesize[0] = naxes[0];  // row-by-row tiles
            fits_set_tile_dim(fptr_, naxis, tilesize.data(), &status);
            if (status != 0) {
                throw std::runtime_error("Failed to set tile dimensions");
            }

            // Determine bitpix/datatype (same as write_image).
            int bitpix = FLOAT_IMG;
            int datatype = TFLOAT;
            nb::dlpack::dtype dt = tensor.dtype();
            if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) { bitpix = BYTE_IMG; datatype = TBYTE; }
            else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 8) { bitpix = SBYTE_IMG; datatype = TSBYTE; }
            else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) { bitpix = SHORT_IMG; datatype = TSHORT; }
            else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) { bitpix = LONG_IMG; datatype = TINT; }
            else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) { bitpix = LONGLONG_IMG; datatype = TLONGLONG; }
            else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) { bitpix = FLOAT_IMG; datatype = TFLOAT; }
            else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) { bitpix = DOUBLE_IMG; datatype = TDOUBLE; }
            else throw std::runtime_error("Unsupported tensor dtype for compressed write");

            // Create compressed image HDU.
            fits_create_img(fptr_, bitpix, naxis, naxes.data(), &status);
            if (status != 0) {
                char err_text[31];
                fits_get_errstatus(status, err_text);
                throw std::runtime_error("Error creating compressed image: status=" + std::to_string(status) + " msg=" + std::string(err_text));
            }

            long nelements = 1;
            for (long dim : naxes) nelements *= dim;
            fits_write_img(fptr_, datatype, 1, nelements, tensor.data(), &status);
            if (status != 0) {
                char err_text[31];
                fits_get_errstatus(status, err_text);
                throw std::runtime_error("Error writing compressed image: status=" + std::to_string(status) + " msg=" + std::string(err_text));
            }

            write_header_dict(hdu_obj);
        }

        return true;
    }


    fitsfile* get_fptr() { return fptr_; }

    std::string read_header_to_string(int hdu_num) {
        int status = 0;
        ensure_hdu(hdu_num, &status);
        char* header_str = nullptr;
        int nkeys = 0;
        if (fits_hdr2str(fptr_, 0, nullptr, 0, &header_str, &nkeys, &status)) {
            return "";
        }
        std::string result(header_str);
        if (header_str) {
            fits_free_memory(header_str, &status);
        }
        return result;
    }

private:
    bool ensure_raw_fd(size_t required_end) {
        if (filename_.find('[') != std::string::npos) {
            return false;
        }
        if (!raw_fd_ready_) {
            raw_fd_ready_ = true;
            raw_fd_ = open_readonly_fd(filename_);
            if (raw_fd_ == -1) {
                return false;
            }
            struct stat sb {};
            if (fstat(raw_fd_, &sb) != 0) {
                ::close(raw_fd_);
                raw_fd_ = -1;
                raw_file_size_ = 0;
                return false;
            }
            raw_file_size_ = sb.st_size;
        }
        return raw_fd_ != -1 && required_end <= static_cast<size_t>(raw_file_size_);
    }

    void close_raw_fd() {
        if (raw_fd_ != -1) {
            ::close(raw_fd_);
            raw_fd_ = -1;
        }
        raw_file_size_ = 0;
        raw_fd_ready_ = false;
    }

    std::string filename_;
    int mode_;
    fitsfile* fptr_ = nullptr;
    bool cached_ = false;
    int start_hdu_ = 1;
    int current_hdu_ = 1;
    int raw_fd_ = -1;
    off_t raw_file_size_ = 0;
    bool raw_fd_ready_ = false;
    bool use_cache_ = false;
    std::unordered_map<int, ScaleInfo> scale_cache_;
    std::unordered_map<int, bool> compressed_cache_;
    std::unordered_map<int, bool> compressed_parallel_cache_;
    std::unordered_map<int, bool> compressed_nulls_cache_;
    std::unordered_map<int, std::tuple<int, int, std::array<LONGLONG, 9>>> image_info_cache_;
    std::shared_ptr<SharedReadMeta> shared_meta_;
};

namespace {
struct CachedHandleGuard {
    std::string path;
    fitsfile* fptr = nullptr;
    bool active = false;
    ~CachedHandleGuard() {
        if (active) {
            torchfits::release_cached(path);
        }
    }
};
}  // namespace

// Fast path: use the unified cached CFITSIO handle directly without constructing a FITSFile wrapper.
// This reduces per-call overhead on the cold path for common "read full image -> torch.Tensor" use.
torch::Tensor read_full_cached(const std::string& path, int hdu_num, bool use_mmap) {
    if (path.find('[') != std::string::npos) {
        // Extended filename syntax: semantics can depend on initial HDU; fall back to the wrapper.
        FITSFile file(path.c_str(), 0);
        return file.read_image(hdu_num, use_mmap);
    }

    fitsfile* fptr = torchfits::get_or_open_cached(path);
    if (!fptr) {
        throw std::runtime_error("Could not open FITS file: " + path);
    }
    CachedHandleGuard guard;
    guard.path = path;
    guard.fptr = fptr;
    guard.active = true;

    int status = 0;
    const int target_hdu = hdu_num + 1;
    int current_hdu = 0;
    fits_get_hdu_num(fptr, &current_hdu);
    if (current_hdu != target_hdu) {
        fits_movabs_hdu(fptr, target_hdu, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Could not move to HDU");
        }
    }

    auto meta = get_shared_meta_for_path(path);

    // Thread-local cache to avoid mutex traffic on repeated reads of the same HDU.
    // Keyed by SharedReadMeta pointer (invalidated on writes by replacing the shared meta).
    struct LocalKey {
        uint64_t meta_uid = 0;
        int hdu = 0;
        bool operator==(const LocalKey& o) const { return meta_uid == o.meta_uid && hdu == o.hdu; }
    };
    struct LocalKeyHash {
        size_t operator()(const LocalKey& k) const noexcept {
            size_t h = std::hash<uint64_t>()(k.meta_uid);
            h ^= (size_t) k.hdu + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct LocalHduMeta {
        bool has_info = false;
        int bitpix = 0;
        int naxis = 0;
        std::array<LONGLONG, 9> naxes_ll{};
        bool has_compressed = false;
        bool compressed = false;
        bool has_compressed_parallel = false;
        bool compressed_parallel = false;
        bool has_nulls = false;
        bool compressed_nulls = false;
        bool has_scale = false;
        bool scaled = false;
        bool trusted = true;
        double bscale = 1.0;
        double bzero = 0.0;
    };
    static thread_local std::unordered_map<LocalKey, LocalHduMeta, LocalKeyHash> tl_cache;

    LocalKey key{meta ? meta->uid : 0, hdu_num};
    LocalHduMeta* local = nullptr;
    {
        auto it = tl_cache.find(key);
        if (it != tl_cache.end()) {
            local = &it->second;
        }
    }
    if (!local) {
        if (tl_cache.size() > 4096) {
            tl_cache.clear();
        }
        auto inserted = tl_cache.emplace(key, LocalHduMeta{});
        local = &inserted.first->second;
    }

    auto get_image_info = [&]() -> void {
        if (local->has_info) return;
        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            auto it = meta->image_info_cache.find(hdu_num);
            if (it != meta->image_info_cache.end()) {
                local->bitpix = std::get<0>(it->second);
                local->naxis = std::get<1>(it->second);
                local->naxes_ll = std::get<2>(it->second);
                local->has_info = true;
                return;
            }
        }
        local->naxes_ll.fill(0);
        status = 0;
        fits_get_img_paramll(fptr, 9, &local->bitpix, &local->naxis, local->naxes_ll.data(), &status);
        if (status != 0) {
            throw std::runtime_error("Could not read image parameters");
        }
        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            meta->image_info_cache[hdu_num] = std::make_tuple(local->bitpix, local->naxis, local->naxes_ll);
        }
        local->has_info = true;
    };

    auto get_compressed = [&]() -> bool {
        if (local->has_compressed) return local->compressed;
        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            auto it = meta->compressed_cache.find(hdu_num);
            if (it != meta->compressed_cache.end()) {
                local->compressed = it->second;
                local->has_compressed = true;
                return local->compressed;
            }
        }
        status = 0;
        const int comp = fits_is_compressed_image(fptr, &status);
        local->compressed = (status == 0) && (comp != 0);
        if (status != 0) {
            status = 0;
        }
        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            meta->compressed_cache[hdu_num] = local->compressed;
        }
        local->has_compressed = true;
        return local->compressed;
    };

    auto get_compressed_parallel = [&]() -> bool {
        if (local->has_compressed_parallel) return local->compressed_parallel;
        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            auto it = meta->compressed_parallel_cache.find(hdu_num);
            if (it != meta->compressed_parallel_cache.end()) {
                local->compressed_parallel = it->second;
                local->has_compressed_parallel = true;
                return local->compressed_parallel;
            }
        }

        bool result = false;
        char zcmptype[FLEN_VALUE];
        std::memset(zcmptype, 0, sizeof(zcmptype));
        status = 0;
        fits_read_key(fptr, TSTRING, "ZCMPTYPE", zcmptype, nullptr, &status);
        if (status == 0) {
            std::string zcmp(zcmptype);
            std::transform(
                zcmp.begin(), zcmp.end(), zcmp.begin(),
                [](unsigned char c) { return static_cast<char>(std::toupper(c)); }
            );
            if (zcmp.find("RICE") != std::string::npos) {
                result = true;
            } else if (compressed_parallel_hcompress_enabled() &&
                       zcmp.find("HCOMPRESS") != std::string::npos) {
                result = true;
            }
        } else {
            status = 0;
        }
        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            meta->compressed_parallel_cache[hdu_num] = result;
        }
        local->compressed_parallel = result;
        local->has_compressed_parallel = true;
        return local->compressed_parallel;
    };

    auto get_compressed_nulls = [&]() -> bool {
        if (local->has_nulls) return local->compressed_nulls;
        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            auto it = meta->compressed_nulls_cache.find(hdu_num);
            if (it != meta->compressed_nulls_cache.end()) {
                local->compressed_nulls = it->second;
                local->has_nulls = true;
                return local->compressed_nulls;
            }
        }
        local->compressed_nulls = has_compressed_nulls(fptr);
        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            meta->compressed_nulls_cache[hdu_num] = local->compressed_nulls;
        }
        local->has_nulls = true;
        return local->compressed_nulls;
    };

    auto get_scale = [&]() -> void {
        if (local->has_scale) return;
        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            auto it = meta->scale_cache.find(hdu_num);
            if (it != meta->scale_cache.end()) {
                local->scaled = std::get<0>(it->second);
                local->trusted = std::get<1>(it->second);
                local->bscale = std::get<2>(it->second);
                local->bzero = std::get<3>(it->second);
                local->has_scale = true;
                return;
            }
        }
        // Need image info (bitpix) before scale decision
        get_image_info();

        if (local->bitpix < 0) {
            local->scaled = false;
            local->trusted = true;
            local->bscale = 1.0;
            local->bzero = 0.0;
        } else {
            double bscale = 1.0;
            double bzero = 0.0;
            status = 0;
            fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, nullptr, &status);
            if (status == 0) {
                local->bscale = bscale;
                if (bscale != 1.0) local->scaled = true;
            } else if (status != KEY_NO_EXIST) {
                local->scaled = true;
                local->trusted = false;
            }

            status = 0;
            fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, nullptr, &status);
            if (status == 0) {
                local->bzero = bzero;
                if (bzero != 0.0) local->scaled = true;
            } else if (status != KEY_NO_EXIST) {
                local->scaled = true;
                local->trusted = false;
            }
        }

        {
            std::lock_guard<std::mutex> lock(meta->mutex);
            meta->scale_cache[hdu_num] = std::make_tuple(
                local->scaled, local->trusted, local->bscale, local->bzero
            );
        }
        local->has_scale = true;
    };

    get_image_info();
    const int bitpix = local->bitpix;
    const int naxis = local->naxis;
    const std::array<LONGLONG, 9> naxes_ll = local->naxes_ll;

    if (naxis == 0) {
        torch::ScalarType dtype = torch::kUInt8;
        switch (bitpix) {
            case BYTE_IMG: dtype = torch::kUInt8; break;
            case SHORT_IMG: dtype = torch::kInt16; break;
            case LONG_IMG: dtype = torch::kInt32; break;
            case LONGLONG_IMG: dtype = torch::kInt64; break;
            case FLOAT_IMG: dtype = torch::kFloat32; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; break;
            default: dtype = torch::kUInt8; break;
        }
        return torch::empty({0}, torch::TensorOptions().dtype(dtype));
    }

    get_scale();
    const bool scaled = local->scaled;
    const double bscale = local->bscale;
    const double bzero = local->bzero;
    const bool compressed = get_compressed();

    LONGLONG nelements = 1;
    for (int i = 0; i < naxis; ++i) {
        nelements *= naxes_ll[i];
    }
    int64_t torch_shape[9];
    for (int i = 0; i < naxis; ++i) {
        torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);
    }

    torch::ScalarType dtype;
    int datatype;
    if (scaled) {
        if (bitpix == BYTE_IMG && bscale == 1.0 && bzero == -128.0) {
            dtype = at::kChar;
            datatype = TSBYTE;
        } else {
            dtype = torch::kFloat32;
            datatype = TFLOAT;
        }
    } else {
        switch (bitpix) {
            case BYTE_IMG: dtype = torch::kUInt8; datatype = TBYTE; break;
            case SHORT_IMG: dtype = torch::kInt16; datatype = TSHORT; break;
            case LONG_IMG: dtype = torch::kInt32; datatype = TINT; break;
            case LONGLONG_IMG: dtype = torch::kInt64; datatype = TLONGLONG; break;
            case FLOAT_IMG: dtype = torch::kFloat32; datatype = TFLOAT; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; datatype = TDOUBLE; break;
            default: throw std::runtime_error("Unsupported BITPIX");
        }
    }

    auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));

    if (compressed && !scaled && get_compressed_parallel()) {
        bool allow_float_parallel = true;
        if (datatype == TFLOAT || datatype == TDOUBLE) {
            allow_float_parallel = !get_compressed_nulls();
        }
        if (try_read_compressed_rows_parallel(
                fptr,
                path,
                target_hdu,
                naxis,
                naxes_ll,
                nelements,
                datatype,
                allow_float_parallel,
                tensor.data_ptr())) {
            return tensor;
        }
    }

    // Fast uncompressed raw path for BYTE images (matches FITSFile::read_image).
    // Also handle common signed-byte scaling (BITPIX=8 with BZERO=-128) by xor'ing
    // the sign bit in-place after reading raw bytes.
    // For multi-byte types, CFITSIO is generally faster (optimized + endian swap).
    const bool signed_byte_scaled = scaled && bitpix == BYTE_IMG && bscale == 1.0 && bzero == -128.0;
    if (use_mmap && !compressed && bitpix == BYTE_IMG && (!scaled || signed_byte_scaled)) {
        LONGLONG headstart = 0, data_offset = 0, dataend = 0;
        status = 0;
        fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
        if (status == 0 && data_offset > 0) {
            const size_t nbytes = static_cast<size_t>(nelements);
            const int fd = get_shared_raw_fd(meta, path);
            if (fd != -1 && read_region_via_fd(fd, static_cast<off_t>(data_offset), tensor.data_ptr(), nbytes)) {
                if (signed_byte_scaled) {
                    _xor_sign_bit_u8(static_cast<uint8_t*>(tensor.data_ptr()), nbytes);
                }
                return tensor;
            }
        } else {
            status = 0;
        }
    }

    // Raw pread path for uncompressed multi-byte native dtypes:
    // read raw bytes directly and byteswap in-place (FITS is big-endian).
    if (use_mmap && !compressed && !scaled && path.find('[') == std::string::npos) {
        size_t elem_size = 0;
        switch (bitpix) {
            case SHORT_IMG: elem_size = sizeof(uint16_t); break;
            case LONG_IMG: elem_size = sizeof(uint32_t); break;
            case LONGLONG_IMG: elem_size = sizeof(uint64_t); break;
            case FLOAT_IMG: elem_size = sizeof(uint32_t); break;
            case DOUBLE_IMG: elem_size = sizeof(uint64_t); break;
            default: elem_size = 0; break;
        }
        if (elem_size > 0) {
            LONGLONG headstart = 0, data_offset = 0, dataend = 0;
            status = 0;
            fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
            if (status == 0 && data_offset > 0) {
                const size_t nbytes = static_cast<size_t>(nelements) * elem_size;
                if (nbytes > 0) {
                    const int fd = get_shared_raw_fd(meta, path);
                    if (fd != -1 &&
                        read_region_via_fd(fd, static_cast<off_t>(data_offset), tensor.data_ptr(), nbytes)) {
                        if (_host_is_little_endian()) {
                            if (elem_size == sizeof(uint16_t)) {
                                auto* p = static_cast<uint16_t*>(tensor.data_ptr());
                                at::parallel_for(0, static_cast<int64_t>(nelements), 1 << 16, [&](int64_t begin, int64_t end) {
                                    for (int64_t i = begin; i < end; ++i) {
                                        p[i] = _bswap16(p[i]);
                                    }
                                });
                            } else if (elem_size == sizeof(uint32_t)) {
                                auto* p = static_cast<uint32_t*>(tensor.data_ptr());
                                at::parallel_for(0, static_cast<int64_t>(nelements), 1 << 16, [&](int64_t begin, int64_t end) {
                                    for (int64_t i = begin; i < end; ++i) {
                                        p[i] = _bswap32(p[i]);
                                    }
                                });
                            } else if (elem_size == sizeof(uint64_t)) {
                                auto* p = static_cast<uint64_t*>(tensor.data_ptr());
                                at::parallel_for(0, static_cast<int64_t>(nelements), 1 << 16, [&](int64_t begin, int64_t end) {
                                    for (int64_t i = begin; i < end; ++i) {
                                        p[i] = _bswap64(p[i]);
                                    }
                                });
                            }
                        }
                        return tensor;
                    }
                }
            } else {
                status = 0;
            }
        }
    }

read_full_cached_fallback:

    int anynul = 0;
    float fnullval = NAN;
    double dnullval = NAN;
    void* nullval_ptr = nullptr;
    if ((datatype == TFLOAT || datatype == TDOUBLE) && compressed) {
        if (get_compressed_nulls()) {
            nullval_ptr = (datatype == TFLOAT) ? (void*) &fnullval : (void*) &dnullval;
        }
    }

    status = 0;
    fits_read_img(
        fptr,
        datatype,
        1,
        nelements,
        nullval_ptr,
        tensor.data_ptr(),
        &anynul,
        &status
    );
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error reading image data: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }

    return tensor;
}


struct HDUInfo {
    int index;
    std::string type;
    std::vector<std::tuple<std::string, std::string, std::string>> header;
};

// Batch open function to reduce FFI overhead
std::pair<FITSFile*, std::vector<HDUInfo>> open_and_read_headers(const std::string& path, int mode) {
    auto* file = new FITSFile(path.c_str(), mode);
    std::vector<HDUInfo> hdus;
    
    int num_hdus = file->get_num_hdus();
    hdus.reserve(num_hdus);
    
    for (int i = 0; i < num_hdus; ++i) {
        HDUInfo info;
        info.index = i;
        info.type = file->get_hdu_type(i);
        info.header = file->get_header(i);
        hdus.push_back(info);
    }
    
    return {file, hdus};
}

// Adaptive batch reading of images (auto-fallback to sequential for tiny reads)
std::vector<torch::Tensor> read_images_batch(const std::vector<std::string>& paths, int hdu_num) {
    size_t n = paths.size();
    std::vector<torch::Tensor> results(n);
    std::vector<std::string> errors(n);

    if (n == 0) {
        return results;
    }

    // Read the first file synchronously to get a per-file cost estimate.
    auto t0 = std::chrono::steady_clock::now();
    try {
        FITSFile file(paths[0].c_str(), 0);
        results[0] = file.read_image(hdu_num);
    } catch (const std::exception& e) {
        errors[0] = e.what();
    }
    auto t1 = std::chrono::steady_clock::now();
    auto first_read_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    if (!errors[0].empty()) {
        throw std::runtime_error("Error reading " + paths[0] + ": " + errors[0]);
    }

    if (n == 1) {
        return results;
    }

    // Measure thread launch overhead on this machine.
    auto t2 = std::chrono::steady_clock::now();
    std::thread overhead_thread([]() {});
    overhead_thread.join();
    auto t3 = std::chrono::steady_clock::now();
    auto thread_overhead_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    bool use_parallel = first_read_us > thread_overhead_us;

    if (!use_parallel) {
        for (size_t i = 1; i < n; ++i) {
            try {
                FITSFile file(paths[i].c_str(), 0);
                results[i] = file.read_image(hdu_num);
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        }
    } else {
        // Use simple std::thread for now. For production, a thread pool is better.
        std::vector<std::thread> threads;
        threads.reserve(n - 1);
        for (size_t i = 1; i < n; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    // Each thread opens its own file handle - CRITICAL for cfitsio thread safety
                    FITSFile file(paths[i].c_str(), 0);
                    results[i] = file.read_image(hdu_num);
                } catch (const std::exception& e) {
                    errors[i] = e.what();
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    // Check for errors
    for (size_t i = 0; i < n; ++i) {
        if (!errors[i].empty()) {
            throw std::runtime_error("Error reading " + paths[i] + ": " + errors[i]);
        }
    }

    return results;
}

// Batch read multiple HDUs from a single file handle
std::vector<torch::Tensor> read_hdus_batch(const std::string& path, const std::vector<int>& hdus, bool use_mmap) {
    FITSFile file(path.c_str(), 0);
    std::vector<torch::Tensor> results;
    results.reserve(hdus.size());
    for (int hdu_num : hdus) {
        results.push_back(file.read_image(hdu_num, use_mmap));
    }
    return results;
}

// Read a sequence of HDUs from one open handle and return only the last tensor.
// This avoids returning a large Python list when benchmarking random extension reads.
torch::Tensor read_hdus_sequence_last(const std::string& path, const std::vector<int>& hdus, bool use_mmap) {
    FITSFile file(path.c_str(), 0);
    torch::Tensor out;
    for (int hdu_num : hdus) {
        out = file.read_image(hdu_num, use_mmap);
    }
    return out;
}

torch::Tensor read_full_unmapped(const std::string& path, int hdu_num) {
    fitsfile* fptr = nullptr;
    int status = 0;
    try {
        fits_open_file(&fptr, path.c_str(), READONLY, &status);
        if (status != 0) {
            throw std::runtime_error("Could not open FITS file: " + path);
        }

        int start_hdu = 1;
        fits_get_hdu_num(fptr, &start_hdu);

        int target_hdu = hdu_num + start_hdu;
        if (!(hdu_num == 0 && start_hdu == 1)) {
            fits_movabs_hdu(fptr, target_hdu, nullptr, &status);
            if (status != 0) throw std::runtime_error("Could not move to HDU");
        }

        int bitpix = 0;
        int naxis = 0;
        std::array<LONGLONG, 9> naxes_ll{};
        bool scaled = false;
        bool compressed = false;
        fits_get_img_paramll(fptr, 9, &bitpix, &naxis, naxes_ll.data(), &status);
        if (status != 0) {
            throw std::runtime_error("Could not read image parameters");
        }

        if (bitpix != FLOAT_IMG && bitpix != DOUBLE_IMG) {
            int key_status = 0;
            double bscale = 1.0;
            double bzero = 0.0;

            key_status = 0;
            fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, nullptr, &key_status);
            if (key_status == 0) {
                if (bscale != 1.0) {
                    scaled = true;
                }
            } else if (key_status != KEY_NO_EXIST) {
                scaled = true;
            }

            key_status = 0;
            fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, nullptr, &key_status);
            if (key_status == 0) {
                if (bzero != 0.0) {
                    scaled = true;
                }
            } else if (key_status != KEY_NO_EXIST) {
                scaled = true;
            }
        }

        int compressed_status = 0;
        int is_compressed = fits_is_compressed_image(fptr, &compressed_status);
        compressed = (compressed_status == 0 && is_compressed);

        torch::ScalarType dtype;
        int datatype;
        if (scaled) {
            dtype = torch::kFloat32;
            datatype = TFLOAT;
        } else {
            switch (bitpix) {
                case BYTE_IMG:   dtype = torch::kUInt8; datatype = TBYTE; break;
                case SHORT_IMG:  dtype = torch::kInt16; datatype = TSHORT; break;
                case LONG_IMG:   dtype = torch::kInt32; datatype = TINT; break;
                case LONGLONG_IMG: dtype = torch::kInt64; datatype = TLONGLONG; break;
                case FLOAT_IMG:  dtype = torch::kFloat32; datatype = TFLOAT; break;
                case DOUBLE_IMG: dtype = torch::kFloat64; datatype = TDOUBLE; break;
                default: throw std::runtime_error("Unsupported BITPIX");
            }
        }

        int64_t torch_shape[9];
        for (int i = 0; i < naxis; ++i) {
            torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);
        }

        auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));
        LONGLONG nelements = 0;
        if (naxis > 0) {
            nelements = 1;
            for (int i = 0; i < naxis; ++i) nelements *= naxes_ll[i];
        }

        int anynul = 0;
        float fnullval = NAN;
        double dnullval = NAN;
        void* nullval_ptr = nullptr;
        if ((datatype == TFLOAT || datatype == TDOUBLE) && compressed) {
            if (has_compressed_nulls(fptr)) {
                if (datatype == TFLOAT) {
                    nullval_ptr = &fnullval;
                } else {
                    nullval_ptr = &dnullval;
                }
            }
        }

        static LONGLONG firstpixels[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        fits_read_pixll(
            fptr,
            datatype,
            firstpixels,
            nelements,
            nullval_ptr,
            tensor.data_ptr(),
            &anynul,
            &status
        );
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error reading image data: status=" + std::to_string(status) + " msg=" + std::string(err_text));
        }

        fits_close_file(fptr, &status);
        fptr = nullptr;
        return tensor;
    } catch (...) {
        if (fptr) {
            int close_status = 0;
            fits_close_file(fptr, &close_status);
        }
        throw;
    }
}

torch::Tensor read_full_unmapped_raw(const std::string& path, int hdu_num) {
    fitsfile* fptr = nullptr;
    int status = 0;
    try {
        fits_open_file(&fptr, path.c_str(), READONLY, &status);
        if (status != 0) {
            throw std::runtime_error("Could not open FITS file: " + path);
        }

        int start_hdu = 1;
        fits_get_hdu_num(fptr, &start_hdu);

        int target_hdu = hdu_num + start_hdu;
        if (!(hdu_num == 0 && start_hdu == 1)) {
            fits_movabs_hdu(fptr, target_hdu, nullptr, &status);
            if (status != 0) throw std::runtime_error("Could not move to HDU");
        }

        int bitpix = 0;
        int naxis = 0;
        std::array<LONGLONG, 9> naxes_ll{};
        fits_get_img_paramll(fptr, 9, &bitpix, &naxis, naxes_ll.data(), &status);
        if (status != 0) {
            throw std::runtime_error("Could not read image parameters");
        }

        torch::ScalarType dtype;
        int datatype;
        switch (bitpix) {
            case BYTE_IMG:   dtype = torch::kUInt8; datatype = TBYTE; break;
            case SHORT_IMG:  dtype = torch::kInt16; datatype = TSHORT; break;
            case LONG_IMG:   dtype = torch::kInt32; datatype = TINT; break;
            case LONGLONG_IMG: dtype = torch::kInt64; datatype = TLONGLONG; break;
            case FLOAT_IMG:  dtype = torch::kFloat32; datatype = TFLOAT; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; datatype = TDOUBLE; break;
            default: throw std::runtime_error("Unsupported BITPIX");
        }

        int64_t torch_shape[9];
        for (int i = 0; i < naxis; ++i) {
            torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);
        }

        auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));
        LONGLONG nelements = 0;
        if (naxis > 0) {
            nelements = 1;
            for (int i = 0; i < naxis; ++i) nelements *= naxes_ll[i];
        }

        // Disable CFITSIO scaling for raw reads and restore after.
        FITSFile::BScaleGuard guard;
        guard.fptr = fptr;
        {
            int key_status = 0;
            double bscale = 1.0;
            double bzero = 0.0;

            key_status = 0;
            fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, nullptr, &key_status);
            if (key_status != 0 && key_status != KEY_NO_EXIST) {
                bscale = 1.0;
            }

            key_status = 0;
            fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, nullptr, &key_status);
            if (key_status != 0 && key_status != KEY_NO_EXIST) {
                bzero = 0.0;
            }

            guard.bscale = bscale;
            guard.bzero = bzero;
        }

        status = 0;
        fits_set_bscale(fptr, 1.0, 0.0, &status);
        if (status == 0) {
            guard.active = true;
        } else {
            status = 0;
        }

        int anynul = 0;
        float fnullval = NAN;
        double dnullval = NAN;
        void* nullval_ptr = nullptr;
        if (datatype == TFLOAT || datatype == TDOUBLE) {
            int compressed_status = 0;
            int is_compressed = fits_is_compressed_image(fptr, &compressed_status);
            if (compressed_status == 0 && is_compressed && has_compressed_nulls(fptr)) {
                if (datatype == TFLOAT) {
                    nullval_ptr = &fnullval;
                } else {
                    nullval_ptr = &dnullval;
                }
            }
        }

        static LONGLONG firstpixels[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        fits_read_pixll(
            fptr,
            datatype,
            firstpixels,
            nelements,
            nullval_ptr,
            tensor.data_ptr(),
            &anynul,
            &status
        );
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error reading image data: status=" + std::to_string(status) + " msg=" + std::string(err_text));
        }

        fits_close_file(fptr, &status);
        fptr = nullptr;
        return tensor;
    } catch (...) {
        if (fptr) {
            int close_status = 0;
            fits_close_file(fptr, &close_status);
        }
        throw;
    }
}

torch::Tensor read_full_nocache(const std::string& path, int hdu_num, bool use_mmap) {
    // True "no-cache" read path: open -> read -> close every call.
    // Unlike the cached paths, this avoids global cache locks and refcounting.
    fitsfile* fptr = nullptr;
    int status = 0;
    std::shared_ptr<SharedReadMeta> shared_meta = get_shared_meta_for_path(path);
    fits_open_file(&fptr, path.c_str(), READONLY, &status);
    if (status != 0 || !fptr) {
        throw std::runtime_error("Could not open FITS file: " + path);
    }

    auto close_guard = [&]() {
        if (fptr) {
            int close_status = 0;
            fits_close_file(fptr, &close_status);
            fptr = nullptr;
        }
    };

    try {
        status = 0;
        fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
        if (status != 0) {
            close_guard();
            throw std::runtime_error("Could not move to HDU");
        }

        int bitpix = 0;
        int naxis = 0;
        std::array<LONGLONG, 9> naxes_ll{};
        naxes_ll.fill(0);
        bool info_cached = false;
        if (shared_meta) {
            std::lock_guard<std::mutex> lock(shared_meta->mutex);
            auto it = shared_meta->image_info_cache.find(hdu_num);
            if (it != shared_meta->image_info_cache.end()) {
                bitpix = std::get<0>(it->second);
                naxis = std::get<1>(it->second);
                naxes_ll = std::get<2>(it->second);
                info_cached = true;
            }
        }
        if (!info_cached) {
            status = 0;
            fits_get_img_paramll(fptr, 9, &bitpix, &naxis, naxes_ll.data(), &status);
            if (status != 0) {
                close_guard();
                throw std::runtime_error("Could not read image parameters");
            }
            if (shared_meta) {
                std::lock_guard<std::mutex> lock(shared_meta->mutex);
                shared_meta->image_info_cache[hdu_num] = std::make_tuple(
                    bitpix, naxis, naxes_ll
                );
            }
        }

        // Empty images: return empty tensor of the appropriate dtype.
        if (naxis == 0) {
            torch::ScalarType dtype;
            switch (bitpix) {
                case BYTE_IMG:     dtype = torch::kUInt8; break;
                case SHORT_IMG:    dtype = torch::kInt16; break;
                case LONG_IMG:     dtype = torch::kInt32; break;
                case LONGLONG_IMG: dtype = torch::kInt64; break;
                case FLOAT_IMG:    dtype = torch::kFloat32; break;
                case DOUBLE_IMG:   dtype = torch::kFloat64; break;
                default:           dtype = torch::kUInt8; break;
            }
            close_guard();
            return torch::empty({0}, torch::TensorOptions().dtype(dtype));
        }

        // Compression check (used for null-handling and to avoid raw pread path).
        bool compressed = false;
        bool compressed_cached = false;
        if (shared_meta) {
            std::lock_guard<std::mutex> lock(shared_meta->mutex);
            auto it = shared_meta->compressed_cache.find(hdu_num);
            if (it != shared_meta->compressed_cache.end()) {
                compressed = it->second;
                compressed_cached = true;
            }
        }
        if (!compressed_cached) {
            status = 0;
            const int is_comp = fits_is_compressed_image(fptr, &status);
            compressed = (status == 0) && (is_comp != 0);
            if (status != 0) {
                status = 0;
            }
            if (shared_meta) {
                std::lock_guard<std::mutex> lock(shared_meta->mutex);
                shared_meta->compressed_cache[hdu_num] = compressed;
            }
        }

        // Scale detection: only relevant for integer images. Float images typically
        // don't use BSCALE/BZERO.
        bool scaled = false;
        bool scale_trusted = true;
        double bscale = 1.0;
        double bzero = 0.0;
        if (bitpix != FLOAT_IMG && bitpix != DOUBLE_IMG) {
            bool scale_cached = false;
            if (shared_meta) {
                std::lock_guard<std::mutex> lock(shared_meta->mutex);
                auto it = shared_meta->scale_cache.find(hdu_num);
                if (it != shared_meta->scale_cache.end()) {
                    scaled = std::get<0>(it->second);
                    scale_trusted = std::get<1>(it->second);
                    bscale = std::get<2>(it->second);
                    bzero = std::get<3>(it->second);
                    scale_cached = true;
                }
            }
            if (!scale_cached) {
                int s1 = 0;
                fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, nullptr, &s1);
                if (s1 == 0 && bscale != 1.0) {
                    scaled = true;
                } else if (s1 != 0 && s1 != KEY_NO_EXIST) {
                    scaled = true;
                    scale_trusted = false;
                }
                int s2 = 0;
                fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, nullptr, &s2);
                if (s2 == 0 && bzero != 0.0) {
                    scaled = true;
                } else if (s2 != 0 && s2 != KEY_NO_EXIST) {
                    scaled = true;
                    scale_trusted = false;
                }
                if (shared_meta) {
                    std::lock_guard<std::mutex> lock(shared_meta->mutex);
                    shared_meta->scale_cache[hdu_num] = std::make_tuple(
                        scaled, scale_trusted, bscale, bzero
                    );
                }
            }
        }

        LONGLONG nelements = 1;
        for (int i = 0; i < naxis; ++i) {
            nelements *= naxes_ll[i];
        }

        int64_t torch_shape[9];
        for (int i = 0; i < naxis; ++i) {
            torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);  // reverse for C-contig
        }

        torch::ScalarType dtype;
        int datatype;
        if (scaled) {
            // Special-case signed bytes encoded as uint8 with BZERO=-128.
            if (bitpix == BYTE_IMG && bscale == 1.0 && bzero == -128.0) {
                dtype = at::kChar;  // int8
                datatype = TSBYTE;
            } else {
                dtype = torch::kFloat32;
                datatype = TFLOAT;
            }
        } else {
            switch (bitpix) {
                case BYTE_IMG:      dtype = torch::kUInt8;  datatype = TBYTE; break;
                case SHORT_IMG:     dtype = torch::kInt16;  datatype = TSHORT; break;
                case LONG_IMG:      dtype = torch::kInt32;  datatype = TINT; break;
                case LONGLONG_IMG:  dtype = torch::kInt64;  datatype = TLONGLONG; break;
                case FLOAT_IMG:     dtype = torch::kFloat32; datatype = TFLOAT; break;
                case DOUBLE_IMG:    dtype = torch::kFloat64; datatype = TDOUBLE; break;
                default:
                    close_guard();
                    throw std::runtime_error("Unsupported BITPIX");
            }
        }

        auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis),
                                   torch::TensorOptions().dtype(dtype));

        // Raw pread path for uncompressed BYTE images. Also handle common signed-byte
        // scaling (BITPIX=8 with BZERO=-128) by xor'ing the sign bit in-place.
        const bool signed_byte_scaled = scaled && bitpix == BYTE_IMG && bscale == 1.0 && bzero == -128.0;
        if (use_mmap && !compressed && bitpix == BYTE_IMG && (!scaled || signed_byte_scaled) &&
            path.find('[') == std::string::npos) {
            LONGLONG headstart = 0, data_offset = 0, dataend = 0;
            status = 0;
            fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
            if (status == 0 && data_offset > 0) {
                const size_t nbytes = static_cast<size_t>(nelements);  // 1 byte/elem
                if (nbytes > 0) {
                    const int fd = get_shared_raw_fd(shared_meta, path);
                    if (fd != -1 && read_region_via_fd(fd, static_cast<off_t>(data_offset), tensor.data_ptr(), nbytes)) {
                        if (signed_byte_scaled) {
                            _xor_sign_bit_u8(static_cast<uint8_t*>(tensor.data_ptr()), nbytes);
                        }
                        close_guard();
                        return tensor;
                    }
                }
            } else {
                status = 0;
            }
        }

        // Raw pread path for uncompressed multi-byte native dtypes:
        // read raw bytes directly and byteswap in-place (FITS is big-endian).
        if (use_mmap && !compressed && !scaled && path.find('[') == std::string::npos) {
            size_t elem_size = 0;
            switch (bitpix) {
                case SHORT_IMG: elem_size = sizeof(uint16_t); break;
                case LONG_IMG: elem_size = sizeof(uint32_t); break;
                case LONGLONG_IMG: elem_size = sizeof(uint64_t); break;
                case FLOAT_IMG: elem_size = sizeof(uint32_t); break;
                case DOUBLE_IMG: elem_size = sizeof(uint64_t); break;
                default: elem_size = 0; break;
            }
            if (elem_size > 0) {
                LONGLONG headstart = 0, data_offset = 0, dataend = 0;
                status = 0;
                fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
                if (status == 0 && data_offset > 0) {
                    const size_t nbytes = static_cast<size_t>(nelements) * elem_size;
                    if (nbytes > 0) {
                        const int fd = get_shared_raw_fd(shared_meta, path);
                        if (fd != -1 &&
                            read_region_via_fd(fd, static_cast<off_t>(data_offset), tensor.data_ptr(), nbytes)) {
                            if (_host_is_little_endian()) {
                                if (elem_size == sizeof(uint16_t)) {
                                    auto* p = static_cast<uint16_t*>(tensor.data_ptr());
                                    at::parallel_for(0, static_cast<int64_t>(nelements), 1 << 16, [&](int64_t begin, int64_t end) {
                                        for (int64_t i = begin; i < end; ++i) {
                                            p[i] = _bswap16(p[i]);
                                        }
                                    });
                                } else if (elem_size == sizeof(uint32_t)) {
                                    auto* p = static_cast<uint32_t*>(tensor.data_ptr());
                                    at::parallel_for(0, static_cast<int64_t>(nelements), 1 << 16, [&](int64_t begin, int64_t end) {
                                        for (int64_t i = begin; i < end; ++i) {
                                            p[i] = _bswap32(p[i]);
                                        }
                                    });
                                } else if (elem_size == sizeof(uint64_t)) {
                                    auto* p = static_cast<uint64_t*>(tensor.data_ptr());
                                    at::parallel_for(0, static_cast<int64_t>(nelements), 1 << 16, [&](int64_t begin, int64_t end) {
                                        for (int64_t i = begin; i < end; ++i) {
                                            p[i] = _bswap64(p[i]);
                                        }
                                    });
                                }
                            }
                            close_guard();
                            return tensor;
                        }
                    }
                } else {
                    status = 0;
                }
            }
        }

        int anynul = 0;
        float fnullval = NAN;
        double dnullval = NAN;
        void* nullval_ptr = nullptr;
        if ((datatype == TFLOAT || datatype == TDOUBLE) && compressed) {
            bool has_nulls = false;
            bool nulls_cached = false;
            if (shared_meta) {
                std::lock_guard<std::mutex> lock(shared_meta->mutex);
                auto it = shared_meta->compressed_nulls_cache.find(hdu_num);
                if (it != shared_meta->compressed_nulls_cache.end()) {
                    has_nulls = it->second;
                    nulls_cached = true;
                }
            }
            if (!nulls_cached) {
                has_nulls = has_compressed_nulls(fptr);
                if (shared_meta) {
                    std::lock_guard<std::mutex> lock(shared_meta->mutex);
                    shared_meta->compressed_nulls_cache[hdu_num] = has_nulls;
                }
            }
            if (has_nulls) {
                nullval_ptr = (datatype == TFLOAT) ? (void*) &fnullval : (void*) &dnullval;
            }
        }

        status = 0;
        fits_read_img(
            fptr,
            datatype,
            1,
            nelements,
            nullval_ptr,
            tensor.data_ptr(),
            &anynul,
            &status
        );
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            close_guard();
            throw std::runtime_error("Error reading image data: status=" + std::to_string(status) +
                                     " msg=" + std::string(err_text));
        }

        close_guard();
        return tensor;
    } catch (...) {
        close_guard();
        throw;
    }
}

void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header, nb::object schema_obj, bool is_ascii) {
    struct ColumnWriteInfo {
        std::string name;
        bool is_vla = false;
        bool is_string = false;
        nb::ndarray<> fixed;
        std::vector<nb::ndarray<>> vla_rows;
        std::vector<std::string> string_values;
        int datatype = 0;
        std::string tform;
        std::string tunit;
        std::string tdim;
        long repeat = 1;
        long width = 0;
        bool has_tnull = false;
        long long tnull = 0;
        bool has_bscale = false;
        bool has_bzero = false;
        double bscale = 1.0;
        double bzero = 0.0;
    };

    struct TFormInfo {
        bool vla = false;
        char code = '\0';
        long repeat = 1;
    };

    auto parse_tform = [](const std::string& tform) -> TFormInfo {
        TFormInfo info;
        std::string s = tform;
        // Trim and uppercase
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
        for (auto& c : s) {
            c = std::toupper(static_cast<unsigned char>(c));
        }

        size_t i = 0;
        long repeat = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            repeat = repeat * 10 + (s[i] - '0');
            i++;
        }
        if (repeat > 0) {
            info.repeat = repeat;
        }
        if (i < s.size() && (s[i] == 'P' || s[i] == 'Q')) {
            info.vla = true;
            i++;
        }
        if (i < s.size()) {
            info.code = s[i];
        }
        return info;
    };

    auto dtype_to_code = [](const nb::dlpack::dtype& dt) -> std::pair<std::string, int> {
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) return {"L", TLOGICAL};
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) return {"B", TBYTE};
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) return {"I", TSHORT};
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) return {"J", TINT};
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) return {"E", TFLOAT};
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) return {"D", TDOUBLE};
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) return {"K", TLONGLONG};
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 64) return {"C", TCOMPLEX};
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 128) return {"M", TDBLCOMPLEX};
        throw std::runtime_error("Unsupported table dtype in write_table_hdu");
    };

    auto ascii_tform = [](const nb::dlpack::dtype& dt, long width_hint) -> std::string {
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) return "L1";
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) return "I3";
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) return "I6";
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) return "I11";
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) return "I20";
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) return "E15.7";
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) return "E25.15";
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex) {
            throw std::runtime_error("ASCII table writing does not support complex columns");
        }
        if (width_hint <= 0) {
            width_hint = 8;
        }
        return std::to_string(width_hint) + "A";
    };

    int status = 0;
    std::vector<ColumnWriteInfo> columns;
    columns.reserve(tensor_dict.size());
    long num_rows = -1;

    nb::dict schema;
    bool has_schema = !schema_obj.is_none();
    if (has_schema) {
        schema = nb::cast<nb::dict>(schema_obj);
    }

    std::vector<std::string> column_names;
    if (has_schema) {
        column_names.reserve(schema.size());
        for (auto item : schema) {
            column_names.push_back(nb::cast<std::string>(item.first));
        }
    } else {
        column_names.reserve(tensor_dict.size());
        for (auto item : tensor_dict) {
            column_names.push_back(nb::cast<std::string>(item.first));
        }
    }

    if (has_schema && tensor_dict.size() != schema.size()) {
        throw std::runtime_error("Schema/data column count mismatch in write_table_hdu");
    }

    for (const auto& col_key : column_names) {
        if (!tensor_dict.contains(col_key.c_str())) {
            throw std::runtime_error("Schema column missing data: " + col_key);
        }
        ColumnWriteInfo col;
        col.name = sanitize_fits_string(col_key);
        nb::handle obj = tensor_dict[col_key.c_str()];

        std::string schema_tform;
        bool has_schema_tform = false;
        if (has_schema) {
            nb::handle meta_obj = schema[col_key.c_str()];
            if (nb::isinstance<nb::dict>(meta_obj)) {
                nb::dict meta = nb::cast<nb::dict>(meta_obj);
                if (meta.contains("format")) {
                    schema_tform = nb::cast<std::string>(meta["format"]);
                    has_schema_tform = true;
                } else if (meta.contains("tform")) {
                    schema_tform = nb::cast<std::string>(meta["tform"]);
                    has_schema_tform = true;
                }
                if (meta.contains("unit")) {
                    col.tunit = nb::cast<std::string>(meta["unit"]);
                } else if (meta.contains("tunit")) {
                    col.tunit = nb::cast<std::string>(meta["tunit"]);
                }
                if (meta.contains("null")) {
                    col.has_tnull = true;
                    col.tnull = nb::cast<long long>(meta["null"]);
                } else if (meta.contains("tnull")) {
                    col.has_tnull = true;
                    col.tnull = nb::cast<long long>(meta["tnull"]);
                }
                if (meta.contains("bscale")) {
                    col.has_bscale = true;
                    col.bscale = nb::cast<double>(meta["bscale"]);
                }
                if (meta.contains("bzero")) {
                    col.has_bzero = true;
                    col.bzero = nb::cast<double>(meta["bzero"]);
                }
                if (meta.contains("dim")) {
                    col.tdim = nb::cast<std::string>(meta["dim"]);
                } else if (meta.contains("tdim")) {
                    col.tdim = nb::cast<std::string>(meta["tdim"]);
                }
            }
        }

        TFormInfo schema_info;
        if (has_schema_tform) {
            schema_info = parse_tform(schema_tform);
        }

        bool force_vla = has_schema_tform && schema_info.vla;
        bool force_string = has_schema_tform && schema_info.code == 'A';

        bool treat_vla = false;
        bool treat_string = false;
        if (force_vla) {
            treat_vla = true;
        } else if (force_string) {
            treat_string = true;
        }

        if (!treat_vla && !treat_string && (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj))) {
            nb::sequence seq = nb::cast<nb::sequence>(obj);
            for (auto elem : seq) {
                if (elem.is_none()) {
                    continue;
                }
                if (nb::isinstance<nb::str>(elem) || nb::isinstance<nb::bytes>(elem)) {
                    treat_string = true;
                    break;
                }
                if (nb::isinstance<nb::ndarray<>>(elem) ||
                    nb::isinstance<nb::list>(elem) ||
                    nb::isinstance<nb::tuple>(elem)) {
                    treat_vla = true;
                    break;
                }
            }
        }

        if (treat_vla) {
            col.is_vla = true;
            nb::sequence seq = nb::cast<nb::sequence>(obj);
            size_t seq_len = static_cast<size_t>(nb::len(seq));
            col.vla_rows.reserve(seq_len);
            nb::dlpack::dtype dt{};
            bool dtype_set = false;
            for (auto elem : seq) {
                nb::ndarray<> arr = nb::cast<nb::ndarray<>>(elem);
                if (arr.ndim() > 1) {
                    throw std::runtime_error("VLA column rows must be 1D");
                }
                if (!dtype_set && arr.size() > 0) {
                    dt = arr.dtype();
                    dtype_set = true;
                }
                col.vla_rows.push_back(arr);
            }
            if (!dtype_set) {
                throw std::runtime_error("VLA column has no data to infer dtype");
            }
            auto code = dtype_to_code(dt);
            col.datatype = code.second;
            if (has_schema_tform) {
                col.tform = schema_tform;
            } else {
                col.tform = "1P" + code.first;
            }

            long rows = static_cast<long>(col.vla_rows.size());
            if (num_rows < 0) {
                num_rows = rows;
            } else if (num_rows != rows) {
                throw std::runtime_error("VLA column row count mismatch");
            }
            columns.push_back(std::move(col));
            continue;
        }

        if (treat_string || nb::isinstance<nb::str>(obj) || nb::isinstance<nb::bytes>(obj)) {
            col.is_string = true;
            std::vector<std::string> values;
            if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
                nb::sequence seq = nb::cast<nb::sequence>(obj);
                values.reserve(static_cast<size_t>(nb::len(seq)));
                for (auto elem : seq) {
                    if (elem.is_none()) {
                        values.emplace_back("");
                    } else {
                        values.emplace_back(sanitize_fits_string(nb::cast<std::string>(elem)));
                    }
                }
            } else {
                values.emplace_back(sanitize_fits_string(nb::cast<std::string>(obj)));
            }
            col.string_values = std::move(values);
            long rows = static_cast<long>(col.string_values.size());
            if (num_rows < 0) {
                num_rows = rows;
            } else if (num_rows != rows) {
                throw std::runtime_error("String column row count mismatch in write_table_hdu");
            }
            long max_len = 1;
            for (const auto& s : col.string_values) {
                if (static_cast<long>(s.size()) > max_len) {
                    max_len = static_cast<long>(s.size());
                }
            }
            if (has_schema_tform) {
                col.tform = schema_tform;
                TFormInfo info = parse_tform(schema_tform);
                if (info.repeat > 0) {
                    col.width = info.repeat;
                }
            } else if (is_ascii) {
                col.tform = ascii_tform(nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, 8, 1}, max_len);
            } else {
                col.tform = std::to_string(max_len) + "A";
            }
            if (col.width <= 0) {
                col.width = max_len;
            }
            columns.push_back(std::move(col));
            continue;
        }

        nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(obj);
        int ndim = tensor.ndim();
        long rows = 1;
        if (ndim == 0) {
            rows = 1;
        } else {
            rows = static_cast<long>(tensor.shape(0));
        }
        if (num_rows < 0) {
            num_rows = rows;
        } else if (num_rows != rows) {
            throw std::runtime_error("Column row count mismatch in write_table_hdu");
        }

        auto code = dtype_to_code(tensor.dtype());
        col.datatype = code.second;
        col.fixed = tensor;
        if (ndim > 1) {
            col.repeat = static_cast<long>(tensor.shape(1));
        } else {
            col.repeat = 1;
        }
        if (has_schema_tform) {
            col.tform = schema_tform;
        } else if (is_ascii) {
            col.tform = ascii_tform(tensor.dtype(), col.repeat);
        } else {
            col.tform = std::to_string(col.repeat) + code.first;
        }
        columns.push_back(std::move(col));
    }

    if (num_rows < 0) {
        num_rows = 0;
    }

    int num_cols = static_cast<int>(columns.size());
    char** ttype = new char*[num_cols];
    char** tform = new char*[num_cols];
    char** tunit = new char*[num_cols];

    for (int i = 0; i < num_cols; ++i) {
        const auto& col = columns[i];
        ttype[i] = new char[col.name.length() + 1];
        strncpy(ttype[i], col.name.c_str(), col.name.length());
        ttype[i][col.name.length()] = '\0';

        tform[i] = new char[col.tform.length() + 1];
        strncpy(tform[i], col.tform.c_str(), col.tform.length());
        tform[i][col.tform.length()] = '\0';

        const std::string unit = col.tunit;
        tunit[i] = new char[unit.length() + 1];
        strncpy(tunit[i], unit.c_str(), unit.length());
        tunit[i][unit.length()] = '\0';
    }

    fits_create_tbl(fptr, is_ascii ? ASCII_TBL : BINARY_TBL, num_rows, num_cols, ttype, tform, tunit, "Table", &status);

    if (status != 0) {
        for(int j=0; j<num_cols; j++) { delete[] ttype[j]; delete[] tform[j]; delete[] tunit[j]; }
        delete[] ttype; delete[] tform; delete[] tunit;
        throw std::runtime_error("Failed to create table");
    }

    for (int i = 0; i < num_cols; ++i) {
        const auto& col = columns[i];
        if (col.is_vla) {
            for (long row = 0; row < num_rows; ++row) {
                const auto& arr = col.vla_rows[static_cast<size_t>(row)];
                long nelements = static_cast<long>(arr.size());
                void* data_ptr = arr.size() ? arr.data() : nullptr;
                std::vector<unsigned char> logical;
                if (col.datatype == TLOGICAL && nelements > 0) {
                    nb::dlpack::dtype dt = arr.dtype();
                    logical.resize(static_cast<size_t>(nelements));
                    if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                        const bool* src = static_cast<const bool*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    } else {
                        const uint8_t* src = static_cast<const uint8_t*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    }
                    data_ptr = logical.data();
                }
                fits_write_col(fptr, col.datatype, i + 1, row + 1, 1, nelements, data_ptr, &status);
            }
        } else if (col.is_string) {
            long width_chars = col.width > 0 ? col.width : 1;
            std::vector<std::string> padded;
            padded.reserve(col.string_values.size());
            for (const auto& v : col.string_values) {
                std::string s = v;
                if (static_cast<long>(s.size()) > width_chars) {
                    s = s.substr(0, static_cast<size_t>(width_chars));
                } else if (static_cast<long>(s.size()) < width_chars) {
                    s.append(static_cast<size_t>(width_chars - s.size()), ' ');
                }
                padded.push_back(std::move(s));
            }
            std::vector<const char*> ptrs;
            ptrs.reserve(padded.size());
            for (const auto& s : padded) {
                ptrs.push_back(s.c_str());
            }
            fits_write_col(fptr, TSTRING, i + 1, 1, 1, static_cast<long>(padded.size()),
                           const_cast<char**>(ptrs.data()), &status);
        } else {
            nb::ndarray<> tensor = col.fixed;
            long nelements = num_rows * col.repeat;
            if (col.datatype == TLOGICAL) {
                const bool* src = static_cast<const bool*>(tensor.data());
                std::vector<unsigned char> logical(nelements);
                for (long idx = 0; idx < nelements; ++idx) {
                    logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                }
                fits_write_col(fptr, col.datatype, i + 1, 1, 1, nelements, logical.data(), &status);
            } else {
                fits_write_col(fptr, col.datatype, i + 1, 1, 1, nelements, tensor.data(), &status);
            }
        }
    }
    
    for (auto item : header) {
        std::string key = nb::cast<std::string>(item.first);
        key = sanitize_fits_key(key);
        try {
            if (nb::isinstance<nb::str>(item.second)) {
                std::string val = nb::cast<std::string>(item.second);
                val = sanitize_fits_string(val);
                fits_update_key(fptr, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &status);
            } else if (nb::isinstance<int>(item.second)) {
                int val = nb::cast<int>(item.second);
                fits_update_key(fptr, TINT, key.c_str(), &val, nullptr, &status);
            } else if (nb::isinstance<float>(item.second)) {
                float val = nb::cast<float>(item.second);
                fits_update_key(fptr, TFLOAT, key.c_str(), &val, nullptr, &status);
            } else if (nb::isinstance<double>(item.second)) {
                double val = nb::cast<double>(item.second);
                fits_update_key(fptr, TDOUBLE, key.c_str(), &val, nullptr, &status);
            } else if (nb::isinstance<bool>(item.second)) {
                int val = nb::cast<bool>(item.second) ? 1 : 0;
                fits_update_key(fptr, TLOGICAL, key.c_str(), &val, nullptr, &status);
            }
        } catch (...) {}
    }
    
    for (int i = 0; i < num_cols; ++i) {
        const auto& col = columns[i];
        if (col.has_tnull) {
            long long tnull = col.tnull;
            fits_update_key(fptr, TLONGLONG, ("TNULL" + std::to_string(i + 1)).c_str(), &tnull, nullptr, &status);
        }
        if (col.has_bscale) {
            double bscale = col.bscale;
            fits_update_key(fptr, TDOUBLE, ("TSCAL" + std::to_string(i + 1)).c_str(), &bscale, nullptr, &status);
        }
        if (col.has_bzero) {
            double bzero = col.bzero;
            fits_update_key(fptr, TDOUBLE, ("TZERO" + std::to_string(i + 1)).c_str(), &bzero, nullptr, &status);
        }
        if (!col.tdim.empty()) {
            std::string tdim = col.tdim;
            fits_update_key(fptr, TSTRING, ("TDIM" + std::to_string(i + 1)).c_str(), (void*)tdim.c_str(), nullptr, &status);
        }
    }

    for(int j=0; j<num_cols; j++) { delete[] ttype[j]; delete[] tform[j]; delete[] tunit[j]; }
    delete[] ttype; delete[] tform; delete[] tunit;

    if (status != 0) {
        throw std::runtime_error("Failed to write table data");
    }
}

void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header) {
    write_table_hdu(fptr, tensor_dict, header, nb::none(), false);
}

}
