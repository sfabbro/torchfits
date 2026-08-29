#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <thread>
#include <array>
#include <cmath>
#include <chrono>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <list>
#include <iterator>
#include <atomic>
#include <limits>
#include <cerrno>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>
#include <cstring>
#include <ATen/Parallel.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include "torchfits_torch.h"
#include "torch_compat.h"
#include "security.h"
#include "internal_utils.h"
#include "hardware.h"
#include "cache.h"
#undef READONLY
#include <fitsio.h>
#include <fitsio2.h>

#include "fits_detail.h"
#include "fits_file.h"
#include "fits_rw.h"

namespace nb = nanobind;

namespace torchfits {

namespace d = detail;  // alias to resolve ambiguity with torch::detail

void invalidate_shared_meta(const std::string& filename) {
    d::invalidate_shared_meta_for_path(filename);
}

void clear_shared_read_meta_cache() {
    d::clear_shared_meta_cache();
}

// ---------------------------------------------------------------------------
// read_full_cached — fast path using cached CFITSIO handle + SharedReadMeta
// ---------------------------------------------------------------------------
torch::Tensor read_full_cached(const std::string& path, int hdu_num, bool use_mmap) {
    // Every other entry point enforces the filename security policy; keep this
    // cached fast path consistent (blocks '|' and 'sh://' command execution).
    check_fits_filename_security(path);
    if (has_cfitsio_extended_filename_syntax(path)) {
        FITSFile file(path.c_str(), 0);
        return file.read_tensor(hdu_num, use_mmap);
    }

    // Private per-call handle (CFITSIO §4 Option A): never share one fitsfile*
    // across threads. SharedReadMeta + shared raw fd remain the shared caches.
    fitsfile* fptr = nullptr;
    int open_status = d::open_fits_readonly(&fptr, path);
    if (open_status != 0 || !fptr) {
        throw std::runtime_error("Could not open FITS file: " + path);
    }
    FitsHandleGuard guard;
    guard.fptr = fptr;

    int status = 0;
    const int target_hdu = hdu_num + 1;
    fits_movabs_hdu(fptr, target_hdu, nullptr, &status);
    if (status != 0) {
        throw std::runtime_error("Could not move to HDU");
    }

    auto meta = d::get_shared_meta_for_path(path);

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
        std::list<LocalKey>::iterator lru_it;
        bool has_info = false;
        int bitpix = 0;
        int naxis = 0;
        std::array<LONGLONG, 9> naxes_ll{};
        bool has_compressed = false;
        bool compressed = false;
        bool has_scale = false;
        bool scaled = false;
        bool trusted = true;
        double bscale = 1.0;
        double bzero = 0.0;
    };
    static thread_local std::unordered_map<LocalKey, LocalHduMeta, LocalKeyHash> tl_cache;
    static thread_local std::list<LocalKey> tl_lru;

    LocalKey key{meta->uid, hdu_num};
    LocalHduMeta* local = nullptr;
    auto cache_it = tl_cache.find(key);
    if (cache_it != tl_cache.end()) {
        tl_lru.splice(tl_lru.end(), tl_lru, cache_it->second.lru_it);
        local = &cache_it->second;
    } else {
        constexpr size_t kTlCacheCapacity = 4096;
        if (tl_cache.size() >= kTlCacheCapacity && !tl_lru.empty()) {
            tl_cache.erase(tl_lru.front());
            tl_lru.pop_front();
        }
        auto inserted = tl_cache.emplace(key, LocalHduMeta{});
        tl_lru.push_back(key);
        inserted.first->second.lru_it = std::prev(tl_lru.end());
        local = &inserted.first->second;
    }

    auto get_image_info = [&]() -> void {
        if (local->has_info) return;
        {
            std::shared_lock<std::shared_mutex> lock(meta->mutex);
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
        d::read_image_params_9d(
            fptr, &local->bitpix, &local->naxis, local->naxes_ll, &status);
        if (status != 0) {
            throw std::runtime_error("Could not read image parameters");
        }
        {
            std::unique_lock<std::shared_mutex> lock(meta->mutex);
            meta->image_info_cache[hdu_num] = std::make_tuple(local->bitpix, local->naxis, local->naxes_ll);
        }
        local->has_info = true;
    };

    auto get_compressed = [&]() -> bool {
        if (local->has_compressed) return local->compressed;
        {
            std::shared_lock<std::shared_mutex> lock(meta->mutex);
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
            std::unique_lock<std::shared_mutex> lock(meta->mutex);
            meta->compressed_cache[hdu_num] = local->compressed;
        }
        local->has_compressed = true;
        return local->compressed;
    };

    auto get_scale = [&]() -> void {
        if (local->has_scale) return;
        get_image_info();
        if (local->bitpix == FLOAT_IMG || local->bitpix == DOUBLE_IMG) {
            local->scaled = false;
            local->trusted = true;
            local->bscale = 1.0;
            local->bzero = 0.0;
            local->has_scale = true;
            return;
        }
        {
            std::shared_lock<std::shared_mutex> lock(meta->mutex);
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
        const auto detected = d::detect_scale_info_fast(fptr, local->bitpix);
        local->scaled = detected.scaled;
        local->trusted = detected.trusted;
        local->bscale = detected.bscale;
        local->bzero = detected.bzero;

        {
            std::unique_lock<std::shared_mutex> lock(meta->mutex);
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

    // Build resolved metadata and delegate to canonical read path
    d::ResolvedFITSMeta resolved;
    resolved.bitpix = bitpix;
    resolved.naxis = naxis;
    resolved.naxes_ll = naxes_ll;
    resolved.scaled = scaled;
    resolved.bscale = bscale;
    resolved.bzero = bzero;
    resolved.compressed = compressed;

    // Hold the refcounted fd for the duration of the canonical read so an
    // invalidation cannot close it underneath pread/mmap.
    auto fd_holder = d::get_shared_raw_fd(meta, path);
    const int fd = fd_holder ? fd_holder->fd : -1;
    return d::read_tensor_canonical(fptr, path, resolved, use_mmap, fd, /*use_chunking=*/true);
}

// ---------------------------------------------------------------------------
// resolve_hdu_name_cached
// ---------------------------------------------------------------------------
int resolve_hdu_name_cached(const std::string& path, const std::string& hdu_name) {
    if (hdu_name.empty()) {
        throw std::runtime_error("HDU name cannot be empty");
    }
    // Keep the filename security policy consistent with every other entry point.
    check_fits_filename_security(path);
    auto normalize_hdu_name = [](const std::string& s) -> std::string {
        size_t i = 0;
        size_t j = s.size();
        while (i < j && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1]))) --j;
        std::string out = s.substr(i, j - i);
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return out;
    };
    const std::string hdu_name_key = normalize_hdu_name(hdu_name);

    if (has_cfitsio_extended_filename_syntax(path)) {
        FITSFile file(path.c_str(), 0);
        fitsfile* fptr = file.get_fptr();
        int status = 0;
        fits_movnam_hdu(
            fptr,
            ANY_HDU,
            const_cast<char*>(hdu_name.c_str()),
            0,
            &status
        );
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error(
                "Could not resolve HDU name '" + hdu_name + "': " + std::string(err_text)
            );
        }
        int current_hdu = 0;
        fits_get_hdu_num(fptr, &current_hdu);
        return std::max(0, current_hdu - 1);
    }

    auto meta = d::get_shared_meta_for_path(path);
    if (meta) {
        std::shared_lock<std::shared_mutex> lock(meta->mutex);
        auto it = meta->hdu_name_cache.find(hdu_name_key);
        if (it != meta->hdu_name_cache.end()) {
            return it->second;
        }
    }

    fitsfile* fptr = nullptr;
    int open_status = d::open_fits_readonly(&fptr, path);
    if (open_status != 0 || !fptr) {
        throw std::runtime_error("Could not open FITS file: " + path);
    }
    FitsHandleGuard guard;
    guard.fptr = fptr;

    int status = 0;
    fits_movnam_hdu(
        fptr,
        ANY_HDU,
        const_cast<char*>(hdu_name.c_str()),
        0,
        &status
    );
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error(
            "Could not resolve HDU name '" + hdu_name + "': " + std::string(err_text)
        );
    }

    int current_hdu = 0;
    fits_get_hdu_num(fptr, &current_hdu);
    const int resolved = std::max(0, current_hdu - 1);
    if (meta) {
        std::unique_lock<std::shared_mutex> lock(meta->mutex);
        meta->hdu_name_cache[hdu_name_key] = resolved;
    }
    return resolved;
}

// ---------------------------------------------------------------------------
// open_and_read_headers — batch header read, returns (FITSFile*, vector<HDUInfo>)
// ---------------------------------------------------------------------------
std::pair<FITSFile*, std::vector<HDUInfo>> open_and_read_headers(const std::string& path, int mode) {
    auto file = std::unique_ptr<FITSFile>(new FITSFile(path.c_str(), mode));
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

    return {file.release(), hdus};
}

// ---------------------------------------------------------------------------
// read_images_batch
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> read_images_batch(const std::vector<std::string>& paths, int hdu_num, bool use_mmap) {
    size_t n = paths.size();
    std::vector<torch::Tensor> results(n);
    std::vector<std::string> errors(n);

    if (n == 0) {
        return results;
    }

    auto t0 = std::chrono::steady_clock::now();
    try {
        FITSFile file(paths[0].c_str(), 0);
        results[0] = file.read_tensor(hdu_num, use_mmap);
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
                results[i] = file.read_tensor(hdu_num, use_mmap);
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        }
    } else {
        std::vector<std::thread> threads;
        threads.reserve(n - 1);
        for (size_t i = 1; i < n; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    FITSFile file(paths[i].c_str(), 0);
                    results[i] = file.read_tensor(hdu_num, use_mmap);
                } catch (const std::exception& e) {
                    errors[i] = e.what();
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    for (size_t i = 0; i < n; ++i) {
        if (!errors[i].empty()) {
            throw std::runtime_error("Error reading " + paths[i] + ": " + errors[i]);
        }
    }

    return results;
}

// ---------------------------------------------------------------------------
// read_hdus_batch
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> read_hdus_batch(const std::string& path, const std::vector<int>& hdus, bool use_mmap) {
    FITSFile file(path.c_str(), 0);
    std::vector<torch::Tensor> results;
    results.reserve(hdus.size());
    for (int hdu_num : hdus) {
        results.push_back(file.read_tensor(hdu_num, use_mmap));
    }
    return results;
}

// ---------------------------------------------------------------------------
// read_hdus_sequence_last
// ---------------------------------------------------------------------------
torch::Tensor read_hdus_sequence_last(const std::string& path, const std::vector<int>& hdus, bool use_mmap) {
    FITSFile file(path.c_str(), 0);
    torch::Tensor out;
    for (int hdu_num : hdus) {
        out = file.read_tensor(hdu_num, use_mmap);
    }
    return out;
}

// ---------------------------------------------------------------------------
// read_full_unmapped — open/read/close, no mmap
// ---------------------------------------------------------------------------
torch::Tensor read_full_unmapped(const std::string& path, int hdu_num) {
    check_fits_filename_security(path);
    fitsfile* fptr = nullptr;
    int status = 0;
    try {
        status = d::open_fits_readonly(&fptr, path);
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
        d::ScaleDetectionResult scale_info;
        bool compressed = false;
        d::read_image_params_9d(fptr, &bitpix, &naxis, naxes_ll, &status);
        if (status != 0) {
            throw std::runtime_error("Could not read image parameters");
        }
        scale_info = d::detect_scale_info_fast(fptr, bitpix);

        int compressed_status = 0;
        int is_compressed = fits_is_compressed_image(fptr, &compressed_status);
        compressed = (compressed_status == 0 && is_compressed);

        const bool unsigned_short = scale_info.scaled && bitpix == SHORT_IMG &&
                                     scale_info.bscale == 1.0 && d::is_unsigned_short_offset(scale_info.bzero);
        const bool unsigned_long  = scale_info.scaled && bitpix == LONG_IMG  &&
                                     scale_info.bscale == 1.0 && d::is_unsigned_long_offset(scale_info.bzero);

        torch::ScalarType dtype;
        int datatype;
        if (scale_info.scaled) {
            if (bitpix == BYTE_IMG && scale_info.bscale == 1.0 && scale_info.bzero == -128.0) {
                dtype = torch::kInt8; datatype = TSBYTE;
            } else if (unsigned_short) {
                dtype = torch::kUInt16; datatype = TUSHORT;
            } else if (unsigned_long) {
                dtype = torch::kUInt32; datatype = TUINT;
            } else {
                dtype = torch::kFloat32; datatype = TFLOAT;
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

        int64_t torch_shape[9];
        for (int i = 0; i < naxis; ++i) {
            torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);
        }

        auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));
        LONGLONG nelements = 0;
        if (naxis > 0) {
            nelements = d::checked_nelements_product(naxes_ll.data(), naxis);
        }

        int anynul = 0;
        float fnullval = NAN;
        double dnullval = NAN;
        void* nullval_ptr = d::cfitsio_float_nulval_ptr(
            bitpix, compressed, datatype, &fnullval, &dnullval);

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

// ---------------------------------------------------------------------------
// read_full_unmapped_raw
// ---------------------------------------------------------------------------
torch::Tensor read_full_unmapped_raw(const std::string& path, int hdu_num) {
    FITSFile file(path.c_str(), 0);
    return file.read_image_raw(hdu_num, false);
}

// ---------------------------------------------------------------------------
// read_full_nocache — cold open/read/close (no handle pool). Float/CompImage
// stays ultra-thin; integer/scaled paths keep shared meta + raw_fd so mmap/pread
// and one-time scale probes still win the scorecard.
// ---------------------------------------------------------------------------
torch::Tensor read_full_nocache(const std::string& path, int hdu_num, bool use_mmap) {
    fitsfile* fptr = nullptr;
    int status = 0;
    auto shared_meta = d::get_shared_meta_for_path(path);
    check_fits_filename_security(path);
    status = d::open_fits_readonly(&fptr, path);
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
            std::shared_lock<std::shared_mutex> lock(shared_meta->mutex);
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
            d::read_image_params_9d(fptr, &bitpix, &naxis, naxes_ll, &status);
            if (status != 0) {
                close_guard();
                throw std::runtime_error("Could not read image parameters");
            }
            if (shared_meta) {
                std::unique_lock<std::shared_mutex> lock(shared_meta->mutex);
                shared_meta->image_info_cache[hdu_num] = std::make_tuple(
                    bitpix, naxis, naxes_ll);
            }
        }

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

        bool compressed = false;
        bool compressed_cached = false;
        if (shared_meta) {
            std::shared_lock<std::shared_mutex> lock(shared_meta->mutex);
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
            if (status != 0) status = 0;
            if (shared_meta) {
                std::unique_lock<std::shared_mutex> lock(shared_meta->mutex);
                shared_meta->compressed_cache[hdu_num] = compressed;
            }
        }

        // Float/double (incl. CompImage): direct CFITSIO→tensor, no mmap/scale probes.
        const bool float_like = (bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG);
        if (float_like) {
            LONGLONG nelements = d::checked_nelements_product(naxes_ll.data(), naxis);
            int64_t torch_shape[9];
            for (int i = 0; i < naxis; ++i)
                torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);
            const auto dtype = (bitpix == FLOAT_IMG) ? torch::kFloat32 : torch::kFloat64;
            const int datatype = (bitpix == FLOAT_IMG) ? TFLOAT : TDOUBLE;
            auto tensor = torch::empty(
                at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));
            int anynul = 0;
            float fnullval = NAN;
            double dnullval = NAN;
            void* nullval_ptr = d::cfitsio_float_nulval_ptr(
                bitpix, compressed, datatype, &fnullval, &dnullval);
            status = 0;
            fits_read_img(
                fptr, datatype, 1, nelements, nullval_ptr, tensor.data_ptr(), &anynul, &status);
            if (status != 0) {
                close_guard();
                char err_text[31];
                fits_get_errstatus(status, err_text);
                throw std::runtime_error(
                    "Error reading image data: status=" + std::to_string(status) +
                    " msg=" + std::string(err_text));
            }
            close_guard();
            return tensor;
        }

        bool scaled = false;
        bool scale_trusted = true;
        double bscale = 1.0;
        double bzero = 0.0;
        bool scale_cached = false;
        if (shared_meta) {
            std::shared_lock<std::shared_mutex> lock(shared_meta->mutex);
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
            const auto detected = d::detect_scale_info_fast(fptr, bitpix);
            scaled = detected.scaled;
            scale_trusted = detected.trusted;
            bscale = detected.bscale;
            bzero = detected.bzero;
            if (shared_meta) {
                std::unique_lock<std::shared_mutex> lock(shared_meta->mutex);
                shared_meta->scale_cache[hdu_num] = std::make_tuple(
                    scaled, scale_trusted, bscale, bzero);
            }
        }

        d::ResolvedFITSMeta resolved;
        resolved.bitpix = bitpix;
        resolved.naxis = naxis;
        resolved.naxes_ll = naxes_ll;
        resolved.scaled = scaled;
        resolved.bscale = bscale;
        resolved.bzero = bzero;
        resolved.compressed = compressed;

        // Hold the refcounted fd for the duration of the canonical read so an
        // invalidation cannot close it underneath pread/mmap.
        std::shared_ptr<d::RawFdHolder> fd_holder;
        if (!resolved.compressed && shared_meta) {
            fd_holder = d::get_shared_raw_fd(shared_meta, path);
        }
        const int fd = fd_holder ? fd_holder->fd : -1;
        auto tensor = d::read_tensor_canonical(
            fptr, path, resolved, resolved.compressed ? false : use_mmap, fd,
            /*use_chunking=*/false);
        close_guard();
        return tensor;
    } catch (...) {
        close_guard();
        throw;
    }
}

// ---------------------------------------------------------------------------
// write_table_hdu
// ---------------------------------------------------------------------------

namespace {

// Shared ndarray helpers live in internal_utils.h (single definition for the
// image and table write paths).
using torchfits::internal::ndarray_is_c_contiguous;
using torchfits::internal::ensure_c_contiguous_ndarray;

}  // namespace

// Schema metadata values may arrive as ints/floats (user-supplied) or as
// numeric strings (reconstructed from FITS header cards, e.g. TNULLn).
// nanobind dropped implicit str->number conversion, so parse explicitly.
long long table_schema_to_ll(const nb::object& obj) {
    if (PyLong_Check(obj.ptr())) {
        return nb::cast<long long>(obj);
    }
    if (PyFloat_Check(obj.ptr())) {
        return static_cast<long long>(nb::cast<double>(obj));
    }
    if (PyUnicode_Check(obj.ptr())) {
        try {
            const std::string s = nb::cast<std::string>(obj);
            size_t pos = 0;
            const long long v = std::stoll(s, &pos);
            while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
                ++pos;
            }
            if (pos == s.size()) {
                return v;
            }
        } catch (const std::exception&) {
        }
    }
    throw std::runtime_error("schema numeric value must be an int or a numeric string");
}

double table_schema_to_double(const nb::object& obj) {
    if (PyLong_Check(obj.ptr()) || PyFloat_Check(obj.ptr())) {
        return nb::cast<double>(obj);
    }
    if (PyUnicode_Check(obj.ptr())) {
        try {
            const std::string s = nb::cast<std::string>(obj);
            size_t pos = 0;
            const double v = std::stod(s, &pos);
            while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
                ++pos;
            }
            if (pos == s.size()) {
                return v;
            }
        } catch (const std::exception&) {
        }
    }
    throw std::runtime_error("schema numeric value must be a number or a numeric string");
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
        // Int8 maps to TBYTE (FITS signed-byte storage, bit-identical with
        // the unsigned interpretation) — same convention as append/update_rows.
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 8) return {"B", TBYTE};
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
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
            // CFITSIO's ASCII_TBL rejects 'L'; ASCII tables have no logical
            // format (only A/I/F/E/D), so fail loudly instead of writing an
            // unreadable TFORM.
            throw std::runtime_error(
                "ASCII table writing does not support logical (bool) columns");
        }
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) return "I3";
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 8) return "I4";
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
        col.name = d::sanitize_fits_string(col_key);
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
                    col.tnull = table_schema_to_ll(meta["null"]);
                } else if (meta.contains("tnull")) {
                    col.has_tnull = true;
                    col.tnull = table_schema_to_ll(meta["tnull"]);
                }
                if (meta.contains("bscale")) {
                    col.has_bscale = true;
                    col.bscale = table_schema_to_double(meta["bscale"]);
                }
                if (meta.contains("bzero")) {
                    col.has_bzero = true;
                    col.bzero = table_schema_to_double(meta["bzero"]);
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
                if (arr.size() > 0) {
                    if (!dtype_set) {
                        dt = arr.dtype();
                        dtype_set = true;
                    } else if (arr.dtype().code != dt.code || arr.dtype().bits != dt.bits) {
                        // The column datatype is inferred from the first
                        // non-empty row; a wider/narrower row buffer would be
                        // reinterpreted byte-wise (or read out of bounds).
                        throw std::runtime_error(
                            "VLA column rows must share a single dtype");
                    }
                }
                col.vla_rows.push_back(arr);
            }
            if (!dtype_set) {
                if (col.vla_rows.empty()) {
                    throw std::runtime_error("VLA column has no data to infer dtype");
                }
                // All rows are empty but typed (e.g. explicit empty arrays or
                // None-normalized rows): any row's dtype names the column type.
                dt = col.vla_rows[0].dtype();
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
                        values.emplace_back(d::sanitize_fits_string(nb::cast<std::string>(elem)));
                    }
                }
            } else {
                values.emplace_back(d::sanitize_fits_string(nb::cast<std::string>(obj)));
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
                // ASCII string columns use "Aw" with the character width
                // (CFITSIO's ASCII_TBL parser requires the code letter first).
                long width = max_len;
                if (width <= 0) {
                    width = 1;
                }
                col.tform = "A" + std::to_string(width);
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
            if (schema_info.code == 'X') {
                col.datatype = TBIT;
                if (schema_info.repeat > 0) {
                    col.repeat = schema_info.repeat;
                }
            }
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
    std::vector<std::string> ttype_store(static_cast<size_t>(num_cols));
    std::vector<std::string> tform_store(static_cast<size_t>(num_cols));
    std::vector<std::string> tunit_store(static_cast<size_t>(num_cols));
    std::vector<char*> ttype(static_cast<size_t>(num_cols));
    std::vector<char*> tform(static_cast<size_t>(num_cols));
    std::vector<char*> tunit(static_cast<size_t>(num_cols));

    for (int i = 0; i < num_cols; ++i) {
        const auto& col = columns[i];
        ttype_store[static_cast<size_t>(i)] = col.name;
        tform_store[static_cast<size_t>(i)] = col.tform;
        tunit_store[static_cast<size_t>(i)] = col.tunit;
        ttype[static_cast<size_t>(i)] = ttype_store[static_cast<size_t>(i)].data();
        tform[static_cast<size_t>(i)] = tform_store[static_cast<size_t>(i)].data();
        tunit[static_cast<size_t>(i)] = tunit_store[static_cast<size_t>(i)].data();
    }

    fits_create_tbl(fptr, is_ascii ? ASCII_TBL : BINARY_TBL, num_rows, num_cols,
                    ttype.data(), tform.data(), tunit.data(), "Table", &status);

    if (status != 0) {
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
            std::vector<std::string> padded =
                torchfits::internal::pad_fits_strings(col.string_values, width_chars);
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
            if (col.datatype == TLOGICAL || col.datatype == TBIT) {
                nb::dlpack::dtype dt = tensor.dtype();
                std::vector<unsigned char> logical(nelements);
                if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                    const bool* src = static_cast<const bool*>(tensor.data());
                    for (long idx = 0; idx < nelements; ++idx) {
                        logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                    }
                } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
                    const uint8_t* src = static_cast<const uint8_t*>(tensor.data());
                    for (long idx = 0; idx < nelements; ++idx) {
                        logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                    }
                } else {
                    throw std::runtime_error("Bit/logical table writes require bool or uint8 data");
                }
                if (col.datatype == TBIT) {
                    // fits_write_col(TBIT) maps a flat element run onto raw
                    // data-unit bits and ignores per-row byte padding when
                    // repeat % 8 != 0 (empirically verified against astropy
                    // ground truth for 8X/12X/16X), silently shifting every
                    // row after the first. Write one row per call so the
                    // element count stays within the row.
                    for (long r = 0; r < num_rows; ++r) {
                        fits_write_col(fptr, TBIT, i + 1, 1 + r, 1, col.repeat,
                                       logical.data() + static_cast<size_t>(r * col.repeat),
                                       &status);
                    }
                } else {
                    fits_write_col(fptr, col.datatype, i + 1, 1, 1, nelements, logical.data(), &status);
                }
            } else {
                nb::ndarray<> tensor = col.fixed;
                std::vector<uint8_t> contig_buf;
                void* data_ptr = ensure_c_contiguous_ndarray(tensor, nelements, contig_buf);
                fits_write_col(fptr, col.datatype, i + 1, 1, 1, nelements, data_ptr, &status);
            }
        }
    }

    for (auto item : header) {
        std::string key = nb::cast<std::string>(item.first);
        key = d::sanitize_fits_key(key);
        if (nb::isinstance<bool>(item.second)) {
            int val = nb::cast<bool>(item.second) ? 1 : 0;
            fits_update_key(fptr, TLOGICAL, key.c_str(), &val, nullptr, &status);
        } else if (nb::isinstance<nb::str>(item.second)) {
            std::string val = nb::cast<std::string>(item.second);
            val = d::sanitize_fits_string(val);
            if (val.size() > 68) {
                fits_update_key_longstr(fptr, key.c_str(), val.c_str(), nullptr, &status);
            } else {
                fits_update_key(fptr, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &status);
            }
        } else if (PyLong_Check(item.second.ptr())) {
            int overflow = 0;
            long long val = PyLong_AsLongLongAndOverflow(item.second.ptr(), &overflow);
            if (overflow != 0 || PyErr_Occurred()) {
                PyErr_Clear();
                throw std::runtime_error("FITS header integer out of long long range: " + key);
            }
            fits_update_key(fptr, TLONGLONG, key.c_str(), &val, nullptr, &status);
        } else if (nb::isinstance<double>(item.second) || nb::isinstance<float>(item.second)) {
            double val = nb::cast<double>(item.second);
            fits_update_key(fptr, TDOUBLE, key.c_str(), &val, nullptr, &status);
        } else {
            throw std::runtime_error(
                "Unsupported FITS header value type for key '" + key +
                "': expected str, bool, int, or float"
            );
        }
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
            double rounded = std::round(bzero);
            if (std::isfinite(bzero) && rounded == bzero &&
                rounded >= static_cast<double>(std::numeric_limits<long long>::min()) &&
                rounded <= static_cast<double>(std::numeric_limits<long long>::max())) {
                long long bzero_int = static_cast<long long>(rounded);
                fits_update_key(fptr, TLONGLONG, ("TZERO" + std::to_string(i + 1)).c_str(), &bzero_int, nullptr, &status);
            } else {
                fits_update_key(fptr, TDOUBLE, ("TZERO" + std::to_string(i + 1)).c_str(), &bzero, nullptr, &status);
            }
        }
        if (!col.tdim.empty()) {
            std::string tdim = col.tdim;
            fits_update_key(fptr, TSTRING, ("TDIM" + std::to_string(i + 1)).c_str(), (void*)tdim.c_str(), nullptr, &status);
        }
    }

    if (status != 0) {
        throw std::runtime_error("Failed to write table data");
    }
}

void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header) {
    write_table_hdu(fptr, tensor_dict, header, nb::none(), false);
}

void* get_fptr_from_python_object(nanobind::object obj) {
    return reinterpret_cast<void*>(nanobind::cast<FITSFile&>(obj).get_fptr());
}

} // namespace torchfits

// ---------------------------------------------------------------------------
// bind_fits — nanobind module bindings
// ---------------------------------------------------------------------------
void bind_fits(nb::module_& m) {
    using namespace torchfits;
    nb::class_<FITSFile>(m, "FITSFile")
        .def(nb::init<const char*, int>(), nb::arg("filename"), nb::arg("mode") = 0)
        .def("read_tensor", [](FITSFile& self, int hdu_num, bool use_mmap) {
            torch::Tensor tensor;
            {
                nb::gil_scoped_release release;
                tensor = self.read_tensor(hdu_num, use_mmap);
            }
            return tensor_to_python(tensor);
        }, nb::arg("hdu_num"), nb::arg("use_mmap") = true)
        .def("read_header", [](FITSFile& self, int hdu_num) {
            auto header = std::vector<std::tuple<std::string, std::string, std::string>>();
            {
                nb::gil_scoped_release release;
                header = self.get_header(hdu_num);
            }
            return header;
        })
        .def("get_num_hdus", [](FITSFile& self) {
            long long n = 0;
            {
                nb::gil_scoped_release release;
                n = self.get_num_hdus();
            }
            return n;
        })
        .def("get_hdu_type", [](FITSFile& self, int hdu_num) {
            std::string type;
            {
                nb::gil_scoped_release release;
                type = self.get_hdu_type(hdu_num);
            }
            return type;
        })
        .def("close", &FITSFile::close)
        .def("write_image", [](FITSFile& self, nb::ndarray<> tensor, int hdu_num, double bscale, double bzero) {
            return self.write_image(tensor, hdu_num, bscale, bzero);
        }, nb::arg("tensor"), nb::arg("hdu_num") = 0, nb::arg("bscale") = 1.0, nb::arg("bzero") = 0.0)
        .def("write_hdus", &FITSFile::write_hdus)
        .def("get_shape", [](FITSFile& self, int hdu_num) {
            std::vector<long> shape;
            {
                nb::gil_scoped_release release;
                shape = self.get_shape(hdu_num);
            }
            return shape;
        }, nb::arg("hdu_num"))
        .def("get_dtype", [](FITSFile& self, int hdu_num) {
            int bitpix = 0;
            {
                nb::gil_scoped_release release;
                bitpix = self.get_dtype(hdu_num);
            }
            return bitpix;
        }, nb::arg("hdu_num"))
        .def("read_subset", [](FITSFile& self, int hdu_num, long x1, long y1, long x2, long y2) {
            torch::Tensor tensor;
            {
                nb::gil_scoped_release release;
                tensor = self.read_subset(hdu_num, x1, y1, x2, y2);
            }
            return tensor_to_python(tensor);
        });

    nb::class_<SubsetReader>(m, "SubsetReader")
        .def(nb::init<const std::string&, int>(), nb::arg("filename"), nb::arg("hdu_num") = 0)
        .def("read", [](SubsetReader& self, long x1, long y1, long x2, long y2) {
            torch::Tensor tensor;
            {
                nb::gil_scoped_release release;
                tensor = self.read(x1, y1, x2, y2);
            }
            return tensor_to_python(tensor);
        }, nb::arg("x1"), nb::arg("y1"), nb::arg("x2"), nb::arg("y2"))
        .def("close", &SubsetReader::close)
        .def_prop_ro("width", &SubsetReader::width)
        .def_prop_ro("height", &SubsetReader::height)
        .def_prop_ro("hdu", &SubsetReader::hdu);

    m.def("read_full", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        {
            // Open + read without the GIL: network opens and disk IO must
            // not stall other Python threads.
            nb::gil_scoped_release release;
            FITSFile file(filename.c_str(), 0);
            tensor = file.read_tensor(hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_cached", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = read_full_cached(filename, hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("resolve_hdu_name_cached",
          [](const std::string& filename, const std::string& hdu_name) {
              int hdu_num = 0;
              {
                  nb::gil_scoped_release release;
                  hdu_num = resolve_hdu_name_cached(filename, hdu_name);
              }
              return hdu_num;
          },
          nb::arg("filename"),
          nb::arg("hdu_name"));

    m.def("read_full_numpy_cached", [](const std::string& filename, int hdu_num, bool use_mmap) -> nb::object {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = read_full_cached(filename, hdu_num, use_mmap);
        }
        return tensor_to_numpy_object(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_numpy", [](const std::string& filename, int hdu_num, bool use_mmap) -> nb::object {
        // Phase 1 (no GIL): open — possibly a network extended filename —
        // move to the HDU, and gather all header-derived metadata.
        std::unique_ptr<FITSFile> file;
        fitsfile* fptr = nullptr;
        int bitpix = 0;
        FITSFile::ScaleInfo scale_info;
        bool compressed = false;
        std::vector<size_t> shape;
        {
            nb::gil_scoped_release release;
            file = std::make_unique<FITSFile>(filename.c_str(), 0);
            fptr = file->get_fptr();

            int status = 0;
            file->ensure_hdu(hdu_num, &status);
            if (status != 0) {
                throw std::runtime_error("Could not move to HDU");
            }

            bitpix = file->get_dtype(hdu_num);
            scale_info = file->get_scale_info_for_hdu(hdu_num);

            status = 0;
            const int is_comp = fits_is_compressed_image(fptr, &status);
            compressed = (status == 0) && (is_comp != 0);

            std::vector<long> shape_long = file->get_shape(hdu_num);
            shape.reserve(shape_long.size());
            for (long d : shape_long) {
                shape.push_back((size_t) d);
            }
        }

        const bool scaled = scale_info.scaled;

        if (shape.empty()) {
            return alloc_numpy_array<uint8_t>({0}).cast();
        }

        int datatype = 0;
        nb::object out;
        void* dst = nullptr;

        const bool unsigned_short = scaled && bitpix == SHORT_IMG &&
            scale_info.bscale == 1.0 && d::is_unsigned_short_offset(scale_info.bzero);
        const bool unsigned_long = scaled && bitpix == LONG_IMG &&
            scale_info.bscale == 1.0 && d::is_unsigned_long_offset(scale_info.bzero);

        if (scaled) {
            if (bitpix == BYTE_IMG && scale_info.bscale == 1.0 && scale_info.bzero == -128.0) {
                auto arr = alloc_numpy_array<int8_t>(shape);
                dst = (void*) arr.data();
                datatype = TSBYTE;
                out = arr.cast();
            } else if (unsigned_short) {
                auto arr = alloc_numpy_array<uint16_t>(shape);
                dst = (void*) arr.data();
                datatype = TUSHORT;
                out = arr.cast();
            } else if (unsigned_long) {
                auto arr = alloc_numpy_array<uint32_t>(shape);
                dst = (void*) arr.data();
                datatype = TUINT;
                out = arr.cast();
            } else {
                auto arr = alloc_numpy_array<float>(shape);
                dst = (void*) arr.data();
                datatype = TFLOAT;
                out = arr.cast();
            }
        } else {
            switch (bitpix) {
                case BYTE_IMG: {
                    auto arr = alloc_numpy_array<uint8_t>(shape);
                    dst = (void*) arr.data();
                    datatype = TBYTE;
                    out = arr.cast();
                    break;
                }
                case SHORT_IMG: {
                    auto arr = alloc_numpy_array<int16_t>(shape);
                    dst = (void*) arr.data();
                    datatype = TSHORT;
                    out = arr.cast();
                    break;
                }
                case LONG_IMG: {
                    auto arr = alloc_numpy_array<int32_t>(shape);
                    dst = (void*) arr.data();
                    datatype = TINT;
                    out = arr.cast();
                    break;
                }
                case LONGLONG_IMG: {
                    auto arr = alloc_numpy_array<int64_t>(shape);
                    dst = (void*) arr.data();
                    datatype = TLONGLONG;
                    out = arr.cast();
                    break;
                }
                case FLOAT_IMG: {
                    auto arr = alloc_numpy_array<float>(shape);
                    dst = (void*) arr.data();
                    datatype = TFLOAT;
                    out = arr.cast();
                    break;
                }
                case DOUBLE_IMG: {
                    auto arr = alloc_numpy_array<double>(shape);
                    dst = (void*) arr.data();
                    datatype = TDOUBLE;
                    out = arr.cast();
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported BITPIX for numpy read");
            }
        }

        const bool signed_byte_scaled =
            scaled && bitpix == BYTE_IMG && scale_info.bscale == 1.0 && scale_info.bzero == -128.0;
        if (use_mmap && !compressed && bitpix == BYTE_IMG && (!scaled || signed_byte_scaled)) {
            if (!has_cfitsio_extended_filename_syntax(filename)) {
                // Phase 2 (no GIL): raw pread/mmap of the full data segment.
                nb::gil_scoped_release release;
                LONGLONG headstart = 0, data_offset = 0, dataend = 0;
                int status = 0;
                fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
                if (status == 0 && data_offset > 0) {
                    // Checked product: a NAXIS product overflow must not silently
                    // truncate the pread/mmap to a smaller byte count.
                    const LONGLONG nelem_ll = d::checked_nelements_product(shape);
                    const size_t nelem = static_cast<size_t>(nelem_ll);
                    const size_t nbytes = nelem;
                    const int fd = d::open_readonly_fd(filename);
                    if (fd != -1) {
                        struct stat sb {};
                        if (fstat(fd, &sb) == 0 &&
                            (size_t) sb.st_size >= (size_t) data_offset + nbytes) {
                            uint8_t* dst8 = static_cast<uint8_t*>(dst);
                            size_t remaining = nbytes;
                            off_t off = (off_t) data_offset;
                            bool ok = true;
                            while (remaining) {
                                ssize_t got = pread(fd, dst8, remaining, off);
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
                                dst8 += (size_t) got;
                                off += (off_t) got;
                                remaining -= (size_t) got;
                            }
                            if (ok) {
                                ::close(fd);
                                if (signed_byte_scaled) {
                                    d::_xor_sign_bit_u8(static_cast<uint8_t*>(dst), nbytes);
                                }
                                return out;
                            }

                            void* map_ptr = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
                            if (map_ptr != MAP_FAILED) {
                                const uint8_t* src = static_cast<const uint8_t*>(map_ptr) + data_offset;
                                std::memcpy(dst, src, nbytes);
                                munmap(map_ptr, sb.st_size);
                                ::close(fd);
                                if (signed_byte_scaled) {
                                    d::_xor_sign_bit_u8(static_cast<uint8_t*>(dst), nbytes);
                                }
                                return out;
                            }
                        }
                        ::close(fd);
                    }
                } else {
                    status = 0;
                }
            }
        }

        int anynul = 0;
        int status = 0;
        float fnullval = NAN;
        double dnullval = NAN;
        void* nullval_ptr = nullptr;

        {
            // Phase 3 (no GIL): CFITSIO read. nulval=NaN only for compressed
            // tiles or integer→float (BLANK); native IEEE keeps Inf / -0.
            nb::gil_scoped_release release;
            nullval_ptr = d::cfitsio_float_nulval_ptr(
                bitpix, compressed, datatype, &fnullval, &dnullval);

            status = 0;
            std::vector<LONGLONG> dims;
            dims.reserve(shape.size());
            for (size_t d_idx : shape) {
                dims.push_back(static_cast<LONGLONG>(d_idx));
            }
            LONGLONG nelements = d::checked_nelements_product(dims);
            fits_read_img(fptr, datatype, 1, nelements, nullval_ptr, dst, &anynul, &status);
        }
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error reading image data (numpy): status=" + std::to_string(status) +
                                     " msg=" + std::string(err_text));
        }
        return out;
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_raw", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            FITSFile file(filename.c_str(), 0);
            tensor = file.read_image_raw(hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_raw_with_scale", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        FITSFile::ScaleInfo scale_info;
        {
            nb::gil_scoped_release release;
            FITSFile file(filename.c_str(), 0);
            tensor = file.read_image_raw(hdu_num, use_mmap);
            scale_info = file.get_scale_info_for_hdu(hdu_num);
        }
        return nb::make_tuple(
            tensor_to_python(tensor),
            scale_info.scaled,
            scale_info.bscale,
            scale_info.bzero
        );
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_scaled_cpu", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        FITSFile::ScaleInfo scale_info;
        {
            nb::gil_scoped_release release;
            FITSFile file(filename.c_str(), 0);
            tensor = file.read_image_raw(hdu_num, use_mmap);
            scale_info = file.get_scale_info_for_hdu(hdu_num);
        }
        if (scale_info.scaled) {
            tensor = tensor.to(torch::kFloat32);
            if (scale_info.bscale != 1.0) {
                tensor.mul_(scale_info.bscale);
            }
            if (scale_info.bzero != 0.0) {
                tensor.add_(scale_info.bzero);
            }
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_unmapped", [](const std::string& filename, int hdu_num) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = read_full_unmapped(filename, hdu_num);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_full_unmapped_raw", [](const std::string& filename, int hdu_num) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = read_full_unmapped_raw(filename, hdu_num);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_full_nocache", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = read_full_nocache(filename, hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full", [](FITSFile& file, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = file.read_tensor(hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("file"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("write_fits_file", [](const std::string& path, nb::list hdus, bool overwrite) {
        std::string final_path = path;
        if (overwrite) {
            final_path = "!" + path;
        }
        invalidate_cached(path);
        invalidate_shared_meta(path);
        FITSFile file(final_path.c_str(), 1);
        file.write_hdus(hdus, overwrite);
    });

    m.def("write_fits_file_compressed_images",
          [](const std::string& path, nb::list hdus, bool overwrite, const std::string& algorithm) {
              std::string final_path = path;
              if (overwrite) {
                  final_path = "!" + path;
              }

              int comptype = RICE_1;
              if (!algorithm.empty()) {
                  std::string a = algorithm;
                  std::transform(a.begin(), a.end(), a.begin(),
                                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                  if (a == "R" || a == "RICE" || a == "RICE_1") {
                      comptype = RICE_1;
                  } else if (a == "G" || a == "GZIP" || a == "GZIP_1") {
                      comptype = GZIP_1;
                  } else if (a == "GZIP_2") {
                      comptype = GZIP_2;
                  } else if (a == "H" || a == "HCOMPRESS" || a == "HCOMPRESS_1") {
                      comptype = HCOMPRESS_1;
                  } else if (a == "P" || a == "PLIO" || a == "PLIO_1") {
                      comptype = PLIO_1;
                  } else if (a == "B" || a == "BZIP2" || a == "BZIP2_1") {
                      comptype = BZIP2_1;
                  } else if (a == "NONE") {
                      comptype = 0;
                  } else {
                      throw std::runtime_error("Unsupported compression algorithm: " + algorithm);
                  }
              }

              invalidate_cached(path);
              invalidate_shared_meta(path);
              FITSFile file(final_path.c_str(), 1);
              file.write_hdus_compressed_images(hdus, comptype);
          },
          nb::arg("path"), nb::arg("hdus"), nb::arg("overwrite"),
          nb::arg("algorithm") = std::string("RICE_1"));

    m.def("write_hdu_checksums", [](const std::string& path, int hdu_num) {
        fitsfile* fptr = nullptr;
        int status = 0;
        check_fits_filename_security(path);
        // Header mutations must not leave stale SharedReadMeta / cached
        // readers behind (same contract as the table-mutation wrappers).
        invalidate_shared_meta(path);
        evict_cached_reader(path);
        status = open_fits_for_write(&fptr, path);
        if (status != 0 || !fptr) {
            throw std::runtime_error("Could not open FITS file for checksum writing");
        }
        FitsHandleGuard guard;
        guard.fptr = fptr;
        auto move_to_hdu = [](fitsfile* fptr, int hdu_num) {
            int status = 0;
            fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
            if (status != 0) {
                throw std::runtime_error("Could not move to HDU");
            }
        };
        move_to_hdu(fptr, hdu_num);
        ffpcks(fptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to write FITS checksums");
        }
    }, nb::arg("path"), nb::arg("hdu_num") = 0);

    m.def("verify_hdu_checksums", [](const std::string& path, int hdu_num) {
        fitsfile* fptr = nullptr;
        int status = 0;
        check_fits_filename_security(path);
        fits_open_file(&fptr, path.c_str(), 0 /* READONLY */, &status);
        if (status != 0 || !fptr) {
            throw std::runtime_error("Could not open FITS file for checksum verification");
        }
        FitsHandleGuard guard;
        guard.fptr = fptr;
        int datastatus = -1;
        int hdustatus = -1;
        auto move_to_hdu = [](fitsfile* fptr, int hdu_num) {
            int status = 0;
            fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
            if (status != 0) {
                throw std::runtime_error("Could not move to HDU");
            }
        };
        move_to_hdu(fptr, hdu_num);
        ffvcks(fptr, &datastatus, &hdustatus, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to verify FITS checksums");
        }
        return nb::make_tuple(datastatus, hdustatus);
    }, nb::arg("path"), nb::arg("hdu_num") = 0);

    m.def("write_hdu_header_cards", [](const std::string& path, int hdu_num, nb::list cards) {
        fitsfile* fptr = nullptr;
        int status = 0;
        check_fits_filename_security(path);
        // Header mutations must not leave stale SharedReadMeta / cached
        // readers behind (e.g. an EXTNAME rename must invalidate the
        // hdu_name_cache immediately).
        invalidate_shared_meta(path);
        evict_cached_reader(path);
        status = open_fits_for_write(&fptr, path);
        if (status != 0 || !fptr) {
            throw std::runtime_error("Could not open FITS file for header-card writing");
        }
        FitsHandleGuard guard;
        guard.fptr = fptr;
        auto move_to_hdu = [](fitsfile* fptr, int hdu_num) {
            int status = 0;
            fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
            if (status != 0) {
                throw std::runtime_error("Could not move to HDU");
            }
        };
        move_to_hdu(fptr, hdu_num);

        auto skip_structural = [](const std::string& key_upper) {
            return key_upper == "END" ||
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
                   key_upper.rfind("NAXIS", 0) == 0;
        };

        for (nb::handle card_h : cards) {
            nb::object card = nb::borrow<nb::object>(card_h);
            std::string key;
            nb::object value = nb::none();
            std::string comment;

            if (nb::hasattr(card, "key")) {
                key = nb::cast<std::string>(card.attr("key"));
                value = nb::borrow<nb::object>(card.attr("value"));
                if (nb::hasattr(card, "comment")) {
                    comment = nb::cast<std::string>(card.attr("comment"));
                }
            } else {
                nb::tuple tup = nb::cast<nb::tuple>(card);
                if (tup.size() < 2) {
                    continue;
                }
                key = nb::cast<std::string>(tup[0]);
                value = nb::borrow<nb::object>(tup[1]);
                if (tup.size() >= 3) {
                    comment = nb::cast<std::string>(tup[2]);
                }
            }

            key = d::sanitize_fits_key(key);
            std::string key_upper = key;
            std::transform(key_upper.begin(), key_upper.end(), key_upper.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            if (skip_structural(key_upper)) {
                continue;
            }

            int key_status = 0;
            std::string sanitized_comment = d::sanitize_fits_string(comment);
            char* comment_ptr = sanitized_comment.empty() ? nullptr : sanitized_comment.data();

            if (key_upper == "HISTORY") {
                std::string text = value.is_none() ? comment : nb::cast<std::string>(value);
                text = d::sanitize_fits_string(text);
                fits_write_history(fptr, text.c_str(), &key_status);
            } else if (key_upper == "COMMENT") {
                std::string text = value.is_none() ? comment : nb::cast<std::string>(value);
                text = d::sanitize_fits_string(text);
                fits_write_comment(fptr, text.c_str(), &key_status);
            } else if (nb::isinstance<nb::str>(value)) {
                std::string val = d::sanitize_fits_string(nb::cast<std::string>(value));
                if (val.size() > 68) {
                    // Long strings: fits_update_key truncates at 68 chars
                    // silently. fits_update_key_longstr splits the value into
                    // CONTINUE cards (FITS standard), matching astropy/fitsio.
                    fits_update_key_longstr(fptr, key.c_str(), val.c_str(), comment_ptr, &key_status);
                } else {
                    fits_update_key(fptr, TSTRING, key.c_str(), (void*)val.c_str(), comment_ptr, &key_status);
                }
            } else if (nb::isinstance<bool>(value)) {
                int val = nb::cast<bool>(value) ? 1 : 0;
                fits_update_key(fptr, TLOGICAL, key.c_str(), &val, comment_ptr, &key_status);
            } else if (PyLong_Check(value.ptr())) {
                int overflow = 0;
                long long val = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
                if (overflow != 0 || PyErr_Occurred()) {
                    PyErr_Clear();
                    throw std::runtime_error("FITS header integer out of long long range: " + key);
                }
                fits_update_key(fptr, TLONGLONG, key.c_str(), &val, comment_ptr, &key_status);
            } else if (nb::isinstance<float>(value) || nb::isinstance<double>(value)) {
                double val = nb::cast<double>(value);
                fits_update_key(fptr, TDOUBLE, key.c_str(), &val, comment_ptr, &key_status);
            } else {
                throw std::runtime_error(
                    "Unsupported FITS header value type for key '" + key +
                    "': expected str, bool, int, or float"
                );
            }
            if (key_status != 0) {
                status = key_status;
                break;
            }
        }

        if (status != 0) {
            throw std::runtime_error("Failed to write FITS header cards");
        }
    }, nb::arg("path"), nb::arg("hdu_num"), nb::arg("cards"));

    m.def("delete_hdu_header_key", [](const std::string& path, int hdu_num, const std::string& key) {
        fitsfile* fptr = nullptr;
        int status = 0;
        check_fits_filename_security(path);
        invalidate_shared_meta(path);
        evict_cached_reader(path);
        status = open_fits_for_write(&fptr, path);
        if (status != 0 || !fptr) {
            throw std::runtime_error("Could not open FITS file for header-key deletion");
        }
        FitsHandleGuard guard;
        guard.fptr = fptr;
        fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Could not move to HDU for header-key deletion");
        }
        std::string sanitized = d::sanitize_fits_key(key);
        std::string key_upper = sanitized;
        std::transform(key_upper.begin(), key_upper.end(), key_upper.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (key_upper == "END" || key_upper == "SIMPLE" || key_upper == "XTENSION" ||
            key_upper == "BITPIX" || key_upper == "NAXIS" || key_upper == "EXTEND" ||
            key_upper == "PCOUNT" || key_upper == "GCOUNT" || key_upper == "TFIELDS" ||
            key_upper == "THEAP" || key_upper == "DATASUM" || key_upper == "CHECKSUM" ||
            key_upper.rfind("NAXIS", 0) == 0) {
            throw std::runtime_error("Refusing to delete structural FITS keyword: " + sanitized);
        }
        fits_delete_key(fptr, sanitized.c_str(), &status);
        if (status != 0) {
            throw std::runtime_error("Failed to delete FITS header keyword: " + sanitized);
        }
    }, nb::arg("path"), nb::arg("hdu_num"), nb::arg("key"));

    m.def("open_and_read_headers", [](const std::string& path, int mode) {
        nb::gil_scoped_release release;
        auto result = open_and_read_headers(path, mode);
        nb::gil_scoped_acquire acquire;

        nb::object file_obj = nb::cast(result.first, nb::rv_policy::take_ownership);
        nb::object infos_obj = nb::cast(result.second);

        return nb::make_tuple(file_obj, infos_obj);
    });

    m.def("open_fits_file", [](const std::string& path, const std::string& mode) {
        int mode_int = (mode == "w" || mode == "w+") ? 1 : 0;
        return new FITSFile(path.c_str(), mode_int);
    }, nb::rv_policy::take_ownership);

    m.def("read_header", [](FITSFile& file, int hdu_num) {
        std::vector<std::tuple<std::string, std::string, std::string>> header;
        {
            nb::gil_scoped_release release;
            header = file.get_header(hdu_num);
        }
        return header;
    });
    m.def("read_header_string", [](FITSFile& file, int hdu_num) {
        std::string text;
        {
            nb::gil_scoped_release release;
            text = file.read_header_to_string(hdu_num);
        }
        return text;
    });

    m.def("get_num_hdus", [](FITSFile& file) {
        long long n = 0;
        {
            nb::gil_scoped_release release;
            n = file.get_num_hdus();
        }
        return n;
    });

    m.def("get_hdu_type", [](FITSFile& file, int hdu_num) {
        std::string type;
        {
            nb::gil_scoped_release release;
            type = file.get_hdu_type(hdu_num);
        }
        return type;
    });

    m.def("read_tensor_from_handle", [](FITSFile& file, int hdu_num) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = file.read_tensor(hdu_num);
        }
        return tensor_to_python(tensor);
    });

    m.def("read_images_batch", [](const std::vector<std::string>& paths, int hdu_num, bool use_mmap) {
        nb::gil_scoped_release release;
        auto tensors = read_images_batch(paths, hdu_num, use_mmap);
        nb::gil_scoped_acquire acquire;

        nb::list result;
        for (const auto& t : tensors) {
            result.append(tensor_to_python(t));
        }
        return result;
    }, nb::arg("paths"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_hdus_batch", [](const std::string& path, const std::vector<int>& hdus, bool use_mmap) {
        nb::gil_scoped_release release;
        auto tensors = read_hdus_batch(path, hdus, use_mmap);
        nb::gil_scoped_acquire acquire;

        nb::list result;
        for (const auto& t : tensors) {
            result.append(tensor_to_python(t));
        }
        return result;
    }, nb::arg("path"), nb::arg("hdus"), nb::arg("use_mmap") = true);

    m.def("read_hdus_sequence_last", [](const std::string& path, const std::vector<int>& hdus, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = read_hdus_sequence_last(path, hdus, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("path"), nb::arg("hdus"), nb::arg("use_mmap") = true);

    nb::class_<HDUInfo>(m, "HDUInfo")
        .def_prop_rw("index", [](HDUInfo& t) { return t.index; }, [](HDUInfo& t, int v) { t.index = v; })
        .def_prop_rw("type", [](HDUInfo& t) { return t.type; }, [](HDUInfo& t, std::string v) { t.type = v; })
        .def_prop_ro("header", [](HDUInfo& t) {
            nb::dict d;
            for (const auto& kv : t.header) {
                d[std::get<0>(kv).c_str()] = std::get<1>(kv);
            }
            return d;
        });

    m.def("read_header_dict", [](const std::string& filename, int hdu_num) -> nb::list {
        // Let open/parse failures propagate: silently returning an empty list
        // turned a broken read into a seemingly-valid empty header.
        nb::gil_scoped_release release;
        FITSFile file(filename.c_str(), 0);
        auto header = file.get_header(hdu_num);
        nb::gil_scoped_acquire acquire;
        nb::list result;
        for (const auto& item : header) {
            result.append(nb::make_tuple(std::get<0>(item), std::get<1>(item), std::get<2>(item)));
        }
        return result;
    });

    // Skinny metadata: open → move HDU → structural/key query. No full header dump.
    m.def(
        "read_nrows",
        [](const std::string& filename, int hdu_num) -> long long {
            nb::gil_scoped_release release;
            FITSFile file(filename.c_str(), 0);
            int status = 0;
            file.ensure_hdu(hdu_num, &status);
            if (status != 0) {
                throw std::runtime_error("read_nrows: could not move to HDU");
            }
            long nrows = 0;
            fits_get_num_rows(file.get_fptr(), &nrows, &status);
            if (status != 0) {
                char err_text[FLEN_ERRMSG];
                fits_get_errstatus(status, err_text);
                throw std::runtime_error(
                    std::string("read_nrows: ") + err_text +
                    " (HDU must be a table)");
            }
            return static_cast<long long>(nrows);
        },
        nb::arg("filename"),
        nb::arg("hdu_num"));

    m.def(
        "read_keys",
        [](const std::string& filename, int hdu_num,
           const std::vector<std::string>& keys) -> nb::dict {
            struct KeyRaw {
                std::string key;
                std::string value;
            };
            std::vector<KeyRaw> raw;
            raw.reserve(keys.size());
            {
                nb::gil_scoped_release release;
                FITSFile file(filename.c_str(), 0);
                int status = 0;
                file.ensure_hdu(hdu_num, &status);
                if (status != 0) {
                    throw std::runtime_error("read_keys: could not move to HDU");
                }
                fitsfile* fptr = file.get_fptr();
                for (const auto& key : keys) {
                    char value[FLEN_VALUE] = {0};
                    char comment[FLEN_COMMENT] = {0};
                    status = 0;
                    if (fits_read_keyword(
                            fptr, key.c_str(), value, comment, &status) != 0) {
                        if (status == KEY_NO_EXIST) {
                            throw std::runtime_error(
                                "read_keys: keyword not found: " + key);
                        }
                        char err_text[FLEN_ERRMSG];
                        fits_get_errstatus(status, err_text);
                        throw std::runtime_error(
                            std::string("read_keys: ") + err_text + " (" + key +
                            ")");
                    }
                    raw.push_back(
                        KeyRaw{key, d::sanitize_fits_string(std::string(value))});
                }
            }
            nb::dict out;
            for (const auto& item : raw) {
                const std::string& val = item.value;
                if (val.empty()) {
                    out[item.key.c_str()] = nb::none();
                    continue;
                }
                if (val == "T") {
                    out[item.key.c_str()] = true;
                    continue;
                }
                if (val == "F") {
                    out[item.key.c_str()] = false;
                    continue;
                }
                if (val.front() == '\'') {
                    std::string s = val;
                    size_t last_quote = s.rfind('\'');
                    if (last_quote != std::string::npos && last_quote > 0) {
                        s = s.substr(1, last_quote - 1);
                        size_t last_char = s.find_last_not_of(' ');
                        if (last_char != std::string::npos)
                            s = s.substr(0, last_char + 1);
                        else
                            s.clear();
                        size_t pos = 0;
                        while ((pos = s.find("''", pos)) != std::string::npos) {
                            s.replace(pos, 2, "'");
                            pos += 1;
                        }
                    }
                    out[item.key.c_str()] = s;
                    continue;
                }
                try {
                    // Full-consumption check: a partially numeric string
                    // ("1999-01-01") must stay a string, not truncate to its
                    // leading number.
                    size_t pos = 0;
                    if (val.find_first_of(".eE") != std::string::npos) {
                        const double dv = std::stod(val, &pos);
                        while (pos < val.size() &&
                               std::isspace(static_cast<unsigned char>(val[pos]))) ++pos;
                        if (pos == val.size()) {
                            out[item.key.c_str()] = dv;
                            continue;
                        }
                    } else {
                        const long long iv = std::stoll(val, &pos);
                        while (pos < val.size() &&
                               std::isspace(static_cast<unsigned char>(val[pos]))) ++pos;
                        if (pos == val.size()) {
                            out[item.key.c_str()] = iv;
                            continue;
                        }
                    }
                } catch (const std::exception&) {
                }
                out[item.key.c_str()] = val;
            }
            return out;
        },
        nb::arg("filename"),
        nb::arg("hdu_num"),
        nb::arg("keys"));

    m.def(
        "read_shape",
        [](const std::string& filename, int hdu_num) -> nb::tuple {
            int bitpix = 0;
            int naxis = 0;
            std::array<LONGLONG, 9> naxes_ll{};
            naxes_ll.fill(0);
            {
                nb::gil_scoped_release release;
                // Warm SharedReadMeta: skip CFITSIO open when image params
                // were already populated by a prior read / SubsetReader.
                auto meta = d::get_shared_meta_for_path(filename);
                bool hit = false;
                if (meta) {
                    std::shared_lock<std::shared_mutex> lock(meta->mutex);
                    auto it = meta->image_info_cache.find(hdu_num);
                    if (it != meta->image_info_cache.end()) {
                        bitpix = std::get<0>(it->second);
                        naxis = std::get<1>(it->second);
                        naxes_ll = std::get<2>(it->second);
                        hit = true;
                    }
                }
                if (!hit) {
                    FITSFile file(filename.c_str(), 0);
                    int status = 0;
                    file.ensure_hdu(hdu_num, &status);
                    if (status != 0) {
                        throw std::runtime_error("read_shape: could not move to HDU");
                    }
                    // get_image_info populates per-handle + SharedReadMeta caches.
                    const auto& info = file.get_image_info(hdu_num);
                    bitpix = std::get<0>(info);
                    naxis = std::get<1>(info);
                    naxes_ll = std::get<2>(info);
                }
            }
            nb::list shape;
            // Torch / row-major order (reverse of FITS NAXISn).
            for (int i = naxis - 1; i >= 0; --i) {
                shape.append(static_cast<long long>(naxes_ll[i]));
            }
            return nb::make_tuple(bitpix, nb::tuple(shape));
        },
        nb::arg("filename"),
        nb::arg("hdu_num"));

    m.def(
        "read_hdu_type",
        [](const std::string& filename, int hdu_num) -> std::string {
            nb::gil_scoped_release release;
            FITSFile file(filename.c_str(), 0);
            return file.get_hdu_type(hdu_num);
        },
        nb::arg("filename"),
        nb::arg("hdu_num"));

    m.def(
        "read_num_hdus",
        [](const std::string& filename) -> int {
            nb::gil_scoped_release release;
            FITSFile file(filename.c_str(), 0);
            return file.get_num_hdus();
        },
        nb::arg("filename"));

    m.def(
        "read_colnames",
        [](const std::string& filename, int hdu_num) -> std::vector<std::string> {
            nb::gil_scoped_release release;
            FITSFile file(filename.c_str(), 0);
            int status = 0;
            file.ensure_hdu(hdu_num, &status);
            if (status != 0) {
                throw std::runtime_error("read_colnames: could not move to HDU");
            }
            fitsfile* fptr = file.get_fptr();
            int ncols = 0;
            fits_get_num_cols(fptr, &ncols, &status);
            if (status != 0) {
                char err_text[FLEN_ERRMSG];
                fits_get_errstatus(status, err_text);
                throw std::runtime_error(
                    std::string("read_colnames: ") + err_text +
                    " (HDU must be a table)");
            }
            std::vector<std::string> names;
            names.reserve(ncols);
            for (int i = 1; i <= ncols; ++i) {
                char ttype[FLEN_VALUE];
                memset(ttype, 0, FLEN_VALUE);
                char keyname[FLEN_KEYWORD];
                snprintf(keyname, FLEN_KEYWORD, "TTYPE%d", i);
                int col_status = 0;
                fits_read_key(fptr, TSTRING, keyname, ttype, nullptr, &col_status);
                if (col_status != 0) {
                    snprintf(ttype, FLEN_VALUE, "COL%d", i);
                } else {
                    // Trim trailing spaces from TSTRING.
                    size_t len = strnlen(ttype, FLEN_VALUE);
                    while (len > 0 && ttype[len - 1] == ' ') --len;
                    ttype[len] = '\0';
                }
                names.emplace_back(ttype);
            }
            return names;
        },
        nb::arg("filename"),
        nb::arg("hdu_num"));

    m.def(
        "read_table_info",
        [](const std::string& filename, int hdu_num) -> nb::dict {
            long nrows = 0;
            std::vector<std::string> names;
            std::vector<std::string> tforms;
            {
                nb::gil_scoped_release release;
                FITSFile file(filename.c_str(), 0);
                int status = 0;
                file.ensure_hdu(hdu_num, &status);
                if (status != 0) {
                    throw std::runtime_error("read_table_info: could not move to HDU");
                }
                fitsfile* fptr = file.get_fptr();
                fits_get_num_rows(fptr, &nrows, &status);
                int ncols = 0;
                if (status == 0) fits_get_num_cols(fptr, &ncols, &status);
                if (status != 0) {
                    char err_text[FLEN_ERRMSG];
                    fits_get_errstatus(status, err_text);
                    throw std::runtime_error(
                        std::string("read_table_info: ") + err_text +
                        " (HDU must be a table)");
                }
                names.reserve(ncols);
                tforms.reserve(ncols);
                for (int i = 1; i <= ncols; ++i) {
                    char ttype[FLEN_VALUE];
                    char tform[FLEN_VALUE];
                    memset(ttype, 0, FLEN_VALUE);
                    memset(tform, 0, FLEN_VALUE);
                    char keyname[FLEN_KEYWORD];
                    snprintf(keyname, FLEN_KEYWORD, "TTYPE%d", i);
                    int col_status = 0;
                    fits_read_key(fptr, TSTRING, keyname, ttype, nullptr, &col_status);
                    if (col_status != 0) {
                        snprintf(ttype, FLEN_VALUE, "COL%d", i);
                    } else {
                        size_t len = strnlen(ttype, FLEN_VALUE);
                        while (len > 0 && ttype[len - 1] == ' ') --len;
                        ttype[len] = '\0';
                    }
                    col_status = 0;
                    snprintf(keyname, FLEN_KEYWORD, "TFORM%d", i);
                    fits_read_key(fptr, TSTRING, keyname, tform, nullptr, &col_status);
                    if (col_status == 0) {
                        size_t len = strnlen(tform, FLEN_VALUE);
                        while (len > 0 && tform[len - 1] == ' ') --len;
                        tform[len] = '\0';
                    }
                    names.emplace_back(ttype);
                    tforms.emplace_back(tform);
                }
            }
            nb::dict out;
            out["nrows"] = static_cast<long long>(nrows);
            out["colnames"] = names;
            out["tforms"] = tforms;
            return out;
        },
        nb::arg("filename"),
        nb::arg("hdu_num"));

    m.def("configure_cache", &configure_cache, nb::arg("max_files"), nb::arg("max_memory_mb"));
    m.def("clear_file_cache", &clear_file_cache);
    m.def("invalidate_file_cache", &invalidate_file_cache, nb::arg("path"));
    m.def("clear_shared_read_meta_cache", &clear_shared_read_meta_cache);
    m.def("get_cache_size", &get_cache_size);

    m.def("echo_tensor", [](nb::object obj) {
        return obj;
    });
}
