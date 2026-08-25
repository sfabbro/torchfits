#undef READONLY
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <limits>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <fitsio.h>

#include "torchfits_torch.h"
#include "torch_compat.h"
#include "security.h"
#include "hardware.h"
#include "fits_detail.h"
#include "fits_file.h"
#include "fits_rw.h"

namespace torchfits {

// ---------------------------------------------------------------------------
// BScaleGuard — temporarily disable BSCALE/BZERO for raw read
// ---------------------------------------------------------------------------
struct BScaleGuard {
    fitsfile* fptr = nullptr;
    double bscale = 1.0;
    double bzero = 0.0;
    bool active = false;
    BScaleGuard() = default;
    explicit BScaleGuard(fitsfile* f) : fptr(f) {}
    ~BScaleGuard() {
        if (!active || !fptr) return;
        int status = 0;
        fits_set_bscale(fptr, bscale, bzero, &status);
    }
};

// ===========================================================================
// FITSFile implementation
// ===========================================================================
FITSFile::FITSFile(const char* filename, int mode) : filename_(filename), mode_(mode) {
    check_fits_filename_security(filename_);
    int status = 0;
    if (mode == 0) {
        // Private per-instance handle (CFITSIO §4 Option A): no shared CHDU.
        status = detail::open_fits_readonly(&fptr_, filename_);
    } else {
        fits_create_file(&fptr_, filename, &status);
    }
    if (status != 0 || !fptr_) {
        // CFITSIO may leave a partially-initialized handle on failure; close it
        // to avoid leaking the underlying file.
        if (fptr_) {
            int close_status = 0;
            fits_close_file(fptr_, &close_status);
            fptr_ = nullptr;
        }
        throw std::runtime_error("Could not open FITS file: " + filename_);
    }
    if (mode == 0) shared_meta_ = detail::get_shared_meta_for_path(filename_);
    const bool has_extension = has_cfitsio_extended_filename_syntax(filename_);
    if (!has_extension) {
        // Private handle owns its own CHDU — do not seed from shared_meta_.
        start_hdu_ = 1;
        current_hdu_ = -1;
    } else {
        fits_get_hdu_num(fptr_, &start_hdu_);
        current_hdu_ = start_hdu_;
    }
}

FITSFile::~FITSFile() { close(); }

void FITSFile::close() {
    close_raw_fd();
    if (fptr_) {
        int status = 0;
        fits_close_file(fptr_, &status);
        fptr_ = nullptr;
    }
}

void FITSFile::ensure_hdu(int hdu_num, int* status) {
    if (!fptr_) throw std::runtime_error("FITSFile is closed");
    int target_hdu = hdu_num + start_hdu_;
    if (current_hdu_ != target_hdu) {
        fits_movabs_hdu(fptr_, target_hdu, nullptr, status);
        if (*status == 0) {
            current_hdu_ = target_hdu;
            if (shared_meta_) {
                std::unique_lock<std::shared_mutex> lock(shared_meta_->mutex);
                shared_meta_->current_fits_hdu = target_hdu;
            }
        }
    }
}

const FITSFile::ScaleInfo& FITSFile::get_scale_info(int hdu_num, int bitpix) {
    auto it = scale_cache_.find(hdu_num);
    if (it != scale_cache_.end()) return it->second;
    if (shared_meta_) {
        std::shared_lock<std::shared_mutex> lock(shared_meta_->mutex);
        auto sit = shared_meta_->scale_cache.find(hdu_num);
        if (sit != shared_meta_->scale_cache.end()) {
            auto [scaled, trusted, bscale, bzero] = sit->second;
            ScaleInfo shared_info;
            shared_info.scaled = scaled; shared_info.trusted = trusted;
            shared_info.bscale = bscale; shared_info.bzero = bzero;
            auto inserted = scale_cache_.emplace(hdu_num, shared_info);
            return inserted.first->second;
        }
    }
    ScaleInfo info;
    const auto detected = detail::detect_scale_info_fast(fptr_, bitpix);
    info.scaled = detected.scaled; info.trusted = detected.trusted;
    info.bscale = detected.bscale; info.bzero = detected.bzero;
    auto inserted = scale_cache_.emplace(hdu_num, info);
    if (shared_meta_) {
        std::unique_lock<std::shared_mutex> lock(shared_meta_->mutex);
        shared_meta_->scale_cache[hdu_num] = std::make_tuple(
            info.scaled, info.trusted, info.bscale, info.bzero);
    }
    return inserted.first->second;
}

FITSFile::ScaleInfo FITSFile::get_scale_info_for_hdu(int hdu_num) {
    const auto& info = get_image_info(hdu_num);
    return get_scale_info(hdu_num, std::get<0>(info));
}

bool FITSFile::is_compressed_image_cached(int hdu_num) {
    auto it = compressed_cache_.find(hdu_num);
    if (it != compressed_cache_.end()) return it->second;
    if (shared_meta_) {
        std::shared_lock<std::shared_mutex> lock(shared_meta_->mutex);
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
        std::unique_lock<std::shared_mutex> lock(shared_meta_->mutex);
        shared_meta_->compressed_cache[hdu_num] = result;
    }
    return result;
}

const std::tuple<int, int, std::array<LONGLONG, 9>>& FITSFile::get_image_info(int hdu_num) {
    auto it = image_info_cache_.find(hdu_num);
    if (it != image_info_cache_.end()) return it->second;
    if (shared_meta_) {
        std::shared_lock<std::shared_mutex> lock(shared_meta_->mutex);
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
    detail::read_image_params_9d(fptr_, &bitpix, &naxis, naxes_ll, &status);
    if (status != 0) throw std::runtime_error("Could not read image parameters");
    auto inserted = image_info_cache_.emplace(hdu_num, std::make_tuple(bitpix, naxis, naxes_ll));
    if (shared_meta_) {
        std::unique_lock<std::shared_mutex> lock(shared_meta_->mutex);
        shared_meta_->image_info_cache[hdu_num] = inserted.first->second;
    }
    return inserted.first->second;
}

torch::Tensor FITSFile::read_tensor(int hdu_num, bool use_mmap) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    int bitpix = 0, naxis = 0;
    std::array<LONGLONG, 9> naxes_ll{};
    naxes_ll.fill(0);
    // Direct paramll (skip shared_meta locks) — local FITSFile caches die with
    // the one-shot wrapper anyway.
    detail::read_image_params_9d(fptr_, &bitpix, &naxis, naxes_ll, &status);
    if (status != 0) throw std::runtime_error("Could not read image parameters");
    if (naxis == 0) {
        torch::ScalarType dtype;
        switch (bitpix) {
            case BYTE_IMG:   dtype = torch::kUInt8; break;
            case SHORT_IMG:  dtype = torch::kInt16; break;
            case LONG_IMG:   dtype = torch::kInt32; break;
            case LONGLONG_IMG: dtype = torch::kInt64; break;
            case FLOAT_IMG:  dtype = torch::kFloat32; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; break;
            default:         dtype = torch::kUInt8; break;
        }
        return torch::empty({0}, torch::TensorOptions().dtype(dtype));
    }
    detail::ResolvedFITSMeta meta;
    meta.bitpix = bitpix;
    meta.naxis = naxis;
    meta.naxes_ll = naxes_ll;

    // Float/double: always thin CFITSIO→tensor. CompImage cannot mmap; uncompressed
    // float mmap+bswap is rarely worth the probe tax on the CompImage scorecard path.
    const bool float_like = (bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG);
    if (float_like) {
        meta.scaled = false;
        meta.bscale = 1.0;
        meta.bzero = 0.0;
        meta.compressed = false;
        return detail::read_tensor_canonical(
            fptr_, filename_, meta, /*use_mmap=*/false, /*raw_fd=*/-1, /*use_chunking=*/false);
    }

    image_info_cache_[hdu_num] = std::make_tuple(bitpix, naxis, naxes_ll);
    if (shared_meta_) {
        std::unique_lock<std::shared_mutex> lock(shared_meta_->mutex);
        shared_meta_->image_info_cache[hdu_num] = image_info_cache_[hdu_num];
    }

    const auto& scale_info = get_scale_info(hdu_num, bitpix);
    meta.scaled = scale_info.scaled;
    meta.bscale = scale_info.bscale;
    meta.bzero = scale_info.bzero;
    meta.compressed = is_compressed_image_cached(hdu_num);

    // Hold the refcounted fd for the duration of the canonical read so an
    // invalidation cannot close it underneath pread/mmap.
    std::shared_ptr<detail::RawFdHolder> fd_holder;
    if (!meta.compressed && shared_meta_) {
        fd_holder = detail::get_shared_raw_fd(shared_meta_, filename_);
    }
    const int fd = fd_holder ? fd_holder->fd : -1;
    return detail::read_tensor_canonical(fptr_, filename_, meta, use_mmap, fd, /*use_chunking=*/false);
}

torch::Tensor FITSFile::read_image_raw(int hdu_num, bool use_mmap) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    const bool want_mmap = use_mmap;
    int bitpix = 0, naxis = 0;
    std::array<LONGLONG, 9> naxes_ll{};
    {
        const auto& info = get_image_info(hdu_num);
        bitpix = std::get<0>(info);
        naxis = std::get<1>(info);
        naxes_ll = std::get<2>(info);
    }
    if (naxis == 0) {
        torch::ScalarType dtype;
        switch (bitpix) {
            case BYTE_IMG:   dtype = torch::kUInt8; break;
            case SHORT_IMG:  dtype = torch::kInt16; break;
            case LONG_IMG:   dtype = torch::kInt32; break;
            case LONGLONG_IMG: dtype = torch::kInt64; break;
            case FLOAT_IMG:  dtype = torch::kFloat32; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; break;
            default:         dtype = torch::kUInt8; break;
        }
        return torch::empty({0}, torch::TensorOptions().dtype(dtype));
    }
    torch::ScalarType dtype;
    int datatype;
    switch (bitpix) {
        case BYTE_IMG:     dtype = torch::kUInt8;  datatype = TBYTE;      break;
        case SHORT_IMG:    dtype = torch::kInt16;  datatype = TSHORT;     break;
        case LONG_IMG:     dtype = torch::kInt32;  datatype = TINT;       break;
        case LONGLONG_IMG: dtype = torch::kInt64;  datatype = TLONGLONG;  break;
        case FLOAT_IMG:    dtype = torch::kFloat32; datatype = TFLOAT;     break;
        case DOUBLE_IMG:   dtype = torch::kFloat64; datatype = TDOUBLE;    break;
        default: throw std::runtime_error("Unsupported BITPIX");
    }
    LONGLONG nelements = 0;
    if (naxis > 0) {
        nelements = torchfits::detail::checked_nelements_product(naxes_ll.data(), naxis);
    }
    int64_t torch_shape[9];
    for (int i = 0; i < naxis; ++i) torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);
    auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));
    const bool compressed = is_compressed_image_cached(hdu_num);
    if (want_mmap && !compressed && bitpix == BYTE_IMG) {
        if (!has_cfitsio_extended_filename_syntax(filename_)) {
            LONGLONG headstart = 0, data_offset = 0, dataend = 0;
            status = 0;
            fits_get_hduaddrll(fptr_, &headstart, &data_offset, &dataend, &status);
            if (status == 0 && data_offset > 0) {
                const size_t nbytes = static_cast<size_t>(nelements);
                if (nbytes > 0) {
                    const size_t end_off = static_cast<size_t>(data_offset) + nbytes;
                    if (ensure_raw_fd(end_off)) {
                        uint8_t* dst = static_cast<uint8_t*>(tensor.data_ptr());
                        size_t remaining = nbytes;
                        off_t off = static_cast<off_t>(data_offset);
                        bool ok = true;
                        while (remaining > 0) {
                            ssize_t got = ::pread(raw_fd_, dst, remaining, off);
                            if (got < 0) { if (errno == EINTR) continue; ok = false; break; }
                            if (got == 0) { ok = false; break; }
                            dst += static_cast<size_t>(got);
                            off += static_cast<off_t>(got);
                            remaining -= static_cast<size_t>(got);
                        }
                        if (ok) return tensor;
                        // pread failed mid-loop: mmap only the page-aligned range
                        // covering [data_offset, data_offset+nbytes), not the whole
                        // file — a whole-file map can OOM on huge files with a small HDU.
                        // Re-stat first: the file may have been truncated since the
                        // handle opened; mapping beyond EOF SIGBUSes on first touch.
                        struct stat fresh_sb {};
                        if (fstat(raw_fd_, &fresh_sb) == 0 &&
                            static_cast<size_t>(fresh_sb.st_size) >= end_off) {
                            static const long kPageSize = sysconf(_SC_PAGESIZE);
                            const off_t page_mask = kPageSize > 0 ? static_cast<off_t>(kPageSize - 1) : 0;
                            const off_t map_page_offset = static_cast<off_t>(data_offset) & ~page_mask;
                            const size_t map_len = nbytes + static_cast<size_t>(
                                static_cast<off_t>(data_offset) - map_page_offset);
                            void* map_ptr = mmap(nullptr, map_len, PROT_READ, MAP_SHARED,
                                                 raw_fd_, map_page_offset);
                            if (map_ptr != MAP_FAILED) {
                                const uint8_t* src = static_cast<const uint8_t*>(map_ptr) +
                                    (static_cast<off_t>(data_offset) - map_page_offset);
                                std::memcpy(tensor.data_ptr(), src, nbytes);
                                munmap(map_ptr, map_len);
                                return tensor;
                            }
                        }
                        // Fall through to the CFITSIO path, which reports a clean error.
                    }
                }
            } else { status = 0; }
        }
    }
    BScaleGuard guard(fptr_);
    const bool needs_bscale_guard = (bitpix != FLOAT_IMG && bitpix != DOUBLE_IMG);
    if (needs_bscale_guard) {
        int key_status = 0;
        double bscale = 1.0, bzero = 0.0;
        key_status = 0;
        fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &key_status);
        if (key_status != 0 && key_status != KEY_NO_EXIST) bscale = 1.0;
        key_status = 0;
        fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &key_status);
        if (key_status != 0 && key_status != KEY_NO_EXIST) bzero = 0.0;
        guard.bscale = bscale; guard.bzero = bzero;
        status = 0;
        fits_set_bscale(fptr_, 1.0, 0.0, &status);
        if (status == 0) guard.active = true;
        else status = 0;
    }
    int anynul = 0;
    float fnullval = NAN;
    double dnullval = NAN;
    void* nullval_ptr = nullptr;
    // Compressed images may hold undefined pixels; always substitute NaN for
    // float reads so nulls never decode as 0 (see read_tensor_canonical).
    if ((datatype == TFLOAT || datatype == TDOUBLE) && compressed)
        nullval_ptr = (datatype == TFLOAT) ? (void*)&fnullval : (void*)&dnullval;
    fits_read_img(fptr_, datatype, 1, nelements, nullval_ptr, tensor.data_ptr(), &anynul, &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error reading image data: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    return tensor;
}

bool FITSFile::write_image(nb::ndarray<> tensor, int hdu_num, double bscale, double bzero) {
    int status = 0;
    int naxis = tensor.ndim();
    std::vector<long> naxes(naxis);
    for (int i = 0; i < naxis; ++i) naxes[i] = tensor.shape(i);
    std::reverse(naxes.begin(), naxes.end());
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
    else throw std::runtime_error("Unsupported tensor dtype");
    long nelements = static_cast<long>(
        torchfits::detail::checked_nelements_product(naxes));
    fits_create_img(fptr_, bitpix, naxis, naxes.data(), &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error creating image: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    fits_write_img(fptr_, datatype, 1, nelements, tensor.data(), &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error writing image: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    return true;
}

std::vector<std::tuple<std::string, std::string, std::string>> FITSFile::get_header(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    int nkeys = 0, morekeys = 0;
    fits_get_hdrspace(fptr_, &nkeys, &morekeys, &status);
    std::vector<std::tuple<std::string, std::string, std::string>> header;
    header.reserve(nkeys);
    char keyname[FLEN_KEYWORD], value[FLEN_VALUE], comment[FLEN_COMMENT];
    for (int i = 1; i <= nkeys; i++) {
        fits_read_keyn(fptr_, i, keyname, value, comment, &status);
        if (status == 0) {
            std::string key_str(keyname), val_str(value), com_str(comment);
            val_str = detail::sanitize_fits_string(val_str);
            if (val_str.length() >= 2 && val_str.front() == '\'') {
                size_t last_quote = val_str.rfind('\'');
                if (last_quote != std::string::npos && last_quote > 0) {
                    val_str = val_str.substr(1, last_quote - 1);
                    size_t last_char = val_str.find_last_not_of(' ');
                    if (last_char != std::string::npos) val_str = val_str.substr(0, last_char + 1);
                    else val_str = "";
                    size_t pos = 0;
                    while ((pos = val_str.find("''", pos)) != std::string::npos) {
                        val_str.replace(pos, 2, "'"); pos += 1;
                    }
                }
            }
            if (key_str == "HISTORY" || key_str == "COMMENT") {
                if (val_str.empty() && !com_str.empty()) {
                    val_str = com_str; com_str = "";
                }
            }
            header.emplace_back(key_str, val_str, com_str);
        } else { status = 0; }
    }
    return header;
}

std::vector<long> FITSFile::get_shape(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    int naxis = 0;
    fits_get_img_dim(fptr_, &naxis, &status);
    std::vector<long> naxes(naxis);
    fits_get_img_size(fptr_, naxis, naxes.data(), &status);
    std::reverse(naxes.begin(), naxes.end());
    return naxes;
}

int FITSFile::get_dtype(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    int bitpix = 0;
    fits_get_img_type(fptr_, &bitpix, &status);
    return bitpix;
}

torch::Tensor FITSFile::read_subset(int hdu_num, long x1, long y1, long x2, long y2) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    int bitpix = 0, naxis = 0;
    long naxes[9] = {0};
    {
        const auto& info = get_image_info(hdu_num);
        bitpix = std::get<0>(info);
        naxis = std::get<1>(info);
        const auto& naxes_ll = std::get<2>(info);
        for (int i = 0; i < 9; ++i) naxes[i] = static_cast<long>(naxes_ll[i]);
    }
    if (naxis < 2) throw std::runtime_error("Subset reading requires at least 2D image");
    long max_x = naxes[0], max_y = naxes[1];
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x2 > max_x) x2 = max_x; if (y2 > max_y) y2 = max_y;
    if (x2 <= x1 || y2 <= y1) {
        // Preserve whichever of width/height is non-degenerate instead of
        // collapsing both to 0: e.g. a zero-width, full-height box should
        // report shape (..., height, 0), not (..., 0, 0).
        long empty_width = x2 > x1 ? x2 - x1 : 0;
        long empty_height = y2 > y1 ? y2 - y1 : 0;
        std::vector<int64_t> empty_shape;
        for (int i = naxis - 1; i >= 2; --i) empty_shape.push_back(naxes[i]);
        empty_shape.push_back(empty_height);
        empty_shape.push_back(empty_width);
        return torch::empty(empty_shape, torch::TensorOptions().dtype(torch::kFloat32));
    }
    const auto& scale_info = get_scale_info(hdu_num, bitpix);
    bool scaled = scale_info.scaled;
    torch::ScalarType dtype;
    int datatype;
    // Mirror the dtype conventions of read_tensor_canonical: BZERO=32768 /
    // 2^31 / -128 are unsigned/signed integer conventions, not float scale.
    const bool signed_byte_scaled =
        scaled && bitpix == BYTE_IMG && scale_info.bscale == 1.0 && scale_info.bzero == -128.0;
    const bool unsigned_short =
        scaled && bitpix == SHORT_IMG && scale_info.bscale == 1.0 && detail::is_unsigned_short_offset(scale_info.bzero);
    const bool unsigned_long =
        scaled && bitpix == LONG_IMG && scale_info.bscale == 1.0 && detail::is_unsigned_long_offset(scale_info.bzero);
    if (signed_byte_scaled) { dtype = torch::kInt8;  datatype = TSBYTE; }
    else if (unsigned_short) { dtype = torch::kUInt16; datatype = TUSHORT; }
    else if (unsigned_long) { dtype = torch::kUInt32; datatype = TUINT; }
    else if (scaled) { dtype = torch::kFloat32; datatype = TFLOAT; }
    else {
        switch (bitpix) {
            case BYTE_IMG:     dtype = torch::kUInt8;  datatype = TBYTE;      break;
            case SHORT_IMG:    dtype = torch::kInt16;  datatype = TSHORT;     break;
            case LONG_IMG:     dtype = torch::kInt32;  datatype = TINT;       break;
            case LONGLONG_IMG: dtype = torch::kInt64;  datatype = TLONGLONG;  break;
            case FLOAT_IMG:    dtype = torch::kFloat32; datatype = TFLOAT;     break;
            case DOUBLE_IMG:   dtype = torch::kFloat64; datatype = TDOUBLE;    break;
            default: throw std::runtime_error("Unsupported BITPIX");
        }
    }
    long width = x2 - x1, height = y2 - y1;
    std::vector<int64_t> shape;
    for (int i = naxis - 1; i >= 2; --i) shape.push_back(naxes[i]);
    shape.push_back(height);
    shape.push_back(width);
    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
    std::vector<long> fpixel(naxis, 1), lpixel(naxis, 1), inc(naxis, 1);
    fpixel[0] = x1 + 1; fpixel[1] = y1 + 1;
    lpixel[0] = x2; lpixel[1] = y2;
    for (int i = 2; i < naxis; ++i) {
        lpixel[i] = naxes[i];
    }
    int anynul = 0;
    fits_read_subset(fptr_, datatype, fpixel.data(), lpixel.data(), inc.data(),
                     nullptr, tensor.data_ptr(), &anynul, &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error reading subset: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    return tensor;
}

int FITSFile::get_num_hdus() {
    int status = 0, nhdus = 0;
    fits_get_num_hdus(fptr_, &nhdus, &status);
    return nhdus;
}

std::string FITSFile::get_hdu_type(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    int hdutype = 0;
    fits_get_hdu_type(fptr_, &hdutype, &status);
    if (hdutype == IMAGE_HDU) return "IMAGE";
    if (hdutype == ASCII_TBL) return "ASCII_TABLE";
    if (hdutype == BINARY_TBL) return "BINARY_TABLE";
    return "UNKNOWN";
}

bool FITSFile::write_hdus(nb::list hdus, bool /*overwrite*/) {
    int hdu_count = 0;
    for (auto handle : hdus) {
        nb::object hdu_obj = nb::cast<nb::object>(handle);
        if (nb::hasattr(hdu_obj, "_raw_data")) {
            nb::dict data_dict = nb::cast<nb::dict>(hdu_obj.attr("_raw_data"));
            nb::dict header_dict;
            if (nb::hasattr(hdu_obj, "header"))
                header_dict = nb::cast<nb::dict>(hdu_obj.attr("header"));
            nb::object schema_obj = nb::none();
            if (nb::hasattr(hdu_obj, "_schema")) {
                schema_obj = hdu_obj.attr("_schema");
                if (schema_obj.is_none()) schema_obj = nb::none();
            }
            write_table_hdu(fptr_, data_dict, header_dict, schema_obj, false);
            hdu_count++;
            continue;
        }
        nb::object data_obj;
        bool has_data = false;
        if (nb::hasattr(hdu_obj, "to_tensor")) {
            data_obj = hdu_obj.attr("to_tensor")();
            has_data = true;
        }
        if (!has_data && nb::hasattr(hdu_obj, "data")) {
            data_obj = hdu_obj.attr("data"); has_data = true;
        } else if (!has_data && nb::isinstance<nb::dict>(hdu_obj)) {
            nb::dict d = nb::cast<nb::dict>(hdu_obj);
            if (d.contains("data")) { data_obj = d["data"]; has_data = true; }
        }
        if (has_data) {
            nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(data_obj);
            write_image(tensor, hdu_count, 1.0, 0.0);
        } else {
            int status = 0;
            long naxes[1] = {0};
            fits_create_img(fptr_, BYTE_IMG, 0, naxes, &status);
            if (status != 0) throw std::runtime_error("Failed to create empty image HDU");
        }
        nb::object header_obj;
        if (nb::hasattr(hdu_obj, "header"))
            header_obj = hdu_obj.attr("header");
        else if (nb::isinstance<nb::dict>(hdu_obj)) {
            nb::dict d = nb::cast<nb::dict>(hdu_obj);
            if (d.contains("header")) header_obj = d["header"];
        }
        if (header_obj.is_valid()) {
            nb::dict header = nb::cast<nb::dict>(header_obj);
            for (auto item : header) {
                std::string key = nb::cast<std::string>(item.first);
                key = detail::sanitize_fits_key(key);
                std::string key_upper = key;
                std::transform(key_upper.begin(), key_upper.end(), key_upper.begin(),
                               [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                if (key_upper == "END" || key_upper == "SIMPLE" || key_upper == "XTENSION" ||
                    key_upper == "BITPIX" || key_upper == "NAXIS" || key_upper == "EXTEND" ||
                    key_upper == "PCOUNT" || key_upper == "GCOUNT" || key_upper == "TFIELDS" ||
                    key_upper == "THEAP" || key_upper.rfind("NAXIS", 0) == 0) continue;
                int key_status = 0;
                if (nb::isinstance<bool>(item.second)) {
                    int val = nb::cast<bool>(item.second) ? 1 : 0;
                    fits_update_key(fptr_, TLOGICAL, key.c_str(), &val, nullptr, &key_status);
                } else if (nb::isinstance<nb::str>(item.second)) {
                    std::string val = detail::sanitize_fits_string(nb::cast<std::string>(item.second));
                    fits_update_key(fptr_, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &key_status);
                } else if (PyLong_Check(item.second.ptr())) {
                    int overflow = 0;
                    long long val = PyLong_AsLongLongAndOverflow(
                        item.second.ptr(), &overflow
                    );
                    if (overflow != 0 || PyErr_Occurred()) {
                        PyErr_Clear();
                        throw std::runtime_error(
                            "FITS header integer out of long long range: " + key
                        );
                    }
                    fits_update_key(fptr_, TLONGLONG, key.c_str(), &val, nullptr, &key_status);
                } else if (nb::isinstance<double>(item.second) || nb::isinstance<float>(item.second)) {
                    double val = nb::cast<double>(item.second);
                    fits_update_key(fptr_, TDOUBLE, key.c_str(), &val, nullptr, &key_status);
                }
                if (key_status != 0) {
                    throw std::runtime_error("Failed to write FITS header keyword: " + key);
                }
            }
        }
        hdu_count++;
    }
    return true;
}

bool FITSFile::write_hdus_compressed_images(nb::list hdus, int compression_type) {
    int status = 0;
    long naxes0[1] = {0};
    fits_create_img(fptr_, BYTE_IMG, 0, naxes0, &status);
    if (status != 0) throw std::runtime_error("Failed to create primary HDU for compressed file");
    auto write_header_dict = [&](nb::object hdu_obj) {
        nb::object header_obj;
        if (nb::hasattr(hdu_obj, "header"))
            header_obj = hdu_obj.attr("header");
        else if (nb::isinstance<nb::dict>(hdu_obj)) {
            nb::dict d = nb::cast<nb::dict>(hdu_obj);
            if (d.contains("header")) header_obj = d["header"];
        }
        if (!header_obj.is_valid()) return;
        nb::dict header = nb::cast<nb::dict>(header_obj);
        for (auto item : header) {
            std::string key = nb::cast<std::string>(item.first);
            key = detail::sanitize_fits_key(key);
            std::string key_upper = key;
            std::transform(key_upper.begin(), key_upper.end(), key_upper.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            if (key_upper == "END" || key_upper == "SIMPLE" || key_upper == "XTENSION" ||
                key_upper == "BITPIX" || key_upper == "NAXIS" || key_upper == "EXTEND" ||
                key_upper == "PCOUNT" || key_upper == "GCOUNT" || key_upper == "TFIELDS" ||
                key_upper == "THEAP" || key_upper == "DATASUM" || key_upper == "CHECKSUM" ||
                key_upper == "ZIMAGE" || key_upper == "ZCMPTYPE" || key_upper == "ZBITPIX" ||
                key_upper == "ZNAXIS" || key_upper == "ZPCOUNT" || key_upper == "ZGCOUNT" ||
                key_upper == "ZHECKSUM" || key_upper == "ZDATASUM" ||
                key_upper.rfind("NAXIS", 0) == 0) continue;
            if (key_upper.rfind("ZNAXIS", 0) == 0 || key_upper.rfind("ZTILE", 0) == 0 ||
                key_upper.rfind("ZNAME", 0) == 0 || key_upper.rfind("ZVAL", 0) == 0) continue;
            int key_status = 0;
            if (nb::isinstance<bool>(item.second)) {
                int val = nb::cast<bool>(item.second) ? 1 : 0;
                fits_update_key(fptr_, TLOGICAL, key.c_str(), &val, nullptr, &key_status);
            } else if (nb::isinstance<nb::str>(item.second)) {
                    std::string val = detail::sanitize_fits_string(nb::cast<std::string>(item.second));
                    fits_update_key(fptr_, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &key_status);
            } else if (PyLong_Check(item.second.ptr())) {
                    int overflow = 0;
                    long long val = PyLong_AsLongLongAndOverflow(
                        item.second.ptr(), &overflow
                    );
                    if (overflow != 0 || PyErr_Occurred()) {
                        PyErr_Clear();
                        throw std::runtime_error(
                            "FITS header integer out of long long range: " + key
                        );
                    }
                    fits_update_key(fptr_, TLONGLONG, key.c_str(), &val, nullptr, &key_status);
            } else if (nb::isinstance<double>(item.second) || nb::isinstance<float>(item.second)) {
                    double val = nb::cast<double>(item.second);
                    fits_update_key(fptr_, TDOUBLE, key.c_str(), &val, nullptr, &key_status);
            }
            if (key_status != 0) {
                throw std::runtime_error("Failed to write FITS header keyword: " + key);
            }
        }
    };
    for (auto handle : hdus) {
        nb::object hdu_obj = nb::cast<nb::object>(handle);
        if (nb::hasattr(hdu_obj, "_raw_data")) {
            nb::dict data_dict = nb::cast<nb::dict>(hdu_obj.attr("_raw_data"));
            nb::dict header_dict;
            if (nb::hasattr(hdu_obj, "header"))
                header_dict = nb::cast<nb::dict>(hdu_obj.attr("header"));
            write_table_hdu(fptr_, data_dict, header_dict, nb::none(), false);
            continue;
        }
        nb::object data_obj;
        bool has_data = false;
        if (nb::hasattr(hdu_obj, "to_tensor")) {
            data_obj = hdu_obj.attr("to_tensor")();
            has_data = true;
        }
        if (!has_data && nb::hasattr(hdu_obj, "data")) {
            data_obj = hdu_obj.attr("data"); has_data = true;
        } else if (!has_data && nb::isinstance<nb::dict>(hdu_obj)) {
            nb::dict d = nb::cast<nb::dict>(hdu_obj);
            if (d.contains("data")) { data_obj = d["data"]; has_data = true; }
        }
        if (!has_data) throw std::runtime_error("Compressed writing requires image data");
        nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(data_obj);
        int naxis = tensor.ndim();
        if (naxis <= 0) throw std::runtime_error("Unsupported image ndim for compressed write");
        std::vector<long> naxes(naxis);
        for (int i = 0; i < naxis; ++i) naxes[i] = tensor.shape(i);
        std::reverse(naxes.begin(), naxes.end());
        status = 0;
        fits_set_compression_type(fptr_, compression_type, &status);
        if (status != 0) throw std::runtime_error("Failed to set compression type");
        // Default to lossless where CFITSIO supports it: CFITSIO quantizes
        // float/double pixels (lossy) unless explicitly disabled, which would
        // silently corrupt round-trips of compressed float images. GZIP and
        // integer RICE compress losslessly with qlevel=0; float RICE and
        // HCOMPRESS cannot be lossless (CFITSIO returns compression errors),
        // so they keep the library default quantization (lossy, matching
        // astropy and fitsio defaults).
        bool lossless_capable = false;
        if (compression_type == GZIP_1) {
            lossless_capable = true;
        } else if (compression_type == RICE_1) {
            nb::dlpack::dtype dt = tensor.dtype();
            lossless_capable =
                dt.code != (uint8_t)nb::dlpack::dtype_code::Float;
        }
        if (lossless_capable) {
            status = 0;
            fits_set_quantize_level(fptr_, 0.0f, &status);
            if (status != 0) throw std::runtime_error("Failed to set compression quantization");
        }
        if (compression_type == PLIO_1) {
            nb::dlpack::dtype dt_plio = tensor.dtype();
            if (dt_plio.code == (uint8_t)nb::dlpack::dtype_code::Float) {
                throw std::runtime_error("PLIO_1 compression algorithm does not support floating-point image data");
            }
        }
        std::vector<long> tilesize(naxis, 1);
        tilesize[0] = naxes[0];
        if (compression_type == HCOMPRESS_1) {
            // HCOMPRESS requires 2D tiles; use 16-row bands like fpack's
            // default so output tiles match the reference CLI.
            if (naxis < 2)
                throw std::runtime_error(
                    "HCOMPRESS compression requires a 2D or higher-dimensional image");
            tilesize[0] = naxes[0];
            tilesize[1] = naxes[1] < 16 ? naxes[1] : 16;
        }
        fits_set_tile_dim(fptr_, naxis, tilesize.data(), &status);
        if (status != 0) throw std::runtime_error("Failed to set tile dimensions");
        int bitpix = FLOAT_IMG, datatype = TFLOAT;
        nb::dlpack::dtype dt = tensor.dtype();
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) { bitpix = BYTE_IMG; datatype = TBYTE; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 8) { bitpix = SBYTE_IMG; datatype = TSBYTE; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) { bitpix = SHORT_IMG; datatype = TSHORT; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) { bitpix = LONG_IMG; datatype = TINT; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) { bitpix = LONGLONG_IMG; datatype = TLONGLONG; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) { bitpix = FLOAT_IMG; datatype = TFLOAT; }
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) { bitpix = DOUBLE_IMG; datatype = TDOUBLE; }
        else throw std::runtime_error("Unsupported tensor dtype for compressed write");
        fits_create_img(fptr_, bitpix, naxis, naxes.data(), &status);
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error creating compressed image: status=" + std::to_string(status) +
                                     " msg=" + std::string(err_text));
        }
        long nelements = static_cast<long>(
            torchfits::detail::checked_nelements_product(naxes));
        fits_write_img(fptr_, datatype, 1, nelements, tensor.data(), &status);
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error writing compressed image: status=" + std::to_string(status) +
                                     " msg=" + std::string(err_text));
        }
        write_header_dict(hdu_obj);
    }
    return true;
}

std::string FITSFile::read_header_to_string(int hdu_num) {
    int status = 0;
    ensure_hdu(hdu_num, &status);
    char* header_str = nullptr;
    int nkeys = 0;
    if (fits_hdr2str(fptr_, 0, nullptr, 0, &header_str, &nkeys, &status)) return "";
    std::string result(header_str);
    if (header_str) fits_free_memory(header_str, &status);
    return result;
}

bool FITSFile::ensure_raw_fd(size_t required_end) {
    if (has_cfitsio_extended_filename_syntax(filename_)) return false;
    if (!raw_fd_ready_) {
        raw_fd_ready_ = true;
        raw_fd_ = detail::open_readonly_fd(filename_);
        if (raw_fd_ == -1) return false;
        struct stat sb {};
        if (fstat(raw_fd_, &sb) != 0) {
            ::close(raw_fd_); raw_fd_ = -1; raw_file_size_ = 0; return false;
        }
        raw_file_size_ = sb.st_size;
    }
    return raw_fd_ != -1 && required_end <= static_cast<size_t>(raw_file_size_);
}

void FITSFile::close_raw_fd() {
    if (raw_fd_ != -1) { ::close(raw_fd_); raw_fd_ = -1; }
    raw_file_size_ = 0;
    raw_fd_ready_ = false;
}

// ===========================================================================
// SubsetReader implementation
// ===========================================================================
SubsetReader::SubsetReader(const std::string& filename, int hdu_num)
    : file_(filename.c_str(), 0), filename_(filename), hdu_num_(hdu_num) {
    if (hdu_num_ < 0) throw std::runtime_error("HDU index must be non-negative");
    init_from_hdu();
}

SubsetReader::~SubsetReader() { release_data_mmap(); }

void SubsetReader::init_from_hdu() {
    int status = 0;
    file_.ensure_hdu(hdu_num_, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    const auto& info = file_.get_image_info(hdu_num_);
    const int bitpix = std::get<0>(info);
    const int naxis = std::get<1>(info);
    const auto& naxes = std::get<2>(info);
    if (naxis < 2) throw std::runtime_error("SubsetReader requires at least 2D image HDU");
    naxis_ = naxis;
    bitpix_ = bitpix;
    naxes_.resize(naxis_);
    for (int i = 0; i < naxis_; ++i) naxes_[i] = static_cast<long>(naxes[i]);
    max_x_ = static_cast<long>(naxes[0]);
    max_y_ = static_cast<long>(naxes[1]);
    const auto scale = file_.get_scale_info_for_hdu(hdu_num_);
    // Match full-image logical dtypes — do not float-promote signed-byte /
    // unsigned integer conventions (that lost to fitsio int8 cutouts).
    // Keep mmap eligible for the three FITS integer conventions (same as
    // read_tensor_canonical); only arbitrary BSCALE/BZERO falls back to
    // fits_read_subset float promotion.
    if (scale.scaled && bitpix == BYTE_IMG && scale.bscale == 1.0 && scale.bzero == -128.0) {
        dtype_ = torch::kInt8; datatype_ = TSBYTE; elem_bytes_ = 1;
        mmap_conv_ = MmapConv::SignedByte;
    } else if (scale.scaled && bitpix == SHORT_IMG && scale.bscale == 1.0 &&
               detail::is_unsigned_short_offset(scale.bzero)) {
        dtype_ = torch::kUInt16; datatype_ = TUSHORT; elem_bytes_ = 2;
        mmap_conv_ = MmapConv::UInt16;
    } else if (scale.scaled && bitpix == LONG_IMG && scale.bscale == 1.0 &&
               detail::is_unsigned_long_offset(scale.bzero)) {
        dtype_ = torch::kUInt32; datatype_ = TUINT; elem_bytes_ = 4;
        mmap_conv_ = MmapConv::UInt32;
    } else if (scale.scaled) {
        dtype_ = torch::kFloat32; datatype_ = TFLOAT; return;
    } else {
        switch (bitpix) {
            case BYTE_IMG:     dtype_ = torch::kUInt8;  datatype_ = TBYTE;      elem_bytes_ = 1; break;
            case SHORT_IMG:    dtype_ = torch::kInt16;  datatype_ = TSHORT;     elem_bytes_ = 2; break;
            case LONG_IMG:     dtype_ = torch::kInt32;  datatype_ = TINT;       elem_bytes_ = 4; break;
            case LONGLONG_IMG: dtype_ = torch::kInt64;  datatype_ = TLONGLONG;  elem_bytes_ = 8; break;
            case FLOAT_IMG:    dtype_ = torch::kFloat32; datatype_ = TFLOAT;    elem_bytes_ = 4; break;
            case DOUBLE_IMG:   dtype_ = torch::kFloat64; datatype_ = TDOUBLE;   elem_bytes_ = 8; break;
            default: throw std::runtime_error("Unsupported BITPIX");
        }
    }

    // Uncompressed primary/local 2D image: row mmap + endian swap beats
    // fits_read_subset on large mosaics (page-cache + memcpy, like fitsio memmap).
    const bool compressed = file_.is_compressed_image_cached(hdu_num_);
    if (!compressed && naxis_ == 2 && elem_bytes_ > 0 &&
        !has_cfitsio_extended_filename_syntax(filename_) &&
        filename_.find("://") == std::string::npos) {
        LONGLONG headstart = 0, data_offset = 0, dataend = 0;
        status = 0;
        fits_get_hduaddrll(file_.get_fptr(), &headstart, &data_offset, &dataend, &status);
        if (status == 0 && data_offset > 0) {
            data_offset_ = data_offset;
            raw_fast_ok_ = true;
            // Amortize first-cutout latency for the open_subset_reader hot path.
            (void)ensure_data_mmap();
        }
    }
}

bool SubsetReader::ensure_data_mmap() {
    if (pixel_base_ != nullptr) return true;
    if (!raw_fast_ok_ || elem_bytes_ == 0 || data_offset_ <= 0) return false;
    auto meta = detail::get_shared_meta_for_path(filename_);
    raw_fd_holder_ = detail::get_shared_raw_fd(meta, filename_);
    if (!raw_fd_holder_ || raw_fd_holder_->fd < 0) return false;

    const size_t nbytes =
        static_cast<size_t>(naxes_[0]) * static_cast<size_t>(naxes_[1]) * elem_bytes_;
    if (nbytes == 0) return false;

    // Never map beyond the actual file: header-claimed sizes on truncated or
    // corrupt files would SIGBUS on first touch. Fall back to fits_read_subset.
    struct stat sb {};
    if (fstat(raw_fd_holder_->fd, &sb) != 0) {
        raw_fd_holder_.reset();
        return false;
    }
    static const long kPageSize = sysconf(_SC_PAGESIZE);
    const off_t page_mask = kPageSize > 0 ? static_cast<off_t>(kPageSize - 1) : 0;
    map_page_offset_ = static_cast<off_t>(data_offset_) & ~page_mask;
    map_len_ = nbytes + static_cast<size_t>(static_cast<off_t>(data_offset_) - map_page_offset_);
    const size_t map_end = static_cast<size_t>(map_page_offset_) + map_len_;
    if (static_cast<size_t>(sb.st_size) < map_end) {
        raw_fd_holder_.reset();
        return false;
    }
    map_ptr_ = mmap(nullptr, map_len_, PROT_READ, MAP_SHARED, raw_fd_holder_->fd, map_page_offset_);
    if (map_ptr_ == MAP_FAILED) {
        map_ptr_ = nullptr;
        map_len_ = 0;
        raw_fd_holder_.reset();
        return false;
    }
#if defined(MADV_RANDOM) && defined(MADV_WILLNEED)
    // Random cutouts over a survey mosaic — WILLNEED is a light prefetch hint.
    madvise(map_ptr_, map_len_, MADV_RANDOM | MADV_WILLNEED);
#endif
    pixel_base_ = static_cast<const uint8_t*>(map_ptr_) +
                  (static_cast<off_t>(data_offset_) - map_page_offset_);
    return true;
}

void SubsetReader::release_data_mmap() {
    if (map_ptr_ != nullptr) {
        munmap(map_ptr_, map_len_);
        map_ptr_ = nullptr;
        map_len_ = 0;
        map_page_offset_ = 0;
        pixel_base_ = nullptr;
    }
}

bool SubsetReader::try_read_via_mmap(
    long x1, long y1, long x2, long y2, torch::Tensor& out
) {
    if (!ensure_data_mmap()) return false;

    const long width = x2 - x1;
    const long height = y2 - y1;
    const long naxis1 = naxes_[0];
    const size_t row_bytes = static_cast<size_t>(width) * elem_bytes_;
    uint8_t* base = static_cast<uint8_t*>(out.data_ptr());
    const bool swap = host_is_little_endian() && elem_bytes_ > 1;
    const size_t n = static_cast<size_t>(width);

    for (long y = y1; y < y2; ++y) {
        const uint8_t* src =
            pixel_base_ +
            (static_cast<size_t>(y) * static_cast<size_t>(naxis1) + static_cast<size_t>(x1)) *
                elem_bytes_;
        uint8_t* dst_row = base + static_cast<size_t>(y - y1) * row_bytes;
        if (elem_bytes_ == 1) {
            std::memcpy(dst_row, src, row_bytes);
            if (mmap_conv_ == MmapConv::SignedByte) {
                detail::_xor_sign_bit_u8(dst_row, row_bytes);
            }
        } else if (elem_bytes_ == 2) {
            auto* dst16 = reinterpret_cast<uint16_t*>(dst_row);
            const auto* src16 = reinterpret_cast<const uint16_t*>(src);
            if (mmap_conv_ == MmapConv::UInt16) {
                if (swap) {
                    internal::bswap16_copy_u16_offset(
                        src16, dst16, n, static_cast<uint16_t>(32768));
                } else {
                    for (size_t i = 0; i < n; ++i) {
                        dst16[i] = static_cast<uint16_t>(src16[i] + 32768);
                    }
                }
            } else if (swap) {
                internal::bswap16_copy(src16, dst16, n);
            } else {
                std::memcpy(dst_row, src, row_bytes);
            }
        } else if (elem_bytes_ == 4) {
            auto* dst32 = reinterpret_cast<uint32_t*>(dst_row);
            const auto* src32 = reinterpret_cast<const uint32_t*>(src);
            if (mmap_conv_ == MmapConv::UInt32) {
                if (swap) {
                    internal::bswap32_copy_u32_offset(src32, dst32, n, 2147483648u);
                } else {
                    for (size_t i = 0; i < n; ++i) {
                        dst32[i] = src32[i] + 2147483648u;
                    }
                }
            } else if (swap) {
                internal::bswap32_copy(src32, dst32, n);
            } else {
                std::memcpy(dst_row, src, row_bytes);
            }
        } else if (elem_bytes_ == 8) {
            if (swap) {
                internal::bswap64_copy(
                    reinterpret_cast<const uint64_t*>(src),
                    reinterpret_cast<uint64_t*>(dst_row),
                    n);
            } else {
                std::memcpy(dst_row, src, row_bytes);
            }
        } else {
            return false;
        }
    }
    return true;
}

torch::Tensor SubsetReader::read(long x1, long y1, long x2, long y2) {
    if (closed_) throw std::runtime_error("SubsetReader is closed");
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x2 > max_x_) x2 = max_x_; if (y2 > max_y_) y2 = max_y_;
    if (x2 <= x1 || y2 <= y1) {
        // See FITSFile::read_subset: keep the non-degenerate dimension's
        // extent instead of always reporting (..., 0, 0).
        long empty_width = x2 > x1 ? x2 - x1 : 0;
        long empty_height = y2 > y1 ? y2 - y1 : 0;
        std::vector<int64_t> empty_shape;
        for (int i = naxis_ - 1; i >= 2; --i) empty_shape.push_back(naxes_[i]);
        empty_shape.push_back(empty_height);
        empty_shape.push_back(empty_width);
        return torch::empty(empty_shape, torch::TensorOptions().dtype(dtype_));
    }
    long width = x2 - x1, height = y2 - y1;
    std::vector<int64_t> shape;
    for (int i = naxis_ - 1; i >= 2; --i) shape.push_back(naxes_[i]);
    shape.push_back(height);
    shape.push_back(width);
    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype_));
    if (try_read_via_mmap(x1, y1, x2, y2, tensor)) {
        return tensor;
    }
    std::vector<long> fpixel(naxis_, 1);
    std::vector<long> lpixel(naxis_, 1);
    std::vector<long> inc(naxis_, 1);
    fpixel[0] = x1 + 1;
    fpixel[1] = y1 + 1;
    lpixel[0] = x2;
    lpixel[1] = y2;
    for (int i = 2; i < naxis_; ++i) {
        lpixel[i] = naxes_[i];
    }
    int status = 0, anynul = 0;
    fits_read_subset(file_.get_fptr(), datatype_, fpixel.data(), lpixel.data(), inc.data(),
                     nullptr, tensor.data_ptr(), &anynul, &status);
    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error reading subset: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }
    return tensor;
}

void SubsetReader::close() {
    if (!closed_) {
        release_data_mmap();
        file_.close();
        closed_ = true;
    }
}

} // namespace torchfits
