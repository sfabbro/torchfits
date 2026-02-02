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
#include <sys/stat.h>
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

struct SharedReadMeta {
    std::unordered_map<int, std::tuple<int, int, std::array<LONGLONG, 9>>> image_info_cache;
    std::unordered_map<int, bool> compressed_cache;
    std::unordered_map<int, bool> compressed_nulls_cache;
    std::unordered_map<int, std::tuple<bool, bool, double, double>> scale_cache;
    bool has_stat = false;
    off_t size = 0;
    time_t mtime = 0;
    std::mutex mutex;
};

std::mutex g_shared_meta_mutex;
std::unordered_map<std::string, std::shared_ptr<SharedReadMeta>> g_shared_meta;

std::shared_ptr<SharedReadMeta> get_shared_meta_for_path(const std::string& filename) {
    bool can_stat = filename.find('[') == std::string::npos;
    struct stat st {};
    bool has_stat = can_stat && stat(filename.c_str(), &st) == 0;

    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    auto it = g_shared_meta.find(filename);
    if (it == g_shared_meta.end()) {
        auto meta = std::make_shared<SharedReadMeta>();
        if (has_stat) {
            meta->has_stat = true;
            meta->size = st.st_size;
            meta->mtime = st.st_mtime;
        }
        g_shared_meta.emplace(filename, meta);
        return meta;
    }

    auto meta = it->second;
    if (has_stat && (!meta->has_stat || meta->size != st.st_size || meta->mtime != st.st_mtime)) {
        std::lock_guard<std::mutex> meta_lock(meta->mutex);
        meta->image_info_cache.clear();
        meta->compressed_cache.clear();
        meta->compressed_nulls_cache.clear();
        meta->scale_cache.clear();
        meta->has_stat = true;
        meta->size = st.st_size;
        meta->mtime = st.st_mtime;
    }
    return meta;
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

void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header);
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
std::vector<torch::Tensor> read_hdus_batch(const std::string& path, const std::vector<int>& hdus);

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
        
        // Store the initial HDU number (important for extended filename syntax/virtual files)
        fits_get_hdu_num(fptr_, &start_hdu_);
        current_hdu_ = start_hdu_;

        // Cached handles may have been left on a non-primary HDU.
        // For regular filenames, normalize to the primary HDU to keep 0-based indexing stable.
        if (use_cache_) {
            bool has_extension = filename_.find('[') != std::string::npos;
            if (!has_extension) {
                int move_status = 0;
                fits_movabs_hdu(fptr_, 1, nullptr, &move_status);
                if (move_status == 0) {
                    start_hdu_ = 1;
                    current_hdu_ = 1;
                }
            }
        }
    }
    
    ~FITSFile() {
        close();
    }
    
    void close() {
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
        
        // CFITSIO-only path: use_mmap currently does not change behavior.
        (void)use_mmap;
        
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
            // If scaled, always read as float32 (or double if needed, but float32 is standard for images)
            dtype = torch::kFloat32;
            datatype = TFLOAT;
        } else {
            switch (bitpix) {
                case BYTE_IMG:   dtype = torch::kUInt8; datatype = TBYTE; break;
                case SHORT_IMG:  dtype = torch::kInt16; datatype = TSHORT; break;
                case LONG_IMG:   dtype = torch::kInt32; datatype = TINT; break;
                case FLOAT_IMG:  dtype = torch::kFloat32; datatype = TFLOAT; break;
                case DOUBLE_IMG: dtype = torch::kFloat64; datatype = TDOUBLE; break;
                default: throw std::runtime_error("Unsupported BITPIX");
            }
        }

        auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));

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
        
        // CFITSIO-only path: use_mmap currently does not change behavior.
        (void)use_mmap;

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
        int status = 0;
        int hdu_count = 0;
        
        for (auto handle : hdus) {
            nb::object hdu_obj = nb::cast<nb::object>(handle);
            
            // Check for TableHDU (has feat_dict)
            if (nb::hasattr(hdu_obj, "feat_dict")) {
                nb::dict data_dict = nb::cast<nb::dict>(hdu_obj.attr("feat_dict"));
                nb::dict header_dict;
                if (nb::hasattr(hdu_obj, "header")) {
                     header_dict = nb::cast<nb::dict>(hdu_obj.attr("header"));
                }
                write_table_hdu(fptr_, data_dict, header_dict);
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
                    long naxes[1] = {0};
                    fits_create_img(fptr_, BYTE_IMG, 0, naxes, &status);
                }
            } else {
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
                        
                        try {
                            if (nb::isinstance<nb::str>(item.second)) {
                                std::string val = nb::cast<std::string>(item.second);
                                val = sanitize_fits_string(val);
                                fits_update_key(fptr_, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &status);
                            } else if (nb::isinstance<int>(item.second)) {
                                int val = nb::cast<int>(item.second);
                                fits_update_key(fptr_, TINT, key.c_str(), &val, nullptr, &status);
                            } else if (nb::isinstance<float>(item.second)) {
                                float val = nb::cast<float>(item.second);
                                fits_update_key(fptr_, TFLOAT, key.c_str(), &val, nullptr, &status);
                            } else if (nb::isinstance<double>(item.second)) {
                                double val = nb::cast<double>(item.second);
                                fits_update_key(fptr_, TDOUBLE, key.c_str(), &val, nullptr, &status);
                            } else if (nb::isinstance<bool>(item.second)) {
                                int val = nb::cast<bool>(item.second) ? 1 : 0;
                                fits_update_key(fptr_, TLOGICAL, key.c_str(), &val, nullptr, &status);
                            }
                        } catch (...) {}
                    }
                } catch (...) {}
            }
            
            hdu_count++;
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
    std::string filename_;
    int mode_;
    fitsfile* fptr_ = nullptr;
    bool cached_ = false;
    int start_hdu_ = 1;
    int current_hdu_ = 1;
    bool use_cache_ = false;
    std::unordered_map<int, ScaleInfo> scale_cache_;
    std::unordered_map<int, bool> compressed_cache_;
    std::unordered_map<int, bool> compressed_nulls_cache_;
    std::unordered_map<int, std::tuple<int, int, std::array<LONGLONG, 9>>> image_info_cache_;
    std::shared_ptr<SharedReadMeta> shared_meta_;
};


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
std::vector<torch::Tensor> read_hdus_batch(const std::string& path, const std::vector<int>& hdus) {
    FITSFile file(path.c_str(), 0);
    std::vector<torch::Tensor> results;
    results.reserve(hdus.size());
    for (int hdu_num : hdus) {
        results.push_back(file.read_image(hdu_num));
    }
    return results;
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
    (void)use_mmap;
    return read_full_unmapped(path, hdu_num);
}

void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header) {
    int status = 0;
    int num_cols = tensor_dict.size();
    long num_rows = 0;
    if (num_cols > 0) {
        for (auto item : tensor_dict) {
             try {
                 nb::ndarray<> col = nb::cast<nb::ndarray<>>(item.second);
                 num_rows = col.shape(0);
                 break;
             } catch (...) {
                 continue;
             }
        }
    }

    char** ttype = new char*[num_cols];
    char** tform = new char*[num_cols];
    char** tunit = new char*[num_cols];

    int i = 0;
    std::vector<std::string> col_names;
    std::vector<std::string> forms;
    
    for (auto item : tensor_dict) {
        std::string col_name = nb::cast<std::string>(item.first);
        col_name = sanitize_fits_string(col_name);
        col_names.push_back(col_name);
        
        nb::ndarray<> tensor;
        try {
            tensor = nb::cast<nb::ndarray<>>(item.second);
        } catch (...) {
            continue;
        }

        ttype[i] = new char[col_name.length() + 1];
        strncpy(ttype[i], col_name.c_str(), col_name.length());
        ttype[i][col_name.length()] = '\0';

        std::string form;
        nb::dlpack::dtype dt = tensor.dtype();
        
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
            form = "B";
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) {
            form = "I";
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) {
            form = "J";
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) {
            form = "E";
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) {
            form = "D";
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) {
            form = "K";
        } else {
            form = "E"; 
        }
        
        if (tensor.ndim() > 1) {
             long width = tensor.shape(1);
             form = std::to_string(width) + form;
        } else {
             form = "1" + form;
        }
        
        forms.push_back(form);
        tform[i] = new char[form.length() + 1];
        strncpy(tform[i], form.c_str(), form.length());
        tform[i][form.length()] = '\0';
        
        tunit[i] = new char[1];
        tunit[i][0] = '\0';
        
        i++;
    }
    
    num_cols = i;
    
    fits_create_tbl(fptr, BINARY_TBL, num_rows, num_cols, ttype, tform, tunit, "Table", &status);
    
    if (status != 0) {
        for(int j=0; j<num_cols; j++) { delete[] ttype[j]; delete[] tform[j]; delete[] tunit[j]; }
        delete[] ttype; delete[] tform; delete[] tunit;
        throw std::runtime_error("Failed to create table");
    }
    
    i = 0;
    for (auto item : tensor_dict) {
        nb::ndarray<> tensor;
        try {
            tensor = nb::cast<nb::ndarray<>>(item.second);
        } catch (...) { continue; }
        
        nb::dlpack::dtype dt = tensor.dtype();
        int datatype = TFLOAT;
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) datatype = TBYTE;
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) datatype = TSHORT;
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) datatype = TINT;
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) datatype = TFLOAT;
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) datatype = TDOUBLE;
        else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) datatype = TLONGLONG;
        
        long nelements = num_rows;
        if (tensor.ndim() > 1) nelements *= tensor.shape(1);
        
        fits_write_col(fptr, datatype, i + 1, 1, 1, nelements, tensor.data(), &status);
        i++;
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
    
    for(int j=0; j<num_cols; j++) { delete[] ttype[j]; delete[] tform[j]; delete[] tunit[j]; }
    delete[] ttype; delete[] tform; delete[] tunit;
    
    if (status != 0) {
        throw std::runtime_error("Failed to write table data");
    }
}

}
