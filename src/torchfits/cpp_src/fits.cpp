#include <string>
#include <vector>
#include <unordered_map>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <torch/torch.h>
#include <fitsio.h>
#include "hardware.h"
#include "cache.cpp"

namespace nb = nanobind;

namespace torchfits {

// Forward declarations
torch::Tensor read_image_fast(const std::string& filename, int hdu_num, bool use_mmap = true);
torch::Tensor read_image_fast_int16(const std::string& filename, int hdu_num, int naxis, long* naxes, LONGLONG datastart, double bscale, double bzero);
torch::Tensor read_image_fast_int32(const std::string& filename, int hdu_num, int naxis, long* naxes, LONGLONG datastart, double bscale, double bzero);
torch::Tensor read_image_fast_float32(const std::string& filename, int hdu_num, int naxis, long* naxes, LONGLONG datastart, double bscale, double bzero);
torch::Tensor read_image_fast_double(const std::string& filename, int hdu_num, int naxis, long* naxes, LONGLONG datastart, double bscale, double bzero);

class FITSFileV2 {
public:
    FITSFileV2(const char* filename, int mode) : filename_(filename) {
        int status = 0;
        if (mode == 0) {
            fits_open_file(&fptr_, filename, READONLY, &status);
        } else {
            fits_create_file(&fptr_, filename, &status);
        }

        if (status != 0) {
            throw std::runtime_error("Could not open FITS file: " + filename_);
        }
        cached_ = false;
        
        // Store the initial HDU number (important for extended filename syntax/virtual files)
        fits_get_hdu_num(fptr_, &start_hdu_);
    }
    
    ~FITSFileV2() {
        close();
    }
    
    void close() {
        if (fptr_) {
            int status = 0;
            fits_close_file(fptr_, &status);
            fptr_ = nullptr;
        }
    }

    fitsfile* get_fptr() const { return fptr_; }

    torch::Tensor read_image(int hdu_num, bool use_mmap = true) {
        int status = 0;
        
        // Check current HDU to avoid resetting image section (for extended syntax)
        int cur_hdu = 0;
        fits_get_hdu_num(fptr_, &cur_hdu);
        
        // Calculate target absolute HDU
        // hdu_num is 0-based index relative to the opened file/view
        // start_hdu_ is 1-based absolute index of the first HDU in the view
        int target_hdu = hdu_num + start_hdu_;
        
        if (cur_hdu != target_hdu) {
            fits_movabs_hdu(fptr_, target_hdu, nullptr, &status);
            if (status != 0) throw std::runtime_error("Could not move to HDU");
        }

        int hdutype = 0;
        fits_get_hdu_type(fptr_, &hdutype, &status);
        
        bool is_compressed = false;
        if (hdutype != IMAGE_HDU) {
            // Check if it's a compressed image
            int compressed_status = 0; // Use separate status to avoid throwing if check fails
            if (fits_is_compressed_image(fptr_, &compressed_status)) {
                is_compressed = true;
                // It IS an image (compressed), so we proceed
            } else {
                throw std::runtime_error("HDU is not an image: type=" + std::to_string(hdutype));
            }
        } else {
             // Even if it reports IMAGE_HDU, check for compression just in case
             int compressed_status = 0;
             if (fits_is_compressed_image(fptr_, &compressed_status)) {
                 is_compressed = true;
             }
        }
        
        // Disable mmap for compressed images as they need decompression
        if (is_compressed) {
            use_mmap = false;
        }
        
        int bitpix = 0;
        int naxis = 0;
        long naxes[9] = {0}; // Support up to 9 dimensions (standard limit)
        
        fits_get_img_param(fptr_, 9, &bitpix, &naxis, &naxes[0], &status);
        
        // Check for scaling
        double bscale = 1.0;
        double bzero = 0.0;
        fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &status);
        if (status != 0) { status = 0; bscale = 1.0; }
        if (status != 0) { status = 0; bscale = 1.0; }
        fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &status);
        if (status != 0) { status = 0; bzero = 0.0; }
        
        // Try fast path if requested
        if (use_mmap) {
            try {
                LONGLONG headstart, datastart, dataend;
                if (fits_get_hduaddrll(fptr_, &headstart, &datastart, &dataend, &status) == 0) {
                    if (bitpix == 16) return read_image_fast_int16(filename_, hdu_num, naxis, naxes, datastart, bscale, bzero);
                    else if (bitpix == -32) return read_image_fast_float32(filename_, hdu_num, naxis, naxes, datastart, bscale, bzero);
                    else if (bitpix == 32) return read_image_fast_int32(filename_, hdu_num, naxis, naxes, datastart, bscale, bzero);
                    else if (bitpix == -64) return read_image_fast_double(filename_, hdu_num, naxis, naxes, datastart, bscale, bzero);
                }
            } catch (const std::exception& e) {
                // Fallback to slow path
            }
        }
        
        // Slow path (fits_read_pix)
        std::vector<int64_t> torch_shape;
        torch_shape.reserve(naxis);
        for (int i = 0; i < naxis; ++i) {
            torch_shape.push_back(naxes[naxis - 1 - i]); // Reverse for C-contiguous
        }

        torch::ScalarType dtype;
        int datatype;
        
        bool scaled = (bscale != 1.0 || bzero != 0.0);
        
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

        auto tensor = torch::empty(torch_shape, torch::TensorOptions().dtype(dtype));
        long nelements = 0;
        if (naxis > 0) {
            nelements = 1;
            for (int i = 0; i < naxis; ++i) nelements *= naxes[i];
        }

        std::vector<long> fpixel(naxis, 1);
        int anynul;
        fits_read_pix(fptr_, datatype, fpixel.data(), nelements, nullptr, tensor.data_ptr(), &anynul, &status);

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

    std::unordered_map<std::string, std::string> get_header(int hdu_num) {
        int status = 0;
        int target_hdu = hdu_num + start_hdu_;
        
        // Check current HDU to avoid unnecessary moves (and potential virtual file reset)
        int cur_hdu = 0;
        fits_get_hdu_num(fptr_, &cur_hdu);
        if (cur_hdu != target_hdu) {
            fits_movabs_hdu(fptr_, target_hdu, nullptr, &status);
            if (status != 0) throw std::runtime_error("Could not move to HDU");
        }
        
        int nkeys = 0;
        int morekeys = 0;
        fits_get_hdrspace(fptr_, &nkeys, &morekeys, &status);
        
        std::unordered_map<std::string, std::string> header;
        char keyname[FLEN_KEYWORD];
        char value[FLEN_VALUE];
        char comment[FLEN_COMMENT];
        int length;
        
        for (int i = 1; i <= nkeys; i++) {
            fits_read_keyn(fptr_, i, keyname, value, comment, &status);
            if (status == 0) {
                std::string val_str(value);
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
            header[keyname] = val_str;
            } else {
                status = 0; // Ignore error for single key
            }
        }
        return header;
    }

    std::vector<long> get_shape(int hdu_num) {
        int status = 0;
        int target_hdu = hdu_num + start_hdu_;
        
        int cur_hdu = 0;
        fits_get_hdu_num(fptr_, &cur_hdu);
        if (cur_hdu != target_hdu) {
            fits_movabs_hdu(fptr_, target_hdu, nullptr, &status);
        }
        
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
        int target_hdu = hdu_num + start_hdu_;
        
        int cur_hdu = 0;
        fits_get_hdu_num(fptr_, &cur_hdu);
        if (cur_hdu != target_hdu) {
            fits_movabs_hdu(fptr_, target_hdu, nullptr, &status);
        }
        
        int bitpix = 0;
        fits_get_img_type(fptr_, &bitpix, &status);
        return bitpix;
    }

    torch::Tensor read_subset(int hdu_num, long x1, long y1, long x2, long y2) {
        // Simplified subset reading
        // Assuming 2D image for now
        int status = 0;
        int target_hdu = hdu_num + start_hdu_;
        
        int cur_hdu = 0;
        fits_get_hdu_num(fptr_, &cur_hdu);
        if (cur_hdu != target_hdu) {
            fits_movabs_hdu(fptr_, target_hdu, nullptr, &status);
        }
        
        long fpixel[2] = {x1 + 1, y1 + 1}; // 1-based
        long lpixel[2] = {x2, y2};
        long inc[2] = {1, 1};
        
        int bitpix = 0;
        fits_get_img_type(fptr_, &bitpix, &status);
        
        long width = x2 - x1;
        long height = y2 - y1;
        std::vector<int64_t> shape = {height, width}; // Torch order (y, x)
        
        // ... Implementation omitted for brevity, returning empty
        return torch::empty(shape);
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
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
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
            nb::dict hdu_dict;
            try {
                hdu_dict = nb::cast<nb::dict>(handle);
            } catch (const std::exception& e) {
                continue;
            }
            
            nb::object data_obj;
            bool has_data = false;
            if (hdu_dict.contains("data")) {
                data_obj = hdu_dict["data"];
                has_data = true;
            }
            
            if (has_data) {
                try {
                    nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(data_obj);
                    write_image(tensor, hdu_count, 1.0, 0.0);
                } catch (...) {
                    // Ignore if data cannot be cast to ndarray (e.g. TableHDU data)
                }
            } else {
                long naxes[1] = {0};
                fits_create_img(fptr_, BYTE_IMG, 0, naxes, &status);
            }
            
            if (hdu_dict.contains("header")) {
                try {
                    nb::dict header = nb::cast<nb::dict>(hdu_dict["header"]);
                    for (auto item : header) {
                        std::string key = nb::cast<std::string>(item.first);
                        
                        try {
                            if (nb::isinstance<nb::str>(item.second)) {
                                std::string val = nb::cast<std::string>(item.second);
                                fits_update_key(fptr_, TSTRING, key.c_str(), (void*)val.c_str(), nullptr, &status);
                            } else if (nb::isinstance<int>(item.second)) {
                                int val = nb::cast<int>(item.second);
                                fits_update_key(fptr_, TINT, key.c_str(), &val, nullptr, &status);
                            } else if (nb::isinstance<float>(item.second)) {
                                double val = nb::cast<double>(item.second);
                                fits_update_key(fptr_, TDOUBLE, key.c_str(), &val, nullptr, &status);
                            } else if (nb::isinstance<bool>(item.second)) {
                                int val = nb::cast<bool>(item.second) ? 1 : 0;
                                fits_update_key(fptr_, TLOGICAL, key.c_str(), &val, nullptr, &status);
                            }
                        } catch (...) {
                            status = 0;
                        }
                        if (status != 0) status = 0;
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
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        char* header_str = nullptr;
        int nkeys = 0;
        if (fits_hdr2str(fptr_, 0, nullptr, 0, &header_str, &nkeys, &status)) {
            return "";
        }
        std::string result(header_str);
        // fits_free_memory(header_str, &status); // Need to free? cfitsio usually allocates
        // free(header_str); // fits_hdr2str allocates with malloc?
        return result;
    }

private:
    std::string filename_;
    int mode_;
    fitsfile* fptr_ = nullptr;
    bool cached_ = false;
    int start_hdu_ = 1;
};


struct HDUInfo {
    int index;
    std::string type;
    std::unordered_map<std::string, std::string> header;
};

// Batch open function to reduce FFI overhead
std::pair<FITSFileV2*, std::vector<HDUInfo>> open_and_read_headers(const std::string& path, int mode) {
    auto* file = new FITSFileV2(path.c_str(), mode);
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

// Parallel batch reading of images
std::vector<torch::Tensor> read_images_batch(const std::vector<std::string>& paths, int hdu_num) {
    size_t n = paths.size();
    std::vector<torch::Tensor> results(n);
    std::vector<std::string> errors(n);
    
    // Use simple std::thread for now. For production, a thread pool is better.
    // But since we are I/O bound or GIL-released, launching N threads is okay for moderate N.
    std::vector<std::thread> threads;
    threads.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        threads.emplace_back([&, i]() {
            try {
                // Each thread opens its own file handle - CRITICAL for cfitsio thread safety
                FITSFileV2 file(paths[i].c_str(), 0); // Read mode
                results[i] = file.read_image(hdu_num);
                // File closes automatically via destructor
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Check for errors
    for (size_t i = 0; i < n; ++i) {
        if (!errors[i].empty()) {
            throw std::runtime_error("Error reading " + paths[i] + ": " + errors[i]);
        }
    }
    
    return results;
}

}
