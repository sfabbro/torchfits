#include <torch/torch.h>
#include <fitsio.h>
#include <omp.h>
#include <unordered_map>
#include <string>
#include <mutex>
#include <chrono>
#include <memory>
#include <nanobind/nanobind.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// Architecture-specific SIMD headers
#ifdef __x86_64__
#include <immintrin.h>  // x86 SIMD intrinsics
#elif defined(__aarch64__)
#include <arm_neon.h>   // ARM NEON intrinsics
#endif

#include "hardware.h"
#include "cache.cpp"

namespace nb = nanobind;

namespace torchfits {
// --- Memory-mapped tensor logic (from mmap_tensor.cpp) ---
class MMapTensorManager {
public:
    struct MMapInfo {
        void* addr = nullptr;
        size_t size = 0;
        int fd = -1;
        std::string filepath;
        size_t data_offset = 0;
    };
    static torch::Tensor create_mmap_tensor(const std::string& filepath, 
                                          const std::vector<int64_t>& shape,
                                          torch::ScalarType dtype,
                                          size_t data_offset = 0) {
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file for memory mapping: " + filepath);
        }
        struct stat st;
        if (fstat(fd, &st) == -1) {
            close(fd);
            throw std::runtime_error("Failed to get file size");
        }
        size_t element_size = torch::elementSize(dtype);
        size_t tensor_elements = 1;
        for (auto dim : shape) tensor_elements *= dim;
        size_t tensor_size = tensor_elements * element_size;
        if (data_offset + tensor_size > (size_t)st.st_size) {
            close(fd);
            throw std::runtime_error("File too small for requested tensor size");
        }
        void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to memory map file");
        }
        auto mmap_info = std::make_shared<MMapInfo>();
        mmap_info->addr = addr;
        mmap_info->size = st.st_size;
        mmap_info->fd = fd;
        mmap_info->filepath = filepath;
        mmap_info->data_offset = data_offset;
        void* tensor_data = static_cast<uint8_t*>(addr) + data_offset;
        auto deleter = [mmap_info](void*) {
            if (mmap_info->addr != nullptr) munmap(mmap_info->addr, mmap_info->size);
            if (mmap_info->fd != -1) close(mmap_info->fd);
        };
        auto tensor = torch::from_blob(tensor_data, shape, deleter, torch::TensorOptions().dtype(dtype));
        return tensor;
    }
    static bool should_use_mmap(size_t file_size, size_t tensor_size) {
        const size_t MMAP_THRESHOLD = 64 * 1024 * 1024; // 64MB
        return file_size > MMAP_THRESHOLD || (tensor_size < file_size / 4 && file_size > MMAP_THRESHOLD / 4);
    }
};

class FITSMMapReader {
public:
    static torch::Tensor read_fits_mmap(const std::string& filepath, int hdu_num = 0) {
        fitsfile* fptr = nullptr;
        int status = 0;
        fits_open_file(&fptr, filepath.c_str(), READONLY, &status);
        if (status != 0) throw std::runtime_error("Failed to open FITS file");
        fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
        if (status != 0) { fits_close_file(fptr, &status); throw std::runtime_error("Failed to move to HDU"); }
        int naxis, bitpix; long naxes[10];
        fits_get_img_param(fptr, 10, &bitpix, &naxis, naxes, &status);
        if (status != 0) { fits_close_file(fptr, &status); throw std::runtime_error("Failed to get image parameters"); }
        long data_offset; fits_get_hduaddr(fptr, &data_offset, nullptr, nullptr, &status);
        fits_close_file(fptr, &status);
        std::vector<int64_t> shape(naxis);
        for (int i = 0; i < naxis; i++) shape[naxis - 1 - i] = naxes[i];
        torch::ScalarType dtype;
        switch (bitpix) {
            case 8: dtype = torch::kUInt8; break;
            case 16: dtype = torch::kInt16; break;
            case 32: dtype = torch::kInt32; break;
            case -32: dtype = torch::kFloat32; break;
            case -64: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32;
        }
        return MMapTensorManager::create_mmap_tensor(filepath, shape, dtype, data_offset);
    }
    static bool is_mmap_suitable(const std::string& filepath, int hdu_num = 0) {
        fitsfile* fptr = nullptr;
        int status = 0;
        fits_open_file(&fptr, filepath.c_str(), READONLY, &status);
        if (status != 0) return false;
        fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
        if (status != 0) { fits_close_file(fptr, &status); return false; }
        char zcmptype[FLEN_VALUE]; int comp_status = 0;
        fits_read_key(fptr, TSTRING, "ZCMPTYPE", zcmptype, nullptr, &comp_status);
        double bscale = 1.0, bzero = 0.0; int scale_status = 0;
        fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, nullptr, &scale_status);
        fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, nullptr, &scale_status);
        bool has_scaling = (bscale != 1.0) || (bzero != 0.0);
        fits_close_file(fptr, &status);
        return (comp_status != 0) && !has_scaling;
    }
};


class FITSFile {
public:
    FITSFile(const std::string& filename, int mode = 0) : filename_(filename), mode_(mode) {
        if (mode == 0) {
            fptr_ = global_cache.get_or_open(filename);
            if (fptr_) {
                cached_ = true;
                return;
            }
        }
        
        int status = 0;
        if (mode == 0) {
            fits_open_file(&fptr_, filename.c_str(), READONLY, &status);
        } else {
            fits_create_file(&fptr_, filename.c_str(), &status);
        }
        
        if (status != 0) {
            throw std::runtime_error("Failed to open FITS file: " + filename);
        }
        cached_ = false;
    }
    
    ~FITSFile() {
        if (fptr_ && !cached_ && mode_ != 0) {
            int status = 0;
            fits_close_file(fptr_, &status);
        }
    }
    
    torch::Tensor read_image(int hdu_num = 0, bool use_mmap = false) {
        return read_image_with_device(hdu_num, use_mmap, torch::kCPU);
    }
    
    torch::Tensor read_image_with_device(int hdu_num, bool use_mmap, torch::Device device) {
        int status = 0;
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("Failed to move to HDU " + std::to_string(hdu_num));
        int hdu_type;
        fits_get_hdu_type(fptr_, &hdu_type, &status);
        if (hdu_type != IMAGE_HDU) throw std::runtime_error("HDU is not an image HDU");
        int naxis, bitpix; long naxes[10];
        fits_get_img_param(fptr_, 10, &bitpix, &naxis, naxes, &status);
        if (status != 0) throw std::runtime_error("Failed to get image parameters");

        // Simplified compression check - single path
        int compression_type = 0;
        fits_get_compression_type(fptr_, &compression_type, &status);
        bool is_compressed = (status == 0 && compression_type != NOCOMPRESS);

        std::vector<int64_t> shape(naxis);
        long total_pixels = 1;
        for (int i = 0; i < naxis; i++) { shape[naxis - 1 - i] = naxes[i]; total_pixels *= naxes[i]; }

        // Single scaling check
        double bscale = 1.0, bzero = 0.0;
        fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &status);
        status = 0; // Reset status
        fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &status);
        bool has_scaling = (bscale != 1.0 || bzero != 0.0);

        // Fast path: mmap for uncompressed, unscaled data
        if (use_mmap && !is_compressed && !has_scaling && device.is_cpu()) {
            return FITSMMapReader::read_fits_mmap(filename_, hdu_num);
        }

        // Single optimized read path for all cases
        return read_image_fast(bitpix, shape, total_pixels, has_scaling, bscale, bzero, device);
    }
    
private:
    // Ultra-fast single-path read function - no branching overhead
    torch::Tensor read_image_fast(int bitpix, const std::vector<int64_t>& shape,
                                   long total_pixels, bool has_scaling,
                                   double bscale, double bzero, torch::Device device) {
        int status = 0;

        // For GPU: read to CPU first, then transfer
        if (!device.is_cpu()) {
            auto cpu_tensor = read_image_fast(bitpix, shape, total_pixels, has_scaling, bscale, bzero, torch::kCPU);
            return cpu_tensor.to(device, /*non_blocking=*/true);
        }

        // Fast CPU path - direct allocation and read
        switch (bitpix) {
            case BYTE_IMG: {
                if (has_scaling) {
                    auto tensor = torch::empty(shape, torch::kFloat32);
                    fits_read_img(fptr_, TFLOAT, 1, total_pixels, nullptr,
                                 tensor.data_ptr<float>(), nullptr, &status);
                    if (status != 0) throw std::runtime_error("Failed to read image data");
                    return tensor;
                } else {
                    auto tensor = torch::empty(shape, torch::kUInt8);
                    fits_read_img(fptr_, TBYTE, 1, total_pixels, nullptr,
                                 tensor.data_ptr<uint8_t>(), nullptr, &status);
                    if (status != 0) throw std::runtime_error("Failed to read image data");
                    return tensor;
                }
            }
            case SHORT_IMG: {
                if (has_scaling) {
                    auto tensor = torch::empty(shape, torch::kFloat32);
                    fits_read_img(fptr_, TFLOAT, 1, total_pixels, nullptr,
                                 tensor.data_ptr<float>(), nullptr, &status);
                    if (status != 0) throw std::runtime_error("Failed to read image data");
                    return tensor;
                } else {
                    auto tensor = torch::empty(shape, torch::kInt16);
                    fits_read_img(fptr_, TSHORT, 1, total_pixels, nullptr,
                                 tensor.data_ptr<int16_t>(), nullptr, &status);
                    if (status != 0) throw std::runtime_error("Failed to read image data");
                    return tensor;
                }
            }
            case LONG_IMG: {
                auto tensor = torch::empty(shape, torch::kInt32);
                fits_read_img(fptr_, TINT, 1, total_pixels, nullptr,
                             tensor.data_ptr<int32_t>(), nullptr, &status);
                if (status != 0) throw std::runtime_error("Failed to read image data");
                return tensor;
            }
            case FLOAT_IMG: {
                auto tensor = torch::empty(shape, torch::kFloat32);
                fits_read_img(fptr_, TFLOAT, 1, total_pixels, nullptr,
                             tensor.data_ptr<float>(), nullptr, &status);
                if (status != 0) throw std::runtime_error("Failed to read image data");
                return tensor;
            }
            case DOUBLE_IMG: {
                auto tensor = torch::empty(shape, torch::kFloat64);
                fits_read_img(fptr_, TDOUBLE, 1, total_pixels, nullptr,
                             tensor.data_ptr<double>(), nullptr, &status);
                if (status != 0) throw std::runtime_error("Failed to read image data");
                return tensor;
            }
            default: {
                auto tensor = torch::empty(shape, torch::kFloat32);
                fits_read_img(fptr_, TFLOAT, 1, total_pixels, nullptr,
                             tensor.data_ptr<float>(), nullptr, &status);
                if (status != 0) throw std::runtime_error("Failed to read image data");
                return tensor;
            }
        }
    }
    
    
    torch::Tensor read_compressed_subset(int bitpix, long* fpixel, long* lpixel, long* inc,
                                        long width, long height, bool has_scaling, int compression_type = 0) {
        int status = 0;
        int anynul;
        
        // Get tile dimensions for optimal reading strategy
        long tile_dims[2] = {0, 0};
        fits_get_tile_dim(fptr_, 2, tile_dims, &status);
        
        // For compressed images, use the standard fits_read_subset which handles compression automatically
        // CFITSIO automatically decompresses tiles as needed, but with tile information we can optimize
        // the reading strategy for better performance on small cutouts
        
        // Log compression info for debugging (only in debug builds)
        #ifdef DEBUG
        printf("Compressed cutout read: type=%d, tiles=[%ld,%ld], region=[%ld,%ld,%ld,%ld]\n", 
               compression_type, tile_dims[0], tile_dims[1], *fpixel, *(fpixel+1), *lpixel, *(lpixel+1));
        #endif
        
        switch (bitpix) {
            case BYTE_IMG: {
                auto options = torch::TensorOptions()
                    .dtype(has_scaling ? torch::kFloat32 : torch::kUInt8)
                    .memory_format(torch::MemoryFormat::Contiguous);
                auto tensor = torch::empty({height, width}, options);
                int fits_type = has_scaling ? TFLOAT : TBYTE;
                void* data_ptr = has_scaling ? (void*)tensor.data_ptr<float>() : (void*)tensor.data_ptr<uint8_t>();
                
                // CFITSIO handles tile decompression automatically in fits_read_subset
                fits_read_subset(fptr_, fits_type, fpixel, lpixel, inc, 
                               nullptr, data_ptr, &anynul, &status);
                if (status != 0) throw std::runtime_error("Failed to read compressed subset");
                return tensor;
            }
            case SHORT_IMG: {
                auto options = torch::TensorOptions()
                    .dtype(has_scaling ? torch::kFloat32 : torch::kInt16)
                    .memory_format(torch::MemoryFormat::Contiguous);
                auto tensor = torch::empty({height, width}, options);
                int fits_type = has_scaling ? TFLOAT : TSHORT;
                void* data_ptr = has_scaling ? (void*)tensor.data_ptr<float>() : (void*)tensor.data_ptr<int16_t>();
                
                fits_read_subset(fptr_, fits_type, fpixel, lpixel, inc,
                               nullptr, data_ptr, &anynul, &status);
                if (status != 0) throw std::runtime_error("Failed to read compressed subset");
                return tensor;
            }
            case FLOAT_IMG: {
                auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .memory_format(torch::MemoryFormat::Contiguous);
                auto tensor = torch::empty({height, width}, options);
                
                fits_read_subset(fptr_, TFLOAT, fpixel, lpixel, inc,
                               nullptr, tensor.data_ptr<float>(), &anynul, &status);
                if (status != 0) throw std::runtime_error("Failed to read compressed subset");
                return tensor;
            }
            case DOUBLE_IMG: {
                auto options = torch::TensorOptions()
                    .dtype(torch::kFloat64)
                    .memory_format(torch::MemoryFormat::Contiguous);
                auto tensor = torch::empty({height, width}, options);
                
                fits_read_subset(fptr_, TDOUBLE, fpixel, lpixel, inc,
                               nullptr, tensor.data_ptr<double>(), &anynul, &status);
                if (status != 0) throw std::runtime_error("Failed to read compressed subset");
                return tensor;
            }
            default: {
                auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .memory_format(torch::MemoryFormat::Contiguous);
                auto tensor = torch::empty({height, width}, options);
                
                fits_read_subset(fptr_, TFLOAT, fpixel, lpixel, inc,
                               nullptr, tensor.data_ptr<float>(), &anynul, &status);
                if (status != 0) throw std::runtime_error("Failed to read compressed subset");
                return tensor;
            }
        }
    }
    


public:
    
    torch::Tensor read_subset(int hdu_num, long x1, long y1, long x2, long y2) {
        int status = 0;
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to HDU");
        }
        
        int bitpix;
        fits_get_img_type(fptr_, &bitpix, &status);
        
        // Check for compression - optimize for tiled data
        bool is_compressed = false;
        char zcmptype[FLEN_VALUE]; int comp_status = 0;
        fits_read_key(fptr_, TSTRING, "ZCMPTYPE", zcmptype, nullptr, &comp_status);
        if (comp_status == 0) is_compressed = true;
        
        // FITS uses 1-based inclusive ranges, but we want Python-style exclusive ranges
        long width = x2 - x1;
        long height = y2 - y1;
        long fpixel[2] = {x1 + 1, y1 + 1};
        long lpixel[2] = {x2, y2};  // x2, y2 are exclusive in Python, so this is correct
        long inc[2] = {1, 1};
        int anynul;
        
        // Check for scaling
        double bscale = 1.0, bzero = 0.0;
        int bscale_status = 0, bzero_status = 0;
        fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &bscale_status);
        fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &bzero_status);
        bool has_scaling = (bscale_status == 0 && bscale != 1.0) || (bzero_status == 0 && bzero != 0.0);
        
        // Optimized path for compressed images - use tile-aware reading
        if (is_compressed) {
            // Get compression type for optimized handling
            int comp_type = 0;
            fits_get_compression_type(fptr_, &comp_type, &status);
            return read_compressed_subset(bitpix, fpixel, lpixel, inc, width, height, has_scaling, comp_type);
        }
        
        // Fast uncompressed reading - direct allocation
        switch (bitpix) {
            case BYTE_IMG: {
                if (has_scaling) {
                    auto tensor = torch::empty({height, width}, torch::kFloat32);
                    fits_read_subset(fptr_, TFLOAT, fpixel, lpixel, inc, nullptr, tensor.data_ptr<float>(), &anynul, &status);
                    return tensor;
                } else {
                    auto tensor = torch::empty({height, width}, torch::kUInt8);
                    fits_read_subset(fptr_, TBYTE, fpixel, lpixel, inc, nullptr, tensor.data_ptr<uint8_t>(), &anynul, &status);
                    return tensor;
                }
            }
            case SHORT_IMG: {
                if (has_scaling) {
                    auto tensor = torch::empty({height, width}, torch::kFloat32);
                    fits_read_subset(fptr_, TFLOAT, fpixel, lpixel, inc, nullptr, tensor.data_ptr<float>(), &anynul, &status);
                    return tensor;
                } else {
                    auto tensor = torch::empty({height, width}, torch::kInt16);
                    fits_read_subset(fptr_, TSHORT, fpixel, lpixel, inc, nullptr, tensor.data_ptr<int16_t>(), &anynul, &status);
                    return tensor;
                }
            }
            case LONG_IMG: {
                auto tensor = torch::empty({height, width}, torch::kInt32);
                fits_read_subset(fptr_, TINT, fpixel, lpixel, inc, nullptr, tensor.data_ptr<int32_t>(), &anynul, &status);
                return tensor;
            }
            case FLOAT_IMG: {
                auto tensor = torch::empty({height, width}, torch::kFloat32);
                fits_read_subset(fptr_, TFLOAT, fpixel, lpixel, inc, nullptr, tensor.data_ptr<float>(), &anynul, &status);
                return tensor;
            }
            case DOUBLE_IMG: {
                auto tensor = torch::empty({height, width}, torch::kFloat64);
                fits_read_subset(fptr_, TDOUBLE, fpixel, lpixel, inc, nullptr, tensor.data_ptr<double>(), &anynul, &status);
                return tensor;
            }
            default: {
                auto tensor = torch::empty({height, width}, torch::kFloat32);
                fits_read_subset(fptr_, TFLOAT, fpixel, lpixel, inc, nullptr, tensor.data_ptr<float>(), &anynul, &status);
                return tensor;
            }
        }
        
        if (status != 0) {
            throw std::runtime_error("Failed to read subset");
        }
    }
    
    // Fast bulk header reading using fits_hdr2str - implements OPTIMIZE.md Task #5
    std::string read_header_to_string(int hdu_num = 0) {
        int status = 0;
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) return "";
        
        char* header_str = nullptr;
        int nkeys = 0;
        
        // Use CFITSIO's bulk header dump function - much faster than keyword-by-keyword
        fits_hdr2str(fptr_, 0, nullptr, 0, &header_str, &nkeys, &status);
        if (status != 0 || !header_str) {
            if (header_str) free(header_str);
            return "";
        }
        
        std::string result(header_str);
        free(header_str);
        return result;
    }
    
    std::unordered_map<std::string, std::string> get_header(int hdu_num = 0) {
        int status = 0;
        std::unordered_map<std::string, std::string> header;

        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) return header;

        // Fast bulk header reading only - no fallback
        std::string header_str = read_header_to_string(hdu_num);
        if (!header_str.empty()) {
            header = parse_header_string(header_str);
        }

        return header;
    }
    
private:
    // Fast C++ header string parser - processes entire header block at once
    std::unordered_map<std::string, std::string> parse_header_string(const std::string& header_str) {
        std::unordered_map<std::string, std::string> header;
        
        // Split header string into 80-character FITS cards
        size_t pos = 0;
        while (pos < header_str.length()) {
            // Extract one 80-character FITS card
            std::string card = header_str.substr(pos, 80);
            pos += 80;
            
            // Skip END cards and empty cards
            if (card.substr(0, 3) == "END" || card.find_first_not_of(' ') == std::string::npos) {
                continue;
            }
            
            // Parse keyword = value
            size_t equals_pos = card.find('=');
            if (equals_pos != std::string::npos && equals_pos < 8) {  // Keyword must be in first 8 chars
                std::string keyword = card.substr(0, equals_pos);
                std::string value_part = card.substr(equals_pos + 1);
                
                // Trim whitespace from keyword
                keyword.erase(keyword.find_last_not_of(' ') + 1);
                
                // Extract value (handle quotes and comments)
                size_t value_start = value_part.find_first_not_of(' ');
                if (value_start != std::string::npos) {
                    std::string value;
                    
                    if (value_part[value_start] == '\'') {
                        // String value - find closing quote
                        size_t quote_end = value_part.find('\'', value_start + 1);
                        if (quote_end != std::string::npos) {
                            value = value_part.substr(value_start + 1, quote_end - value_start - 1);
                        }
                    } else {
                        // Numeric or logical value - read until comment or end
                        size_t value_end = value_part.find('/', value_start);
                        if (value_end == std::string::npos) value_end = value_part.length();
                        value = value_part.substr(value_start, value_end - value_start);
                        
                        // Trim trailing whitespace
                        value.erase(value.find_last_not_of(' ') + 1);
                    }
                    
                    if (!keyword.empty()) {
                        header[keyword] = value;
                    }
                }
            }
        }
        
        return header;
    }

public:
    
    std::vector<int64_t> get_shape(int hdu_num = 0) {
        int status = 0;
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        
        int naxis;
        long naxes[10];
        fits_get_img_param(fptr_, 10, nullptr, &naxis, naxes, &status);

        bool is_compressed = false;
        char keyname[FLEN_KEYWORD];
        int key_status = 0;
        fits_read_keyn(fptr_, 2, keyname, nullptr, nullptr, &key_status);
        if (key_status == 0 && strcmp(keyname, "ZCMPTYPE") == 0) {
            is_compressed = true;
        }

        if (is_compressed) {
            fits_get_img_param(fptr_, 10, nullptr, &naxis, naxes, &status);
        }
        
        std::vector<int64_t> shape(naxis);
        for (int i = 0; i < naxis; i++) {
            shape[naxis - 1 - i] = naxes[i];
        }
        return shape;
    }
    
    torch::Dtype get_dtype(int hdu_num = 0) {
        return torch::kFloat32;
    }
    
    std::unordered_map<std::string, double> compute_stats(int hdu_num = 0) {
        auto tensor = read_image(hdu_num);
        return {
            {"mean", tensor.mean().item<double>()},
            {"std", tensor.std().item<double>()},
            {"min", tensor.min().item<double>()},
            {"max", tensor.max().item<double>()}
        };
    }
    
    int get_num_hdus() {
        int num_hdus = 0;
        int status = 0;
        fits_get_num_hdus(fptr_, &num_hdus, &status);
        return num_hdus;
    }
    
    std::string get_hdu_type(int hdu_num) {
        int status = 0;
        int hdu_type;
        fits_movabs_hdu(fptr_, hdu_num + 1, &hdu_type, &status);
        return (hdu_type == IMAGE_HDU) ? "IMAGE" : "TABLE";
    }
    
    void write_hdus(nb::list hdus, bool overwrite) {
        int status = 0;

        for (auto hdu_obj : hdus) {
            if (nb::isinstance<nb::dict>(hdu_obj)) {
                nb::dict hdu_dict = nb::cast<nb::dict>(hdu_obj);
                if (hdu_dict.contains("data")) {
                    auto tensor = nb::cast<torch::Tensor>(hdu_dict["data"]);
                    write_image(tensor);
                } else {
                    // TODO: Implement table writing
                    throw std::runtime_error("Table writing not implemented");
                }
            } else {
                throw std::runtime_error("Unsupported HDU type");
            }
        }

        fits_close_file(fptr_, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to close FITS file");
        }
    }
    
    void write_image(const torch::Tensor& tensor, int hdu_num = 0, 
                    double bscale = 1.0, double bzero = 0.0) {
        int status = 0;
        
        // Ensure tensor is contiguous for direct memory access
        auto contiguous_tensor = tensor.contiguous();
        auto shape = contiguous_tensor.sizes();
        int naxis = shape.size();
        
        if (naxis == 0) {
            throw std::runtime_error("Cannot write scalar tensor as FITS image");
        }
        
        // FITS uses reversed axis order (FORTRAN-style)
        long* naxes = new long[naxis];
        for (int i = 0; i < naxis; i++) {
            naxes[i] = shape[naxis - 1 - i];
        }
        
        int bitpix;
        int fits_type;
        void* data_ptr;
        
        // Map PyTorch dtypes to FITS types with proper BITPIX values
        switch (contiguous_tensor.scalar_type()) {
            case torch::kUInt8:
                bitpix = BYTE_IMG;
                fits_type = TBYTE;
                data_ptr = contiguous_tensor.data_ptr<uint8_t>();
                break;
            case torch::kInt16:
                bitpix = SHORT_IMG;
                fits_type = TSHORT;
                data_ptr = contiguous_tensor.data_ptr<int16_t>();
                break;
            case torch::kInt32:
                bitpix = LONG_IMG;
                fits_type = TINT;
                data_ptr = contiguous_tensor.data_ptr<int32_t>();
                break;
            case torch::kFloat32:
                bitpix = FLOAT_IMG;
                fits_type = TFLOAT;
                data_ptr = contiguous_tensor.data_ptr<float>();
                break;
            case torch::kFloat64:
                bitpix = DOUBLE_IMG;
                fits_type = TDOUBLE;
                data_ptr = contiguous_tensor.data_ptr<double>();
                break;
            default:
                delete[] naxes;
                throw std::runtime_error("Unsupported tensor dtype for FITS writing");
        }
        
        // Create image HDU
        fits_create_img(fptr_, bitpix, naxis, naxes, &status);
        if (status != 0) {
            delete[] naxes;
            char error_text[FLEN_ERRMSG];
            fits_get_errstatus(status, error_text);
            throw std::runtime_error(std::string("Failed to create image HDU: ") + error_text);
        }
        
        // Write scaling keywords if provided
        if (bscale != 1.0) {
            fits_write_key(fptr_, TDOUBLE, "BSCALE", &bscale, "Linear scaling factor", &status);
        }
        if (bzero != 0.0) {
            fits_write_key(fptr_, TDOUBLE, "BZERO", &bzero, "Zero point offset", &status);
        }
        
        // Calculate total number of pixels
        long total_pixels = 1;
        for (int i = 0; i < naxis; i++) {
            total_pixels *= naxes[i];
        }
        
        // Write image data directly from tensor memory (zero-copy)
        fits_write_img(fptr_, fits_type, 1, total_pixels, data_ptr, &status);
        
        delete[] naxes;
        
        if (status != 0) {
            char error_text[FLEN_ERRMSG];
            fits_get_errstatus(status, error_text);
            throw std::runtime_error(std::string("Failed to write image data: ") + error_text);
        }
    }

    fitsfile* get_fptr() { return fptr_; }
    
private:
    std::string filename_;
    int mode_;
    bool cached_ = false;
    fitsfile* fptr_ = nullptr;
};

}