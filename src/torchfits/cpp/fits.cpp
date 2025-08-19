#include <torch/torch.h>
#include <fitsio.h>
#include <omp.h>
#include <unordered_map>
#include <string>
#include <mutex>
#include <chrono>
#include <pybind11/pybind11.h>
#include "hardware.cpp"
#include "cache.cpp"

namespace py = pybind11;

namespace torchfits {

// Hardware info cache
static HardwareInfo hw_info;
static bool hw_detected = false;
static std::mutex hw_mutex;

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
        int status = 0;
        
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to HDU " + std::to_string(hdu_num));
        }

        int hdu_type;
        fits_get_hdu_type(fptr_, &hdu_type, &status);

        if (hdu_type != IMAGE_HDU) {
            throw std::runtime_error("HDU is not an image HDU");
        }

        int naxis, bitpix;
        long naxes[10];
        fits_get_img_param(fptr_, 10, &bitpix, &naxis, naxes, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to get image parameters");
        }

        bool is_compressed = false;
        char keyname[FLEN_KEYWORD];
        int key_status = 0;
        fits_read_keyn(fptr_, 2, keyname, nullptr, nullptr, &key_status);
        if (key_status == 0 && strcmp(keyname, "ZCMPTYPE") == 0) {
            is_compressed = true;
        }

        if (is_compressed) {
            fits_get_img_param(fptr_, 10, &bitpix, &naxis, naxes, &status);
        }
        
        std::vector<int64_t> shape(naxis);
        long total_pixels = 1;
        for (int i = 0; i < naxis; i++) {
            shape[naxis - 1 - i] = naxes[i];
            total_pixels *= naxes[i];
        }
        
        // Check for BSCALE/BZERO scaling once
        double bscale = 1.0, bzero = 0.0;
        int bscale_status = 0, bzero_status = 0;
        fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &bscale_status);
        fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &bzero_status);
        bool has_scaling = (bscale_status == 0 && bscale != 1.0) || (bzero_status == 0 && bzero != 0.0);
        
        // Direct tensor creation - minimal overhead
        switch (bitpix) {
            case BYTE_IMG: {
                auto tensor = torch::empty(shape, has_scaling ? torch::kFloat32 : torch::kUInt8);
                int fits_type = has_scaling ? TFLOAT : TBYTE;
                void* data_ptr = has_scaling ? (void*)tensor.data_ptr<float>() : (void*)tensor.data_ptr<uint8_t>();
                fits_read_img(fptr_, fits_type, 1, total_pixels, nullptr, data_ptr, nullptr, &status);
                return tensor;
            }
            case SHORT_IMG: {
                auto tensor = torch::empty(shape, has_scaling ? torch::kFloat32 : torch::kInt16);
                int fits_type = has_scaling ? TFLOAT : TSHORT;
                void* data_ptr = has_scaling ? (void*)tensor.data_ptr<float>() : (void*)tensor.data_ptr<int16_t>();
                fits_read_img(fptr_, fits_type, 1, total_pixels, nullptr, data_ptr, nullptr, &status);
                return tensor;
            }
            case LONG_IMG: {
                auto tensor = torch::empty(shape, torch::kInt32);
                fits_read_img(fptr_, TINT, 1, total_pixels, nullptr, tensor.data_ptr<int32_t>(), nullptr, &status);
                return tensor;
            }
            case FLOAT_IMG: {
                auto tensor = torch::empty(shape, torch::kFloat32);
                fits_read_img(fptr_, TFLOAT, 1, total_pixels, nullptr, tensor.data_ptr<float>(), nullptr, &status);
                return tensor;
            }
            case DOUBLE_IMG: {
                auto tensor = torch::empty(shape, torch::kFloat64);
                fits_read_img(fptr_, TDOUBLE, 1, total_pixels, nullptr, tensor.data_ptr<double>(), nullptr, &status);
                return tensor;
            }
            default: {
                auto tensor = torch::empty(shape, torch::kFloat32);
                fits_read_img(fptr_, TFLOAT, 1, total_pixels, nullptr, tensor.data_ptr<float>(), nullptr, &status);
                return tensor;
            }
        }
        
        if (status != 0) {
            throw std::runtime_error("Failed to read image data");
        }
    }
    
    torch::Tensor read_subset(int hdu_num, long x1, long y1, long x2, long y2) {
        int status = 0;
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to HDU");
        }
        
        int bitpix;
        fits_get_img_type(fptr_, &bitpix, &status);
        
        long width = x2 - x1;
        long height = y2 - y1;
        long fpixel[2] = {x1 + 1, y1 + 1};
        long lpixel[2] = {x2, y2};
        long inc[2] = {1, 1};
        int anynul;
        
        // Check for scaling
        double bscale = 1.0, bzero = 0.0;
        int bscale_status = 0, bzero_status = 0;
        fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &bscale_status);
        fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &bzero_status);
        bool has_scaling = (bscale_status == 0 && bscale != 1.0) || (bzero_status == 0 && bzero != 0.0);
        
        switch (bitpix) {
            case BYTE_IMG: {
                auto tensor = torch::empty({height, width}, has_scaling ? torch::kFloat32 : torch::kUInt8);
                int fits_type = has_scaling ? TFLOAT : TBYTE;
                void* data_ptr = has_scaling ? (void*)tensor.data_ptr<float>() : (void*)tensor.data_ptr<uint8_t>();
                fits_read_subset(fptr_, fits_type, fpixel, lpixel, inc, nullptr, data_ptr, &anynul, &status);
                return tensor;
            }
            case SHORT_IMG: {
                auto tensor = torch::empty({height, width}, has_scaling ? torch::kFloat32 : torch::kInt16);
                int fits_type = has_scaling ? TFLOAT : TSHORT;
                void* data_ptr = has_scaling ? (void*)tensor.data_ptr<float>() : (void*)tensor.data_ptr<int16_t>();
                fits_read_subset(fptr_, fits_type, fpixel, lpixel, inc, nullptr, data_ptr, &anynul, &status);
                return tensor;
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
    
    std::unordered_map<std::string, std::string> get_header(int hdu_num = 0) {
        int status = 0;
        std::unordered_map<std::string, std::string> header;
        
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) return header;
        
        int nkeys;
        fits_get_hdrspace(fptr_, &nkeys, nullptr, &status);
        
        for (int i = 1; i <= nkeys; i++) {
            char keyname[FLEN_KEYWORD];
            char value[FLEN_VALUE];
            char comment[FLEN_COMMENT];
            
            fits_read_keyn(fptr_, i, keyname, value, comment, &status);
            if (status == 0) {
                header[keyname] = value;
            }
        }
        
        return header;
    }
    
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
    
    void write_hdus(py::list hdus, bool overwrite) {
        int status = 0;

        for (auto hdu_obj : hdus) {
            if (py::isinstance<py::dict>(hdu_obj)) {
                py::dict hdu_dict = hdu_obj.cast<py::dict>();
                if (hdu_dict.contains("data")) {
                    auto tensor = hdu_dict["data"].cast<torch::Tensor>();
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
        
        auto shape = tensor.sizes();
        int naxis = shape.size();
        long* naxes = new long[naxis];
        for (int i = 0; i < naxis; i++) {
            naxes[i] = shape[naxis - 1 - i];
        }

        int bitpix;
        int fits_type;
        void* data_ptr;

        if (tensor.dtype() == torch::kUInt8) {
            bitpix = BYTE_IMG;
            fits_type = TBYTE;
            data_ptr = tensor.data_ptr<uint8_t>();
        } else if (tensor.dtype() == torch::kInt16) {
            bitpix = SHORT_IMG;
            fits_type = TSHORT;
            data_ptr = tensor.data_ptr<int16_t>();
        } else if (tensor.dtype() == torch::kInt32) {
            bitpix = LONG_IMG;
            fits_type = TINT;
            data_ptr = tensor.data_ptr<int32_t>();
        } else if (tensor.dtype() == torch::kFloat32) {
            bitpix = FLOAT_IMG;
            fits_type = TFLOAT;
            data_ptr = tensor.data_ptr<float>();
        } else if (tensor.dtype() == torch::kFloat64) {
            bitpix = DOUBLE_IMG;
            fits_type = TDOUBLE;
            data_ptr = tensor.data_ptr<double>();
        } else {
            throw std::runtime_error("Unsupported tensor data type");
        }

        fits_create_img(fptr_, bitpix, naxis, naxes, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to create image HDU");
        }

        if (bscale != 1.0) {
            fits_write_key(fptr_, TDOUBLE, "BSCALE", &bscale, "", &status);
        }
        if (bzero != 0.0) {
            fits_write_key(fptr_, TDOUBLE, "BZERO", &bzero, "", &status);
        }

        fits_write_img(fptr_, fits_type, 1, tensor.numel(), data_ptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to write image data");
        }

        delete[] naxes;
    }

    fitsfile* get_fptr() { return fptr_; }
    
private:
    std::string filename_;
    int mode_;
    bool cached_ = false;
    fitsfile* fptr_ = nullptr;
};

}