#include <torch/torch.h>
#include <fitsio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <iostream>
#include <iostream>
#include <cstring>
#include "hardware.h"

#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace torchfits {



// Generic fast reader implementation
template<typename InputT, typename OutputT, typename ReaderFunc>
torch::Tensor read_image_mmap_impl(const std::string& filename, int naxis, long* naxes, LONGLONG datastart, ReaderFunc reader) {
    MMapHandle mmap_handle(filename);
    
    long nelements = 0;
    std::vector<int64_t> shape;
    if (naxis > 0) {
        nelements = 1;
        for (int i = naxis - 1; i >= 0; --i) {
            shape.push_back(naxes[i]);
            nelements *= naxes[i];
        }
    }
    
    // Check bounds
    if (datastart + nelements * sizeof(InputT) > mmap_handle.size) {
        throw std::runtime_error("File too small for requested data");
    }
    
    // Map C++ type to Torch scalar type
    torch::ScalarType dtype;
    if (std::is_same<OutputT, float>::value) dtype = torch::kFloat32;
    else if (std::is_same<OutputT, double>::value) dtype = torch::kFloat64;
    else if (std::is_same<OutputT, int32_t>::value) dtype = torch::kInt32;
    else if (std::is_same<OutputT, int16_t>::value) dtype = torch::kInt16;
    else dtype = torch::kFloat32; // Default fallback

    auto options = torch::TensorOptions().dtype(dtype);
    torch::Tensor tensor = torch::empty(shape, options);
    OutputT* out_ptr = tensor.data_ptr<OutputT>();
    
    const InputT* in_ptr = (const InputT*)((char*)mmap_handle.ptr + datastart);
    
    reader(in_ptr, out_ptr, nelements);
    
    return tensor;
}

// Implementations for specific types
torch::Tensor read_image_fast_int16(const std::string& filename, int hdu_num, int naxis, long* naxes, LONGLONG datastart, double bscale, double bzero) {
    return read_image_mmap_impl<int16_t, float>(filename, naxis, naxes, datastart, [&](const int16_t* in, float* out, long n) {
        #ifdef __aarch64__
        // SIMD implementation
        long i = 0;
        float32x4_t vscale = vdupq_n_f32((float)bscale);
        float32x4_t vzero = vdupq_n_f32((float)bzero);
        
        for (; i <= n - 8; i += 8) {
            int16x8_t raw = vld1q_s16(in + i);
            int16x8_t swapped = vreinterpretq_s16_s8(vrev16q_s8(vreinterpretq_s8_s16(raw)));
            int16x4_t low = vget_low_s16(swapped);
            int16x4_t high = vget_high_s16(swapped);
            int32x4_t low32 = vmovl_s16(low);
            int32x4_t high32 = vmovl_s16(high);
            float32x4_t flow = vcvtq_f32_s32(low32);
            float32x4_t fhigh = vcvtq_f32_s32(high32);
            flow = vmlaq_f32(vzero, flow, vscale);
            fhigh = vmlaq_f32(vzero, fhigh, vscale);
            vst1q_f32(out + i, flow);
            vst1q_f32(out + i + 4, fhigh);
        }
        for (; i < n; ++i) {
            int16_t val = bswap_16(in[i]);
            out[i] = val * bscale + bzero;
        }
        #else
        for (long i = 0; i < n; ++i) {
            int16_t val = bswap_16(in[i]);
            out[i] = val * bscale + bzero;
        }
        #endif
    });
}

torch::Tensor read_image_fast_int32(const std::string& filename, int hdu_num, int naxis, long* naxes, LONGLONG datastart, double bscale, double bzero) {
    return read_image_mmap_impl<int32_t, float>(filename, naxis, naxes, datastart, [&](const int32_t* in, float* out, long n) {
        #ifdef __aarch64__
        long i = 0;
        float32x4_t vscale = vdupq_n_f32((float)bscale);
        float32x4_t vzero = vdupq_n_f32((float)bzero);
        for (; i <= n - 4; i += 4) {
            int32x4_t raw = vld1q_s32(in + i);
            int32x4_t swapped = vreinterpretq_s32_s8(vrev32q_s8(vreinterpretq_s8_s32(raw)));
            float32x4_t fval = vcvtq_f32_s32(swapped);
            fval = vmlaq_f32(vzero, fval, vscale);
            vst1q_f32(out + i, fval);
        }
        for (; i < n; ++i) {
            int32_t val = bswap_32(in[i]);
            out[i] = val * bscale + bzero;
        }
        #else
        for (long i = 0; i < n; ++i) {
            int32_t val = bswap_32(in[i]);
            out[i] = val * bscale + bzero;
        }
        #endif
    });
}

torch::Tensor read_image_fast_float32(const std::string& filename, int hdu_num, int naxis, long* naxes, LONGLONG datastart, double bscale, double bzero) {
    return read_image_mmap_impl<int32_t, float>(filename, naxis, naxes, datastart, [&](const int32_t* in, float* out, long n) {
        // Note: we read as int32 then reinterpret bits
        #ifdef __aarch64__
        long i = 0;
        float32x4_t vscale = vdupq_n_f32((float)bscale);
        float32x4_t vzero = vdupq_n_f32((float)bzero);
        for (; i <= n - 4; i += 4) {
            int32x4_t raw = vld1q_s32(in + i);
            int32x4_t swapped = vreinterpretq_s32_s8(vrev32q_s8(vreinterpretq_s8_s32(raw)));
            float32x4_t fval = vreinterpretq_f32_s32(swapped);
            fval = vmlaq_f32(vzero, fval, vscale);
            vst1q_f32(out + i, fval);
        }
        for (; i < n; ++i) {
            int32_t val = bswap_32(in[i]);
            float fval;
            std::memcpy(&fval, &val, 4);
            out[i] = fval * bscale + bzero;
        }
        #else
        for (long i = 0; i < n; ++i) {
            int32_t val = bswap_32(in[i]);
            float fval;
            std::memcpy(&fval, &val, 4);
            out[i] = fval * bscale + bzero;
        }
        #endif
    });
}

torch::Tensor read_image_fast_double(const std::string& filename, int hdu_num, int naxis, long* naxes, LONGLONG datastart, double bscale, double bzero) {
    return read_image_mmap_impl<int64_t, double>(filename, naxis, naxes, datastart, [&](const int64_t* in, double* out, long n) {
        // Read as int64, reinterpret as double
        for (long i = 0; i < n; ++i) {
            int64_t val = bswap_64(in[i]);
            double dval;
            std::memcpy(&dval, &val, 8);
            out[i] = dval * bscale + bzero;
        }
    });
}

torch::Tensor read_image_fast(const std::string& filename, int hdu_num, bool use_mmap) {
    if (!use_mmap) {
        throw std::runtime_error("Standard I/O not implemented in fast path");
    }

    int status = 0;
    fitsfile* fptr = nullptr;
    
    // Use try-catch to ensure fits_close_file is called
    try {
        fits_open_file(&fptr, filename.c_str(), READONLY, &status);
        if (status != 0) {
            throw std::runtime_error("Could not open FITS file: " + filename);
        }
        
        fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("Could not move to HDU");
        
        int hdutype = 0;
        fits_get_hdu_type(fptr, &hdutype, &status);
        if (hdutype != IMAGE_HDU) {
             int is_compressed = fits_is_compressed_image(fptr, &status);
             if (is_compressed) {
                 throw std::runtime_error("Compressed image reading temporarily disabled");
             }
             throw std::runtime_error("HDU is not an image");
        }
        
        int bitpix = 0;
        int naxis = 0;
        long naxes[9] = {0};
        fits_get_img_param(fptr, 9, &bitpix, &naxis, &naxes[0], &status);
        
        double bscale = 1.0;
        double bzero = 0.0;
        fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, nullptr, &status);
        if (status) { status = 0; bscale = 1.0; }
        fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, nullptr, &status);
        if (status) { status = 0; bzero = 0.0; }
        
        LONGLONG headstart, datastart, dataend;
        if (fits_get_hduaddrll(fptr, &headstart, &datastart, &dataend, &status)) {
            throw std::runtime_error("Failed to get HDU address");
        }
        
        fits_close_file(fptr, &status);
        fptr = nullptr; // Prevent double close
        
        if (bitpix == 16) return read_image_fast_int16(filename, hdu_num, naxis, naxes, datastart, bscale, bzero);
        else if (bitpix == -32) return read_image_fast_float32(filename, hdu_num, naxis, naxes, datastart, bscale, bzero);
        else if (bitpix == 32) return read_image_fast_int32(filename, hdu_num, naxis, naxes, datastart, bscale, bzero);
        else if (bitpix == -64) return read_image_fast_double(filename, hdu_num, naxis, naxes, datastart, bscale, bzero);
        else throw std::runtime_error("Unsupported BITPIX: " + std::to_string(bitpix));
        
    } catch (...) {
        if (fptr) fits_close_file(fptr, &status);
        throw;
    }
}

}
