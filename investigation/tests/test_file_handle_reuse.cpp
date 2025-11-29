/**
 * Test if file handle reuse reduces the int16 overhead
 *
 * Compare: open-read-close each time vs open once, read many times
 */
#include <torch/torch.h>
#include <fitsio.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std::chrono;

// Strategy 1: Open/close each time (what our code does)
double benchmark_with_reopen(const char* filename, torch::ScalarType torch_dtype, int cfitsio_dtype, int iterations) {
    std::vector<double> times;

    for (int iter = 0; iter < iterations; iter++) {
        auto start = high_resolution_clock::now();

        fitsfile* fptr = nullptr;
        int status = 0;
        fits_open_file(&fptr, filename, READONLY, &status);
        fits_movabs_hdu(fptr, 1, nullptr, &status);

        int naxis, bitpix;
        long long naxes[10];
        fits_get_img_paramll(fptr, 10, &bitpix, &naxis, naxes, &status);

        std::vector<int64_t> shape = {naxes[1], naxes[0]};
        long long total_pixels = naxes[0] * naxes[1];

        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch_dtype));
        fits_read_img(fptr, cfitsio_dtype, 1, total_pixels, nullptr, tensor.data_ptr(), nullptr, &status);
        fits_close_file(fptr, &status);

        auto end = high_resolution_clock::now();
        times.push_back(duration<double, std::milli>(end - start).count());
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

// Strategy 2: Open once, read many times (what fitsio might do with caching)
double benchmark_with_reuse(const char* filename, torch::ScalarType torch_dtype, int cfitsio_dtype, int iterations) {
    // Open file once
    fitsfile* fptr = nullptr;
    int status = 0;
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status != 0) return -1.0;

    fits_movabs_hdu(fptr, 1, nullptr, &status);
    int naxis, bitpix;
    long long naxes[10];
    fits_get_img_paramll(fptr, 10, &bitpix, &naxis, naxes, &status);

    std::vector<int64_t> shape = {naxes[1], naxes[0]};
    long long total_pixels = naxes[0] * naxes[1];

    std::vector<double> times;

    for (int iter = 0; iter < iterations; iter++) {
        auto start = high_resolution_clock::now();

        // Just allocate tensor and read
        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch_dtype));
        fits_read_img(fptr, cfitsio_dtype, 1, total_pixels, nullptr, tensor.data_ptr(), nullptr, &status);

        auto end = high_resolution_clock::now();
        times.push_back(duration<double, std::milli>(end - start).count());

        // Rewind to start of image for next read
        fits_movabs_hdu(fptr, 1, nullptr, &status);
    }

    fits_close_file(fptr, &status);

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <uint8_file> <int16_file>" << std::endl;
        return 1;
    }

    const char* uint8_file = argv[1];
    const char* int16_file = argv[2];
    int iterations = 20;

    std::cout << "========================================" << std::endl;
    std::cout << "File Handle Reuse Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Strategy 1: Open/close each time" << std::endl;
    std::cout << "--------------------" << std::endl;
    auto uint8_reopen = benchmark_with_reopen(uint8_file, torch::kUInt8, TBYTE, iterations);
    auto int16_reopen = benchmark_with_reopen(int16_file, torch::kInt16, TSHORT, iterations);
    std::cout << "  uint8:   " << uint8_reopen << "ms" << std::endl;
    std::cout << "  int16:   " << int16_reopen << "ms" << std::endl;
    std::cout << "  Ratio:   " << (int16_reopen / uint8_reopen) << "x" << std::endl;
    std::cout << std::endl;

    std::cout << "Strategy 2: Reuse file handle" << std::endl;
    std::cout << "--------------------" << std::endl;
    auto uint8_reuse = benchmark_with_reuse(uint8_file, torch::kUInt8, TBYTE, iterations);
    auto int16_reuse = benchmark_with_reuse(int16_file, torch::kInt16, TSHORT, iterations);
    std::cout << "  uint8:   " << uint8_reuse << "ms" << std::endl;
    std::cout << "  int16:   " << int16_reuse << "ms" << std::endl;
    std::cout << "  Ratio:   " << (int16_reuse / uint8_reuse) << "x" << std::endl;
    std::cout << std::endl;

    std::cout << "========================================" << std::endl;
    std::cout << "Analysis" << std::endl;
    std::cout << "========================================" << std::endl;

    double uint8_savings = uint8_reopen - uint8_reuse;
    double int16_savings = int16_reopen - int16_reuse;

    std::cout << "Time saved by reusing handle:" << std::endl;
    std::cout << "  uint8:   " << uint8_savings << "ms" << std::endl;
    std::cout << "  int16:   " << int16_savings << "ms" << std::endl;
    std::cout << std::endl;

    if (int16_savings > uint8_savings * 1.5) {
        std::cout << "✅ File handle reuse helps int16 MORE than uint8!" << std::endl;
        std::cout << "   This could explain why fitsio is relatively faster on int16." << std::endl;
    } else {
        std::cout << "❌ File handle reuse doesn't explain the int16 gap." << std::endl;
    }

    return 0;
}
