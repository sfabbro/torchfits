/**
 * Direct CFITSIO benchmark - NO PyTorch
 *
 * This tests pure CFITSIO performance for uint8 vs int16
 * to isolate whether the slowdown is in CFITSIO or torch::empty()
 */
#include <fitsio.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstdint>

using namespace std::chrono;

double benchmark_read(const char* filename, int datatype, void* buffer, size_t pixels) {
    fitsfile* fptr = nullptr;
    int status = 0;

    auto start = high_resolution_clock::now();

    // Open file
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status != 0) {
        std::cerr << "Failed to open file, status=" << status << std::endl;
        return -1.0;
    }

    // Move to HDU (already at primary, but matches our code)
    fits_movabs_hdu(fptr, 1, nullptr, &status);
    if (status != 0) {
        std::cerr << "Failed to move HDU, status=" << status << std::endl;
        return -1.0;
    }

    // Get parameters
    int naxis, bitpix;
    long long naxes[10];
    fits_get_img_paramll(fptr, 10, &bitpix, &naxis, naxes, &status);
    if (status != 0) {
        std::cerr << "Failed to get params, status=" << status << std::endl;
        return -1.0;
    }

    // Read image
    fits_read_img(fptr, datatype, 1, pixels, nullptr, buffer, nullptr, &status);
    if (status != 0) {
        std::cerr << "Failed to read image, status=" << status << std::endl;
        return -1.0;
    }

    // Close
    fits_close_file(fptr, &status);

    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <uint8_file> <int16_file>" << std::endl;
        return 1;
    }

    const char* uint8_file = argv[1];
    const char* int16_file = argv[2];
    const size_t pixels = 1000 * 1000;

    std::cout << "========================================" << std::endl;
    std::cout << "Pure CFITSIO Benchmark (No PyTorch)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Allocate buffers
    std::vector<uint8_t> uint8_buffer(pixels);
    std::vector<int16_t> int16_buffer(pixels);

    // Benchmark uint8
    std::cout << "Benchmarking uint8 (10 iterations)..." << std::endl;
    std::vector<double> uint8_times;
    for (int i = 0; i < 10; i++) {
        double t = benchmark_read(uint8_file, TBYTE, uint8_buffer.data(), pixels);
        if (t > 0) {
            uint8_times.push_back(t);
            if (i % 2 == 0) std::cout << "  Run " << i << ": " << t << "ms" << std::endl;
        }
    }

    std::cout << std::endl;

    // Benchmark int16
    std::cout << "Benchmarking int16 (10 iterations)..." << std::endl;
    std::vector<double> int16_times;
    for (int i = 0; i < 10; i++) {
        double t = benchmark_read(int16_file, TSHORT, int16_buffer.data(), pixels);
        if (t > 0) {
            int16_times.push_back(t);
            if (i % 2 == 0) std::cout << "  Run " << i << ": " << t << "ms" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << "========================================" << std::endl;

    // Compute medians
    std::sort(uint8_times.begin(), uint8_times.end());
    std::sort(int16_times.begin(), int16_times.end());

    double uint8_median = uint8_times[uint8_times.size() / 2];
    double int16_median = int16_times[int16_times.size() / 2];

    std::cout << "uint8  median: " << uint8_median << "ms" << std::endl;
    std::cout << "int16  median: " << int16_median << "ms" << std::endl;
    std::cout << "Ratio (int16/uint8): " << (int16_median / uint8_median) << "x" << std::endl;
    std::cout << std::endl;

    double diff = int16_median - uint8_median;
    std::cout << "int16 is " << diff << "ms slower than uint8" << std::endl;

    return 0;
}
