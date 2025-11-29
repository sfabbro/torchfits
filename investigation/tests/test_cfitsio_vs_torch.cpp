// Test to isolate CFITSIO vs torch overhead for int16 vs uint8
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <fitsio.h>
#include <torch/torch.h>

using namespace std::chrono;

double benchmark_cfitsio_only(const char* filename, int datatype, size_t num_pixels, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        fitsfile* fptr;
        int status = 0;

        // Open file
        fits_open_file(&fptr, filename, READONLY, &status);
        if (status != 0) {
            std::cerr << "Failed to open file\n";
            return -1;
        }

        // Allocate buffer
        void* buffer = nullptr;
        if (datatype == TBYTE) {
            buffer = malloc(num_pixels * sizeof(uint8_t));
        } else {
            buffer = malloc(num_pixels * sizeof(int16_t));
        }

        auto start = high_resolution_clock::now();

        // Read data
        fits_read_img(fptr, datatype, 1, num_pixels, nullptr, buffer, nullptr, &status);

        auto end = high_resolution_clock::now();

        if (status != 0) {
            std::cerr << "Failed to read data\n";
            return -1;
        }

        times.push_back(duration<double>(end - start).count() * 1000.0);

        free(buffer);
        fits_close_file(fptr, &status);
    }

    // Return median
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double benchmark_torch_allocation(torch::ScalarType dtype, std::vector<int64_t> shape, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();

        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));

        auto end = high_resolution_clock::now();

        times.push_back(duration<double>(end - start).count() * 1000.0);
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double benchmark_full_path(const char* filename, int fits_type, torch::ScalarType torch_type,
                          std::vector<int64_t> shape, size_t num_pixels, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        fitsfile* fptr;
        int status = 0;

        fits_open_file(&fptr, filename, READONLY, &status);
        if (status != 0) return -1;

        auto start = high_resolution_clock::now();

        // Allocate tensor
        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch_type));

        // Read data
        if (fits_type == TBYTE) {
            fits_read_img(fptr, fits_type, 1, num_pixels, nullptr,
                         tensor.data_ptr<uint8_t>(), nullptr, &status);
        } else {
            fits_read_img(fptr, fits_type, 1, num_pixels, nullptr,
                         tensor.data_ptr<int16_t>(), nullptr, &status);
        }

        auto end = high_resolution_clock::now();

        if (status != 0) return -1;

        times.push_back(duration<double>(end - start).count() * 1000.0);

        fits_close_file(fptr, &status);
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <uint8_file> <int16_file>\n";
        return 1;
    }

    const char* uint8_file = argv[1];
    const char* int16_file = argv[2];
    const int iterations = 100;
    const size_t num_pixels = 1000 * 1000;
    std::vector<int64_t> shape = {1000, 1000};

    std::cout << "Testing with " << iterations << " iterations, " << num_pixels << " pixels\n";
    std::cout << "================================================================\n\n";

    // Test 1: CFITSIO only (no torch)
    std::cout << "1. CFITSIO read only (raw malloc buffer):\n";
    double uint8_cfitsio = benchmark_cfitsio_only(uint8_file, TBYTE, num_pixels, iterations);
    double int16_cfitsio = benchmark_cfitsio_only(int16_file, TSHORT, num_pixels, iterations);
    std::cout << "   uint8:  " << uint8_cfitsio << "ms\n";
    std::cout << "   int16:  " << int16_cfitsio << "ms\n";
    std::cout << "   Ratio:  " << int16_cfitsio / uint8_cfitsio << "x\n\n";

    // Test 2: Torch allocation only
    std::cout << "2. torch::empty() allocation:\n";
    double uint8_alloc = benchmark_torch_allocation(torch::kUInt8, shape, iterations);
    double int16_alloc = benchmark_torch_allocation(torch::kInt16, shape, iterations);
    std::cout << "   uint8:  " << uint8_alloc << "ms\n";
    std::cout << "   int16:  " << int16_alloc << "ms\n";
    std::cout << "   Ratio:  " << int16_alloc / uint8_alloc << "x\n\n";

    // Test 3: Full path (allocation + read)
    std::cout << "3. Full path (torch::empty + fits_read_img):\n";
    double uint8_full = benchmark_full_path(uint8_file, TBYTE, torch::kUInt8, shape, num_pixels, iterations);
    double int16_full = benchmark_full_path(int16_file, TSHORT, torch::kInt16, shape, num_pixels, iterations);
    std::cout << "   uint8:  " << uint8_full << "ms\n";
    std::cout << "   int16:  " << int16_full << "ms\n";
    std::cout << "   Ratio:  " << int16_full / uint8_full << "x\n\n";

    // Analysis
    std::cout << "================================================================\n";
    std::cout << "ANALYSIS:\n";
    std::cout << "================================================================\n\n";

    double cfitsio_ratio = int16_cfitsio / uint8_cfitsio;
    double alloc_ratio = int16_alloc / uint8_alloc;
    double full_ratio = int16_full / uint8_full;

    std::cout << "CFITSIO overhead (pure I/O):     " << cfitsio_ratio << "x\n";
    std::cout << "Allocation overhead:              " << alloc_ratio << "x\n";
    std::cout << "Full path overhead:               " << full_ratio << "x\n\n";

    double expected_ratio = cfitsio_ratio;  // Allocation should be ~1.0x
    double unexplained = full_ratio - expected_ratio;

    std::cout << "Expected ratio (CFITSIO only):    " << expected_ratio << "x\n";
    std::cout << "Actual ratio (full path):         " << full_ratio << "x\n";
    std::cout << "Unexplained overhead:             " << unexplained << "x\n";

    if (unexplained > 0.5) {
        std::cout << "\n⚠️  PROBLEM: Unexplained overhead of " << unexplained << "x detected!\n";
        std::cout << "This suggests an issue in the read path beyond CFITSIO.\n";
    } else {
        std::cout << "\n✅ Overhead is explained by CFITSIO's int16 handling.\n";
    }

    return 0;
}
