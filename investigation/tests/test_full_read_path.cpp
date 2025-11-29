/**
 * Full read path benchmark: CFITSIO + torch::empty + data copy
 *
 * This simulates exactly what our read function does
 */
#include <torch/torch.h>
#include <fitsio.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std::chrono;

double benchmark_full_read(const char* filename, torch::ScalarType torch_dtype, int cfitsio_dtype) {
    std::vector<double> times;

    for (int iter = 0; iter < 20; iter++) {
        auto start = high_resolution_clock::now();

        // 1. Open file
        fitsfile* fptr = nullptr;
        int status = 0;
        fits_open_file(&fptr, filename, READONLY, &status);
        if (status != 0) return -1.0;

        // 2. Move to HDU
        fits_movabs_hdu(fptr, 1, nullptr, &status);
        if (status != 0) return -1.0;

        // 3. Get parameters
        int naxis, bitpix;
        long long naxes[10];
        fits_get_img_paramll(fptr, 10, &bitpix, &naxis, naxes, &status);
        if (status != 0) return -1.0;

        std::vector<int64_t> shape = {naxes[1], naxes[0]};  // Reversed for row-major
        long long total_pixels = naxes[0] * naxes[1];

        // 4. Allocate tensor
        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch_dtype));

        // 5. Read data
        void* data_ptr = tensor.data_ptr();
        fits_read_img(fptr, cfitsio_dtype, 1, total_pixels, nullptr, data_ptr, nullptr, &status);
        if (status != 0) return -1.0;

        // 6. Close file
        fits_close_file(fptr, &status);

        auto end = high_resolution_clock::now();
        times.push_back(duration<double, std::milli>(end - start).count());
    }

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

    std::cout << "========================================" << std::endl;
    std::cout << "Full Read Path Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    auto uint8_time = benchmark_full_read(uint8_file, torch::kUInt8, TBYTE);
    auto int16_time = benchmark_full_read(int16_file, torch::kInt16, TSHORT);

    std::cout << "Median times (20 iterations):" << std::endl;
    std::cout << "  uint8:   " << uint8_time << "ms" << std::endl;
    std::cout << "  int16:   " << int16_time << "ms" << std::endl;
    std::cout << std::endl;

    double ratio = int16_time / uint8_time;
    double diff = int16_time - uint8_time;

    std::cout << "Ratio (int16/uint8): " << ratio << "x" << std::endl;
    std::cout << "Absolute difference: " << diff << "ms" << std::endl;
    std::cout << std::endl;

    // Breakdown (from previous tests)
    std::cout << "Overhead breakdown (approximate):" << std::endl;
    std::cout << "  CFITSIO pure:     0.069ms  (from test_cfitsio_direct)" << std::endl;
    std::cout << "  torch::empty:     0.000ms  (from test_torch_allocation)" << std::endl;
    std::cout << "  Unexplained:      " << (diff - 0.069) << "ms" << std::endl;

    return 0;
}
