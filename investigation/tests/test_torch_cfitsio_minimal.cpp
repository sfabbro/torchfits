/*
 * Minimal test: Does linking against PyTorch make CFITSIO slow?
 *
 * This tests if PyTorch's memory allocator or threading interferes with CFITSIO.
 */
#include <torch/torch.h>
#include <fitsio.h>
#include <iostream>
#include <chrono>

double benchmark_cfitsio(const char* filename, int iterations) {
    fitsfile *fptr;
    int status = 0;
    int bitpix, naxis;
    long naxes[10];
    int anynull;
    LONGLONG fpixel[10] = {1,1,1,1,1,1,1,1,1,1};
    LONGLONG nelements;

    // Get image info
    fits_open_file(&fptr, filename, READONLY, &status);
    fits_get_img_param(fptr, 10, &bitpix, &naxis, naxes, &status);
    fits_close_file(fptr, &status);

    nelements = naxes[0] * naxes[1];
    std::cout << "Image: " << naxes[0] << "x" << naxes[1] << ", bitpix=" << bitpix << std::endl;

    // Allocate buffer
    int16_t* buffer = (int16_t*)malloc(nelements * sizeof(int16_t));

    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < iterations; i++) {
        status = 0;
        fits_open_file(&fptr, filename, READONLY, &status);

        auto t0 = std::chrono::high_resolution_clock::now();
        fits_read_pixll(fptr, TSHORT, fpixel, nelements, NULL, buffer, &anynull, &status);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(dt);

        fits_close_file(fptr, &status);
    }

    // Calculate median
    std::sort(times.begin(), times.end());
    double median = times[iterations / 2];

    free(buffer);
    return median;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <fits_file>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];

    std::cout << "Minimal C++ + PyTorch + CFITSIO Test" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::endl;

    // Test 1: CFITSIO only (no PyTorch initialization)
    std::cout << "Test 1: CFITSIO without PyTorch init" << std::endl;
    double time_no_torch = benchmark_cfitsio(filename, 100);
    std::cout << "  Median: " << time_no_torch << "ms" << std::endl;
    std::cout << std::endl;

    // Test 2: Initialize PyTorch, then CFITSIO
    std::cout << "Test 2: Initialize PyTorch first" << std::endl;
    auto tensor = torch::empty({100, 100}, torch::kInt16);  // Initialize PyTorch
    std::cout << "  PyTorch initialized (created tensor)" << std::endl;

    double time_with_torch = benchmark_cfitsio(filename, 100);
    std::cout << "  Median: " << time_with_torch << "ms" << std::endl;
    std::cout << std::endl;

    // Test 3: Read into PyTorch tensor memory
    std::cout << "Test 3: Read into torch::Tensor memory" << std::endl;

    fitsfile *fptr;
    int status = 0;
    int bitpix, naxis;
    long naxes[10];
    int anynull;
    LONGLONG fpixel[10] = {1,1,1,1,1,1,1,1,1,1};
    LONGLONG nelements;

    fits_open_file(&fptr, filename, READONLY, &status);
    fits_get_img_param(fptr, 10, &bitpix, &naxis, naxes, &status);
    fits_close_file(fptr, &status);
    nelements = naxes[0] * naxes[1];

    std::vector<double> times;
    for (int i = 0; i < 100; i++) {
        // Create tensor
        auto tensor = torch::empty({naxes[0], naxes[1]}, torch::kInt16);

        status = 0;
        fits_open_file(&fptr, filename, READONLY, &status);

        auto t0 = std::chrono::high_resolution_clock::now();
        fits_read_pixll(fptr, TSHORT, fpixel, nelements, NULL,
                       tensor.data_ptr<int16_t>(), &anynull, &status);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(dt);

        fits_close_file(fptr, &status);
    }

    std::sort(times.begin(), times.end());
    double time_tensor_memory = times[50];
    std::cout << "  Median: " << time_tensor_memory << "ms" << std::endl;
    std::cout << std::endl;

    // Summary
    std::cout << "=====================================" << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "No PyTorch:            " << time_no_torch << "ms" << std::endl;
    std::cout << "With PyTorch init:     " << time_with_torch << "ms" << std::endl;
    std::cout << "Into tensor memory:    " << time_tensor_memory << "ms" << std::endl;
    std::cout << std::endl;

    if (time_tensor_memory > time_no_torch * 2.0) {
        std::cout << "⚠️  FOUND IT! torch::Tensor memory causes slow CFITSIO!" << std::endl;
    } else if (time_with_torch > time_no_torch * 1.5) {
        std::cout << "⚠️  PyTorch initialization slows CFITSIO" << std::endl;
    } else {
        std::cout << "✅ PyTorch doesn't slow CFITSIO - issue must be elsewhere" << std::endl;
    }

    return 0;
}
