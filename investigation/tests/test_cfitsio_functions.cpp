// Compare fits_read_img vs fits_read_pixll to see which is faster
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fitsio.h>

using namespace std::chrono;

double benchmark_fits_read_img(const char* filename, int datatype, size_t num_pixels, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        fitsfile* fptr;
        int status = 0;

        fits_open_file(&fptr, filename, READONLY, &status);
        if (status != 0) {
            std::cerr << "Failed to open file\n";
            return -1;
        }

        void* buffer = malloc(num_pixels * (datatype == TBYTE ? 1 : 2));

        auto start = high_resolution_clock::now();
        fits_read_img(fptr, datatype, 1, num_pixels, nullptr, buffer, nullptr, &status);
        auto end = high_resolution_clock::now();

        if (status != 0) {
            std::cerr << "fits_read_img failed with status " << status << "\n";
            return -1;
        }

        times.push_back(duration<double>(end - start).count() * 1000.0);

        free(buffer);
        fits_close_file(fptr, &status);
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double benchmark_fits_read_pixll(const char* filename, int datatype, size_t num_pixels, int naxis, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        fitsfile* fptr;
        int status = 0;

        fits_open_file(&fptr, filename, READONLY, &status);
        if (status != 0) {
            std::cerr << "Failed to open file\n";
            return -1;
        }

        void* buffer = malloc(num_pixels * (datatype == TBYTE ? 1 : 2));

        // fits_read_pixll requires firstpix array (1-indexed)
        LONGLONG firstpix[2] = {1, 1};

        auto start = high_resolution_clock::now();
        int anynul;
        fits_read_pixll(fptr, datatype, firstpix, num_pixels, nullptr, buffer, &anynul, &status);
        auto end = high_resolution_clock::now();

        if (status != 0) {
            std::cerr << "fits_read_pixll failed with status " << status << "\n";
            return -1;
        }

        times.push_back(duration<double>(end - start).count() * 1000.0);

        free(buffer);
        fits_close_file(fptr, &status);
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double benchmark_fits_read_pix(const char* filename, int datatype, size_t num_pixels, int naxis, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        fitsfile* fptr;
        int status = 0;

        fits_open_file(&fptr, filename, READONLY, &status);
        if (status != 0) {
            std::cerr << "Failed to open file\n";
            return -1;
        }

        void* buffer = malloc(num_pixels * (datatype == TBYTE ? 1 : 2));

        // fits_read_pix requires firstpix array (1-indexed)
        long firstpix[2] = {1, 1};

        auto start = high_resolution_clock::now();
        int anynul;
        fits_read_pix(fptr, datatype, firstpix, num_pixels, nullptr, buffer, &anynul, &status);
        auto end = high_resolution_clock::now();

        if (status != 0) {
            std::cerr << "fits_read_pix failed with status " << status << "\n";
            return -1;
        }

        times.push_back(duration<double>(end - start).count() * 1000.0);

        free(buffer);
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

    std::cout << "Comparing CFITSIO read functions (100 iterations)\n";
    std::cout << "================================================================\n\n";

    // Test uint8
    std::cout << "UINT8 (TBYTE):\n";
    std::cout << "----------------------------------------\n";
    double u8_img = benchmark_fits_read_img(uint8_file, TBYTE, num_pixels, iterations);
    double u8_pix = benchmark_fits_read_pix(uint8_file, TBYTE, num_pixels, 2, iterations);
    double u8_pixll = benchmark_fits_read_pixll(uint8_file, TBYTE, num_pixels, 2, iterations);

    std::cout << "  fits_read_img:   " << u8_img << "ms\n";
    std::cout << "  fits_read_pix:   " << u8_pix << "ms\n";
    std::cout << "  fits_read_pixll: " << u8_pixll << "ms (fitsio uses this)\n";
    std::cout << "  Fastest: " << (u8_img < u8_pix ? (u8_img < u8_pixll ? "img" : "pixll") : (u8_pix < u8_pixll ? "pix" : "pixll")) << "\n";
    std::cout << "\n";

    // Test int16
    std::cout << "INT16 (TSHORT):\n";
    std::cout << "----------------------------------------\n";
    double i16_img = benchmark_fits_read_img(int16_file, TSHORT, num_pixels, iterations);
    double i16_pix = benchmark_fits_read_pix(int16_file, TSHORT, num_pixels, 2, iterations);
    double i16_pixll = benchmark_fits_read_pixll(int16_file, TSHORT, num_pixels, 2, iterations);

    std::cout << "  fits_read_img:   " << i16_img << "ms\n";
    std::cout << "  fits_read_pix:   " << i16_pix << "ms\n";
    std::cout << "  fits_read_pixll: " << i16_pixll << "ms (fitsio uses this)\n";
    std::cout << "  Fastest: " << (i16_img < i16_pix ? (i16_img < i16_pixll ? "img" : "pixll") : (i16_pix < i16_pixll ? "pix" : "pixll")) << "\n";
    std::cout << "\n";

    // Analysis
    std::cout << "================================================================\n";
    std::cout << "ANALYSIS:\n";
    std::cout << "================================================================\n\n";

    std::cout << "Performance ratios (int16/uint8):\n";
    std::cout << "  fits_read_img:   " << i16_img / u8_img << "x\n";
    std::cout << "  fits_read_pix:   " << i16_pix / u8_pix << "x\n";
    std::cout << "  fits_read_pixll: " << i16_pixll / u8_pixll << "x\n";
    std::cout << "\n";

    // Which function is fastest for int16?
    double best_int16 = std::min({i16_img, i16_pix, i16_pixll});
    std::string best_func = (best_int16 == i16_img ? "fits_read_img" :
                             (best_int16 == i16_pix ? "fits_read_pix" : "fits_read_pixll"));

    std::cout << "Best function for int16: " << best_func << " (" << best_int16 << "ms)\n";
    std::cout << "We currently use: fits_read_img (" << i16_img << "ms)\n";

    if (best_int16 < i16_img) {
        double improvement = (i16_img - best_int16) / i16_img * 100.0;
        std::cout << "Potential improvement: " << improvement << "% faster\n";
    } else {
        std::cout << "✅ We're already using the best function!\n";
    }

    return 0;
}
