/**
 * Compare CFITSIO APIs: fits_read_img vs fits_read_pixll
 *
 * We use fits_read_img (empirically tested as faster for float32)
 * fitsio uses fits_read_pixll
 *
 * Test if fits_read_pixll is better for int16 specifically
 */
#include <fitsio.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>

using namespace std::chrono;

double benchmark_read_img(const char* filename, int datatype, void* buffer, size_t pixels) {
    std::vector<double> times;

    for (int iter = 0; iter < 20; iter++) {
        fitsfile* fptr = nullptr;
        int status = 0;

        auto start = high_resolution_clock::now();

        fits_open_file(&fptr, filename, READONLY, &status);
        fits_movabs_hdu(fptr, 1, nullptr, &status);

        int naxis, bitpix;
        long long naxes[10];
        fits_get_img_paramll(fptr, 10, &bitpix, &naxis, naxes, &status);

        // Use fits_read_img
        fits_read_img(fptr, datatype, 1, pixels, nullptr, buffer, nullptr, &status);

        fits_close_file(fptr, &status);

        auto end = high_resolution_clock::now();
        times.push_back(duration<double, std::milli>(end - start).count());
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double benchmark_read_pixll(const char* filename, int datatype, void* buffer, size_t pixels) {
    std::vector<double> times;

    for (int iter = 0; iter < 20; iter++) {
        fitsfile* fptr = nullptr;
        int status = 0;

        auto start = high_resolution_clock::now();

        fits_open_file(&fptr, filename, READONLY, &status);
        fits_movabs_hdu(fptr, 1, nullptr, &status);

        int naxis, bitpix;
        long long naxes[10];
        fits_get_img_paramll(fptr, 10, &bitpix, &naxis, naxes, &status);

        // Use fits_read_pixll (what fitsio uses)
        long long firstpixel[2] = {1, 1};
        int anynul = 0;
        fits_read_pixll(fptr, datatype, firstpixel, pixels, nullptr, buffer, &anynul, &status);

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
    const size_t pixels = 1000 * 1000;

    std::vector<uint8_t> uint8_buffer(pixels);
    std::vector<int16_t> int16_buffer(pixels);

    std::cout << "==========================================" << std::endl;
    std::cout << "CFITSIO API Comparison" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Testing fits_read_img (what we use):" << std::endl;
    std::cout << "--------------------" << std::endl;
    auto uint8_img = benchmark_read_img(uint8_file, TBYTE, uint8_buffer.data(), pixels);
    auto int16_img = benchmark_read_img(int16_file, TSHORT, int16_buffer.data(), pixels);
    std::cout << "  uint8:   " << uint8_img << "ms" << std::endl;
    std::cout << "  int16:   " << int16_img << "ms" << std::endl;
    std::cout << "  Ratio:   " << (int16_img / uint8_img) << "x" << std::endl;
    std::cout << std::endl;

    std::cout << "Testing fits_read_pixll (what fitsio uses):" << std::endl;
    std::cout << "--------------------" << std::endl;
    auto uint8_pixll = benchmark_read_pixll(uint8_file, TBYTE, uint8_buffer.data(), pixels);
    auto int16_pixll = benchmark_read_pixll(int16_file, TSHORT, int16_buffer.data(), pixels);
    std::cout << "  uint8:   " << uint8_pixll << "ms" << std::endl;
    std::cout << "  int16:   " << int16_pixll << "ms" << std::endl;
    std::cout << "  Ratio:   " << (int16_pixll / uint8_pixll) << "x" << std::endl;
    std::cout << std::endl;

    std::cout << "==========================================" << std::endl;
    std::cout << "Analysis" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Speed comparison (lower is better):" << std::endl;
    std::cout << "  uint8 - read_img vs read_pixll: " << uint8_img << " vs " << uint8_pixll;
    if (uint8_img < uint8_pixll) {
        std::cout << " ✅ read_img is faster" << std::endl;
    } else {
        std::cout << " ❌ read_pixll is faster" << std::endl;
    }

    std::cout << "  int16 - read_img vs read_pixll: " << int16_img << " vs " << int16_pixll;
    if (int16_img < int16_pixll) {
        std::cout << " ✅ read_img is faster" << std::endl;
    } else {
        std::cout << " ❌ read_pixll is faster" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Ratio comparison:" << std::endl;
    double ratio_img = int16_img / uint8_img;
    double ratio_pixll = int16_pixll / uint8_pixll;
    std::cout << "  read_img ratio:   " << ratio_img << "x" << std::endl;
    std::cout << "  read_pixll ratio: " << ratio_pixll << "x" << std::endl;

    if (ratio_pixll < ratio_img * 0.9) {
        std::cout << std::endl;
        std::cout << "✅ fits_read_pixll has BETTER int16/uint8 ratio!" << std::endl;
        std::cout << "   This could explain fitsio's better int16 performance." << std::endl;
        std::cout << "   Consider switching to fits_read_pixll for better int16." << std::endl;
    } else {
        std::cout << std::endl;
        std::cout << "❌ API choice doesn't explain the difference." << std::endl;
    }

    return 0;
}
