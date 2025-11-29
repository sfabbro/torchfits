// Test different CFITSIO buffer sizes for int16
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fitsio.h>

using namespace std::chrono;

double benchmark_with_bufsize(const char* filename, int datatype, size_t num_pixels,
                              int bufsize, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        fitsfile* fptr;
        int status = 0;

        fits_open_file(&fptr, filename, READONLY, &status);
        if (status != 0) {
            std::cerr << "Failed to open file\n";
            return -1;
        }

        // Set I/O buffer size (in 2880-byte blocks)
        if (bufsize > 0) {
            fits_set_bscale(fptr, 1.0, 0.0, &status);  // This also sets buffer size internally
            // Note: There's no public API to set buffer size directly in newer CFITSIO
            // It's set automatically based on file type and access pattern
        }

        void* buffer = malloc(num_pixels * (datatype == TBYTE ? 1 : 2));
        LONGLONG firstpix[2] = {1, 1};
        int anynul;

        auto start = high_resolution_clock::now();
        fits_read_pixll(fptr, datatype, firstpix, num_pixels, nullptr, buffer, &anynul, &status);
        auto end = high_resolution_clock::now();

        if (status != 0) {
            std::cerr << "Read failed with status " << status << "\n";
            free(buffer);
            fits_close_file(fptr, &status);
            return -1;
        }

        times.push_back(duration<double>(end - start).count() * 1000.0);

        free(buffer);
        fits_close_file(fptr, &status);
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double benchmark_readall_mode(const char* filename, int datatype, size_t num_pixels, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        fitsfile* fptr;
        int status = 0;

        // Open with "readall" mode (reads entire file into memory)
        std::string readall_path = std::string(filename) + "[readall]";
        fits_open_file(&fptr, readall_path.c_str(), READONLY, &status);
        if (status != 0) {
            std::cerr << "Failed to open file with readall: " << status << "\n";
            return -1;
        }

        void* buffer = malloc(num_pixels * (datatype == TBYTE ? 1 : 2));
        LONGLONG firstpix[2] = {1, 1};
        int anynul;

        auto start = high_resolution_clock::now();
        fits_read_pixll(fptr, datatype, firstpix, num_pixels, nullptr, buffer, &anynul, &status);
        auto end = high_resolution_clock::now();

        if (status != 0) {
            std::cerr << "Read failed\n";
            free(buffer);
            fits_close_file(fptr, &status);
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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <int16_file>\n";
        return 1;
    }

    const char* int16_file = argv[1];
    const int iterations = 100;
    const size_t num_pixels = 1000 * 1000;

    std::cout << "Testing CFITSIO buffer configurations for int16\n";
    std::cout << "================================================================\n\n";

    // Test different configurations
    std::cout << "1. Default settings:\n";
    double default_time = benchmark_with_bufsize(int16_file, TSHORT, num_pixels, 0, iterations);
    std::cout << "   Time: " << default_time << "ms\n\n";

    std::cout << "2. With readall mode (loads entire file):\n";
    double readall_time = benchmark_readall_mode(int16_file, TSHORT, num_pixels, iterations);
    std::cout << "   Time: " << readall_time << "ms\n\n";

    // Try with explicit file handle reuse
    std::cout << "3. With handle reuse (100 reads, same handle):\n";
    fitsfile* fptr;
    int status = 0;
    fits_open_file(&fptr, int16_file, READONLY, &status);

    std::vector<double> reuse_times;
    void* buffer = malloc(num_pixels * 2);
    LONGLONG firstpix[2] = {1, 1};
    int anynul;

    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        fits_read_pixll(fptr, TSHORT, firstpix, num_pixels, nullptr, buffer, &anynul, &status);
        auto end = high_resolution_clock::now();
        reuse_times.push_back(duration<double>(end - start).count() * 1000.0);
    }

    std::sort(reuse_times.begin(), reuse_times.end());
    double reuse_time = reuse_times[reuse_times.size() / 2];
    std::cout << "   Time: " << reuse_time << "ms\n\n";

    free(buffer);
    fits_close_file(fptr, &status);

    // Analysis
    std::cout << "================================================================\n";
    std::cout << "ANALYSIS:\n";
    std::cout << "================================================================\n\n";

    std::cout << "Default:       " << default_time << "ms\n";
    std::cout << "Readall:       " << readall_time << "ms";
    if (readall_time < default_time) {
        std::cout << " (faster by " << (default_time - readall_time) / default_time * 100 << "%)\n";
    } else {
        std::cout << " (slower)\n";
    }

    std::cout << "Handle reuse:  " << reuse_time << "ms";
    if (reuse_time < default_time) {
        std::cout << " (faster by " << (default_time - reuse_time) / default_time * 100 << "%)\n";
    } else {
        std::cout << " (SLOWER by " << (reuse_time - default_time) / default_time * 100 << "%!) ⚠️\n";
    }

    std::cout << "\n";
    if (reuse_time > default_time) {
        std::cout << "⚠️  WARNING: Handle reuse is SLOWER for int16!\n";
        std::cout << "This confirms our earlier finding about file handle caching.\n";
    }

    return 0;
}
