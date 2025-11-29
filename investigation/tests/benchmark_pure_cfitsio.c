/*
 * Pure C benchmark of CFITSIO fits_read_pixll performance
 * This eliminates ALL Python/C++ overhead
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <fitsio.h>

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void benchmark_read(const char* filename, int iterations) {
    fitsfile *fptr;
    int status = 0;
    int bitpix, naxis;
    long naxes[10];
    int anynull;
    LONGLONG fpixel[10] = {1,1,1,1,1,1,1,1,1,1};
    LONGLONG nelements;

    // Determine image parameters
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        printf("Error opening file: %d\n", status);
        return;
    }

    fits_get_img_param(fptr, 10, &bitpix, &naxis, naxes, &status);
    if (status) {
        printf("Error getting image params: %d\n", status);
        return;
    }

    nelements = 1;
    for (int i = 0; i < naxis; i++) {
        nelements *= naxes[i];
    }

    printf("Image: %ldx%ld, bitpix=%d\n", naxes[0], naxes[1], bitpix);

    // Determine data type
    int datatype;
    size_t element_size;
    if (bitpix == BYTE_IMG) {
        datatype = TBYTE;
        element_size = 1;
    } else if (bitpix == SHORT_IMG) {
        datatype = TSHORT;
        element_size = 2;
    } else {
        printf("Unsupported bitpix: %d\n", bitpix);
        return;
    }

    // Allocate buffer
    void *buffer = malloc(nelements * element_size);
    if (!buffer) {
        printf("Failed to allocate buffer\n");
        return;
    }

    fits_close_file(fptr, &status);

    // Benchmark
    double *times = malloc(iterations * sizeof(double));

    for (int i = 0; i < iterations; i++) {
        status = 0;
        fits_open_file(&fptr, filename, READONLY, &status);

        double t0 = get_time_ms();
        fits_read_pixll(fptr, datatype, fpixel, nelements, NULL, buffer, &anynull, &status);
        double t1 = get_time_ms();

        times[i] = t1 - t0;

        fits_close_file(fptr, &status);
    }

    // Calculate median
    for (int i = 0; i < iterations - 1; i++) {
        for (int j = i + 1; j < iterations; j++) {
            if (times[j] < times[i]) {
                double tmp = times[i];
                times[i] = times[j];
                times[j] = tmp;
            }
        }
    }

    double median = times[iterations / 2];
    double min = times[0];
    double max = times[iterations - 1];

    printf("Iterations: %d\n", iterations);
    printf("  Min:    %.4fms\n", min);
    printf("  Median: %.4fms\n", median);
    printf("  Max:    %.4fms\n", max);

    free(buffer);
    free(times);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <fits_file> <iterations>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int iterations = atoi(argv[2]);

    printf("Pure CFITSIO benchmark\n");
    printf("======================\n");
    printf("File: %s\n", filename);

    benchmark_read(filename, iterations);

    return 0;
}
