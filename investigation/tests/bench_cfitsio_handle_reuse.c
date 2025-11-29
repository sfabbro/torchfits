/*
 * Test if keeping file handle open causes slow CFITSIO reads
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fitsio.h>

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <fits_file>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
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
        return 1;
    }

    fits_get_img_param(fptr, 10, &bitpix, &naxis, naxes, &status);
    nelements = naxes[0] * naxes[1];

    printf("Testing CFITSIO with handle reuse\n");
    printf("==================================\n");
    printf("File: %s\n", filename);
    printf("Image: %ldx%ld, bitpix=%d\n", naxes[0], naxes[1], bitpix);
    printf("\n");

    int datatype = (bitpix == BYTE_IMG) ? TBYTE : TSHORT;
    size_t element_size = (bitpix == BYTE_IMG) ? 1 : 2;

    void *buffer = malloc(nelements * element_size);

    // Test 1: Keep file open, multiple reads
    printf("Test 1: Handle reuse (file kept open)\n");
    printf("--------------------------------------\n");

    for (int i = 0; i < 10; i++) {
        double t0 = get_time_ms();
        status = 0;
        fits_read_pixll(fptr, datatype, fpixel, nelements, NULL, buffer, &anynull, &status);
        double t1 = get_time_ms();

        printf("  Read %2d: %.4fms (status=%d)\n", i+1, t1-t0, status);
    }

    fits_close_file(fptr, &status);
    printf("\n");

    // Test 2: Open/close each time
    printf("Test 2: No reuse (open/close each time)\n");
    printf("----------------------------------------\n");

    for (int i = 0; i < 10; i++) {
        status = 0;
        fits_open_file(&fptr, filename, READONLY, &status);

        double t0 = get_time_ms();
        fits_read_pixll(fptr, datatype, fpixel, nelements, NULL, buffer, &anynull, &status);
        double t1 = get_time_ms();

        fits_close_file(fptr, &status);

        printf("  Read %2d: %.4fms (status=%d)\n", i+1, t1-t0, status);
    }

    free(buffer);

    return 0;
}
