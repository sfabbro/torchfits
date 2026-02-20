
#include "compression.h"
#include <fitsio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string.h>

// -----------------------------------------------------------------------------
// Vendored Rice Decompression Algorithm (from CFITSIO ricecomp.c)
// -----------------------------------------------------------------------------

static const int nonzero_count[256] = {
0, 
1, 
2, 2, 
3, 3, 3, 3, 
4, 4, 4, 4, 4, 4, 4, 4, 
5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};

static int tf_rdecomp(const unsigned char *c, int clen, unsigned int array[], int nx, int nblock) {
    int i, k, imax;
    int nbits, nzero, fs;
    const unsigned char *cend;
    unsigned char bytevalue;
    unsigned int b, diff, lastpix;
    int fsmax, fsbits, bbits;

    /* Constants for bsize=4 */
    fsbits = 5;
    fsmax = 25;
    bbits = 1<<fsbits;

    if (clen < 4) return 1;

    /* First 4 bytes = first value */
    lastpix = 0;
    bytevalue = c[0]; lastpix |= (bytevalue<<24);
    bytevalue = c[1]; lastpix |= (bytevalue<<16);
    bytevalue = c[2]; lastpix |= (bytevalue<<8);
    bytevalue = c[3]; lastpix |= bytevalue;

    c += 4;  
    cend = c + clen - 4;

    b = *c++;		    /* bit buffer */
    nbits = 8;		    /* bits remaining */

    for (i = 0; i<nx; ) {
        nbits -= fsbits;
        while (nbits < 0) {
            b = (b<<8) | (*c++);
            nbits += 8;
        }
        fs = (b >> nbits) - 1;
        b &= (1<<nbits)-1;

        imax = i + nblock;
        if (imax > nx) imax = nx;

        if (fs<0) {
            for ( ; i<imax; i++) array[i] = lastpix;
        } else if (fs==fsmax) {
            for ( ; i<imax; i++) {
                k = bbits - nbits;
                diff = b<<k;
                for (k -= 8; k >= 0; k -= 8) {
                    b = *c++;
                    diff |= b<<k;
                }
                if (nbits>0) {
                    b = *c++;
                    diff |= b>>(-k);
                    b &= (1<<nbits)-1;
                } else {
                    b = 0;
                }
                if ((diff & 1) == 0) diff = diff>>1;
                else diff = ~(diff>>1);
                
                array[i] = diff+lastpix;
                lastpix = array[i];
            }
        } else {
            for ( ; i<imax; i++) {
                while (b == 0) {
                    nbits += 8;
                    b = *c++;
                }
                nzero = nbits - nonzero_count[b];
                nbits -= nzero+1;
                b ^= 1<<nbits;
                nbits -= fs;
                while (nbits < 0) {
                    b = (b<<8) | (*c++);
                    nbits += 8;
                }
                diff = (nzero<<fs) | (b>>nbits);
                b &= (1<<nbits)-1;

                if ((diff & 1) == 0) diff = diff>>1;
                else diff = ~(diff>>1);
                
                array[i] = diff+lastpix;
                lastpix = array[i];
            }
        }
    }
    return 0;
}

// -----------------------------------------------------------------------------
// End Vendored Code
// -----------------------------------------------------------------------------

// Helper class for mmap RAII
class MMapFile {
public:
    void* ptr = MAP_FAILED;
    size_t size = 0;
    int fd = -1;

    MMapFile(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("Failed to open file: " + path);
        struct stat st;
        if (fstat(fd, &st) == -1) { close(fd); throw std::runtime_error("Failed to stat file"); }
        size = st.st_size;
        ptr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) { close(fd); throw std::runtime_error("mmap failed"); }
    }

    ~MMapFile() {
        if (ptr != MAP_FAILED) munmap(ptr, size);
        if (fd != -1) close(fd);
    }
};

inline void check_status(int status, const char* msg) {
    if (status) {
        char err_text[30];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error(std::string(msg) + ": " + err_text);
    }
}

inline uint32_t bswap32(uint32_t x) { return __builtin_bswap32(x); }
inline uint64_t bswap64(uint64_t x) { return __builtin_bswap64(x); }

torch::Tensor read_rice_parallel(const std::string& path, int hdu, int num_threads) {
    int status = 0;
    fitsfile* fptr = nullptr;
    fits_open_file(&fptr, path.c_str(), READONLY, &status);
    check_status(status, "open_file");
    
    struct FileGuard { fitsfile* f; ~FileGuard() { int s=0; if(f) fits_close_file(f, &s); } } file_guard{fptr};

    fits_movabs_hdu(fptr, hdu + 1, nullptr, &status);
    check_status(status, "move_hdu");

    char zcmptype[FLEN_VALUE];
    fits_read_key(fptr, TSTRING, "ZCMPTYPE", zcmptype, nullptr, &status);
    if (status || strncmp(zcmptype, "RICE_1", 6) != 0) {
        throw std::runtime_error("Only RICE_1 compression is supported");
    }

    long znaxis = 0;
    fits_read_key(fptr, TLONG, "ZNAXIS", &znaxis, nullptr, &status);
    if (znaxis != 2) throw std::runtime_error("Only 2D images supported for now");

    long width=0, height=0;
    fits_read_key(fptr, TLONG, "ZNAXIS1", &width, nullptr, &status);
    fits_read_key(fptr, TLONG, "ZNAXIS2", &height, nullptr, &status);
    
    long tile_w=0, tile_h=0;
    fits_read_key(fptr, TLONG, "ZTILE1", &tile_w, nullptr, &status);
    fits_read_key(fptr, TLONG, "ZTILE2", &tile_h, nullptr, &status);

    int zbitpix = 0;
    fits_read_key(fptr, TINT, "ZBITPIX", &zbitpix, nullptr, &status);

    long long headstart, data_start, data_end;
    fits_get_hduaddrll(fptr, &headstart, &data_start, &data_end, &status);
    
    long table_width = 0;
    fits_read_key(fptr, TLONG, "NAXIS1", &table_width, nullptr, &status);
    
    long nrows = 0;
    fits_read_key(fptr, TLONG, "NAXIS2", &nrows, nullptr, &status);
    
    long theap = 0;
    int s2 = 0;
    fits_read_key(fptr, TLONG, "THEAP", &theap, nullptr, &s2);
    if (s2 != 0) theap = nrows * table_width;
    
    int64_t heap_abs_start = data_start + theap;

    int col_idx = 0;
    int ncols = 0;
    fits_get_num_cols(fptr, &ncols, &status);
    
    char colname[FLEN_VALUE];
    fits_get_colname(fptr, FALSE, (char*)"*COMPRESSED_DATA*", colname, &col_idx, &status);
    if (col_idx == 0) throw std::runtime_error("COMPRESSED_DATA column not found");
    
    int typecode;
    long repeat, col_w;
    fits_get_coltype(fptr, col_idx, &typecode, &repeat, &col_w, &status);
    
    char tform[FLEN_VALUE];
    char key[20]; snprintf(key, 20, "TFORM%d", col_idx);
    fits_read_key(fptr, TSTRING, key, tform, nullptr, &status);
    bool descriptor_is_64bit = (strchr(tform, 'Q') != nullptr);
    
    MMapFile mapped(path);
    uint8_t* map_ptr = (uint8_t*)mapped.ptr;
    
    auto options = torch::TensorOptions().dtype(torch::kInt32); 
    if (zbitpix == 16) options = options.dtype(torch::kShort);
    else if (zbitpix == 8) options = options.dtype(torch::kUInt8);
    else if (zbitpix == -32) options= options.dtype(torch::kFloat32); 
    
    torch::Tensor output;
    if (zbitpix == -32) output = torch::empty({height, width}, torch::kFloat32);
    else output = torch::empty({height, width}, torch::kInt32); 
    
    
    long tiles_x = (width + tile_w - 1) / tile_w;
    int64_t table_start_offset = data_start; 
    
    if (num_threads < 1) num_threads = at::get_num_threads();
    
    // Debug Prints
    // if (num_threads > 1) {
    //       std::cout << "Debug: Image (" << width << "x" << height << ") Tiles(" << tile_w << "x" << tile_h << ") Count=" << nrows << " Threads=" << num_threads << std::endl;
    // }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    int64_t chunk_size = (nrows + num_threads - 1) / num_threads;

    auto worker_func = [&](int64_t begin, int64_t end, int tid) {
        std::vector<unsigned int> scratch(tile_w * tile_h);
        // if (tid == 0) std::cout << "Debug: Thread " << tid << " processing " << begin << " to " << end << " (chunk size " << (end-begin) << ")" << std::endl;
        
        for (int64_t r = begin; r < end; ++r) {
            uint64_t len = 0;
            uint64_t off = 0;
            uint8_t* desc_ptr = map_ptr + table_start_offset + r * table_width;
            
            if (descriptor_is_64bit) {
                uint64_t* p = (uint64_t*)desc_ptr;
                len = bswap64(p[0]);
                off = bswap64(p[1]);
            } else {
                uint32_t* p = (uint32_t*)desc_ptr;
                len = bswap32(p[0]);
                off = bswap32(p[1]);
            }
            
            if (len == 0) continue;
            
            uint8_t* cdata = map_ptr + heap_abs_start + off;
            
            long ty = r / tiles_x;
            long tx = r % tiles_x;
            long pix_y = ty * tile_h;
            long pix_x = tx * tile_w;
            long valid_w = std::min(tile_w, width - pix_x);
            long valid_h = std::min(tile_h, height - pix_y);
            long npix = valid_w * valid_h;
            
            // Call vendored logic
            int err = tf_rdecomp(cdata, (int)len, scratch.data(), (int)npix, 32);
            if (err) {
                 memset(scratch.data(), 0, npix * sizeof(unsigned int));
            }
            
            if (output.scalar_type() == torch::kFloat32) {
                float* out_f = output.data_ptr<float>();
                for (int y=0; y<valid_h; ++y) {
                    float* dst_row = out_f + (pix_y + y) * width + pix_x;
                    unsigned int* src_row = scratch.data() + y * valid_w;
                    for (int x=0; x<valid_w; ++x) {
                        dst_row[x] = (float)((int)src_row[x]); 
                    }
                }
            } else if (output.scalar_type() == torch::kInt32) {
                int32_t* out_i = output.data_ptr<int32_t>();
                for (int y=0; y<valid_h; ++y) {
                    int32_t* dst_row = out_i + (pix_y + y) * width + pix_x;
                    unsigned int* src_row = scratch.data() + y * valid_w;
                    memcpy(dst_row, src_row, valid_w * sizeof(int32_t));
                }
            }
        }
    };

    for (int t=0; t<num_threads; ++t) {
        int64_t begin = t * chunk_size;
        int64_t end = std::min<int64_t>(begin + chunk_size, nrows);
        if (begin >= end) break;
        workers.emplace_back(worker_func, begin, end, t);
    }
    
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    return output;
}
