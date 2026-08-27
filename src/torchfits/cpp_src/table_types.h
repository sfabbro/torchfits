#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <ATen/ATen.h>
#include <fitsio.h>

#include "torchfits_torch.h"
#include "security.h"
#include "cache.h"

namespace torchfits {

enum class FITSColumnType {
    LOGICAL,    // L
    BIT,        // X (bit array)
    BYTE,       // B
    SHORT,      // I
    INT,        // J
    LONG,       // K
    FLOAT,      // E
    DOUBLE,     // D
    COMPLEX_FLOAT,   // C
    COMPLEX_DOUBLE,  // M
    STRING,     // A
    VARIABLE    // P/Q - variable length arrays
};

struct ColumnInfo {
    std::string name;
    FITSColumnType type;
    int repeat;
    int width;
    torch::ScalarType torch_type;
    long byte_offset; // Offset in bytes from start of row
    double tscale = 1.0;
    double tzero = 0.0;
    bool scaled = false;
    bool scale_resolved = false;  // lazy TSCAL/TZERO load
    bool has_tnull = false;
    long long tnull = 0;
    bool is_unsigned_int = false;  // uint16/uint32 FITS convention (TZERO offset)
    int64_t unsigned_offset = 0;   // 32768 or 2147483648
    torch::ScalarType unsigned_target_type = torch::kInt64;  // kUInt16 or kUInt32
    long storage_bytes = 0;        // Physical bytes occupied by one table row
    int fits_typecode = 0;         // CFITSIO typecode from analyze_table
};

// Filter operations
enum class FilterOp {
    EQ, NE, GT, LT, GE, LE
};

struct TableFilter {
    std::string col_name;
    FilterOp op;
    double val_d = 0.0;
    int64_t val_i = 0;
    std::string val_s;
    // 0=double, 1=int, 2=string
    int type_idx = 0;
};

// Helper to check if buffered row reading is enabled
inline bool table_buffered_read_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("TORCHFITS_TABLE_BUFFERED");
        if (!env || env[0] == '\0') {
            return true;
        }
        return !(env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' || env[0] == 'F');
    }();
    return enabled;
}

} // namespace torchfits
