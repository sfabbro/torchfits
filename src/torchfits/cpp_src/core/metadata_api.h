#pragma once

// Path-level FITS metadata for the torch-free core.
//
// Two layers, both free of libtorch and of nanobind so the same code serves
// the nanobind bindings in _core and in _C:
//
//   * ``fitsfile*``-level functions (``header_cards``, ``hdu_type_name``, ...)
//     assume the caller has already positioned the HDU cursor. ``FITSFile`` in
//     _C calls these, so the tensor extension and the metadata extension share
//     one implementation of every header/shape/dtype rule.
//   * path-level functions (``header_cards_path``, ``nrows_path``, ...)
//     open, query, and close around the shared per-path metadata cache. These
//     are the ones bound in _core: a header read never has to load libtorch.

#include <array>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fitsio.h>

#include "core/core_export.h"
#include "core/fits_core.h"

namespace torchfits {
namespace core {

using HeaderCard = std::tuple<std::string, std::string, std::string>;

// --------------------------------------------------------------------------
// fitsfile*-level inspection. The HDU cursor must already point at the HDU.
// --------------------------------------------------------------------------

// Full card list, duplicates preserved (HISTORY/COMMENT may repeat).
TORCHFITS_CORE_API std::vector<HeaderCard> header_cards(fitsfile* fptr);
// Bulk header text, ready for the Python fast parser.
TORCHFITS_CORE_API std::string header_text(fitsfile* fptr);
// Row-major (torch order) shape, i.e. NAXISn reversed.
TORCHFITS_CORE_API std::vector<long> image_shape(fitsfile* fptr);
TORCHFITS_CORE_API int image_bitpix(fitsfile* fptr);
// "IMAGE" / "ASCII_TABLE" / "BINARY_TABLE" / "UNKNOWN".
TORCHFITS_CORE_API std::string hdu_type_name(fitsfile* fptr);
TORCHFITS_CORE_API long table_nrows(fitsfile* fptr);
TORCHFITS_CORE_API std::vector<std::string> table_colnames(fitsfile* fptr);

struct TableInfo {
    long nrows = 0;
    std::vector<std::string> colnames;
    std::vector<std::string> tforms;
};
TORCHFITS_CORE_API TableInfo table_info(fitsfile* fptr);

// A header keyword plus the FITS type it round-trips as. Typing happens here
// (not in the bindings) so _C and _core cannot disagree about whether "17" is
// an int and "T" is a bool.
struct KeyValue {
    enum class Kind { None, Bool, Int, Double, Str };
    std::string key;
    std::string raw;
    Kind kind = Kind::Str;
    bool bool_value = false;
    long long int_value = 0;
    double double_value = 0.0;
    std::string str_value;
};
TORCHFITS_CORE_API std::vector<KeyValue> read_keywords(
    fitsfile* fptr, const std::vector<std::string>& keys);

// num_hdus, then the truncation check: fits_get_num_hdus stops silently at the
// first unparsable HDU, so a file truncated mid-header under-reports its
// inventory. `start_hdu` is the handle's absolute first HDU (1, or N for
// extended-syntax paths).
TORCHFITS_CORE_API int checked_num_hdus(fitsfile* fptr, int start_hdu);

// --------------------------------------------------------------------------
// Path-level entry points (open / query / close, via SharedReadMeta).
// --------------------------------------------------------------------------

TORCHFITS_CORE_API std::vector<HeaderCard> header_cards_path(const std::string& path, int hdu);
TORCHFITS_CORE_API std::string header_text_path(const std::string& path, int hdu);
TORCHFITS_CORE_API int num_hdus_path(const std::string& path);
TORCHFITS_CORE_API std::string hdu_type_path(const std::string& path, int hdu);
TORCHFITS_CORE_API long long nrows_path(const std::string& path, int hdu);
TORCHFITS_CORE_API std::vector<std::string> colnames_path(const std::string& path, int hdu);
TORCHFITS_CORE_API TableInfo table_info_path(const std::string& path, int hdu);
TORCHFITS_CORE_API std::vector<KeyValue> keywords_path(
    const std::string& path, int hdu, const std::vector<std::string>& keys);
// (bitpix, row-major shape) for an image HDU.
TORCHFITS_CORE_API std::pair<int, std::vector<long>> image_shape_path(const std::string& path, int hdu);

// --------------------------------------------------------------------------
// FitsReader: an open read-only handle for the _core module.
// --------------------------------------------------------------------------

class TORCHFITS_CORE_API FitsReader {
public:
    explicit FitsReader(const std::string& path);
    ~FitsReader();
    FitsReader(const FitsReader&) = delete;
    FitsReader& operator=(const FitsReader&) = delete;

    void close();
    bool closed() const { return fptr_ == nullptr; }
    fitsfile* fptr() const { return fptr_; }

    int num_hdus();
    std::string hdu_type(int hdu);
    std::vector<long> shape(int hdu);
    int bitpix(int hdu);
    std::vector<HeaderCard> header(int hdu);
    std::string header_text(int hdu);
    long nrows(int hdu);
    std::vector<std::string> colnames(int hdu);
    TableInfo table_info(int hdu);
    std::vector<KeyValue> keywords(int hdu, const std::vector<std::string>& keys);
    // (scaled, trusted, bscale, bzero)
    std::tuple<bool, bool, double, double> scale_info(int hdu);
    // (bitpix, naxis, NAXIS1..NAXIS9)
    std::tuple<int, int, std::array<LONGLONG, 9>> image_info(int hdu);
    bool is_compressed_image(int hdu);

private:
    void move_to(int hdu);

    std::string path_;
    fitsfile* fptr_ = nullptr;
    int start_hdu_ = 1;
    int current_hdu_ = -1;
    // CFITSIO exposes one mutable current-HDU cursor per fitsfile handle, so a
    // FitsReader is only safe to share if every accessor holds this across the
    // whole move-then-read, not just across the move. It is recursive because
    // the accessors call move_to() / image_info(), which take it again.
    // ``closed()`` and ``fptr()`` are the deliberate exception: they read the
    // pointer without locking, which is a benign race against close() and worth
    // avoiding a lock in a hot trivial accessor.
    mutable std::recursive_mutex io_mutex_;
};

}  // namespace core
}  // namespace torchfits
