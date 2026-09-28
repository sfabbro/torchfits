#include "core/metadata_api.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include <fitsio2.h>

#include "security.h"

namespace torchfits {
namespace core {
namespace {

namespace d = torchfits::detail;

// FITS string values arrive quoted: strip the quotes, the trailing padding,
// and the doubled '' escape. Applied to every card value so the Python side
// sees the same text astropy would.
void unquote_fits_string(std::string& value) {
    if (value.length() < 2 || value.front() != '\'') return;
    const size_t last_quote = value.rfind('\'');
    if (last_quote == std::string::npos || last_quote == 0) return;
    value = value.substr(1, last_quote - 1);
    const size_t last_char = value.find_last_not_of(' ');
    value = (last_char == std::string::npos) ? std::string()
                                              : value.substr(0, last_char + 1);
    size_t pos = 0;
    while ((pos = value.find("''", pos)) != std::string::npos) {
        value.replace(pos, 2, "'");
        pos += 1;
    }
}

[[noreturn]] void throw_fits_error(int status, const char* what) {
    char err_text[FLEN_ERRMSG] = {0};
    fits_get_errstatus(status, err_text);
    throw std::runtime_error(std::string(what) + ": " + err_text);
}

std::string read_tstring_card(fitsfile* fptr, const char* keyname) {
    char value[FLEN_VALUE];
    std::memset(value, 0, FLEN_VALUE);
    int status = 0;
    fits_read_key(fptr, TSTRING, keyname, value, nullptr, &status);
    if (status != 0) return std::string();
    size_t len = strnlen(value, FLEN_VALUE);
    while (len > 0 && value[len - 1] == ' ') --len;
    return std::string(value, len);
}

// Open for reading, run `body`, always close. Path-level metadata must not leak
// a fitsfile* even when the query throws (a bad EXTNAME, a non-table HDU).
template <typename F>
auto with_open(const std::string& path, F&& body) -> decltype(body(std::declval<fitsfile*>())) {
    check_fits_filename_security(path);
    fitsfile* fptr = nullptr;
    const int status = d::open_fits_readonly(&fptr, path);
    if (status != 0 || fptr == nullptr) {
        throw std::runtime_error("Could not open FITS file: " + path);
    }
    struct Guard {
        fitsfile* fptr;
        ~Guard() {
            int close_status = 0;
            fits_close_file(fptr, &close_status);
        }
    } guard{fptr};
    return body(fptr);
}

}  // namespace

// ---------------------------------------------------------------------------
// fitsfile*-level inspection
// ---------------------------------------------------------------------------

std::vector<HeaderCard> header_cards(fitsfile* fptr) {
    int status = 0;
    int nkeys = 0;
    int morekeys = 0;
    fits_get_hdrspace(fptr, &nkeys, &morekeys, &status);
    std::vector<HeaderCard> header;
    header.reserve(static_cast<size_t>(nkeys < 0 ? 0 : nkeys));
    char keyname[FLEN_KEYWORD];
    char value[FLEN_VALUE];
    char comment[FLEN_COMMENT];
    for (int i = 1; i <= nkeys; ++i) {
        status = 0;
        fits_read_keyn(fptr, i, keyname, value, comment, &status);
        if (status != 0) {
            status = 0;
            continue;
        }
        std::string val_str = d::sanitize_fits_string(std::string(value));
        unquote_fits_string(val_str);
        std::string com_str(comment);
        if (std::string(keyname) == "HISTORY" || std::string(keyname) == "COMMENT") {
            if (val_str.empty() && !com_str.empty()) {
                val_str = com_str;
                com_str.clear();
            }
        }
        header.emplace_back(std::string(keyname), val_str, com_str);
    }
    return header;
}

std::string header_text(fitsfile* fptr) {
    int status = 0;
    char* text = nullptr;
    int nkeys = 0;
    const int hdr_status = fits_hdr2str(fptr, 0, nullptr, 0, &text, &nkeys, &status);
    if (hdr_status != 0 || status != 0 || text == nullptr) {
        if (text != nullptr) fits_free_memory(text, &status);
        return std::string();
    }
    std::string result(text);
    fits_free_memory(text, &status);
    return result;
}

std::vector<long> image_shape(fitsfile* fptr) {
    int status = 0;
    int naxis = 0;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status != 0) throw std::runtime_error("Could not read image dimensions");
    d::validate_image_naxis(naxis);
    std::vector<long> naxes(static_cast<size_t>(naxis));
    if (naxis > 0) {
        fits_get_img_size(fptr, naxis, naxes.data(), &status);
        if (status != 0) throw std::runtime_error("Could not read image size");
    }
    std::reverse(naxes.begin(), naxes.end());
    return naxes;
}

int image_bitpix(fitsfile* fptr) {
    int status = 0;
    int bitpix = 0;
    fits_get_img_type(fptr, &bitpix, &status);
    if (status != 0) throw std::runtime_error("Could not read image type");
    return bitpix;
}

std::string hdu_type_name(fitsfile* fptr) {
    int status = 0;
    int hdutype = 0;
    fits_get_hdu_type(fptr, &hdutype, &status);
    if (status != 0) throw std::runtime_error("Could not read HDU type");
    if (hdutype == IMAGE_HDU) return "IMAGE";
    if (hdutype == ASCII_TBL) return "ASCII_TABLE";
    if (hdutype == BINARY_TBL) return "BINARY_TABLE";
    return "UNKNOWN";
}

long table_nrows(fitsfile* fptr) {
    int status = 0;
    long nrows = 0;
    fits_get_num_rows(fptr, &nrows, &status);
    if (status != 0) throw_fits_error(status, "read_nrows (HDU must be a table)");
    return nrows;
}

std::vector<std::string> table_colnames(fitsfile* fptr) {
    int status = 0;
    int ncols = 0;
    fits_get_num_cols(fptr, &ncols, &status);
    if (status != 0) throw_fits_error(status, "read_colnames (HDU must be a table)");
    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(ncols));
    for (int i = 1; i <= ncols; ++i) {
        char keyname[FLEN_KEYWORD];
        std::snprintf(keyname, FLEN_KEYWORD, "TTYPE%d", i);
        std::string name = read_tstring_card(fptr, keyname);
        if (name.empty()) {
            char fallback[FLEN_VALUE];
            std::snprintf(fallback, FLEN_VALUE, "COL%d", i);
            name = fallback;
        }
        names.emplace_back(std::move(name));
    }
    return names;
}

TableInfo table_info(fitsfile* fptr) {
    TableInfo info;
    info.nrows = table_nrows(fptr);
    int status = 0;
    int ncols = 0;
    fits_get_num_cols(fptr, &ncols, &status);
    if (status != 0) throw_fits_error(status, "read_table_info (HDU must be a table)");
    info.colnames.reserve(static_cast<size_t>(ncols));
    info.tforms.reserve(static_cast<size_t>(ncols));
    for (int i = 1; i <= ncols; ++i) {
        char keyname[FLEN_KEYWORD];
        std::snprintf(keyname, FLEN_KEYWORD, "TTYPE%d", i);
        std::string name = read_tstring_card(fptr, keyname);
        if (name.empty()) {
            char fallback[FLEN_VALUE];
            std::snprintf(fallback, FLEN_VALUE, "COL%d", i);
            name = fallback;
        }
        info.colnames.emplace_back(std::move(name));
        std::snprintf(keyname, FLEN_KEYWORD, "TFORM%d", i);
        info.tforms.emplace_back(read_tstring_card(fptr, keyname));
    }
    return info;
}

std::vector<KeyValue> read_keywords(fitsfile* fptr, const std::vector<std::string>& keys) {
    std::vector<KeyValue> out;
    out.reserve(keys.size());
    for (const auto& key : keys) {
        char value[FLEN_VALUE] = {0};
        char comment[FLEN_COMMENT] = {0};
        int status = 0;
        if (fits_read_keyword(fptr, key.c_str(), value, comment, &status) != 0) {
            if (status == KEY_NO_EXIST) {
                throw std::runtime_error("read_keys: keyword not found: " + key);
            }
            char err_text[FLEN_ERRMSG] = {0};
            fits_get_errstatus(status, err_text);
            throw std::runtime_error(
                std::string("read_keys: ") + err_text + " (" + key + ")");
        }
        KeyValue kv;
        kv.key = key;
        kv.raw = d::sanitize_fits_string(std::string(value));
        const std::string& val = kv.raw;
        if (val.empty()) {
            kv.kind = KeyValue::Kind::None;
        } else if (val == "T" || val == "F") {
            kv.kind = KeyValue::Kind::Bool;
            kv.bool_value = (val == "T");
        } else if (val.front() == '\'') {
            std::string s = val;
            unquote_fits_string(s);
            kv.kind = KeyValue::Kind::Str;
            kv.str_value = s;
        } else {
            bool typed = false;
            try {
                // Full-consumption check: a partially numeric string
                // ("1999-01-01") must stay a string, not truncate to its
                // leading number.
                size_t pos = 0;
                if (val.find_first_of(".eE") != std::string::npos) {
                    const double dv = std::stod(val, &pos);
                    while (pos < val.size() &&
                           std::isspace(static_cast<unsigned char>(val[pos]))) ++pos;
                    if (pos == val.size()) {
                        kv.kind = KeyValue::Kind::Double;
                        kv.double_value = dv;
                        typed = true;
                    }
                } else {
                    const long long iv = std::stoll(val, &pos);
                    while (pos < val.size() &&
                           std::isspace(static_cast<unsigned char>(val[pos]))) ++pos;
                    if (pos == val.size()) {
                        kv.kind = KeyValue::Kind::Int;
                        kv.int_value = iv;
                        typed = true;
                    }
                }
            } catch (const std::exception&) {
                typed = false;
            }
            if (!typed) {
                kv.kind = KeyValue::Kind::Str;
                kv.str_value = val;
            }
        }
        out.push_back(std::move(kv));
    }
    return out;
}

int checked_num_hdus(fitsfile* fptr, int start_hdu) {
    int status = 0;
    int num_hdus = 0;
    fits_get_num_hdus(fptr, &num_hdus, &status);
    if (status != 0) throw std::runtime_error("Could not read number of HDUs");
    if (num_hdus <= 0 || fptr == nullptr) {
        return num_hdus;
    }

    status = 0;
    char name[FLEN_FILENAME] = {0};
    fits_file_name(fptr, name, &status);
    if (status != 0 || name[0] == '\0') {
        return num_hdus;
    }
    const std::string path(name);
    if (has_cfitsio_extended_filename_syntax(path) ||
        path.find("://") != std::string::npos) {
        return num_hdus;  // no reliable on-disk extent to compare
    }
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        return num_hdus;
    }

    int mstatus = 0;
    fits_movabs_hdu(fptr, start_hdu + num_hdus - 1, nullptr, &mstatus);
    if (mstatus != 0) {
        return num_hdus;
    }
    LONGLONG headstart = 0;
    LONGLONG datastart = 0;
    LONGLONG dataend = 0;
    int astatus = 0;
    fits_get_hduaddrll(fptr, &headstart, &datastart, &dataend, &astatus);
    if (astatus != 0) {
        return num_hdus;
    }
    // ffghadll: dataend is the byte offset where the next HDU would begin --
    // the exclusive, record-aligned end of the last parsed HDU.
    const LONGLONG fsize = static_cast<LONGLONG>(st.st_size);
    if (dataend <= 0 || dataend >= fsize) {
        return num_hdus;  // no trailing bytes (a short file is data truncation)
    }

    unsigned char probe[8] = {0};
    const int fd = d::open_readonly_fd(path);
    if (fd == -1) {
        return num_hdus;
    }
    const ssize_t got = ::pread(fd, probe, sizeof(probe), static_cast<off_t>(dataend));
    ::close(fd);
    if (got <= 0) {
        return num_hdus;
    }
    auto starts_hdu_header = [probe, got](const char* word) {
        const size_t want = std::strlen(word);
        const size_t have = std::min(static_cast<size_t>(got), want);
        return std::memcmp(probe, word, have) == 0;
    };
    if (starts_hdu_header("SIMPLE") || starts_hdu_header("XTENSION")) {
        throw std::runtime_error(
            "FITS file appears truncated or corrupt: the HDU scan parsed " +
            std::to_string(num_hdus) + " HDU(s) but another HDU header at byte " +
            std::to_string(dataend) + " could not be parsed: " + path);
    }
    return num_hdus;
}

// ---------------------------------------------------------------------------
// Path-level entry points
// ---------------------------------------------------------------------------

std::vector<HeaderCard> header_cards_path(const std::string& path, int hdu) {
    return with_open(path, [hdu](fitsfile* fptr) {
        int status = 0;
        fits_movabs_hdu(fptr, hdu + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("read_header_dict: could not move to HDU");
        return header_cards(fptr);
    });
}

std::string header_text_path(const std::string& path, int hdu) {
    return with_open(path, [hdu](fitsfile* fptr) {
        int status = 0;
        fits_movabs_hdu(fptr, hdu + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("read_header_string: could not move to HDU");
        return d::drop_non_ascii_bytes(header_text(fptr));
    });
}

int num_hdus_path(const std::string& path) {
    check_fits_filename_security(path);
    auto meta = d::get_shared_meta_for_path(path);
    if (meta) {
        std::shared_lock<std::shared_mutex> lock(meta->mutex);
        if (meta->num_hdus >= 0) return meta->num_hdus;
    }
    const int num_hdus = with_open(path, [](fitsfile* fptr) {
        return checked_num_hdus(fptr, 1);
    });
    if (meta) {
        std::unique_lock<std::shared_mutex> lock(meta->mutex);
        meta->num_hdus = num_hdus;
    }
    return num_hdus;
}

std::string hdu_type_path(const std::string& path, int hdu) {
    check_fits_filename_security(path);
    auto meta = d::get_shared_meta_for_path(path);
    if (meta) {
        std::shared_lock<std::shared_mutex> lock(meta->mutex);
        auto it = meta->hdu_type_cache.find(hdu);
        if (it != meta->hdu_type_cache.end()) return it->second;
    }
    const std::string hdu_type = with_open(path, [hdu](fitsfile* fptr) {
        int status = 0;
        fits_movabs_hdu(fptr, hdu + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("read_hdu_type: could not move to HDU");
        return hdu_type_name(fptr);
    });
    if (meta) {
        std::unique_lock<std::shared_mutex> lock(meta->mutex);
        meta->hdu_type_cache[hdu] = hdu_type;
    }
    return hdu_type;
}

long long nrows_path(const std::string& path, int hdu) {
    // Guard before get_shared_meta_for_path: it stats the path and inserts a
    // global-map entry.
    check_fits_filename_security(path);
    auto meta = d::get_shared_meta_for_path(path);
    if (meta) {
        std::shared_lock<std::shared_mutex> lock(meta->mutex);
        auto it = meta->nrows_cache.find(hdu);
        if (it != meta->nrows_cache.end()) return it->second;
    }
    const long nrows = with_open(path, [hdu](fitsfile* fptr) {
        int status = 0;
        fits_movabs_hdu(fptr, hdu + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("read_nrows: could not move to HDU");
        return table_nrows(fptr);
    });
    if (meta) {
        std::unique_lock<std::shared_mutex> lock(meta->mutex);
        meta->nrows_cache[hdu] = static_cast<long long>(nrows);
    }
    return static_cast<long long>(nrows);
}

std::vector<std::string> colnames_path(const std::string& path, int hdu) {
    check_fits_filename_security(path);
    auto meta = d::get_shared_meta_for_path(path);
    if (meta) {
        std::shared_lock<std::shared_mutex> lock(meta->mutex);
        auto it = meta->colnames_cache.find(hdu);
        if (it != meta->colnames_cache.end()) return it->second;
    }
    const std::vector<std::string> names = with_open(path, [hdu](fitsfile* fptr) {
        int status = 0;
        fits_movabs_hdu(fptr, hdu + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("read_colnames: could not move to HDU");
        return table_colnames(fptr);
    });
    if (meta) {
        std::unique_lock<std::shared_mutex> lock(meta->mutex);
        meta->colnames_cache[hdu] = names;
    }
    return names;
}

TableInfo table_info_path(const std::string& path, int hdu) {
    return with_open(path, [hdu](fitsfile* fptr) {
        int status = 0;
        fits_movabs_hdu(fptr, hdu + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("read_table_info: could not move to HDU");
        return table_info(fptr);
    });
}

std::vector<KeyValue> keywords_path(
    const std::string& path, int hdu, const std::vector<std::string>& keys) {
    return with_open(path, [&](fitsfile* fptr) {
        int status = 0;
        fits_movabs_hdu(fptr, hdu + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("read_keys: could not move to HDU");
        return read_keywords(fptr, keys);
    });
}

std::pair<int, std::vector<long>> image_shape_path(const std::string& path, int hdu) {
    check_fits_filename_security(path);
    // Warm SharedReadMeta: skip the CFITSIO open when image params were
    // already populated by a prior read / SubsetReader.
    auto meta = d::get_shared_meta_for_path(path);
    if (meta) {
        std::shared_lock<std::shared_mutex> lock(meta->mutex);
        auto it = meta->image_info_cache.find(hdu);
        if (it != meta->image_info_cache.end()) {
            const auto& cached = it->second;
            const int bitpix = std::get<0>(cached);
            const int naxis = std::get<1>(cached);
            const std::array<LONGLONG, 9>& naxes_ll = std::get<2>(cached);
            std::vector<long> shape;
            shape.reserve(static_cast<size_t>(naxis));
            // Torch / row-major order (reverse of FITS NAXISn).
            for (int i = naxis - 1; i >= 0; --i) {
                shape.push_back(static_cast<long>(naxes_ll[static_cast<size_t>(i)]));
            }
            return {bitpix, std::move(shape)};
        }
    }
    return with_open(path, [hdu, &path](fitsfile* fptr) {
        int status = 0;
        fits_movabs_hdu(fptr, hdu + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("read_shape: could not move to HDU");
        int bitpix = 0;
        int naxis = 0;
        std::array<LONGLONG, 9> naxes_ll{};
        naxes_ll.fill(0);
        d::read_image_params_9d(fptr, &bitpix, &naxis, naxes_ll, &status);
        if (status != 0) throw std::runtime_error("Could not read image parameters");
        auto cache = d::get_shared_meta_for_path(path);
        if (cache) {
            std::unique_lock<std::shared_mutex> lock(cache->mutex);
            cache->image_info_cache[hdu] = std::make_tuple(bitpix, naxis, naxes_ll);
        }
        std::vector<long> shape;
        shape.reserve(static_cast<size_t>(naxis));
        for (int i = naxis - 1; i >= 0; --i) {
            shape.push_back(static_cast<long>(naxes_ll[static_cast<size_t>(i)]));
        }
        return std::make_pair(bitpix, std::move(shape));
    });
}

// ---------------------------------------------------------------------------
// FitsReader
// ---------------------------------------------------------------------------

FitsReader::FitsReader(const std::string& path) : path_(path) {
    check_fits_filename_security(path_);
    int status = 0;
    status = d::open_fits_readonly(&fptr_, path_);
    if (status != 0 || fptr_ == nullptr) {
        if (fptr_ != nullptr) {
            int close_status = 0;
            fits_close_file(fptr_, &close_status);
            fptr_ = nullptr;
        }
        throw std::runtime_error("Could not open FITS file: " + path_);
    }
    if (has_cfitsio_extended_filename_syntax(path_)) {
        fits_get_hdu_num(fptr_, &start_hdu_);
        current_hdu_ = start_hdu_;
    } else {
        // Private handle owns its own CHDU -- do not seed from the shared meta.
        start_hdu_ = 1;
        current_hdu_ = -1;
    }
}

FitsReader::~FitsReader() { close(); }

void FitsReader::close() {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    if (fptr_ != nullptr) {
        int status = 0;
        fits_close_file(fptr_, &status);
        fptr_ = nullptr;
    }
}

void FitsReader::move_to(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    if (fptr_ == nullptr) throw std::runtime_error("FitsReader is closed");
    const int target_hdu = hdu + start_hdu_;
    if (current_hdu_ == target_hdu) return;
    int status = 0;
    fits_movabs_hdu(fptr_, target_hdu, nullptr, &status);
    if (status != 0) throw std::runtime_error("Could not move to HDU");
    current_hdu_ = target_hdu;
}

int FitsReader::num_hdus() {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    if (fptr_ == nullptr) throw std::runtime_error("FitsReader is closed");
    return core::checked_num_hdus(fptr_, start_hdu_);
}

// Every accessor below takes io_mutex_ for the *whole* move-then-read, not just
// for the move. CFITSIO keeps one mutable current-HDU cursor per fitsfile, so
// releasing the lock between fits_movabs_hdu() and the fits_* query that
// depends on it lets another thread change the cursor underneath the read.
// Measured with two threads on one FitsReader before this was fixed: wrong
// answers from other HDUs, spurious "Could not read image dimensions", and a
// segfault. The lock is recursive because move_to() and image_info() take it
// again on the way through.
std::string FitsReader::hdu_type(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    return core::hdu_type_name(fptr_);
}
std::vector<long> FitsReader::shape(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    return core::image_shape(fptr_);
}
int FitsReader::bitpix(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    return core::image_bitpix(fptr_);
}
std::vector<HeaderCard> FitsReader::header(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    return core::header_cards(fptr_);
}
std::string FitsReader::header_text(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    return d::drop_non_ascii_bytes(core::header_text(fptr_));
}
long FitsReader::nrows(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    return core::table_nrows(fptr_);
}
std::vector<std::string> FitsReader::colnames(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    return core::table_colnames(fptr_);
}
TableInfo FitsReader::table_info(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    return core::table_info(fptr_);
}

std::vector<KeyValue> FitsReader::keywords(
    int hdu, const std::vector<std::string>& keys) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    return core::read_keywords(fptr_, keys);
}

std::tuple<bool, bool, double, double> FitsReader::scale_info(int hdu) {
    // Held across both halves: the BSCALE/BZERO query reads the same cursor the
    // image-parameter read just positioned, so letting go in between would let
    // a concurrent call move it out from under detect_scale_info_fast().
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    const auto info = image_info(hdu);
    const int bitpix = std::get<0>(info);
    const d::ScaleDetectionResult detected = d::detect_scale_info_fast(fptr_, bitpix);
    return std::make_tuple(detected.scaled, detected.trusted, detected.bscale, detected.bzero);
}

std::tuple<int, int, std::array<LONGLONG, 9>> FitsReader::image_info(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    int status = 0;
    int bitpix = 0;
    int naxis = 0;
    std::array<LONGLONG, 9> naxes_ll{};
    naxes_ll.fill(0);
    d::read_image_params_9d(fptr_, &bitpix, &naxis, naxes_ll, &status);
    if (status != 0) throw std::runtime_error("Could not read image parameters");
    return std::make_tuple(bitpix, naxis, naxes_ll);
}

bool FitsReader::is_compressed_image(int hdu) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    move_to(hdu);
    int status = 0;
    const int is_compressed = fits_is_compressed_image(fptr_, &status);
    return status == 0 && is_compressed;
}

}  // namespace core
}  // namespace torchfits
