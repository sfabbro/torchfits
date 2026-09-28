#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>
#include <ATen/ATen.h>
#include <fitsio.h>

#include "torchfits_torch.h"
#include "fits_file.h"

namespace torchfits {

struct HDUInfo {
    int index;
    std::string type;
    std::vector<std::tuple<std::string, std::string, std::string>> header;
};

torch::Tensor read_full_cached(const std::string& path, int hdu_num, bool use_mmap);
torch::Tensor read_full_unmapped(const std::string& path, int hdu_num);
torch::Tensor read_full_unmapped_raw(const std::string& path, int hdu_num);
torch::Tensor read_full_nocache(const std::string& path, int hdu_num, bool use_mmap);
int resolve_hdu_name_cached(const std::string& filename, const std::string& hdu_name);
std::vector<torch::Tensor> read_images_batch(const std::vector<std::string>& paths, int hdu_num, bool use_mmap = true);
std::vector<torch::Tensor> read_hdus_batch(const std::string& path, const std::vector<int>& hdus, bool use_mmap);
torch::Tensor read_hdus_sequence_last(const std::string& path, const std::vector<int>& hdus, bool use_mmap);
std::pair<FITSFile*, std::vector<HDUInfo>> open_and_read_headers(const std::string& path, int mode);
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header, nb::object schema_obj, bool is_ascii);
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header);
void* get_fptr_from_python_object(nanobind::object obj);

void invalidate_shared_meta(const std::string& filename);
void clear_shared_read_meta_cache();

// Drop every thread's cached TableReader for `filename` (defined in
// table_bindings.cpp). Writers call this before opening a file READWRITE:
//
// a cached reader holds a CFITSIO handle open READONLY, and CFITSIO refuses to
// reopen a file READWRITE while such a handle is still registered in its
// FptrTable. Measured: the second open returns FILE_NOT_OPENED (104), and
// returns 0 again once the READONLY handle is closed.
//
// The status is raised inside CFITSIO's fits_already_open() (vendored
// extern/cfitsio/cfileio.c), which is the *function* that performs the
// same-file lookup -- not a status code. The status it sets is the generic
// FILE_NOT_OPENED, whose own name ("could not open the named file") points away
// from the cache-conflict meaning. That mismatch is why this retry used to carry
// a bare 104 and a comment naming a constant that does not exist.
void evict_cached_reader(const std::string& filename);

// Open `path` for writing, retrying once if a cached READONLY handle for it
// was registered in-process. The retry drops that handle and re-opens, so
// read-then-mutate sequences work without the caller knowing about the cache.
//
// Known limitation, measured: CFITSIO reuses FILE_NOT_OPENED for the
// READONLY-blocks-READWRITE refusal, so this retry also fires on any *ordinary*
// failure to open -- a missing directory, a path that is a directory, a
// permissions error. When the retry cannot fix the conflict, the two are
// indistinguishable by status alone, so the failure is reported with both
// explanations rather than CFITSIO's bare "could not open the named file",
// which names neither. Measured: the most common cause is a *live* reader from
// torchfits.open_table_reader(), which owns its own CFITSIO handle; eviction
// only drops the cache, so the retry cannot release it and the write must wait
// for the caller to close the reader.
//
// Throws std::runtime_error in that one case, so every caller's message improves
// at once. Callers' `if (status != 0)` branches still handle every other status;
// this is the only path that throws from here, and it is documented as such.
inline int open_fits_for_write(fitsfile** fptr, const std::string& path) {
    int status = 0;
    *fptr = nullptr;
    fits_open_file(fptr, path.c_str(), 1 /* READWRITE */, &status);
    if (status == FILE_NOT_OPENED) {
        evict_cached_reader(path);
        status = 0;
        *fptr = nullptr;
        fits_open_file(fptr, path.c_str(), 1 /* READWRITE */, &status);
        if (status == FILE_NOT_OPENED) {
            throw std::runtime_error(
                "Failed to open '" + path + "' for writing: CFITSIO still reports "
                "the file as already open after every cached reader was dropped. "
                "The usual cause is a reader that is still open -- one obtained "
                "from torchfits.open_table_reader() holds its own CFITSIO handle, "
                "which cache eviction cannot release; close it (or drop the "
                "reference) and retry. Otherwise the path is probably not "
                "writable: a missing parent directory, or insufficient "
                "permissions.");
        }
    }
    return status;
}

} // namespace torchfits
