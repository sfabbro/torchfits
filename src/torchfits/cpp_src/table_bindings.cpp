/**
 * Table read/write/mutation bindings for the _C module.
 */

#include <string>
#include <cstdlib>
#include <vector>
#include <list>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <fitsio.h>

#include "torchfits_torch.h"
#include "torch_compat.h"
#include "fits_rw.h"
#include "fits_detail.h"
#include "cache.h"
#include "table_types.h"
#include "table_reader.h"
#include "table_ops.h"

namespace nb = nanobind;

namespace {

struct ReaderCacheKey {
    std::string filename;
    int hdu_num = 0;
    bool operator==(const ReaderCacheKey& o) const {
        return filename == o.filename && hdu_num == o.hdu_num;
    }
};

struct ReaderCacheKeyHash {
    size_t operator()(const ReaderCacheKey& k) const noexcept {
        size_t h = std::hash<std::string>()(k.filename);
        return h ^ ((size_t)k.hdu_num * 0x9e3779b97f4a7c15ULL);
    }
};

// Thread-local LRU of owned TableReader handles. Repeated cold table reads of
// the same (file, hdu) reuse the CFITSIO handle, the ~16 MiB row scratch
// buffer and the pread fd, so steady-state reads stop re-opening the file,
// re-parsing the header and re-faulting ~24 MiB of anonymous memory per call
// (mirrors fitsio's persistent-handle behavior).
//
// Every cache registers in a process-wide registry so writers can evict the
// file's readers from ALL threads: a cached reader holds a CFITSIO handle
// open READONLY, and CFITSIO refuses to reopen a file READWRITE in the same
// process while such a handle is registered (status 104 in fits_already_open).
// Cache entries are only ever inserted/removed by their owning thread, but
// evict() from another thread may destroy an idle cached reader, so map_/lru_
// are mutex-guarded.
struct ThreadLocalReaderCache {
    struct Entry {
        std::unique_ptr<torchfits::TableReader> reader;
        std::list<ReaderCacheKey>::iterator lru_it;
    };

    ThreadLocalReaderCache() {
        std::lock_guard<std::mutex> g(g_registry_mu);
        g_registry.push_back(this);
    }

    ~ThreadLocalReaderCache() {
        std::lock_guard<std::mutex> g(g_registry_mu);
        auto it = std::find(g_registry.begin(), g_registry.end(), this);
        if (it != g_registry.end()) {
            g_registry.erase(it);
        }
    }

    std::mutex mu_;
    std::unordered_map<ReaderCacheKey, Entry, ReaderCacheKeyHash> map_;
    std::list<ReaderCacheKey> lru_;
    static constexpr size_t kCapacity = 8;

    std::unique_ptr<torchfits::TableReader> acquire(const ReaderCacheKey& key) {
        std::lock_guard<std::mutex> l(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) {
            return std::unique_ptr<torchfits::TableReader>();
        }
        lru_.splice(lru_.end(), lru_, it->second.lru_it);
        auto reader = std::move(it->second.reader);
        map_.erase(it);
        if (!reader->file_unchanged()) {
            // The file was replaced or rewritten since this reader was cached
            // (e.g. a writer on another thread whose eviction raced this
            // thread's release): the handle, cached offset and pread fd are
            // stale. Drop it and let the caller open fresh.
            return std::unique_ptr<torchfits::TableReader>();
        }
        return reader;
    }

    void release(ReaderCacheKey key, std::unique_ptr<torchfits::TableReader> reader) {
        if (!reader) {
            return;
        }
        std::lock_guard<std::mutex> l(mu_);
        if (map_.size() >= kCapacity) {
            const ReaderCacheKey victim = lru_.front();
            lru_.pop_front();
            map_.erase(victim);
        }
        lru_.push_back(key);
        map_[key] = Entry{std::move(reader), std::prev(lru_.end())};
    }

    // Drop any cached reader for `filename` from this cache. A reader is only
    // destroyed here while idle (not acquired), so a concurrent read on the
    // owning thread never touches an evicted reader.
    void evict(const std::string& filename) {
        std::lock_guard<std::mutex> l(mu_);
        for (auto it = map_.begin(); it != map_.end();) {
            if (it->first.filename == filename) {
                lru_.erase(it->second.lru_it);
                it = map_.erase(it);
            } else {
                ++it;
            }
        }
    }

    static void evict_everywhere(const std::string& filename) {
        std::lock_guard<std::mutex> g(g_registry_mu);
        for (ThreadLocalReaderCache* cache : g_registry) {
            cache->evict(filename);
        }
    }

private:
    static std::mutex g_registry_mu;
    static std::vector<ThreadLocalReaderCache*> g_registry;
};

std::mutex ThreadLocalReaderCache::g_registry_mu;
std::vector<ThreadLocalReaderCache*> ThreadLocalReaderCache::g_registry;

static thread_local ThreadLocalReaderCache g_reader_cache;

nb::dict tensor_map_to_python(
    const std::vector<std::pair<std::string, torch::Tensor>>& result_map
) {
    // Must be called with the GIL held: wraps each C++ tensor as a Python
    // object. The heavy read work itself happens GIL-free in the callers.
    // Input order (file/request column order) is preserved in the dict.
    nb::dict result_dict;
    for (auto& [key, tensor] : result_map) {
        result_dict[key.c_str()] = tensor_to_python(tensor);
    }
    return result_dict;
}

nb::dict table_result_to_python(
    const std::vector<std::pair<std::string, torchfits::TableReader::ColumnData>>& result_map,
    bool as_numpy
) {
    nb::dict result_dict;
    for (auto& [key, col_data] : result_map) {
        if (col_data.is_vla) {
            if (col_data.vla_offsets.defined() && col_data.fixed_data.defined()) {
                if (as_numpy) {
                    result_dict[key.c_str()] = nb::make_tuple(
                        tensor_to_numpy_object(col_data.fixed_data),
                        tensor_to_numpy_object(col_data.vla_offsets)
                    );
                    continue;
                }
                // One flat buffer → list of row views (avoids per-row CFITSIO + extra C++ vector).
                const torch::Tensor& values = col_data.fixed_data;
                const int64_t* op = col_data.vla_offsets.data_ptr<int64_t>();
                const long n = col_data.vla_offsets.size(0) - 1;
                nb::list vla_list;
                for (long i = 0; i < n; i++) {
                    vla_list.append(tensor_to_python(values.slice(0, op[i], op[i + 1])));
                }
                result_dict[key.c_str()] = vla_list;
                continue;
            }
            nb::list vla_list;
            for (const auto& tensor : col_data.vla_data) {
                vla_list.append(as_numpy ? tensor_to_numpy_object(tensor) : tensor_to_python(tensor));
            }
            result_dict[key.c_str()] = vla_list;
        } else {
            result_dict[key.c_str()] = as_numpy ? tensor_to_numpy_object(col_data.fixed_data)
                                                : tensor_to_python(col_data.fixed_data);
        }
    }
    return result_dict;
}

} // anonymous namespace

namespace torchfits {

void evict_cached_reader(const std::string& filename) {
    ThreadLocalReaderCache::evict_everywhere(filename);
}

}  // namespace torchfits

// Forward declare invalidation functions (defined in cache.cpp)
namespace torchfits {
void invalidate_cached(const std::string& filepath);
void invalidate_shared_meta(const std::string& filepath);
}

void bind_table(nb::module_& m) {
    nb::class_<torchfits::TableReader>(m, "TableReader")
        .def("__init__", [](torchfits::TableReader* self, const std::string& filename, int hdu_num) {
            new (self) torchfits::TableReader(filename, hdu_num);
        }, nb::arg("filename"), nb::arg("hdu_num") = 1)
        .def("__init__", [](torchfits::TableReader* self, nb::object file_obj, int hdu_num) {
            fitsfile* fptr = reinterpret_cast<fitsfile*>(torchfits::get_fptr_from_python_object(file_obj));
            new (self) torchfits::TableReader(fptr, hdu_num);
        }, nb::arg("file_obj"), nb::arg("hdu_num") = 1)
        .def_prop_ro("num_rows", &torchfits::TableReader::get_num_rows)
        .def("read_rows", [](torchfits::TableReader& self,
                             const std::vector<std::string>& column_names,
                             long start_row, long num_rows) -> nb::object {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> io_lock(self.io_mutex_);
            auto result_map = self.read_columns(column_names, start_row, num_rows, true);
            nb::gil_scoped_acquire acquire;
            return table_result_to_python(result_map, false);
        }, nb::arg("column_names") = std::vector<std::string>(),
           nb::arg("start_row") = 1, nb::arg("num_rows") = -1)
        .def("read_rows_numpy", [](torchfits::TableReader& self,
                                   const std::vector<std::string>& column_names,
                                   long start_row, long num_rows) -> nb::object {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> io_lock(self.io_mutex_);
            auto result_map = self.read_columns(column_names, start_row, num_rows, true);
            nb::gil_scoped_acquire acquire;
            return table_result_to_python(result_map, true);
        }, nb::arg("column_names") = std::vector<std::string>(),
           nb::arg("start_row") = 1, nb::arg("num_rows") = -1)
        .def_prop_ro("num_cols", &torchfits::TableReader::get_num_cols);

    m.def("evict_cached_reader", &torchfits::evict_cached_reader, nb::arg("path"));

    m.def("write_fits_table", [](const std::string& filename, nb::dict tensor_dict, nb::dict header, bool overwrite,
                                 nb::object schema, const std::string& table_type) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        torchfits::evict_cached_reader(filename);
        write_fits_table(filename.c_str(), tensor_dict, header, overwrite, schema, table_type);
    }, nb::arg("filename"), nb::arg("tensor_dict"), nb::arg("header"), nb::arg("overwrite"),
       nb::arg("schema") = nb::none(), nb::arg("table_type") = "binary");

    m.def("append_fits_table_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        torchfits::evict_cached_reader(filename);
        append_rows(filename.c_str(), hdu_num, tensor_dict);
    });

    m.def("insert_fits_table_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict,
                                       long start_row) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        torchfits::evict_cached_reader(filename);
        insert_rows(filename.c_str(), hdu_num, tensor_dict, start_row);
    });

    m.def("update_fits_table_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict,
                                       long start_row, long num_rows) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        torchfits::evict_cached_reader(filename);
        update_rows(filename.c_str(), hdu_num, tensor_dict, start_row, num_rows);
    });

    m.def("update_fits_table_rows_mmap", [](const std::string& filename, int hdu_num, nb::dict tensor_dict,
                                            long start_row, long num_rows) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        torchfits::evict_cached_reader(filename);
        update_rows_mmap(filename.c_str(), hdu_num, tensor_dict, start_row, num_rows);
    });

    m.def("rename_fits_table_columns", [](const std::string& filename, int hdu_num, nb::dict mapping) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        torchfits::evict_cached_reader(filename);
        rename_columns(filename.c_str(), hdu_num, mapping);
    });

    m.def("drop_fits_table_columns", [](const std::string& filename, int hdu_num, nb::list columns) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        torchfits::evict_cached_reader(filename);
        drop_columns(filename.c_str(), hdu_num, columns);
    });

    m.def("delete_fits_table_rows", [](const std::string& filename, int hdu_num, long start_row,
                                       long num_rows) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        delete_rows(filename.c_str(), hdu_num, start_row, num_rows);
    });

    m.def("read_fits_table", [](const std::string& filename, int hdu_num) -> nb::object {
        nb::gil_scoped_release release;
        torchfits::TableReader reader(filename, hdu_num);
        auto result_map = reader.read_columns({}, 1, -1, true);
        nb::gil_scoped_acquire acquire;
        nb::dict result_dict;
        for (auto& [key, col_data] : result_map) {
            if (col_data.is_vla) {
                nb::list vla_list;
                for (const auto& tensor : col_data.vla_data) {
                    vla_list.append(tensor_to_python(tensor));
                }
                result_dict[key.c_str()] = vla_list;
            } else {
                result_dict[key.c_str()] = tensor_to_python(col_data.fixed_data);
            }
        }
        return result_dict;
    });

    m.def("read_fits_table_from_handle", [](nb::object file_obj, int hdu_num) -> nb::object {
        fitsfile* fptr = reinterpret_cast<fitsfile*>(torchfits::get_fptr_from_python_object(file_obj));
        nb::gil_scoped_release release;
        torchfits::TableReader reader(fptr, hdu_num);
        auto result_map = reader.read_columns({}, 1, -1, true);
        nb::gil_scoped_acquire acquire;
        return table_result_to_python(result_map, false);
    });

    m.def("read_fits_table_rows_from_handle", [](nb::object file_obj, int hdu_num,
                                                 const std::vector<std::string>& column_names,
                                                 long start_row, long num_rows) -> nb::object {
        fitsfile* fptr = reinterpret_cast<fitsfile*>(torchfits::get_fptr_from_python_object(file_obj));
        nb::gil_scoped_release release;
        torchfits::TableReader reader(fptr, hdu_num);
        auto result_map = reader.read_columns(column_names, start_row, num_rows, true);
        nb::gil_scoped_acquire acquire;
        return table_result_to_python(result_map, false);
    }, nb::arg("file"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1);

    m.def("read_fits_table", [](const std::string& filename, int hdu_num, const std::vector<std::string>& column_names, bool mmap) -> nb::object {
        nb::gil_scoped_release release;
        ReaderCacheKey key{filename, hdu_num};
        std::unique_ptr<torchfits::TableReader> reader = g_reader_cache.acquire(key);
        if (!reader) {
            reader = std::make_unique<torchfits::TableReader>(filename, hdu_num);
        }
        if (mmap) {
            auto result = reader->read_columns_mmap(column_names);
            g_reader_cache.release(key, std::move(reader));
            nb::gil_scoped_acquire acquire;
            return tensor_map_to_python(result);
        } else {
            auto result_map = reader->read_columns(column_names, 1, -1, true);
            g_reader_cache.release(key, std::move(reader));
            nb::gil_scoped_acquire acquire;
            nb::object out = nb::object(table_result_to_python(result_map, false));
            return out;
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1, nb::arg("column_names") = std::vector<std::string>(), nb::arg("mmap") = false);

    m.def("read_fits_table_rows", [](const std::string& filename, int hdu_num,
                                     const std::vector<std::string>& column_names,
                                     long start_row, long num_rows, bool mmap) -> nb::object {
        nb::gil_scoped_release release;
        ReaderCacheKey key{filename, hdu_num};
        std::unique_ptr<torchfits::TableReader> reader = g_reader_cache.acquire(key);
        if (!reader) {
            reader = std::make_unique<torchfits::TableReader>(filename, hdu_num);
        }
        if (mmap) {
            auto result = reader->read_columns_mmap(column_names, start_row, num_rows);
            g_reader_cache.release(key, std::move(reader));
            nb::gil_scoped_acquire acquire;
            return tensor_map_to_python(result);
        } else {
            auto result_map = reader->read_columns(column_names, start_row, num_rows, true);
            g_reader_cache.release(key, std::move(reader));
            nb::gil_scoped_acquire acquire;
            nb::object out = table_result_to_python(result_map, false);
            return out;
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1, nb::arg("mmap") = false);

    m.def("read_fits_table_rows_numpy_from_handle", [](nb::object file_obj, int hdu_num,
                                                       const std::vector<std::string>& column_names,
                                                       long start_row, long num_rows) -> nb::object {
        fitsfile* fptr = reinterpret_cast<fitsfile*>(torchfits::get_fptr_from_python_object(file_obj));
        nb::gil_scoped_release release;
        torchfits::TableReader reader(fptr, hdu_num);
        auto result_map = reader.read_columns(column_names, start_row, num_rows, true);
        nb::gil_scoped_acquire acquire;
        return table_result_to_python(result_map, true);
    }, nb::arg("file"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1);

    m.def("read_fits_table_rows_numpy", [](const std::string& filename, int hdu_num,
                                           const std::vector<std::string>& column_names,
                                           long start_row, long num_rows, bool mmap) -> nb::object {
        nb::gil_scoped_release release;
        ReaderCacheKey key{filename, hdu_num};
        std::unique_ptr<torchfits::TableReader> reader = g_reader_cache.acquire(key);
        if (!reader) {
            reader = std::make_unique<torchfits::TableReader>(filename, hdu_num);
        }
        if (mmap) {
            auto result_map = reader->read_columns_mmap(column_names, start_row, num_rows);
            g_reader_cache.release(key, std::move(reader));
            nb::gil_scoped_acquire acquire;
            nb::dict mapped = tensor_map_to_python(result_map);
            nb::dict numpy_result;
            for (auto item : mapped) {
                nb::handle key_h = item.first;
                nb::handle value = item.second;
                if (PyObject_HasAttrString(value.ptr(), "numpy")) {
                    PyObject* np_obj = PyObject_CallMethod(value.ptr(), "numpy", nullptr);
                    if (!np_obj) {
                        throw nb::python_error();
                    }
                    numpy_result[key_h] = nb::steal(np_obj);
                } else {
                    numpy_result[key_h] = nb::borrow(value);
                }
            }
            return nb::object(numpy_result);
        } else {
            auto result_map = reader->read_columns(column_names, start_row, num_rows, true);
            g_reader_cache.release(key, std::move(reader));
            nb::gil_scoped_acquire acquire;
            nb::object out = table_result_to_python(result_map, true);
            return out;
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1, nb::arg("mmap") = false);

    m.def("open_fits_mmap_reader", [](const std::string& path, int hdu_num) -> nb::capsule {
        // Persistent TableReader for batch mmap scans: opens + parses the
        // header once; each read_rows call maps/unmaps per batch but never
        // re-opens the file.
        torchfits::TableReader* reader = new torchfits::TableReader(path, hdu_num);
        return nb::capsule(reader, [](void* p) noexcept {
            delete static_cast<torchfits::TableReader*>(p);
        });
    }, nb::arg("path"), nb::arg("hdu_num") = 1);

    m.def("read_fits_table_rows_mmap_from_reader", [](nb::capsule reader_cap,
                                                      const std::vector<std::string>& column_names,
                                                      long start_row, long num_rows) -> nb::object {
        auto* reader = static_cast<torchfits::TableReader*>(reader_cap.data());
        if (reader == nullptr) {
            throw std::runtime_error("Invalid mmap reader capsule");
        }
        nb::gil_scoped_release release;
        // A persistent capsule reader can be shared across threads (e.g.
        // DataLoader workers); serialize access to its CFITSIO cursor.
        std::lock_guard<std::mutex> io_lock(reader->io_mutex_);
        auto result = reader->read_columns_mmap(column_names, start_row, num_rows);
        nb::gil_scoped_acquire acquire;
        return tensor_map_to_python(result);
    }, nb::arg("reader"), nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1);

    m.def("read_fits_table_filtered", [](const std::string& filename, int hdu_num,
                                         const std::vector<std::string>& column_names,
                                         nb::list filters_py) -> nb::object {
        std::vector<torchfits::TableFilter> filters;
        for (auto handle : filters_py) {
            nb::tuple item = nb::cast<nb::tuple>(handle);
            if (item.size() != 3) throw std::runtime_error("Filter must be (col, op, val)");

            torchfits::TableFilter f;
            f.col_name = nb::cast<std::string>(item[0]);
            std::string op = nb::cast<std::string>(item[1]);

            if (op == "==" || op == "eq") f.op = torchfits::FilterOp::EQ;
            else if (op == "!=" || op == "ne") f.op = torchfits::FilterOp::NE;
            else if (op == ">" || op == "gt") f.op = torchfits::FilterOp::GT;
            else if (op == "<" || op == "lt") f.op = torchfits::FilterOp::LT;
            else if (op == ">=" || op == "ge") f.op = torchfits::FilterOp::GE;
            else if (op == "<=" || op == "le") f.op = torchfits::FilterOp::LE;
            else throw std::runtime_error("Unknown operator: " + op);

            nb::handle val = item[2];
            if (nb::isinstance<float>(val)) {
                 f.val_d = nb::cast<double>(val);
                 f.val_i = (int64_t)f.val_d;
                 f.type_idx = 0;
            } else if (nb::isinstance<int>(val)) {
                 f.val_i = nb::cast<int64_t>(val);
                 f.val_d = (double)f.val_i;
                 f.type_idx = 1;
            } else {
                 throw std::runtime_error("Unsupported filter value type (only float/int)");
            }
            filters.push_back(f);
        }

        nb::gil_scoped_release release;
        torchfits::TableReader reader(filename, hdu_num);
        auto result_map = reader.read_columns_mmap_filtered(column_names, filters);
        nb::gil_scoped_acquire acquire;

        nb::dict result;
        for (auto& [key, val] : result_map) {
             result[key.c_str()] = tensor_to_python(val);
        }
        return result;
    }, nb::arg("filename"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("filters"));
}
