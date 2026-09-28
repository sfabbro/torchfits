// nanobind surface for libtorchfits_core, exposed as ``torchfits._core``.
//
// This module is the metadata half of the native stack. Unlike ``_C`` it does
// not link libtorch, so ``import torchfits._core`` never dlopens libtorch and
// never imports the Python ``torch`` module -- which is what makes a header or
// shape read cheap enough to sit on an interactive prompt's critical path.
//
// ``_C`` and ``_core`` share one implementation (core/metadata_api.cpp) and one
// shared-cache (core/fits_core.cpp), so the two entry points cannot disagree
// about a header, a dtype, or a scale factor.

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <Python.h>

#include <array>

#include "core/fits_core.h"
#include "core/metadata_api.h"
#include "core/parallel.h"

#ifndef TORCHFITS_CORE_BUILD_ID
#define TORCHFITS_CORE_BUILD_ID "unknown"
#endif

namespace nb = nanobind;

namespace {

namespace c = torchfits::core;
namespace d = torchfits::detail;

nb::dict key_values_to_dict(const std::vector<c::KeyValue>& values) {
    nb::dict out;
    for (const auto& kv : values) {
        const char* key = kv.key.c_str();
        switch (kv.kind) {
            case c::KeyValue::Kind::None:
                out[key] = nb::none();
                break;
            case c::KeyValue::Kind::Bool:
                out[key] = kv.bool_value;
                break;
            case c::KeyValue::Kind::Int:
                out[key] = kv.int_value;
                break;
            case c::KeyValue::Kind::Double:
                out[key] = kv.double_value;
                break;
            case c::KeyValue::Kind::Str:
                out[key] = kv.str_value;
                break;
        }
    }
    return out;
}

nb::list header_cards_to_list(const std::vector<c::HeaderCard>& cards) {
    nb::list out;
    for (const auto& card : cards) {
        out.append(nb::make_tuple(std::get<0>(card), std::get<1>(card), std::get<2>(card)));
    }
    return out;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "torchfits metadata core: CFITSIO inspection with no libtorch linkage";

    // Build identity. ``_C`` embeds the same string; torchfits/__init__.py
    // compares the two so a stale libtorchfits_core next to a fresh _C fails
    // loudly instead of crossing an ABI boundary.
    m.attr("__build_id__") = TORCHFITS_CORE_BUILD_ID;
    m.attr("__core_api_version__") = 1;
    // Asserted by tests/test_torch_boundary.py: this module must never gain a
    // libtorch dependency.
    m.attr("TORCH_FREE") = true;
    // Identity of the shared object this module actually loaded, which can
    // differ from the identity the module itself was compiled with.
    m.def("core_library_build_id", []() { return std::string(d::core_library_build_id()); });
    m.def("thread_count", []() { return c::thread_count(); });

    nb::class_<c::FitsReader>(m, "Metadata",
        "Read-only FITS handle for metadata queries. Holds no tensor state.\n\n"
        "Axis order differs between two of the accessors on purpose: ``shape()``\n"
        "is row-major (NAXISn reversed), ``image_info()`` is FITS NAXISn order.\n"
        "See each method's docstring before mixing them.")
        .def(nb::init<const std::string&>(), nb::arg("path"))          .def("num_hdus", &c::FitsReader::num_hdus)
          .def("hdu_type", &c::FitsReader::hdu_type, nb::arg("hdu"))
          // Axis order is the one thing a caller cannot check for itself, so it
          // is stated on every accessor that hands back a shape.
          .def("shape", &c::FitsReader::shape, nb::arg("hdu"),
               "Row-major (torch order) shape: NAXISn reversed.\n\n"
               "This is the shape the tensor paths use. ``image_info()`` returns\n"
               "the same dimensions in FITS NAXISn order instead.")
          .def("bitpix", &c::FitsReader::bitpix, nb::arg("hdu"))
        .def("nrows", &c::FitsReader::nrows, nb::arg("hdu"))
        .def("colnames", &c::FitsReader::colnames, nb::arg("hdu"))
        .def("table_info",
             [](c::FitsReader& self, int hdu) {
                 const c::TableInfo info = self.table_info(hdu);
                 nb::dict out;
                 out["nrows"] = static_cast<long long>(info.nrows);
                 out["colnames"] = info.colnames;
                 out["tforms"] = info.tforms;
                 return out;
             },
             nb::arg("hdu"))
        .def("keywords",
             [](c::FitsReader& self, int hdu, const std::vector<std::string>& keys) {
                 return key_values_to_dict(self.keywords(hdu, keys));
             },
             nb::arg("hdu"), nb::arg("keys"))
        .def("header",
             [](c::FitsReader& self, int hdu) { return header_cards_to_list(self.header(hdu)); },
             nb::arg("hdu"))
        .def("header_text", &c::FitsReader::header_text, nb::arg("hdu"))
        .def("scale_info", &c::FitsReader::scale_info, nb::arg("hdu"))
        .def("image_info",
             [](c::FitsReader& self, int hdu) {
                 const auto info = self.image_info(hdu);
                 const int naxis = std::get<1>(info);
                 const std::array<LONGLONG, 9>& naxes = std::get<2>(info);
                 nb::list dims;
                 for (int i = 0; i < naxis && i < 9; ++i) {
                     dims.append(static_cast<long long>(naxes[static_cast<size_t>(i)]));
                 }
                 return nb::make_tuple(std::get<0>(info), naxis, nb::tuple(dims));
             },               nb::arg("hdu"),
               "(bitpix, naxis, (NAXIS1 .. NAXISn)) in FITS order, NOT row-major.\n\n"
               "The axis tuple is *not* the shape: it is NAXIS1 first, so for a\n"
               "non-square HDU it is the transpose of ``shape()``. Callers that\n"
               "want a torch shape want ``shape()``; this one exists to report\n"
               "NAXISn as written, e.g. to compare against a header.")
          .def("is_compressed_image", &c::FitsReader::is_compressed_image, nb::arg("hdu"))
        .def("close", &c::FitsReader::close)
        .def("closed", &c::FitsReader::closed)
        .def("__enter__", [](c::FitsReader& self) -> c::FitsReader& { return self; },
             nb::rv_policy::reference_internal)
        // `.none(true)`: nanobind's object caster rejects None by default,
        // and all three values CPython passes here are None on a clean exit.
        // Without it every `with Metadata(...)` raises TypeError.
        .def("__exit__",
             [](c::FitsReader& self, nb::object, nb::object, nb::object) {
                 self.close();
                 return false;
             },
             nb::arg("exc_type").none(),
             nb::arg("exc_value").none(),
             nb::arg("traceback").none())
        .def("__repr__", [](const c::FitsReader& self) {
            return std::string("<torchfits._core.Metadata ") +
                   (self.closed() ? "closed" : "open") + ">";
        });

    m.def("read_header_dict",
          [](const std::string& path, int hdu) {
              std::vector<c::HeaderCard> cards;
              {
                  nb::gil_scoped_release release;
                  cards = c::header_cards_path(path, hdu);
              }
              return header_cards_to_list(cards);
          },
          nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_header_string",
          [](const std::string& path, int hdu) {
              std::string text;
              {
                  nb::gil_scoped_release release;
                  text = c::header_text_path(path, hdu);
              }
              return text;
          },
          nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_num_hdus",
          [](const std::string& path) {
              int n = 0;
              {
                  nb::gil_scoped_release release;
                  n = c::num_hdus_path(path);
              }
              return n;
          },
          nb::arg("filename"));

    m.def("read_hdu_type",
          [](const std::string& path, int hdu) {
              std::string type;
              {
                  nb::gil_scoped_release release;
                  type = c::hdu_type_path(path, hdu);
              }
              return type;
          },
          nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_nrows",
          [](const std::string& path, int hdu) {
              long long nrows = 0;
              {
                  nb::gil_scoped_release release;
                  nrows = c::nrows_path(path, hdu);
              }
              return nrows;
          },
          nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_colnames",
          [](const std::string& path, int hdu) {
              std::vector<std::string> names;
              {
                  nb::gil_scoped_release release;
                  names = c::colnames_path(path, hdu);
              }
              return names;
          },
          nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_table_info",
          [](const std::string& path, int hdu) {
              c::TableInfo info;
              {
                  nb::gil_scoped_release release;
                  info = c::table_info_path(path, hdu);
              }
              nb::dict out;
              out["nrows"] = static_cast<long long>(info.nrows);
              out["colnames"] = info.colnames;
              out["tforms"] = info.tforms;
              return out;
          },
          nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_keys",
          [](const std::string& path, int hdu, const std::vector<std::string>& keys) {
              std::vector<c::KeyValue> values;
              {
                  nb::gil_scoped_release release;
                  values = c::keywords_path(path, hdu, keys);
              }
              return key_values_to_dict(values);
          },
          nb::arg("filename"), nb::arg("hdu_num"), nb::arg("keys"));

    m.def("read_shape",
          [](const std::string& path, int hdu) {
              std::pair<int, std::vector<long>> shape;
              {
                  nb::gil_scoped_release release;
                  shape = c::image_shape_path(path, hdu);
              }
              nb::list dims;
              for (long dim : shape.second) {
                  dims.append(static_cast<long long>(dim));
              }
              return nb::make_tuple(shape.first, nb::tuple(dims));
          },               nb::arg("filename"), nb::arg("hdu_num"),
               "(bitpix, row-major shape) for an image HDU.\n\n"
               "Row-major means NAXISn reversed, matching ``Metadata.shape()``\n"
               "and the tensor paths -- not ``Metadata.image_info()``.");

      m.def("clear_shared_read_meta_cache", []() {
        d::clear_shared_meta_cache();
    });

    // How many paths the shared read-metadata cache currently holds. Not part
    // of any public contract: it exists so a test can assert that _C and _core
    // really do resolve to the same registry (two caches would both "work" and
    // quietly disagree after a file is rewritten).
    m.def("shared_meta_entry_count", []() {
        std::lock_guard<std::mutex> lock(d::g_shared_meta_mutex);
        return static_cast<long long>(d::g_shared_meta.size());
    });
}
