#include <nanobind/nanobind.h>
#include <Python.h>
#include <string>

namespace nb = nanobind;

#ifndef TORCHFITS_CORE_BUILD_ID
#define TORCHFITS_CORE_BUILD_ID "unknown"
#endif

void bind_fits(nb::module_& m);
void bind_table(nb::module_& m);

namespace {

// Importing the extension is also the metadata-only import boundary.  The
// extension is linked to libtorch, but linking a shared library does not import
// the Python ``torch`` module.  Keep the ABI guard eager when the caller has
// already imported torch (the normal tensor path), while allowing a metadata
// caller to import this module without paying for the Python runtime.
void check_torch_abi(bool force_import) {
    PyObject* torch_module = nullptr;
    bool owns_torch_module = false;
    if (force_import) {
        torch_module = PyImport_ImportModule("torch");
        if (torch_module == nullptr) {
            throw nb::python_error();
        }
        owns_torch_module = true;
    } else {
        PyObject* modules = PyImport_GetModuleDict();
        if (modules == nullptr) {
            return;
        }
        torch_module = PyDict_GetItemString(modules, "torch");
        if (torch_module == nullptr) {
            return;
        }
    }

    PyObject* version_object = PyObject_GetAttrString(torch_module, "__version__");
    if (version_object == nullptr) {
        if (owns_torch_module) {
            Py_DECREF(torch_module);
        }
        throw nb::python_error();
    }
    const char* version_text = PyUnicode_AsUTF8(version_object);
    if (version_text == nullptr) {
        Py_DECREF(version_object);
        if (owns_torch_module) {
            Py_DECREF(torch_module);
        }
        throw nb::python_error();
    }
    const std::string runtime_version(version_text);
    Py_DECREF(version_object);
    if (owns_torch_module) {
        Py_DECREF(torch_module);
    }

    const std::string required_abi = TORCHFITS_TORCH_ABI;
    const bool matching_abi =
        runtime_version.compare(0, required_abi.size(), required_abi) == 0
        && (runtime_version.size() == required_abi.size()
            || runtime_version[required_abi.size()] == '.'
            || runtime_version[required_abi.size()] == '+');
    if (!matching_abi) {
        PyErr_Format(
            PyExc_ImportError,
            "torchfits was built for PyTorch %s.x but found PyTorch %s",
            required_abi.c_str(),
            runtime_version.c_str()
        );
        throw nb::python_error();
    }
}

} // namespace

void ensure_torch_abi() {
    static bool checked = false;
    if (!checked) {
        check_torch_abi(true);
        checked = true;
    }
}

NB_MODULE(_C, m) {
    check_torch_abi(false);
#ifdef TORCHFITS_HAVE_BZIP2
    m.attr("HAS_BZIP2") = true;
#else
    m.attr("HAS_BZIP2") = false;
#endif
    // libtorchfits_core owns CFITSIO and the metadata cache; _C calls into it.
    // torchfits/__init__.py compares this against torchfits._core.__build_id__
    // so a stale core next to a fresh extension fails at import, not later.
    m.attr("__core_build_id__") = TORCHFITS_CORE_BUILD_ID;
    // Python calls this after loading torch when a process first entered via
    // the metadata-only path and the extension was already imported without
    // torch.  Keep it private; it is an ABI guard, not public API.
    m.def("_check_torch_abi", []() { check_torch_abi(true); });
    bind_fits(m);
    bind_table(m);
}
