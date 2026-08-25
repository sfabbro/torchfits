#include <nanobind/nanobind.h>
#include <string>

namespace nb = nanobind;

void bind_fits(nb::module_& m);
void bind_table(nb::module_& m);

NB_MODULE(_C, m) {
    const nb::str runtime_version_object(
        nb::module_::import_("torch").attr("__version__")
    );
    const std::string runtime_version(runtime_version_object.c_str());
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
#ifdef TORCHFITS_HAVE_BZIP2
    m.attr("HAS_BZIP2") = true;
#else
    m.attr("HAS_BZIP2") = false;
#endif
    bind_fits(m);
    bind_table(m);
}
