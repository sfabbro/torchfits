#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_fits(nb::module_& m);
void bind_table(nb::module_& m);
void bind_compression(nb::module_& m);

NB_MODULE(_C, m) {
    bind_fits(m);
    bind_table(m);
    bind_compression(m);
}
