#pragma once
#include <nanobind/nanobind.h>

namespace torchfits {
    void* get_fptr_from_python_object(nanobind::object obj);
}
