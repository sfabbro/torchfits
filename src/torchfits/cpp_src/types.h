#pragma once

#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

namespace torchfits {

namespace nb = nanobind;

// Define Tensor as a nanobind ndarray with pytorch compatibility
using Tensor = nb::ndarray<nb::pytorch, nb::c_contig>;
using ScalarType = nb::dlpack::dtype;

} // namespace torchfits