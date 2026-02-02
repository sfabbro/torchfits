#pragma once

#include <nanobind/nanobind.h>
#include "torchfits_torch.h"
#include <Python.h>

namespace nb = nanobind;

// Manual definitions to interface with libtorch_python without including headers that pull in pybind11
// This allows us to use the native PyTorch C++ API for tensor conversion without conflicts.

// Forward declare THPVariableClass
extern PyObject* THPVariableClass;

// Check if object is a Tensor
inline bool THPVariable_Check(PyObject* obj) {
    return PyObject_IsInstance(obj, THPVariableClass);
}

// THPVariable struct definition (must match torch/csrc/autograd/python_variable.h)
// This is necessary because THPVariable_Unpack is inline and accesses members directly.
struct THPVariable {
    PyObject_HEAD
    c10::MaybeOwned<at::Tensor> cdata;
    PyObject* backward_hooks;
    PyObject* post_accumulate_grad_hooks;
};

// Extern declaration for THPVariable_Wrap (exported by libtorch_python)
extern PyObject* THPVariable_Wrap(const at::TensorBase& var);

// Unpack helper
inline const at::Tensor& THPVariable_Unpack(PyObject* obj) {
    return *(((THPVariable*)obj)->cdata);
}

// Helper function to convert torch::Tensor to Python object - FAST PATH
inline nb::object tensor_to_python(const torch::Tensor& tensor) {
    PyObject* tensor_obj = THPVariable_Wrap(tensor);
    if (!tensor_obj) {
        throw std::runtime_error("Failed to wrap tensor");
    }
    return nb::steal(tensor_obj);
}

// Helper function to convert Python object to torch::Tensor - FAST PATH
inline torch::Tensor python_to_tensor(nb::object obj) {
    if (!THPVariable_Check(obj.ptr())) {
        throw std::runtime_error("Object is not a PyTorch tensor");
    }
    return THPVariable_Unpack(obj.ptr());
}
