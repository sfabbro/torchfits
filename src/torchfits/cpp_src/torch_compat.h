#pragma once

#include <nanobind/nanobind.h>
#include "torchfits_torch.h"
#include <ATen/DLConvertor.h>
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

// Extern declaration for THPVariable_Wrap (exported by libtorch_python)
extern PyObject* THPVariable_Wrap(const at::TensorBase& var);

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
    if (!PyObject_HasAttrString(obj.ptr(), "__dlpack__")) {
        throw std::runtime_error("Object does not implement __dlpack__");
    }
    PyObject* capsule_obj = PyObject_CallMethod(obj.ptr(), "__dlpack__", nullptr);
    if (!capsule_obj) {
        throw nb::python_error();
    }
    nb::object capsule = nb::steal(capsule_obj);

    auto* dl_managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor")
    );
    if (!dl_managed) {
        throw nb::python_error();
    }
    auto t = at::fromDLPack(dl_managed);

    // Mark the capsule as consumed to avoid the capsule destructor calling deleter
    // a second time after ATen takes ownership.
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") != 0) {
        PyErr_Clear();
    }
    return t;
}
