// Profile the tensor return path to isolate THPVariable_Wrap overhead
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <torch/torch.h>
#include <torch/csrc/autograd/python_variable.h>
#include <Python.h>

using namespace std::chrono;

double benchmark_tensor_wrap(torch::ScalarType dtype, std::vector<int64_t> shape, int iterations) {
    std::vector<double> times;

    // Initialize Python (needed for THPVariable_Wrap)
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    for (int i = 0; i < iterations; i++) {
        // Create tensor with some data
        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));

        auto start = high_resolution_clock::now();

        // This is what we do in bindings.cpp
        PyObject* tensor_obj = THPVariable_Wrap(tensor);

        auto end = high_resolution_clock::now();

        // Clean up
        Py_XDECREF(tensor_obj);

        times.push_back(duration<double>(end - start).count() * 1e6);  // microseconds
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double benchmark_full_path(torch::ScalarType dtype, std::vector<int64_t> shape, int iterations) {
    std::vector<double> times;

    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();

        // Full path: create tensor + wrap
        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
        PyObject* tensor_obj = THPVariable_Wrap(tensor);

        auto end = high_resolution_clock::now();

        Py_XDECREF(tensor_obj);

        times.push_back(duration<double>(end - start).count() * 1e6);  // microseconds
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double benchmark_tensor_creation(torch::ScalarType dtype, std::vector<int64_t> shape, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
        auto end = high_resolution_clock::now();

        times.push_back(duration<double>(end - start).count() * 1e6);  // microseconds
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main() {
    std::cout << "Profiling Tensor Return Path (1000x1000 tensors, 1000 iterations)\n";
    std::cout << "================================================================\n\n";

    std::vector<int64_t> shape = {1000, 1000};
    const int iterations = 1000;

    // Test uint8
    std::cout << "UINT8:\n";
    std::cout << "----------------------------------------\n";
    double u8_create = benchmark_tensor_creation(torch::kUInt8, shape, iterations);
    double u8_wrap = benchmark_tensor_wrap(torch::kUInt8, shape, iterations);
    double u8_full = benchmark_full_path(torch::kUInt8, shape, iterations);

    std::cout << "  torch::empty():       " << u8_create << "μs\n";
    std::cout << "  THPVariable_Wrap():   " << u8_wrap << "μs\n";
    std::cout << "  Full path (both):     " << u8_full << "μs\n";
    std::cout << "  Overhead (wrap only): " << (u8_wrap) << "μs\n";
    std::cout << "\n";

    // Test int16
    std::cout << "INT16:\n";
    std::cout << "----------------------------------------\n";
    double i16_create = benchmark_tensor_creation(torch::kInt16, shape, iterations);
    double i16_wrap = benchmark_tensor_wrap(torch::kInt16, shape, iterations);
    double i16_full = benchmark_full_path(torch::kInt16, shape, iterations);

    std::cout << "  torch::empty():       " << i16_create << "μs\n";
    std::cout << "  THPVariable_Wrap():   " << i16_wrap << "μs\n";
    std::cout << "  Full path (both):     " << i16_full << "μs\n";
    std::cout << "  Overhead (wrap only): " << (i16_wrap) << "μs\n";
    std::cout << "\n";

    // Analysis
    std::cout << "================================================================\n";
    std::cout << "ANALYSIS:\n";
    std::cout << "================================================================\n\n";

    std::cout << "Tensor creation overhead (int16/uint8): " << i16_create / u8_create << "x\n";
    std::cout << "THPVariable_Wrap overhead (int16/uint8): " << i16_wrap / u8_wrap << "x\n";
    std::cout << "Full path overhead (int16/uint8):        " << i16_full / u8_full << "x\n";
    std::cout << "\n";

    double wrap_overhead_diff = i16_wrap - u8_wrap;
    std::cout << "Extra wrap overhead for int16: " << wrap_overhead_diff << "μs\n";

    if (wrap_overhead_diff > 10.0) {
        std::cout << "\n⚠️  WARNING: THPVariable_Wrap has significant int16 overhead!\n";
        std::cout << "This could explain the int16 performance gap.\n";
    } else {
        std::cout << "\n✅ THPVariable_Wrap overhead is similar for both types.\n";
        std::cout << "The bottleneck must be elsewhere.\n";
    }

    // Cleanup
    if (Py_IsInitialized()) {
        Py_Finalize();
    }

    return 0;
}
