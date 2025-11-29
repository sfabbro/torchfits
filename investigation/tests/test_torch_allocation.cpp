/**
 * Benchmark torch::empty() for different types
 *
 * Tests if int16 tensor allocation is slower than uint8
 */
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std::chrono;

double benchmark_allocation(torch::ScalarType dtype, const std::vector<int64_t>& shape, int iterations) {
    std::vector<double> times;

    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
        auto end = high_resolution_clock::now();

        times.push_back(duration<double, std::milli>(end - start).count());

        // Force tensor to actually allocate by touching first element
        auto ptr = tensor.data_ptr();
        (void)ptr;
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];  // median
}

int main() {
    std::vector<int64_t> shape = {1000, 1000};
    int iterations = 100;

    std::cout << "========================================" << std::endl;
    std::cout << "PyTorch Tensor Allocation Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Shape: 1000x1000" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << std::endl;

    auto uint8_time = benchmark_allocation(torch::kUInt8, shape, iterations);
    auto int16_time = benchmark_allocation(torch::kInt16, shape, iterations);
    auto int32_time = benchmark_allocation(torch::kInt32, shape, iterations);
    auto float32_time = benchmark_allocation(torch::kFloat32, shape, iterations);
    auto float64_time = benchmark_allocation(torch::kFloat64, shape, iterations);

    std::cout << "Median allocation times:" << std::endl;
    std::cout << "  uint8:   " << uint8_time << "ms" << std::endl;
    std::cout << "  int16:   " << int16_time << "ms" << std::endl;
    std::cout << "  int32:   " << int32_time << "ms" << std::endl;
    std::cout << "  float32: " << float32_time << "ms" << std::endl;
    std::cout << "  float64: " << float64_time << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Ratios vs uint8:" << std::endl;
    std::cout << "  int16:   " << (int16_time / uint8_time) << "x" << std::endl;
    std::cout << "  int32:   " << (int32_time / uint8_time) << "x" << std::endl;
    std::cout << "  float32: " << (float32_time / uint8_time) << "x" << std::endl;
    std::cout << "  float64: " << (float64_time / uint8_time) << "x" << std::endl;
    std::cout << std::endl;

    double int16_overhead = int16_time - uint8_time;
    std::cout << "int16 allocation overhead: " << int16_overhead << "ms" << std::endl;

    return 0;
}
