/*
cd tests
nvcc -std=c++17 -O2 -I../include -I../include/layer -I../include/model -I../include/kernel \
    test_residual_add.cpp ../src/layer/ResidualAdd.cpp ../kernel/residual_add_kernel.cu \
    -o ../build/tests/test_residual_add.exe
./../build/tests/test_residual_add.exe
*/

#include "layer/ResidualAdd.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) && (count > 0);
}

void CheckCuda(cudaError_t err) {
    assert(err == cudaSuccess);
}

bool AlmostEqual(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

void RunAndCheckResidualAdd(bool use_prefill) {
    // input and residual (stored in output) both shape [2, 3].
    const std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    const std::vector<float> h_residual = {
        10.0f, 20.0f, 30.0f,
        40.0f, 50.0f, 60.0f
    };

    float* d_input = nullptr;
    float* d_output = nullptr;

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), h_input.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), h_residual.size() * sizeof(float)));

    CheckCuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_output, h_residual.data(), h_residual.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor input(
        h_input.size(),
        d_input,
        {2, 3},
        DataType::FLOAT32,
        "gpu"
    );

    Tensor output(
        h_residual.size(),
        d_output,
        {2, 3},
        DataType::FLOAT32,
        "gpu"
    );

    ResidualAdd residual_add;
    ForwardContext context{};

    if (use_prefill) {
        residual_add.prefill_forward(input, output, context);
    } else {
        residual_add.decode_forward(input, output, context);
    }

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_output(h_residual.size(), 0.0f);
    CheckCuda(cudaMemcpy(
        h_output.data(),
        d_output,
        h_output.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const std::vector<float> expected = {
        11.0f, 22.0f, 33.0f,
        44.0f, 55.0f, 66.0f
    };

    assert(h_output.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(AlmostEqual(h_output[i], expected[i]));
    }

    cudaFree(d_output);
    cudaFree(d_input);
}

void TestResidualAddPrefillForward() {
    RunAndCheckResidualAdd(true);
}

void TestResidualAddDecodeForward() {
    RunAndCheckResidualAdd(false);
}

} // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "No CUDA device found, skipping test_residual_add\n";
        return 0;
    }

    TestResidualAddPrefillForward();
    TestResidualAddDecodeForward();

    std::cout << "test_residual_add passed\n";
    return 0;
}
