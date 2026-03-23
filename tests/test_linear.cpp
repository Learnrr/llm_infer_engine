/*
cd tests
nvcc -std=c++17 -O2 -I../include -I../include/layer -I../include/model -I../include/kernel \
    test_linear.cpp ../src/layer/Linear.cpp ../kernel/linear_kernel.cu \
    -o ../build/tests/test_linear.exe
./../build/tests/test_linear.exe
*/

#include "layer/Linear.h"

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

Linear BuildLinearLayer(Tensor& weight) {
    LinearConfig cfg;
    cfg.in_features = 3;
    cfg.out_features = 2;
    return Linear(cfg, 0, weight);
}

void RunAndCheckLinear(bool use_prefill) {
    // Input shape [2, 3].
    const std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    // Weight shape [3, 2], row-major by in_feature.
    // [ [1,0], [0,1], [1,1] ]
    const std::vector<float> h_weight = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };

    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), h_input.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_weight), h_weight.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), 4 * sizeof(float)));

    CheckCuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemset(d_output, 0, 4 * sizeof(float)));

    Tensor input(
        h_input.size(),
        d_input,
        {2, 3},
        DataType::FLOAT32,
        "gpu"
    );

    Tensor weight(
        h_weight.size(),
        d_weight,
        {3, 2},
        DataType::FLOAT32,
        "gpu"
    );

    Tensor output(
        4,
        d_output,
        {2, 2},
        DataType::FLOAT32,
        "gpu"
    );

    Linear linear = BuildLinearLayer(weight);

    Batch batch;
    batch.num_tokens = 2;

    ForwardContext context{};
    context.layer_id = 0;
    context.batch = &batch;
    context.workspace = nullptr;
    context.config = nullptr;

    if (use_prefill) {
        linear.prefill_forward(input, output, context);
    } else {
        linear.decode_forward(input, output, context);
    }

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_output(4, 0.0f);
    CheckCuda(cudaMemcpy(h_output.data(), d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // Expected output:
    // token0: [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]
    // token1: [4*1 + 5*0 + 6*1, 4*0 + 5*1 + 6*1] = [10, 11]
    const std::vector<float> expected = {4.0f, 5.0f, 10.0f, 11.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(AlmostEqual(h_output[i], expected[i]));
    }

    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_input);
}

void TestLinearPrefillForward() {
    RunAndCheckLinear(true);
}

void TestLinearDecodeForward() {
    RunAndCheckLinear(false);
}

} // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "No CUDA device found, skipping test_linear\n";
        return 0;
    }

    TestLinearPrefillForward();
    TestLinearDecodeForward();

    std::cout << "test_linear passed\n";
    return 0;
}
