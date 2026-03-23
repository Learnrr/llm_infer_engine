/*
cd tests
nvcc -std=c++17 -O2 -I../include -I../include/layer -I../include/layer/activation \
-I../include/model -I../include/kernel test_swiglu.cpp ../src/layer/activation/SwiGLU.cpp ../kernel/swiglu_kernel.cu \
    -o ../build/tests/test_swiglu.exe
./../build/tests/test_swiglu.exe
*/

#include "layer/activation/SwiGLU.h"
#include "Batch.h"
#include "ForwardContext.h"

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

bool AlmostEqual(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> ComputeSwiGluReference(
    const std::vector<float>& gate,
    const std::vector<float>& up,
    size_t num_tokens,
    size_t hidden_size
) {
    std::vector<float> out(num_tokens * hidden_size, 0.0f);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = gate[i] * Sigmoid(gate[i]) * up[i];
    }
    return out;
}

void RunAndCheckSwiGlu() {
    const size_t num_tokens = 2;
    const size_t hidden_size = 3;

    const std::vector<float> h_gate = {
        1.0f, -1.0f, 0.5f,
        0.0f, 2.0f, -2.0f
    };
    const std::vector<float> h_up = {
        2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f
    };

    float* d_gate = nullptr;
    float* d_up = nullptr;
    float* d_output = nullptr;

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_gate), h_gate.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_up), h_up.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), num_tokens * hidden_size * sizeof(float)));

    CheckCuda(cudaMemcpy(d_gate, h_gate.data(), h_gate.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_up, h_up.data(), h_up.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemset(d_output, 0, num_tokens * hidden_size * sizeof(float)));

    Tensor gate(
        h_gate.size(),
        d_gate,
        {num_tokens, hidden_size},
        DataType::FLOAT32,
        "gpu"
    );

    Tensor up(
        h_up.size(),
        d_up,
        {num_tokens, hidden_size},
        DataType::FLOAT32,
        "gpu"
    );

    Tensor output(
        num_tokens * hidden_size,
        d_output,
        {num_tokens, hidden_size},
        DataType::FLOAT32,
        "gpu"
    );

    Batch batch;
    batch.num_tokens = num_tokens;

    ForwardContext context;
    context.layer_id = 0;
    context.batch = &batch;
    context.workspace = nullptr;
    context.config = nullptr;

    SwiGLU swiglu(hidden_size);
    swiglu.forward(gate, up, output, context);

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_output(num_tokens * hidden_size, 0.0f);
    CheckCuda(cudaMemcpy(
        h_output.data(),
        d_output,
        h_output.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const std::vector<float> expected = ComputeSwiGluReference(h_gate, h_up, num_tokens, hidden_size);
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(AlmostEqual(h_output[i], expected[i]));
    }

    cudaFree(d_output);
    cudaFree(d_up);
    cudaFree(d_gate);
}

} // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "No CUDA device found, skipping test_swiglu\n";
        return 0;
    }

    RunAndCheckSwiGlu();

    std::cout << "test_swiglu passed\n";
    return 0;
}
