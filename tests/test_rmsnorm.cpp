/*
cd tests
nvcc -std=c++17 -O2 -I../include -I../include/layer -I../include/model -I../include/kernel \
    test_rmsnorm.cpp ../src/layer/RMSNorm.cpp ../kernel/rmsnorm_kernel.cu \
    -o ../build/tests/test_rmsnorm.exe
./../build/tests/test_rmsnorm.exe
*/

#include "layer/RMSNorm.h"

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

std::vector<float> ComputeRmsNormReference(
    const std::vector<float>& input,
    const std::vector<float>& gamma,
    size_t num_tokens,
    size_t hidden_size,
    float eps
) {
    std::vector<float> out(input.size(), 0.0f);
    for (size_t t = 0; t < num_tokens; ++t) {
        float sum_sq = 0.0f;
        for (size_t h = 0; h < hidden_size; ++h) {
            const float v = input[t * hidden_size + h];
            sum_sq += v * v;
        }
        const float mean_sq = sum_sq / static_cast<float>(hidden_size);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (size_t h = 0; h < hidden_size; ++h) {
            const float v = input[t * hidden_size + h];
            out[t * hidden_size + h] = v * inv_rms * gamma[h];
        }
    }
    return out;
}

void RunAndCheckRmsNorm(bool use_prefill) {
    const size_t num_tokens = 2;
    const size_t hidden_size = 4;

    const std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 0.0f, 2.0f, 0.0f
    };
    const std::vector<float> h_gamma = {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    float* d_input = nullptr;
    float* d_gamma = nullptr;
    float* d_output = nullptr;

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), h_input.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_gamma), h_gamma.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), h_input.size() * sizeof(float)));

    CheckCuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemset(d_output, 0, h_input.size() * sizeof(float)));

    Tensor input(
        h_input.size(),
        d_input,
        {num_tokens, hidden_size},
        DataType::FLOAT32,
        "gpu"
    );

    Tensor output(
        h_input.size(),
        d_output,
        {num_tokens, hidden_size},
        DataType::FLOAT32,
        "gpu"
    );

    LayerNormLayerConfig cfg;
    cfg.norm_size = hidden_size;

    LayerNormLayerWeightLayout weight_layout;
    weight_layout.norm_weight.data = d_gamma;
    weight_layout.norm_weight.num_elements = h_gamma.size();
    weight_layout.norm_weight.size = h_gamma.size() * sizeof(float);
    weight_layout.norm_weight.shape = {hidden_size};
    weight_layout.norm_weight.dtype = DataType::FLOAT32;
    weight_layout.norm_weight.device = "gpu";
    weight_layout.gamma = d_gamma;

    RMSNorm rmsnorm(cfg, weight_layout);

    Batch batch;
    batch.num_tokens = num_tokens;

    ForwardContext context{};
    context.layer_id = 0;
    context.batch = &batch;
    context.workspace = nullptr;
    context.config = nullptr;

    if (use_prefill) {
        rmsnorm.prefill_forward(input, output, context);
    } else {
        rmsnorm.decode_forward(input, output, context);
    }

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_output(h_input.size(), 0.0f);
    CheckCuda(cudaMemcpy(
        h_output.data(),
        d_output,
        h_output.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const std::vector<float> expected =
        ComputeRmsNormReference(h_input, h_gamma, num_tokens, hidden_size, 1e-5f);

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(AlmostEqual(h_output[i], expected[i]));
    }

    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_input);
}

void TestRmsNormPrefillForward() {
    RunAndCheckRmsNorm(true);
}

void TestRmsNormDecodeForward() {
    RunAndCheckRmsNorm(false);
}

} // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "No CUDA device found, skipping test_rmsnorm\n";
        return 0;
    }

    TestRmsNormPrefillForward();
    TestRmsNormDecodeForward();

    std::cout << "test_rmsnorm passed\n";
    return 0;
}
