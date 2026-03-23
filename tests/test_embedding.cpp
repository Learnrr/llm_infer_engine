/*
cd tests
nvcc -std=c++17 -O2 -I../include -I../include/layer -I../include/model -I../include/kernel \
    test_embedding.cpp ../src/layer/Embedding.cpp ../kernel/embedding_kernel.cu \
    -o ../build/tests/test_embedding.exe
./../build/tests/test_embedding.exe
*/

#include "layer/Embedding.h"

#include <cassert>
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

ModelConfig BuildTinyConfig() {
    ModelConfig cfg;
    cfg.vocab_size = 5;
    cfg.hidden_size = 4;
    return cfg;
}

void TestEmbeddingForwardGathersRows() {
    const ModelConfig cfg = BuildTinyConfig();

    // embedding_table[token_id][hidden_idx] = token_id * 10 + hidden_idx.
    const std::vector<float> h_table = {
        0.0f, 1.0f, 2.0f, 3.0f,
        10.0f, 11.0f, 12.0f, 13.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
        30.0f, 31.0f, 32.0f, 33.0f,
        40.0f, 41.0f, 42.0f, 43.0f
    };

    float* d_table = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_table), h_table.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(
        d_table,
        h_table.data(),
        h_table.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    Tensor embedding_weight(
        h_table.size(),
        d_table,
        {cfg.vocab_size, cfg.hidden_size},
        DataType::FLOAT32,
        "gpu"
    );

    Embedding embedding(cfg, embedding_weight);

    const std::vector<size_t> token_ids = {3, 1, 4};
    const size_t num_tokens = token_ids.size();

    float* d_output = nullptr;
    const size_t output_elems = num_tokens * cfg.hidden_size;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), output_elems * sizeof(float)));

    Tensor output(
        output_elems,
        d_output,
        {num_tokens, cfg.hidden_size},
        DataType::FLOAT32,
        "gpu"
    );

    embedding.forward(const_cast<size_t*>(token_ids.data()), output, num_tokens);
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_output(output_elems, 0.0f);
    CheckCuda(cudaMemcpy(
        h_output.data(),
        d_output,
        output_elems * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const std::vector<float> expected = {
        30.0f, 31.0f, 32.0f, 33.0f,
        10.0f, 11.0f, 12.0f, 13.0f,
        40.0f, 41.0f, 42.0f, 43.0f
    };

    assert(h_output.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(h_output[i] == expected[i]);
    }

    cudaFree(d_output);
    cudaFree(d_table);
}

void TestEmbeddingForwardSupportsRepeatedTokens() {
    const ModelConfig cfg = BuildTinyConfig();

    const std::vector<float> h_table = {
        0.5f, 1.5f, 2.5f, 3.5f,
        10.5f, 11.5f, 12.5f, 13.5f,
        20.5f, 21.5f, 22.5f, 23.5f,
        30.5f, 31.5f, 32.5f, 33.5f,
        40.5f, 41.5f, 42.5f, 43.5f
    };

    float* d_table = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_table), h_table.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(
        d_table,
        h_table.data(),
        h_table.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    Tensor embedding_weight(
        h_table.size(),
        d_table,
        {cfg.vocab_size, cfg.hidden_size},
        DataType::FLOAT32,
        "gpu"
    );
    Embedding embedding(cfg, embedding_weight);

    const std::vector<size_t> token_ids = {2, 2};
    const size_t num_tokens = token_ids.size();
    const size_t output_elems = num_tokens * cfg.hidden_size;

    float* d_output = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), output_elems * sizeof(float)));

    Tensor output(
        output_elems,
        d_output,
        {num_tokens, cfg.hidden_size},
        DataType::FLOAT32,
        "gpu"
    );

    embedding.forward(const_cast<size_t*>(token_ids.data()), output, num_tokens);
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_output(output_elems, 0.0f);
    CheckCuda(cudaMemcpy(
        h_output.data(),
        d_output,
        output_elems * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const std::vector<float> expected = {
        20.5f, 21.5f, 22.5f, 23.5f,
        20.5f, 21.5f, 22.5f, 23.5f
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(h_output[i] == expected[i]);
    }

    cudaFree(d_output);
    cudaFree(d_table);
}

} // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "No CUDA device found, skipping test_embedding\n";
        return 0;
    }

    TestEmbeddingForwardGathersRows();
    TestEmbeddingForwardSupportsRepeatedTokens();

    std::cout << "test_embedding passed\n";
    return 0;
}
