#include <cstdint>
#include "cuda_runtime.h"
#include "kernel/transpose_kernel.h"

__global__ void transpose_last2d_float32_kernel(
    const float* input,
    float* output,
    size_t outer,
    size_t rows,
    size_t cols
) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = outer * rows * cols;
    if (linear >= total) {
        return;
    }

    const size_t block_elems = rows * cols;
    const size_t o = linear / block_elems;
    const size_t rem = linear % block_elems;
    const size_t i = rem / cols;
    const size_t j = rem % cols;

    const size_t out_idx = o * block_elems + j * rows + i;
    output[out_idx] = input[linear];
}

__global__ void transpose_last2d_float16_kernel(
    const uint16_t* input,
    uint16_t* output,
    size_t outer,
    size_t rows,
    size_t cols
) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = outer * rows * cols;
    if (linear >= total) {
        return;
    }

    const size_t block_elems = rows * cols;
    const size_t o = linear / block_elems;
    const size_t rem = linear % block_elems;
    const size_t i = rem / cols;
    const size_t j = rem % cols;

    const size_t out_idx = o * block_elems + j * rows + i;
    output[out_idx] = input[linear];
}

void launch_transpose_last2d_kernel(
    const void* input,
    void* output,
    size_t outer,
    size_t rows,
    size_t cols,
    DataType dtype
) {
    const size_t total = outer * rows * cols;
    if (total == 0) {
        return;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

    if (dtype == DataType::FLOAT32) {
        transpose_last2d_float32_kernel<<<blocks, threads>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            outer,
            rows,
            cols
        );
    } else {
        transpose_last2d_float16_kernel<<<blocks, threads>>>(
            static_cast<const uint16_t*>(input),
            static_cast<uint16_t*>(output),
            outer,
            rows,
            cols
        );
    }
}
