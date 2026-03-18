#include "residual_add_kernel.h"
#include <cuda_runtime.h>

__global__ void residual_add_kernel(
    const float* residual,
    const float* input,
    float* output,
    size_t num_elements
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    output[idx] = residual[idx] + input[idx];
}

void launch_residual_add_kernel(
    const float* residual,
    const float* input,
    float* output,
    size_t num_elements
) {
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);
    residual_add_kernel<<<blocks, threads_per_block>>>(
        residual,
        input,
        output,
        num_elements
    );
}
