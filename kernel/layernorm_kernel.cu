#include "layernorm_kernel.h"

#include <cuda_runtime.h>

__global__ void layernorm_kernel(
    const float* input,
    const float* gamma,
    float* output,
    size_t hidden_size,
    float eps
) {
    const size_t token_idx = static_cast<size_t>(blockIdx.x);
    const size_t tid = static_cast<size_t>(threadIdx.x);

    const float* x = input + token_idx * hidden_size;
    float* y = output + token_idx * hidden_size;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sq_sum = sdata + blockDim.x;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        const float v = x[i];
        local_sum += v;
        local_sq_sum += v * v;
    }

    s_sum[tid] = local_sum;
    s_sq_sum[tid] = local_sq_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq_sum[tid] += s_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    const float mean = s_sum[0] / static_cast<float>(hidden_size);
    const float var = s_sq_sum[0] / static_cast<float>(hidden_size) - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        const float normalized = (x[i] - mean) * inv_std;
        const float w = (gamma != nullptr) ? gamma[i] : 1.0f;
        y[i] = normalized * w;
    }
}

void launch_layernorm_kernel(
    const float* input,
    const float* gamma,
    float* output,
    size_t num_tokens,
    size_t hidden_size,
    float eps
) {
    constexpr unsigned int kThreads = 256;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>(num_tokens));
    const size_t shared_bytes = static_cast<size_t>(2 * kThreads) * sizeof(float);
    layernorm_kernel<<<grid, block, shared_bytes>>>(
        input,
        gamma,
        output,
        hidden_size,
        eps
    );
}
