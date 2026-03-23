#include "rmsnorm_kernel.h"

#include <cuda_runtime.h>

__global__ void rmsnorm_kernel(
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

    extern __shared__ float s_sq_sum[];

    float local_sq_sum = 0.0f;
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        const float v = x[i];
        local_sq_sum += v * v;
    }

    s_sq_sum[tid] = local_sq_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sq_sum[tid] += s_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    const float mean_sq = s_sq_sum[0] / static_cast<float>(hidden_size);
    const float inv_rms = rsqrtf(mean_sq + eps);

    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        const float w = (gamma != nullptr) ? gamma[i] : 1.0f;
        y[i] = x[i] * inv_rms * w;
    }
}

void launch_rmsnorm_kernel(
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
    const size_t shared_bytes = static_cast<size_t>(kThreads) * sizeof(float);
    rmsnorm_kernel<<<grid, block, shared_bytes>>>(
        input,
        gamma,
        output,
        hidden_size,
        eps
    );
}
