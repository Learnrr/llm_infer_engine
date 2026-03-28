#include "kernel/rope_kernel.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ inline float rope_inv_freq(int pair_idx, int head_dim, float rope_theta) {
    return powf(rope_theta, -2.0f * static_cast<float>(pair_idx) / static_cast<float>(head_dim));
}

template <typename T>
__device__ inline float rope_to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float rope_to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ inline float rope_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ inline T rope_from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ inline __half rope_from_float<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ inline __nv_bfloat16 rope_from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__global__ void apply_rope_inplace_float_kernel(
    float* tensor,
    const size_t* positions,
    int num_tokens,
    int num_heads,
    int head_dim,
    float rope_theta
) {
    const int half_dim = head_dim / 2;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(num_tokens) * num_heads * half_dim;
    if (idx >= total) return;

    const int pair_idx = static_cast<int>(idx % half_dim);
    const size_t tmp = idx / half_dim;
    const int head_idx = static_cast<int>(tmp % num_heads);
    const int token_idx = static_cast<int>(tmp / num_heads);

    const float inv_freq = rope_inv_freq(pair_idx, head_dim, rope_theta);
    const float angle = static_cast<float>(positions[token_idx]) * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);

    const size_t base = (static_cast<size_t>(token_idx) * num_heads + head_idx) * head_dim;
    const size_t even_idx = base + (2 * pair_idx);
    const size_t odd_idx = even_idx + 1;

    const float x0 = tensor[even_idx];
    const float x1 = tensor[odd_idx];

    tensor[even_idx] = x0 * c - x1 * s;
    tensor[odd_idx] = x0 * s + x1 * c;
}

template <typename T>
__global__ void apply_rope_inplace_16bit_kernel(
    T* tensor,
    const size_t* positions,
    int num_tokens,
    int num_heads,
    int head_dim,
    float rope_theta
) {
    const int half_dim = head_dim / 2;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(num_tokens) * num_heads * half_dim;
    if (idx >= total) return;

    const int pair_idx = static_cast<int>(idx % half_dim);
    const size_t tmp = idx / half_dim;
    const int head_idx = static_cast<int>(tmp % num_heads);
    const int token_idx = static_cast<int>(tmp / num_heads);

    const float inv_freq = rope_inv_freq(pair_idx, head_dim, rope_theta);
    const float angle = static_cast<float>(positions[token_idx]) * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);

    const size_t base = (static_cast<size_t>(token_idx) * num_heads + head_idx) * head_dim;
    const size_t even_idx = base + (2 * pair_idx);
    const size_t odd_idx = even_idx + 1;

    const float x0 = rope_to_float<T>(tensor[even_idx]);
    const float x1 = rope_to_float<T>(tensor[odd_idx]);

    tensor[even_idx] = rope_from_float<T>(x0 * c - x1 * s);
    tensor[odd_idx] = rope_from_float<T>(x0 * s + x1 * c);
}

void launch_apply_rope_inplace(
    void* tensor,
    const size_t* positions,
    size_t num_tokens,
    size_t num_heads,
    size_t head_dim,
    float rope_theta,
    DataType dtype
) {
    if (head_dim == 0 || (head_dim % 2) != 0) {
        return;
    }
    constexpr int threads = 256;
    const size_t total = num_tokens * num_heads * (head_dim / 2);
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    if (dtype == DataType::FLOAT32) {
        apply_rope_inplace_float_kernel<<<blocks, threads>>>(
            static_cast<float*>(tensor),
            positions,
            static_cast<int>(num_tokens),
            static_cast<int>(num_heads),
            static_cast<int>(head_dim),
            rope_theta
        );
    } else if (dtype == DataType::FLOAT16) {
        apply_rope_inplace_16bit_kernel<__half><<<blocks, threads>>>(
            static_cast<__half*>(tensor),
            positions,
            static_cast<int>(num_tokens),
            static_cast<int>(num_heads),
            static_cast<int>(head_dim),
            rope_theta
        );
    } else {
        apply_rope_inplace_16bit_kernel<__nv_bfloat16><<<blocks, threads>>>(
            static_cast<__nv_bfloat16*>(tensor),
            positions,
            static_cast<int>(num_tokens),
            static_cast<int>(num_heads),
            static_cast<int>(head_dim),
            rope_theta
        );
    }
}
