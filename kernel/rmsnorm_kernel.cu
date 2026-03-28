#include "kernel/rmsnorm_kernel.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__device__ inline float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ inline T from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ inline __half from_float<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__global__ void rmsnorm_kernel(
    const T* input,
    const T* gamma,
    T* output,
    size_t hidden_size,
    float eps
) {
    const size_t token_idx = static_cast<size_t>(blockIdx.x);
    const size_t tid = static_cast<size_t>(threadIdx.x);

    const T* x = input + token_idx * hidden_size;
    T* y = output + token_idx * hidden_size;

    extern __shared__ float s_sq_sum[];

    float local_sq_sum = 0.0f;
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        const float v = to_float<T>(x[i]);
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
        const float w = (gamma != nullptr) ? to_float<T>(gamma[i]) : 1.0f;
        const float xv = to_float<T>(x[i]);
        y[i] = from_float<T>(xv * inv_rms * w);
    }
}

void launch_rmsnorm_kernel(
    const void* input,
    const void* gamma,
    void* output,
    size_t num_tokens,
    size_t hidden_size,
    float eps,
    DataType dtype
) {
    constexpr unsigned int kThreads = 256;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>(num_tokens));
    const size_t shared_bytes = static_cast<size_t>(kThreads) * sizeof(float);
    if (dtype == DataType::FLOAT32) {
        rmsnorm_kernel<float><<<grid, block, shared_bytes>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(gamma),
            static_cast<float*>(output),
            hidden_size,
            eps
        );
    } else if (dtype == DataType::FLOAT16) {
        rmsnorm_kernel<__half><<<grid, block, shared_bytes>>>(
            static_cast<const __half*>(input),
            static_cast<const __half*>(gamma),
            static_cast<__half*>(output),
            hidden_size,
            eps
        );
    } else {
        rmsnorm_kernel<__nv_bfloat16><<<grid, block, shared_bytes>>>(
            static_cast<const __nv_bfloat16*>(input),
            static_cast<const __nv_bfloat16*>(gamma),
            static_cast<__nv_bfloat16*>(output),
            hidden_size,
            eps
        );
    }
}
