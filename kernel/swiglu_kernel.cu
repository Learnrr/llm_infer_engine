#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "kernel/swiglu_kernel.h"

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
__global__ void swiglu_kernel_from_gate_up(
    const T* gate,
    const T* up,
    T* output,
    size_t hidden_size,
    size_t total_elements
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    const float gate_val = to_float<T>(gate[idx]);
    const float up_val = to_float<T>(up[idx]);
    const float sigmoid_gate = 1.0f / (1.0f + __expf(-gate_val));
    output[idx] = from_float<T>((gate_val * sigmoid_gate) * up_val);
}


void launch_swiglu_kernel_from_gate_up(
    const void* gate,
    const void* up,
    void* output,
    size_t num_tokens,
    size_t hidden_size,
    DataType dtype
) {
    const size_t total_elements = num_tokens * hidden_size;
    if (total_elements == 0) {
        return;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    if (dtype == DataType::FLOAT32) {
        swiglu_kernel_from_gate_up<float><<<blocks, threads>>>(
            static_cast<const float*>(gate),
            static_cast<const float*>(up),
            static_cast<float*>(output),
            hidden_size,
            total_elements
        );
    } else if (dtype == DataType::FLOAT16) {
        swiglu_kernel_from_gate_up<__half><<<blocks, threads>>>(
            static_cast<const __half*>(gate),
            static_cast<const __half*>(up),
            static_cast<__half*>(output),
            hidden_size,
            total_elements
        );
    } else {
        swiglu_kernel_from_gate_up<__nv_bfloat16><<<blocks, threads>>>(
            static_cast<const __nv_bfloat16*>(gate),
            static_cast<const __nv_bfloat16*>(up),
            static_cast<__nv_bfloat16*>(output),
            hidden_size,
            total_elements
        );
    }
}