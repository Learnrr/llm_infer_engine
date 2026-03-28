#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "kernel/linear_kernel.h"

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
__global__ void mlp_linear_kernel(
    const T* input,
    const T* weight,
    T* output,
    size_t batch_seq_len,
    size_t in_features,
    size_t out_features
){
    int token = blockIdx.x;
    int out_col = blockIdx.y * blockDim.x + threadIdx.x;
    if(token < batch_seq_len && out_col < out_features){
        float sum = 0.0f;
        for(int in_d = 0; in_d < in_features; ++in_d){
            float in_val = to_float<T>(input[token * in_features + in_d]);
            float w_val = to_float<T>(weight[in_d * out_features + out_col]);
            sum += in_val * w_val;
        }
        output[token * out_features + out_col] = from_float<T>(sum);
    }
}

void launch_mlp_linear_kernel(
    const void* input,
    const void* weight,
    void* output,
    size_t batch_seq_len,
    size_t in_features,
    size_t out_features,
    DataType dtype
) {
    dim3 block(256);
    dim3 grid(batch_seq_len, (out_features + block.x - 1) / block.x);
    if (dtype == DataType::FLOAT32) {
        mlp_linear_kernel<float><<<grid, block>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(weight),
            static_cast<float*>(output),
            batch_seq_len,
            in_features,
            out_features
        );
    } else if (dtype == DataType::FLOAT16) {
        mlp_linear_kernel<__half><<<grid, block>>>(
            static_cast<const __half*>(input),
            static_cast<const __half*>(weight),
            static_cast<__half*>(output),
            batch_seq_len,
            in_features,
            out_features
        );
    } else {
        mlp_linear_kernel<__nv_bfloat16><<<grid, block>>>(
            static_cast<const __nv_bfloat16*>(input),
            static_cast<const __nv_bfloat16*>(weight),
            static_cast<__nv_bfloat16*>(output),
            batch_seq_len,
            in_features,
            out_features
        );
    }
}