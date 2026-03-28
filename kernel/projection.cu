#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "kernel/projection_kernel.h"

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
__global__ void projection_kernel(
    const void* input, 
    const void* weight, 
    const void* bias, 
    void* output,
    size_t batch_seq_len,
    size_t num_attention_heads,
    size_t num_kv_heads,
    size_t head_dim
){
    const int token = blockIdx.x;
    const int out_col = blockIdx.y;
    if (token >= static_cast<int>(batch_seq_len)) return;

    const int in_features = static_cast<int>(num_attention_heads * head_dim);
    const int q_size = static_cast<int>(num_attention_heads * head_dim);
    const int k_size = static_cast<int>(num_kv_heads * head_dim);
    const int v_size = static_cast<int>(num_kv_heads * head_dim);
    const int out_features = q_size + k_size + v_size;
    if (out_col >= out_features) return;

    float sum = 0.0f;
    const T* in_ptr = static_cast<const T*>(input);
    const T* w_ptr = static_cast<const T*>(weight);
    T* out_ptr = static_cast<T*>(output);
    for (int in_d = 0; in_d < in_features; ++in_d) {
        const float in_val = to_float<T>(in_ptr[token * in_features + in_d]);
        const float w_val = to_float<T>(w_ptr[in_d * out_features + out_col]);
        sum += in_val * w_val;
    }

    // Store as [Q_all_tokens][K_all_tokens][V_all_tokens] so split_qkv can view by pointer.
    int dst_idx = 0;
    if (out_col < q_size) {
        dst_idx = token * q_size + out_col;
    } else if (out_col < q_size + k_size) {
        dst_idx = static_cast<int>(batch_seq_len) * q_size + token * k_size + (out_col - q_size);
    } else {
        dst_idx = static_cast<int>(batch_seq_len) * (q_size + k_size) + token * v_size + (out_col - q_size - k_size);
    }
    out_ptr[dst_idx] = from_float<T>(sum);

}

void launch_projection_kernel(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    size_t batch_seq_len,
    size_t num_attention_heads,
    size_t num_kv_heads,
    size_t head_dim,
    DataType dtype
) {
    size_t out_features = (num_attention_heads + 2 * num_kv_heads) * head_dim;
    dim3 grid(batch_seq_len, out_features);
    dim3 block(1);
    if (dtype == DataType::FLOAT32) {
        projection_kernel<float><<<grid, block>>>(
            input,
            weight,
            bias,
            output,
            batch_seq_len,
            num_attention_heads,
            num_kv_heads,
            head_dim
        );
    } else if (dtype == DataType::FLOAT16) {
        projection_kernel<__half><<<grid, block>>>(
            input,
            weight,
            bias,
            output,
            batch_seq_len,
            num_attention_heads,
            num_kv_heads,
            head_dim
        );
    } else {
        projection_kernel<__nv_bfloat16><<<grid, block>>>(
            input,
            weight,
            bias,
            output,
            batch_seq_len,
            num_attention_heads,
            num_kv_heads,
            head_dim
        );
    }
}