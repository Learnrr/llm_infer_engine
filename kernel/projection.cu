#include "cuda_runtime.h"
#include "projection_kernel.h"

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
    for (int in_d = 0; in_d < in_features; ++in_d) {
        const float in_val = ((float*)input)[token * in_features + in_d];
        const float w_val = ((float*)weight)[in_d * out_features + out_col];
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
    ((float*)output)[dst_idx] = sum;

}

void launch_projection_kernel(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    size_t batch_seq_len,
    size_t num_attention_heads,
    size_t num_kv_heads,
    size_t head_dim
) {
    size_t out_features = (num_attention_heads + 2 * num_kv_heads) * head_dim;
    dim3 grid(batch_seq_len, out_features);
    dim3 block(1);
    projection_kernel<<<grid, block>>>(
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