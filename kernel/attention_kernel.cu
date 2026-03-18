
#include "cuda_runtime.h"
#include "attention_kernel.h"

__global__ void attention_qk_softmax_Pv_kernel(
    const float* q,                    // [num_tokens, num_q_heads, head_dim]
    float** kcache_block_ptrs,         // [num_blocks][block_size, layers, heads, head_dim]
    float** vcache_block_ptrs,         // [num_blocks][block_size, layers, heads, head_dim]
    const size_t* block_ids,           // [num_tokens]
    const size_t* block_offsets,       // [num_tokens]
    float* attn_output,                // [num_tokens, num_q_heads, head_dim]
    int num_tokens,
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    int layer_id
) {
    int token = blockIdx.x;
    int head = blockIdx.y;
    int dim = threadIdx.x;
    if (token >= num_tokens || head >= num_q_heads || dim >= head_dim) return;

    const int group_size = max(1, num_q_heads / max(1, num_kv_heads));
    const int kv_head = min(num_kv_heads - 1, head / group_size);

    int q_offset = token * num_q_heads * head_dim + head * head_dim + dim;
    float q_val = q[q_offset];

    float max_score = -1e30f;
    extern __shared__ float scores[]; //size: max_seq_len
    for (int t = 0; t <= token && t < max_seq_len; ++t) {
        size_t off = block_offsets[t];
        // Pointer arrays are per-token entries, not global block-id tables.
        float* k_block = kcache_block_ptrs[t];
        int k_offset = off * num_layers * num_kv_heads * head_dim + layer_id * num_kv_heads * head_dim + kv_head * head_dim + dim;
        float k_val = k_block[k_offset];
        float score = q_val * k_val; 
        scores[t] = score;
        if (score > max_score) max_score = score;
    }
    __syncthreads();

    float sum_exp = 0.0f;
    for (int t = 0; t <= token && t < max_seq_len; ++t) {
        scores[t] = expf(scores[t] - max_score);
        sum_exp += scores[t];
    }
    __syncthreads();

    float out = 0.0f;
    for (int t = 0; t <= token && t < max_seq_len; ++t) {
        size_t off = block_offsets[t];
        float* v_block = vcache_block_ptrs[t];
        int v_offset = off * num_layers * num_kv_heads * head_dim + layer_id * num_kv_heads * head_dim + kv_head * head_dim + dim;
        float v_val = v_block[v_offset];
        out += scores[t] / sum_exp * v_val;
    }
    int out_offset = token * num_q_heads * head_dim + head * head_dim + dim;
    attn_output[out_offset] = out;
}

void launch_attention_qk_softmax_pv_kernel(
    const float* q,
    float** kcache_block_ptrs,
    float** vcache_block_ptrs,
    const size_t* block_ids,
    const size_t* block_offsets,
    float* attn_output,
    int num_tokens,
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    int layer_id
) {
    dim3 grid(num_tokens, num_q_heads);
    dim3 block(head_dim);
    attention_qk_softmax_Pv_kernel<<<grid, block, max_seq_len * sizeof(float)>>>(
        q,
        kcache_block_ptrs,
        vcache_block_ptrs,
        block_ids,
        block_offsets,
        attn_output,
        num_tokens,
        num_layers,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_seq_len,
        layer_id
    );
}