#pragma once

#include <cstddef>

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
);
