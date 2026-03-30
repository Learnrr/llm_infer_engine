#pragma once

#include <cstddef>
#include "define.h"

void launch_attention_qk_softmax_pv_kernel(
    const void* q,
    void** kcache_block_ptrs,
    void** vcache_block_ptrs,
    const size_t* block_offsets,
    const size_t* query_hist_start,
    const size_t* query_hist_len,
    void* attn_output,
    int num_tokens,
    int num_sequences,
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    int layer_id,
    DataType dtype
);

void launch_attention_qk_softmax_pv_kernel_decode(
    const void* q,
    void** history_kcache_block_ptrs,
    void** history_vcache_block_ptrs,
    const size_t* history_block_offsets,
    const size_t* query_hist_start,
    const size_t* query_hist_len,
    void* attn_output,
    int num_queries,
    int total_history_tokens,
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    int layer_id,
    DataType dtype
);
