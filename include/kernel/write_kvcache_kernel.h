#pragma once

#include <cstddef>

void launch_write_kvcache_kernel(
    float** kcache_block_ptrs,
    float** vcache_block_ptrs,
    const size_t* block_ids,
    const size_t* block_offsets,
    const float* key_data,
    const float* value_data,
    int num_tokens,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int layer_id
);
