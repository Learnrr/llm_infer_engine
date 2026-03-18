#pragma once

#include <cstddef>

void launch_projection_kernel(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    size_t batch_seq_len,
    size_t num_attention_heads,
    size_t num_kv_heads,
    size_t head_dim
);
