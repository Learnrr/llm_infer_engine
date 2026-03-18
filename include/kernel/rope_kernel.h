#pragma once

#include <cstddef>
#include <cstdint>

void launch_apply_rope_inplace_float(
    float* tensor,
    const size_t* positions,
    size_t num_tokens,
    size_t num_heads,
    size_t head_dim,
    float rope_theta
);

void launch_apply_rope_inplace_half(
    uint16_t* tensor,
    const size_t* positions,
    size_t num_tokens,
    size_t num_heads,
    size_t head_dim,
    float rope_theta
);
