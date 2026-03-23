#pragma once

#include <cstddef>

void launch_rmsnorm_kernel(
    const float* input,
    const float* gamma,
    float* output,
    size_t num_tokens,
    size_t hidden_size,
    float eps
);
