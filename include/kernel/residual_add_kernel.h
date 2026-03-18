#pragma once

#include <cstddef>

void launch_residual_add_kernel(
    const float* residual,
    const float* input,
    float* output,
    size_t num_elements
);
