#pragma once

#include <cstddef>
#include "define.h"

void launch_transpose_last2d_kernel(
    const void* input,
    void* output,
    size_t outer,
    size_t rows,
    size_t cols,
    DataType dtype
);
