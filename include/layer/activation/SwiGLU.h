#pragma once
#include "Tensor.h"
#include "ForwardContext.h"
#include <cstddef>

class SwiGLU {
public:
    SwiGLU(size_t hidden_size) : hidden_size(hidden_size) {}

    void forward(
        const Tensor& gate,
        const Tensor& up,
        Tensor& output,
        ForwardContext& context
    );

private:
    size_t hidden_size;
};