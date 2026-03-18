#pragma once

#include "Layer.h"
#include "Tensor.h"

class ResidualAdd : public Layer {
public:
    void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
    void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
};