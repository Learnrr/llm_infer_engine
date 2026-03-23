#pragma once

#include "ForwardContext.h"
#include "Layer.h"
#include "ModelConfig.h"
#include "ModelWeights.h"
#include "Tensor.h"

class RMSNorm : public Layer {
public:
    RMSNorm(const LayerNormLayerConfig& config, LayerNormLayerWeightLayout& norm_weight)
        : config(config), norm_weight(norm_weight) {}

    void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
    void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;

private:
    LayerNormLayerConfig config;
    LayerNormLayerWeightLayout& norm_weight;
    static constexpr float kDefaultEps = 1e-5f;
};