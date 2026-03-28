#pragma once

#include "ForwardContext.h"
#include "layer/Layer.h"
#include "model/ModelConfig.h"
#include "Tensor.h"

class RMSNorm : public Layer {
public:
    RMSNorm(
        const LayerNormLayerConfig& config,
        Tensor& norm_weight,
        void* gamma)
        : config(config), norm_weight(norm_weight), gamma(gamma) {}

    void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
    void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;

private:
    LayerNormLayerConfig config;
    Tensor& norm_weight;
    void* gamma = nullptr;
    static constexpr float kDefaultEps = 1e-5f;
};