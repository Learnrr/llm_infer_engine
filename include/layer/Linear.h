#pragma once
#include "define.h"
#include "Tensor.h"
#include "Workspace.h"
#include "Layer.h"
#include "ForwardContext.h"
#include "ModelWeights.h"
#include "ModelConfig.h"
class Linear: public Layer {
    public:
        Linear(const LinearConfig& config, 
            size_t layer_id, 
            Tensor& linear_weight)
            : layer_id(layer_id), 
            linear_weight(linear_weight), 
            config(config) {}
        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
    private:
        size_t layer_id;
        Tensor& linear_weight;
        LinearConfig config;
};