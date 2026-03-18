#pragma once

#include "define.h"
#include "Tensor.h"
#include "Attention.h"
#include "MLP.h"
#include "ModelWeights.h"
#include "ModelConfig.h"
#include "Workspace.h"
#include "Layer.h"
#include "ForwardContext.h"
#include "ResidualAdd.h"
#include <memory>

class TransformerLayer: public Layer {
    public:
        TransformerLayer(
            int hidden_size, 
            int num_heads, 
            std::shared_ptr<TransformerLayerWeightLayout> layer_layout,
            const std::shared_ptr<TransformerLayerConfig>& layer_config
        ) {
            const AttentionLayerConfig& attn_config = layer_config->attention_config;
            const MLPLayerConfig& mlp_config = layer_config->mlp_config;

            attention = std::make_unique<Attention>(attn_config, layer_layout->attention_weights);
            mlp = std::make_unique<MLP>(mlp_config, layer_layout->mlp_weights);

            this->layer_layout = layer_layout;
            residual_add = std::make_unique<ResidualAdd>();
        }

        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;

        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;

    private:
        std::unique_ptr<Attention> attention;
        std::unique_ptr<MLP> mlp;
        std::unique_ptr<ResidualAdd> residual_add;
        std::shared_ptr<TransformerLayerWeightLayout> layer_layout;


};