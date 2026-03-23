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
#include "RMSNorm.h"
#include <memory>

class TransformerLayer: public Layer {
    public:
        TransformerLayer(
            int hidden_size, 
            int num_heads, 
            std::shared_ptr<TransformerLayerWeightLayout> layer_layout,
            const std::shared_ptr<TransformerLayerConfig>& layer_config
        ) {
            this->layer_layout = layer_layout;            

            //create attention module
            const AttentionLayerConfig& attn_config = layer_config->attention_config;
            attention = std::make_unique<Attention>(attn_config, layer_layout->attention_weights);

            //create mlp module
            const MLPLayerConfig& mlp_config = layer_config->mlp_config;
            mlp = std::make_unique<MLP>(mlp_config, layer_layout->mlp_weights);

            //create norm layers based on the number of norm configs provided
            norm_layers.resize(layer_config->norm_configs.size());
            for(size_t i = 0; i<layer_config->norm_configs.size(); ++i){
                const auto& norm_cfg = layer_config->norm_configs[i];
                const auto& norm_weight = layer_layout->norm_weights[i];
                const auto& norm_weight_layout = layer_layout->norm_weights[i];
                norm_layers.emplace_back(std::make_unique<RMSNorm>(norm_cfg, norm_weight_layout));
            }
            //create residual add module
            residual_add = std::make_unique<ResidualAdd>();
        }

        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;

        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;

    private:
        std::unique_ptr<Attention> attention;
        std::vector<std::unique_ptr<RMSNorm>> norm_layers;
        std::unique_ptr<MLP> mlp;
        std::unique_ptr<ResidualAdd> residual_add;
        std::shared_ptr<TransformerLayerWeightLayout> layer_layout;


};