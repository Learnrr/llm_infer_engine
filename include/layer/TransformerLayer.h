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
#include "utils/logger.h"
#include <algorithm>
#include <memory>

class TransformerLayer: public Layer {
    public:
        TransformerLayer(
            int hidden_size, 
            int num_heads, 
            const std::shared_ptr<TransformerLayerWeightLayout>& layer_layout,
            std::shared_ptr<TransformerLayerConfig>& layer_config
        ) {
            this->layer_layout = layer_layout;            

            if (layer_layout == nullptr || layer_config == nullptr) {
                LOG_ERROR("TransformerLayer received null layout/config");
                return;
            }

            //create attention module
            const AttentionLayerConfig& attn_config = layer_config->attention_config;
            attention = std::make_unique<Attention>(attn_config, layer_layout->attention_weights);

            //create mlp module
            const MLPLayerConfig& mlp_config = layer_config->mlp_config;
            mlp = std::make_unique<MLP>(mlp_config, layer_layout->mlp_weights);

            //create norm layers based on the number of norm configs provided
            size_t num_norm_layers = std::min(layer_config->norm_configs.size(), 
                                            layer_layout->norm_weights.size());
            norm_layers.reserve(num_norm_layers);
            for(size_t i = 0; i < num_norm_layers; ++i){
                const auto& norm_cfg = layer_config->norm_configs[i];
                auto& norm_weight_layout = layer_layout->norm_weights[i];
                norm_layers.emplace_back(std::make_unique<RMSNorm>(
                    norm_cfg,
                    norm_weight_layout.norm_weight,
                    norm_weight_layout.gamma
                ));
            }
            if (norm_layers.size() < 2) {
                LOG_ERROR("TransformerLayer initialized with insufficient norm layers");
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