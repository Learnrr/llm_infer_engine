#pragma once

#include "define.h"
#include "embedding.h"
#include "ModelConfig.h"
#include "ModelWeights.h"
#include "Batch.h"
#include "Workspace.h"
#include "Layer.h"
#include "Tensor.h"
#include "TransformerLayer.h"
#include "LayerNorm.h"
#include "Linear.h"
class Model {
    public:
        Model(const char* model_path) {     
            init(model_path);
        } 

        void init(const char* model_path) {
            config = ModelConfig();
            weights = std::make_unique<ModelWeights>();
            weights->init(config);
            weights->load_weights(model_path);
            embedding =  std::make_unique<Embedding>(
                config, 
                weights->layout.embedding_weights.data
            );            
            transformer_layers = std::make_unique<TransformerLayer[]>(config.num_hidden_layers);
            for(size_t i = 0; i < config.num_hidden_layers; ++i) {
                transformer_layers[i] = TransformerLayer(
                    config.hidden_size, 
                    config.num_attention_heads,
                    &weights->layout.layers[i]
                );
            }

            layer_norm = std::make_unique<LayerNorm>(config.hidden_size);
            lm_head = std::make_unique<Linear>();

        }

        void prefill_forward(Batch& batch, Workspace& workspace) {
            // Implement the logic for the prefill forward pass
            Tensor hidden(
                batch.num_tokens * config.hidden_size, 
                workspace->get_embedding_workspace(),
                {batch.num_tokens, config.hidden_size}, 
                DataType::FLOAT16
            );
            embedding->forward(batch.token_ids, hidden, batch.num_tokens);

            for(size_t i = 0; i < config.num_hidden_layers; ++i) {
                transformer_layers[i].prefill_forward(hidden, hidden, workspace);
            }

            
        }

        void decode_forward(Batch& batch, Workspace& workspace) {
            // Implement the logic for the decode forward pass
            Tensor hidden(
                batch.num_tokens * config.hidden_size, 
                workspace->get_embedding_workspace(),
                {batch.num_tokens, config.hidden_size}, 
                DataType::FLOAT16
            );
            embedding->forward(batch.token_ids, hidden, batch.num_tokens);

            for(size_t i = 0; i < config.num_hidden_layers; ++i) {
                transformer_layers[i].decode_forward(hidden, hidden, workspace);
            }
            
        }

    private:
     
        std::unique_ptr<Embedding> embedding;
        std::unique_ptr<ModelWeights> weights;
        ModelConfig config;
        std::unique_ptr<TransformerLayer[]> transformer_layers; // Array of layers (e.g., attention, MLP, etc.)
        std::unique_ptr<LayerNorm> layer_norm;
        std::unique_ptr<Linear> lm_head;
}