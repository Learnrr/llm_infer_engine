#include "QWEN_Model.h"
#include <utility>

void QWEN_Model::init(ModelConfig config) {
    this->config = config;
    weights = std::make_unique<ModelWeights>();
    ErrorCode error = weights->init(config);
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to initialize model weights");
        return;
    }

    // Create embedding layer
    embedding =  std::make_unique<Embedding>(
        config, 
        weights->layout.embedding_weights
    );            
    layers.reserve(config.num_hidden_layers + 2); // +2 for layer norm and lm head

    // Create transformer layers
    for(size_t i = 0; i < config.num_hidden_layers; ++i) {
        auto layer_config = config.get_layer_config<TransformerLayerConfig>(i);
        if (layer_config == nullptr) {
            LOG_ERROR("Missing TransformerLayerConfig in model config");
            return;
        }
        while (layer_config->norm_configs.size() < 2) {
            LayerNormLayerConfig norm_cfg;
            norm_cfg.norm_size = config.hidden_size;
            layer_config->norm_configs.push_back(norm_cfg);
        }

        auto layer_layout = weights->layout.get_layer_layout<TransformerLayerWeightLayout>(i);
        if (layer_layout == nullptr) {
            LOG_ERROR("Missing TransformerLayerWeightLayout in weight layout");
            return;
        }

        layers.emplace_back(std::make_unique<TransformerLayer>(
            config.hidden_size, 
            config.num_heads,
            layer_layout,
            layer_config
        ));
    }

    // Create final layer norm
    LayerNormLayerConfig layernorm_config;
    layernorm_config.norm_size = config.hidden_size;
    auto layernorm_config_ptr = config.get_layer_config<LayerNormLayerConfig>(config.num_hidden_layers);
    if (layernorm_config_ptr != nullptr) {
        layernorm_config = *layernorm_config_ptr;
    }

    auto layernorm_layout = weights->layout.get_layer_layout<LayerNormLayerWeightLayout>(config.num_hidden_layers);
    if (layernorm_layout == nullptr) {
        LOG_ERROR("Failed to get final layernorm layout");
        return;
    }

    layers.emplace_back(std::make_unique<RMSNorm>(
        layernorm_config,
        layernorm_layout->norm_weight,
        layernorm_layout->gamma
    ));
    
    // Create LM head
    auto lmhead_config = config.get_layer_config<LinearLayerConfig>(config.num_hidden_layers + 1);
    auto lm_head_layout = weights->layout.get_layer_layout<LinearLayerWeightLayout>(config.num_hidden_layers + 1);
    if (lmhead_config == nullptr || lm_head_layout == nullptr) {
        LOG_ERROR("Failed to get LM head config/layout");
        return;
    }
    layers.emplace_back(std::make_unique<Linear>(
        lmhead_config->linear_config, 
        config.num_hidden_layers + 1, 
        lm_head_layout->linear_weight
    ));

    post_processor = std::make_unique<PostProcessor>(config);

}
void QWEN_Model::load_weights(const char* model_path) {
    ErrorCode error = weights->load_weights(model_path);
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to load model weights");
    }
}
void QWEN_Model::prefill_forward(Batch& batch, Workspace& workspace) {
    // Implement the logic for the prefill forward pass
    if (!embedding || layers.size() < config.num_hidden_layers + 2) {
        LOG_ERROR("QWEN_Model is not fully initialized before prefill_forward");
        return;
    }

    Tensor hidden(
        batch.num_tokens * config.hidden_size, 
        workspace.get_embedding_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT32
    );


    embedding->forward(batch.token_ids, hidden, batch.num_tokens);

    Tensor hidden2(
        batch.num_tokens * config.hidden_size, 
        workspace.get_hidden2_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT32
    );

    ForwardContext context;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &config;

    for(size_t i = 0; i < config.num_hidden_layers; ++i) {
        context.layer_id = i;
        layers[i]->prefill_forward(hidden, hidden2, context);
        std::swap(hidden.data, hidden2.data);
    }

    Tensor logits_output(
        batch.num_tokens * config.vocab_size, 
        workspace.get_logits_workspace(),
        {batch.num_tokens, config.vocab_size}, 
        DataType::FLOAT32
    );
    

    layers[config.num_hidden_layers]->prefill_forward(hidden, hidden, context);
    layers[config.num_hidden_layers + 1]->prefill_forward(hidden, logits_output, context);

    
}

void QWEN_Model::decode_forward(Batch& batch, Workspace& workspace) {
    // Implement the logic for the decode forward pass
    if (!embedding || !post_processor || layers.size() < config.num_hidden_layers + 2) {
        LOG_ERROR("QWEN_Model is not fully initialized before decode_forward");
        return;
    }

    Tensor hidden(
        batch.num_tokens * config.hidden_size, 
        workspace.get_embedding_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT32
    );
    embedding->forward(batch.token_ids, hidden, batch.num_tokens);

    Tensor hidden2(
        batch.num_tokens * config.hidden_size, 
        workspace.get_hidden2_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT32
    );
    ForwardContext context;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &config;
    for(size_t i = 0; i < config.num_hidden_layers; ++i) {
        context.layer_id = i;
        layers[i]->decode_forward(hidden, hidden2, context);
        std::swap(hidden.data, hidden2.data);
    }

    Tensor logits_output(
        batch.num_tokens * config.vocab_size, 
        workspace.get_logits_workspace(),
        {batch.num_tokens, config.vocab_size}, 
        DataType::FLOAT32
    );

    layers[config.num_hidden_layers]->decode_forward(hidden, hidden, context);
    layers[config.num_hidden_layers + 1]->decode_forward(hidden, logits_output, context);

    post_processor->process(logits_output, context);

    for(size_t i = 0; i < batch.sequences.size(); ++i) {
        batch.token_ids.push_back(batch.sampled_token_ids[i]);
    }

}