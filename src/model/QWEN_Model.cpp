#include "QWEN_Model.h"

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

        layers.emplace_back(std::make_unique<TransformerLayer>(
            config.hidden_size, 
            config.num_attention_heads,
            weights->layout.get_layer_layout<TransformerLayerWeightLayout>(i),
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

    auto layernorm_layout_ptr = weights->layout.get_layer_layout<LayerNormLayerWeightLayout>(config.num_hidden_layers);
    LayerNormLayerWeightLayout default_norm_layout;
    default_norm_layout.norm_weight = Tensor();
    LayerNormLayerWeightLayout* norm_layout = &default_norm_layout;
    if (layernorm_layout_ptr != nullptr) {
        norm_layout = layernorm_layout_ptr.get();
    }

    layers.emplace_back(std::make_unique<RMSNorm>(layernorm_config, *norm_layout));
    
    // Create LM head
    auto lmhead_config = config.get_layer_config<LinearLayerConfig>(config.num_hidden_layers + 1);
    auto lm_head_layout = weights->layout.get_layer_layout<LinearLayerWeightLayout>(config.num_hidden_layers + 1);
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
    Tensor hidden(
        batch.num_tokens * config.hidden_size, 
        workspace.get_embedding_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT16
    );


    embedding->forward(batch.token_ids, hidden, batch.num_tokens);

    Tensor hidden2(
        batch.num_tokens * config.hidden_size, 
        workspace.get_hidden2_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT16
    );

    ForwardContext context;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &config;

    for(size_t i = 0; i < config.num_hidden_layers; ++i) {
        context.layer_id = i;
        layers[i]->prefill_forward(hidden, hidden2, context);
        Tensor tmp = hidden2;
        hidden2 = hidden;
        hidden = tmp;
    }

    Tensor logits_output(
        batch.num_tokens * config.vocab_size, 
        workspace.get_logits_workspace(),
        {batch.num_tokens, config.vocab_size}, 
        DataType::FLOAT16
    );
    

    layers[config.num_hidden_layers]->prefill_forward(hidden, hidden, context);
    layers[config.num_hidden_layers + 1]->prefill_forward(hidden, logits_output, context);

    
}

void QWEN_Model::decode_forward(Batch& batch, Workspace& workspace) {
    // Implement the logic for the decode forward pass
    Tensor hidden(
        batch.num_tokens * config.hidden_size, 
        workspace.get_embedding_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT16
    );
    embedding->forward(batch.token_ids, hidden, batch.num_tokens);

    Tensor hidden2(
        batch.num_tokens * config.hidden_size, 
        workspace.get_hidden2_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT16
    );
    ForwardContext context;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &config;
    for(size_t i = 0; i < config.num_hidden_layers; ++i) {
        context.layer_id = i;
        layers[i]->decode_forward(hidden, hidden2, context);
        Tensor tmp = hidden2;
        hidden2 = hidden;
        hidden = tmp;
    }

    Tensor logits_output(
        batch.num_tokens * config.vocab_size, 
        workspace.get_logits_workspace(),
        {batch.num_tokens, config.vocab_size}, 
        DataType::FLOAT16
    );

    layers[config.num_hidden_layers]->decode_forward(hidden, hidden, context);
    layers[config.num_hidden_layers + 1]->decode_forward(hidden, logits_output, context);

    post_processor->process(logits_output, context);

    for(size_t i = 0; i < batch.sequences.size(); ++i) {
        batch.token_ids.push_back(batch.sampled_token_ids[i]);
    }

}