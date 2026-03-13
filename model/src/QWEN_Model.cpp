#include "QWEN_Model.h"

void QWEN_Model::init(ModelConfig config) {
    this->config = config;
    weights = std::make_unique<ModelWeights>();
    embedding =  std::make_unique<Embedding>(
        config, 
        weights->layout.embedding_weights
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
void QWEN_Model::load_weights(const char* model_path) {
    weights->load_weights(model_path);
}
void QWEN_Model::prefill_forward(Batch& batch, Workspace& workspace) {
    // Implement the logic for the prefill forward pass
    Tensor hidden(
        batch.num_tokens * config.hidden_size, 
        workspace->get_embedding_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT16
    );
    embedding->forward(batch.token_ids, hidden, batch.num_tokens);

    Tensor hidden2(
        batch.num_tokens * config.hidden_size, 
        workspace->get_hidden2_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT16
    );

    ForwardContext context;
    
    context.batch = &batch;
    context.workspace = &workspace;
    for(size_t i = 0; i < config.num_hidden_layers; ++i) {
        context.layer_id = i;
        transformer_layers[i].prefill_forward(hidden, hidden2, context);
        Tensor tmp = hidden2;
        hidden2 = hidden;
        hidden = tmp;
    }

    layer_norm->forward(hidden.data, hidden.data, context);
    lm_head->forward(hidden, context);

    
}

void QWEN_Model::decode_forward(Batch& batch, Workspace& workspace) {
    // Implement the logic for the decode forward pass
    Tensor hidden(
        batch.num_tokens * config.hidden_size, 
        workspace->get_embedding_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT16
    );
    embedding->forward(batch.token_ids, hidden, batch.num_tokens);

    Tensor hidden2(
        batch.num_tokens * config.hidden_size, 
        workspace->get_hidden2_workspace(),
        {batch.num_tokens, config.hidden_size}, 
        DataType::FLOAT16
    );
    ForwardContext context;
    context.batch = &batch;
    context.workspace = &workspace;

    for(size_t i = 0; i < config.num_hidden_layers; ++i) {
        context.layer_id = i;
        transformer_layers[i].decode_forward(hidden, hidden2, context);
        Tensor tmp = hidden2;
        hidden2 = hidden;
        hidden = tmp;
    }

    layer_norm->forward(hidden.data, hidden.data, context);
    lm_head->forward(hidden, context);
    
}