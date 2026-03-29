#include "model/QWEN_Model.h"
#include <utility>
#include <cmath>
#include "utils/logger.h"
#include <stdlib.h>
#include "half_float/half.hpp"
#include "utils/tensor_debug.h"

void QWEN_Model::init(LLMEngineConfig& config) {
    this->config = config;
    weights = std::make_unique<ModelWeights>();
    ErrorCode error = weights->init(config.model_config);
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to initialize model weights");
        return;
    }
    LOG_INFO("Model weights initialized successfully");
    // Create embedding layer
    embedding =  std::make_unique<Embedding>(
        config.model_config, 
        weights->layout.embedding_weights
    );            
    LOG_INFO("Initialized Embedding layer");
    
    layers.reserve(config.model_config.num_hidden_layers + 2); // +2 for layer norm and lm head

    // Create transformer layers
    for(size_t i = 0; i < config.model_config.num_hidden_layers; ++i) {
        auto layer_config = config.model_config.get_layer_config<TransformerLayerConfig>(i);
        if (layer_config == nullptr) {
            LOG_ERROR("Missing TransformerLayerConfig in model config");
            return;
        }
        while (layer_config->norm_configs.size() < 2) {
            LayerNormLayerConfig norm_cfg;
            norm_cfg.norm_size = config.model_config.hidden_size;
            layer_config->norm_configs.push_back(norm_cfg);
        }

        auto layer_layout = weights->layout.get_layer_layout<TransformerLayerWeightLayout>(i);
        if (layer_layout == nullptr) {
            LOG_ERROR("Missing TransformerLayerWeightLayout in weight layout");
            return;
        }

        layers.emplace_back(std::make_unique<TransformerLayer>(
            config.model_config.hidden_size, 
            config.model_config.num_heads,
            layer_layout,
            layer_config
        ));
        LOG_DEBUG("Initialized TransformerLayer " + std::to_string(i));
    }
    LOG_INFO("Initialized all Transformer layers");
    // Create final layer norm
    LayerNormLayerConfig layernorm_config;
    layernorm_config.norm_size = config.model_config.hidden_size;
    auto layernorm_config_ptr = config.model_config.get_layer_config<LayerNormLayerConfig>(config.model_config.num_hidden_layers);
    if (layernorm_config_ptr != nullptr) {
        layernorm_config = *layernorm_config_ptr;
    }

    auto layernorm_layout = weights->layout.get_layer_layout<LayerNormLayerWeightLayout>(config.model_config.num_hidden_layers);
    if (layernorm_layout == nullptr) {
        LOG_ERROR("Failed to get final layernorm layout");
        return;
    }

    layers.emplace_back(std::make_unique<RMSNorm>(
        layernorm_config,
        layernorm_layout->norm_weight,
        layernorm_layout->gamma
    ));
    LOG_INFO("Initialized final layernorm");

    // Create LM head
    auto lmhead_config = config.model_config.get_layer_config<LinearLayerConfig>(config.model_config.num_hidden_layers + 1);
    auto lm_head_layout = weights->layout.get_layer_layout<LinearLayerWeightLayout>(config.model_config.num_hidden_layers + 1);
    if (lmhead_config == nullptr || lm_head_layout == nullptr) {
        LOG_ERROR("Failed to get LM head config/layout");
        return;
    }
    layers.emplace_back(std::make_unique<Linear>(
        lmhead_config->linear_config, 
        config.model_config.num_hidden_layers + 1, 
        lm_head_layout->linear_weight
    ));

    post_processor = std::make_unique<PostProcessor>(config.model_config);
    LOG_INFO("Initialized PostProcessor");

}
void QWEN_Model::load_weights(const char* model_path) {
    if (!weights || weights->layout.weights == nullptr || weights->layout.total_size == 0) {
        LOG_ERROR("QWEN_Model::load_weights called before successful init");
        return;
    }
    ErrorCode error = weights->load_weights(model_path);
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to load model weights");
    }
}
void QWEN_Model::prefill_forward(Batch& batch, Workspace& workspace) {
    // Implement the logic for the prefill forward pass
    LOG_DEBUG("Entered QWEN_Model::prefill_forward");
    if (!embedding || layers.size() < config.model_config.num_hidden_layers + 2) {
        std::ostringstream oss;
        oss << "QWEN_Model is not fully initialized before prefill_forward"<<
             " embedding: " << (embedding ? "initialized" : "null") <<
             " layers: " << layers.size() << "/" << (config.model_config.num_hidden_layers + 2);
        LOG_ERROR(oss.str());
        return;
    }

    if (workspace.get_embedding_workspace() == nullptr ||
        workspace.get_hidden2_workspace() == nullptr ||
        workspace.get_logits_workspace() == nullptr) {
        LOG_ERROR("Workspace buffers are not initialized before prefill_forward");
        return;
    }

    if (batch.num_tokens == 0 
        || batch.token_ids.size() != batch.num_tokens 
        || batch.token_positions.size() != batch.num_tokens 
        || batch.sequences.size() != batch.num_tokens) {
        std::ostringstream oss;
        oss << "Invalid prefill batch: num_tokens=" << batch.num_tokens
            << " token_ids=" << batch.token_ids.size()
            << " token_positions=" << batch.token_positions.size()
            << " sequences=" << batch.sequences.size();
        LOG_ERROR(oss.str());
        return;
    }

    Tensor hidden(
        batch.num_tokens * config.model_config.hidden_size, 
        workspace.get_embedding_workspace(),
        {batch.num_tokens, config.model_config.hidden_size}, 
        config.model_config.data_type
    );

    {
        std::ostringstream oss;
        oss << "Running Embedding prefill forward with batch.num_tokens=" << batch.num_tokens <<
             " hidden_size=" << config.model_config.hidden_size <<
             " data_type=" << static_cast<int>(config.model_config.data_type);
        LOG_DEBUG(oss.str());
    }
    LOG_DEBUG("Calling embedding->forward in prefill");
    embedding->forward(batch.token_ids, hidden, batch.num_tokens);
    //log_tensor_anomaly(hidden, std::string("after_embedding"));
    Tensor hidden2(
        batch.num_tokens * config.model_config.hidden_size, 
        workspace.get_hidden2_workspace(),
        {batch.num_tokens, config.model_config.hidden_size}, 
        config.model_config.data_type
    );

    ForwardContext context;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &config;

    for(size_t i = 0; i < config.model_config.num_hidden_layers; ++i) {
        context.layer_id = i;
        LOG_DEBUG("Calling TransformerLayer prefill_forward layer=" + std::to_string(i));
        {
            std::ostringstream oss;
            oss << "Running TransformerLayer " << i << " prefill_forward with batch.num_tokens=" << batch.num_tokens <<
                 " hidden_size=" << config.model_config.hidden_size <<
                 " num_heads=" << config.model_config.num_heads <<
                 " data_type=" << static_cast<int>(config.model_config.data_type);
            LOG_DEBUG(oss.str());
        }
        layers[i]->prefill_forward(hidden, hidden2, context);
        std::swap(hidden.data, hidden2.data);
    }

    Tensor logits_output(
        batch.num_tokens * config.model_config.vocab_size, 
        workspace.get_logits_workspace(),
        {batch.num_tokens, config.model_config.vocab_size}, 
        config.model_config.data_type
    );
    
    {
        std::ostringstream oss;
        oss << "Running LM head prefill_forward with batch.num_tokens=" << batch.num_tokens <<
             " hidden_size=" << config.model_config.hidden_size <<
             " vocab_size=" << config.model_config.vocab_size <<
             " data_type=" << static_cast<int>(config.model_config.data_type);
        LOG_DEBUG(oss.str());
    }
    layers[config.model_config.num_hidden_layers]->prefill_forward(hidden, hidden, context);
    layers[config.model_config.num_hidden_layers + 1]->prefill_forward(hidden, logits_output, context);
    LOG_DEBUG("Finished QWEN_Model::prefill_forward");

    
}

void QWEN_Model::decode_forward(Batch& batch, Workspace& workspace) {
    // Implement the logic for the decode forward pass
    LOG_DEBUG("Entered QWEN_Model::decode_forward");
    if (!embedding 
        || !post_processor 
        || layers.size() < config.model_config.num_hidden_layers + 2) {
        std::ostringstream oss;
        oss << "QWEN_Model is not fully initialized before decode_forward"<<
             " embedding: " << (embedding ? "initialized" : "null") <<
             " post_processor: " << (post_processor ? "initialized" : "null") <<
             " layers: " << layers.size() << "/" << (config.model_config.num_hidden_layers + 2);
        LOG_ERROR(oss.str());
        return;
    }

    if (workspace.get_embedding_workspace() == nullptr ||
        workspace.get_hidden2_workspace() == nullptr ||
        workspace.get_logits_workspace() == nullptr) {
        LOG_ERROR("Workspace buffers are not initialized before decode_forward");
        return;
    }

    if (batch.num_tokens == 0 ||
        batch.token_ids.size() != batch.num_tokens ||
        batch.token_positions.size() != batch.num_tokens ||
        batch.sequences.size() != batch.num_tokens) {
        std::ostringstream oss;
        oss << "Invalid decode batch: num_tokens=" << batch.num_tokens
            << " token_ids=" << batch.token_ids.size()
            << " token_positions=" << batch.token_positions.size()
            << " sequences=" << batch.sequences.size();
        LOG_ERROR(oss.str());
        return;
    }

    Tensor hidden(
        batch.num_tokens * config.model_config.hidden_size,
        workspace.get_embedding_workspace(),
        {batch.num_tokens, config.model_config.hidden_size},
        config.model_config.data_type
    );
    {
        std::ostringstream oss;
        oss << "Running Embedding decode forward with batch.num_tokens=" << batch.num_tokens <<
            " hidden_size=" << config.model_config.hidden_size <<
            " data_type=" << static_cast<int>(config.model_config.data_type);
        LOG_DEBUG(oss.str());
    }
    embedding->forward(batch.token_ids, hidden, batch.num_tokens);
    LOG_DEBUG("Finished embedding->forward in decode");
    // log_tensor_nan_stats(hidden, "after_embedding");

    Tensor hidden2(
        batch.num_tokens * config.model_config.hidden_size,
        workspace.get_hidden2_workspace(),
        {batch.num_tokens, config.model_config.hidden_size},
        config.model_config.data_type
    );
    ForwardContext context;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &config;
    for(size_t i = 0; i < config.model_config.num_hidden_layers; ++i) {
        context.layer_id = i;
        LOG_DEBUG("Calling TransformerLayer decode_forward layer=" + std::to_string(i));
        {
            std::ostringstream oss;
            oss << "Running TransformerLayer " << i << " decode_forward with batch.num_tokens=" << batch.num_tokens <<
                 " hidden_size=" << config.model_config.hidden_size <<
                 " num_heads=" << config.model_config.num_heads <<
                 " data_type=" << static_cast<int>(config.model_config.data_type);
            LOG_DEBUG(oss.str());
        }
        layers[i]->decode_forward(hidden, hidden2, context);
        std::swap(hidden.data, hidden2.data);
        // log_tensor_nan_stats(hidden, (std::string("after_transformer_layer_") + std::to_string(i)).c_str());
    }

    Tensor logits_output(
        batch.num_tokens * config.model_config.vocab_size,
        workspace.get_logits_workspace(),
        {batch.num_tokens, config.model_config.vocab_size},
        config.model_config.data_type
    );
    {
        std::ostringstream oss;
        oss << "Running LM head decode_forward with batch.num_tokens=" << batch.num_tokens <<
             " hidden_size=" << config.model_config.hidden_size <<
             " vocab_size=" << config.model_config.vocab_size <<
             " data_type=" << static_cast<int>(config.model_config.data_type);
        LOG_DEBUG(oss.str());
    }
    layers[config.model_config.num_hidden_layers]->decode_forward(hidden, hidden, context);
    // log_tensor_nan_stats(hidden, "after_final_norm");
    layers[config.model_config.num_hidden_layers + 1]->decode_forward(hidden, logits_output, context);
    // log_tensor_nan_stats(logits_output, "after_lm_head");

    {
        std::ostringstream oss;
        oss << "Running PostProcessor with batch.num_tokens=" << batch.num_tokens <<
             " vocab_size=" << config.model_config.vocab_size <<
             " data_type=" << static_cast<int>(config.model_config.data_type);
        LOG_DEBUG(oss.str());
    }

    // Copy logits to CPU for post-processing
    Tensor logits_on_CPU(
        batch.num_tokens * config.model_config.vocab_size,
        nullptr,
        {batch.num_tokens, config.model_config.vocab_size},
        config.model_config.data_type,
        "cpu"
    );
    logits_on_CPU.data = malloc(logits_output.size);
    if(logits_on_CPU.data == nullptr) {
        LOG_ERROR("Failed to allocate CPU memory for logits in decode_forward");
        return;
    }
    cudaError_t cudaError = cudaMemcpy(
        logits_on_CPU.data, 
        logits_output.data, 
        logits_output.size, 
        cudaMemcpyDeviceToHost
    );
    if (cudaError != cudaSuccess) {
        LOG_ERROR("Failed to copy logits from GPU to CPU for post-processing");
        free(logits_on_CPU.data);
        return;
    }
    
    LOG_DEBUG("running post-processing on CPU");
    post_processor->process(logits_on_CPU, context);
    free(logits_on_CPU.data);

    for(size_t i = 0; i < batch.sequences.size(); ++i) {
        batch.token_ids.push_back(batch.sampled_token_ids[i]);
        {
            std::ostringstream oss;
            oss << "Sequence " << i << " sampled token id: " << batch.sampled_token_ids[i] <<
                 " token position: " << batch.token_positions[i] + 1;
            LOG_INFO(oss.str());
        }
    }

}