#include "Workspace.h"

ErrorCode Workspace::init(const LLMEngineConfig& engine_config) {
    // Re-init should not leak the previous device buffer.
    free();
    size_t datatype = engine_config.model_config.data_type == DataType::FLOAT16 ? 2 : 4; // Assuming FLOAT16 is 2 bytes and FLOAT32 is 4 bytes

    size_t hidden_size = engine_config.model_config.max_seq_len * engine_config.model_config.hidden_size * datatype;
    size_t qkv_size = engine_config.model_config.max_seq_len * 3 * engine_config.model_config.hidden_size * datatype;
    size_t attention_norm_size = engine_config.model_config.max_seq_len * engine_config.model_config.hidden_size * datatype;
    size_t attn_out_size = engine_config.model_config.max_seq_len * engine_config.model_config.hidden_size * datatype;
    size_t context_size = engine_config.model_config.max_seq_len * engine_config.model_config.hidden_size * datatype;
    size_t mlp_norm_size = engine_config.model_config.max_seq_len * engine_config.model_config.hidden_size * datatype;
    // MLP workspace stores packed [gate | up], so it needs 2 * intermediate_size.
    size_t mlp_size = engine_config.model_config.max_seq_len * 2 * engine_config.model_config.mlp_intermediate_size * datatype;
    size_t mlp_out_size = engine_config.model_config.max_seq_len * engine_config.model_config.hidden_size * datatype;
    size_t logits_size = engine_config.model_config.max_seq_len * engine_config.model_config.vocab_size * datatype;
    size_t temp_size = engine_config.model_config.max_seq_len * engine_config.model_config.hidden_size * datatype;

    size_t offset = 0;
    layout.hidden_offset = offset;
    offset += align_size(hidden_size, 256);
    layout.hidden2_offset = offset;
    offset += align_size(hidden_size, 256);
    layout.layer_workspace.attn_norm_offset = offset;
    offset += align_size(attention_norm_size, 256);

    layout.layer_workspace.attention_workspace.qkv_offset = offset;
    offset += align_size(qkv_size, 256);
    layout.layer_workspace.attention_workspace.attn_out_offset = offset;
    offset += align_size(attn_out_size, 256);
    layout.layer_workspace.attention_workspace.context_offset = offset;
    offset += align_size(context_size, 256);

    layout.layer_workspace.mlp_workspace.mlp_offset = offset;
    offset += align_size(mlp_size, 256);
    layout.layer_workspace.mlp_norm_offset = offset;
    offset += align_size(mlp_norm_size, 256);
    layout.layer_workspace.mlp_workspace.mlp_out_offset = offset;
    offset += align_size(mlp_out_size, 256);

    layout.temp_offset = offset;
    offset += align_size(temp_size, 256);
    layout.logits_offset = offset;
    offset += align_size(logits_size, 256);
    layout.total_size = offset;

    cudaError_t cuda_error = cudaMalloc(&workspace, layout.total_size);
    if (cuda_error != cudaSuccess) {
        workspace = nullptr;
        return ErrorCode::CUDA_FAILURE;
    }

    return ErrorCode::SUCCESS;
}