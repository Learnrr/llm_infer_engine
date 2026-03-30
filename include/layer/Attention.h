#pragma once
#include "layer/Layer.h"
#include "Tensor.h"
#include "Workspace.h"
#include "model/ModelWeights.h"
#include "KVCache.h"
#include "Batch.h"
#include "ForwardContext.h"
#include "model/ModelConfig.h"
#include "layer/position/RoPE.h"
#include "error.h"
class Attention: public Layer {
    public:
        Attention(const AttentionLayerConfig& attention_config, 
            AttentionLayerWeightLayout& layer_layout)
        : attention_config(attention_config), 
          layer_layout(layer_layout) {
            rope = std::make_unique<RoPE>();
          }

        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        
    private:
        AttentionLayerConfig attention_config;
        AttentionLayerWeightLayout& layer_layout;
        std::unique_ptr<RoPE> rope;

        ErrorCode write_cache(
            ForwardContext& context, 
            const Tensor& key, 
            const Tensor& value
        );
        ErrorCode split_qkv(
            const Tensor& qkv, 
            Tensor& q, Tensor& k, Tensor& v, 
            size_t batch_seq_len, 
            size_t num_heads, 
            size_t num_kv_heads, 
            size_t head_dim
        );
        ErrorCode qkv_projection(
            const Tensor& input, 
            const Tensor& weight, 
            Tensor& qkv, 
            size_t batch_seq_len, 
            size_t num_heads, 
            size_t num_kv_heads, 
            size_t head_dim
        );
        ErrorCode build_read_cache(
            ForwardContext& context, 
            std::vector<size_t>& block_ids, 
            std::vector<size_t>& block_offsets, 
            std::vector<size_t>& query_hist_start,
            std::vector<size_t>& query_hist_len,
            std::vector<void*>& kcache_block_ptrs, 
            std::vector<void*>& vcache_block_ptrs
        );
        ErrorCode build_decode_read_cache(
            ForwardContext& context, 
            std::vector<size_t>& history_block_offsets,
            std::vector<size_t>& query_hist_start,
            std::vector<size_t>& query_hist_len,
            std::vector<void*>& history_kcache_block_ptrs,
            std::vector<void*>& history_vcache_block_ptrs
        );
        ErrorCode output_projection(
            const Tensor& input, 
            const Tensor& weight, 
            Tensor& output
        );

};