#include "Layer.h"
#include "Tensor.h"
#include "Workspace.h"
#include "ModelWeights.h"
#include "KVCache.h"
#include "Batch.h"
#include "ForwardContext.h"
#include "ModelConfig.h"
class Attention: public Layer {
    public:
        Attention(const AttentionLayerConfig& attention_config, 
            AttentionLayerWeightLayout& attention_layout): 
        : attention_config(attention_config), 
        attention_layout(attention_layout) {}

        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        
    private:
        AttentionLayerConfig attention_config;
        AttentionLayerWeightLayout& attention_layout;

        void write_cache(ForwardContext& context, const Tensor& key, const Tensor& value);
        void split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v, size_t batch_seq_len, size_t num_heads, size_t head_dim);
        void qkv_projection(const Tensor& input, const Tensor& weight, Tensor& qkv, size_t batch_seq_len, size_t num_heads, size_t head_dim);
        void build_read_cache(ForwardContext& context, size_t* block_ids, size_t* block_offsets);
        void output_projection(const Tensor& input, const Tensor& weight, Tensor& output);

};