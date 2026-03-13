#include "Layer.h"
#include "Tensor.h"
#include "Workspace.h"
#include "ModelWeights.h"
#include "KVCache.h"
#include "Batch.h"
#include "ForwardContext.h"
class Attention: public Layer {
    public:
        Attention(int hidden_size, int num_heads, LayerWeightLayout* layer_layout) : layer_layout(layer_layout) {}

        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void write_cache(ForwardContext& context, void* key_data, void* value_data);
        void split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v);
        void qkv_projection(const Tensor& input, const Tensor& weight, Tensor& qkv);
        void build_read_cache(ForwardContext& context, size_t* block_ids, size_t* block_offsets);
        void output_projection(const Tensor& input, const Tensor& weight, Tensor& output);
    private:
        LayerWeightLayout* layer_layout;

};