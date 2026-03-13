#include "Attention.h"


void Attention::prefill_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) override{

    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();
    qkv_projection(input, layer_layout->qkv_proj_weight, qkv);

    Tensor q;
    Tensor k;
    Tensor v;

    split_qkv(qkv, q, k, v);
    
    //write k and v to blocked cache
    write_cache(context, k, v);

    Tensor attn_output;
    attn_output.data = context.workspace->get_attn_context_workspace();
    attention_qk_softmax_Pv_kernel(q, block_ids, block_offsets, attn_output);

    output_projection(attn_output, layer_layout->o_proj_weight, output);

};

void Attention::decode_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) override{
    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();
    qkv_projection(input, layer_layout->qkv_proj_weight, qkv);

    Tensor q;
    Tensor k;
    Tensor v;

    split_qkv(qkv, q, k, v);

    //write k and v to blocked cache
    write_cache(context, k, v);

    //build read config for k and v
    size_t* block_ids;
    size_t* block_offsets;
    build_read_cache(context, block_ids, block_offsets);

    //read k and v from blocked cache in kernel through config
    Tensor attn_output;
    attn_output.data = context.workspace->get_attn_context_workspace();
    attention_qk_softmax_Pv_kernel(q, block_ids, block_offsets, attn_output);

    delete[] block_ids;
    delete[] block_offsets;

    output_projection(attn_output, layer_layout->o_proj_weight, output);

};

void Attention::write_cache(ForwardContext& context, const Tensor& key, const Tensor& value) {
    size_t* block_ids = new size_t[context.batch->num_tokens];
    size_t* block_offsets = new size_t[context.batch->num_tokens];

    for(size_t i = 0; i < context.batch->num_tokens; ++i) {
        Batch* batch = context.batch;
        auto seq = batch->sequences[i];
        size_t pos = batch->token_positions[i];

        size_t block_idx = pos / BLOCK_SIZE;
        size_t offset = pos % BLOCK_SIZE;

        block_ids[i] = seq->blocks[block_idx];
        block_offsets[i] = offset;
    }

    write_kvcache_kernel(block_ids, block_offsets, key.data, value.data);

    delete[] block_ids;
    delete[] block_offsets;

}

void Attention::build_read_cache(ForwardContext& context, size_t* block_ids, size_t* block_offsets) {

    block_ids = new int[contact.batch->num_tokens];
    block_offsets = new size_t[contect->batch->num_tokens];
    for(size_t i = 0; i < context.batch->num_tokens; ++i) {
        Batch* batch = context.batch;
        auto seq = batch->sequences[i];
        size_t pos = batch->token_positions[i];

        size_t block_idx = pos / BLOCK_SIZE;
        size_t offset = pos % BLOCK_SIZE;

        block_ids[i] = seq->blocks[block_idx];
        block_offsets[i] = offset;
    }
}
void Attention::split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) {
    // Implement logic to split the combined qkv tensor into separate q, k, v tensors
    q.data = qkv.data;
    q.size = qkv.size / 3; 
    q.shape = {qkv.shape[0], qkv.shape[1], qkv.shape[2] / 3}; 
    q.dtype = qkv.dtype;

    k.data = qkv.data + (qkv.size / 3); 
    k.size = qkv.size / 3; 
    k.shape = {qkv.shape[0], qkv.shape[1], qkv.shape[2] / 3}; 
    k.dtype = qkv.dtype;

    v.data = qkv.data + (2 * qkv.size / 3);
    v.size = qkv.size / 3; 
    v.shape = {qkv.shape[0], qkv.shape[1], qkv.shape[2] / 3}; 
    v.dtype = qkv.dtype;
}
void Attention::qkv_projection(const Tensor& input, const Tensor& weight, Tensor& qkv) {
    // Implement logic to project the input tensor into a combined qkv tensor
    qkv.size = input.shape[0] * weight.shape[1] * 3;
    qkv.shape = {input.shape[0], weight.shape[1] * 3};
    qkv.dtype = input.dtype;
    projection_kernel(input.data, weight.data, nullptr, qkv.data);
    

}

void Attention::output_projection(const Tensor& input, const Tensor& weight, Tensor& output) {
    projection_kernel(input.data, weight.data, nullptr, output.data);
}