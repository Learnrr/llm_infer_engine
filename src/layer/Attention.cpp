#include "Attention.h"
#include "attention_kernel.h"
#include "projection_kernel.h"
#include "output_projection_kernel.h"
#include "write_kvcache_kernel.h"
#include "utils/cuda_deleter.h"
#include <cuda_runtime.h>
#include <vector>

void Attention::prefill_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) {

    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();

    size_t batch_seq_len = input.shape[0];

    qkv_projection(
        input, 
        layer_layout.qkv_proj_weight, 
        qkv, 
        batch_seq_len, 
        context.config->num_heads, 
        context.config->num_kv_heads,
        context.config->head_dim
    );

    Tensor q;
    Tensor k;
    Tensor v;

    split_qkv(
        qkv, q, k, v, 
        batch_seq_len, 
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim
    );

    rope->apply(
        q,
        k,
        context,
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim
    );
    
    //write k and v to blocked cache
    write_cache(context, k, v);

    Tensor attn_output;
    attn_output.data = context.workspace->get_attn_context_workspace();

    // block_ids， block_offsets
    size_t num_tokens = context.batch->num_tokens;
    std::vector<size_t> h_block_ids(num_tokens);
    std::vector<size_t> h_block_offsets(num_tokens);
    std::vector<float*> h_kcache_block_ptrs(num_tokens);
    std::vector<float*> h_vcache_block_ptrs(num_tokens);
    build_read_cache(context, h_block_ids.data(), h_block_offsets.data(), (void**)h_kcache_block_ptrs.data(), (void**)h_vcache_block_ptrs.data());

    size_t* d_block_ids;
    size_t* d_block_offsets;
    float** d_kcache_block_ptrs;
    float** d_vcache_block_ptrs;
    cudaMalloc(reinterpret_cast<void**>(&d_block_ids), num_tokens * sizeof(size_t));
    cudaMalloc(reinterpret_cast<void**>(&d_block_offsets), num_tokens * sizeof(size_t));
    cudaMalloc(reinterpret_cast<void**>(&d_kcache_block_ptrs), num_tokens * sizeof(float*));
    cudaMalloc(reinterpret_cast<void**>(&d_vcache_block_ptrs), num_tokens * sizeof(float*));
    cudaMemcpy(d_block_ids, h_block_ids.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets, h_block_offsets.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kcache_block_ptrs, h_kcache_block_ptrs.data(), num_tokens * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vcache_block_ptrs, h_vcache_block_ptrs.data(), num_tokens * sizeof(float*), cudaMemcpyHostToDevice);
    CudaUniquePtr<float*> d_kcache_block_ptrs_dev(d_kcache_block_ptrs);
    CudaUniquePtr<float*> d_vcache_block_ptrs_dev(d_vcache_block_ptrs);
    CudaUniquePtr<size_t> d_block_ids_dev(d_block_ids);
    CudaUniquePtr<size_t> d_block_offsets_dev(d_block_offsets);

    size_t layer_id = context.layer_id;
    launch_attention_qk_softmax_pv_kernel(
        (float*)q.data,
        d_kcache_block_ptrs_dev.get(),
        d_vcache_block_ptrs_dev.get(),
        d_block_ids_dev.get(),
        d_block_offsets_dev.get(),
        (float*)attn_output.data,
        batch_seq_len,
        context.config->num_hidden_layers,
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim,
        BLOCK_SIZE,
        context.config->max_seq_len,
        layer_id
    );

    output_projection(attn_output, layer_layout.o_proj_weight, output);

};

void Attention::decode_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
){

    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();
    
    size_t batch_seq_len = input.shape[0];
    qkv_projection(
        input,                          //input tensor
        layer_layout.qkv_proj_weight,  //projection weight
        qkv,                            //output qkv tensor           
        batch_seq_len,                  // batch_seq_len
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim
    );

    Tensor q;
    Tensor k;
    Tensor v;

    split_qkv(
        qkv, q, k, v, 
        batch_seq_len, 
        context.config->num_heads, 
        context.config->num_kv_heads,
        context.config->head_dim
    );

    rope->apply(
        q,
        k,
        context,
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim
    );

    //write k and v to blocked cache
    write_cache(context, k, v);

    size_t num_tokens = context.batch->num_tokens;
    std::vector<size_t> h_block_ids(num_tokens);
    std::vector<size_t> h_block_offsets(num_tokens);
    std::vector<float*> h_kcache_block_ptrs(num_tokens);
    std::vector<float*> h_vcache_block_ptrs(num_tokens);
    build_read_cache(context, h_block_ids.data(), h_block_offsets.data(), (void**)h_kcache_block_ptrs.data(), (void**)h_vcache_block_ptrs.data());

    size_t* d_block_ids_raw = nullptr;
    size_t* d_block_offsets_raw = nullptr;
    float** d_kcache_block_ptrs_raw = nullptr;
    float** d_vcache_block_ptrs_raw = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_block_ids_raw), num_tokens * sizeof(size_t));
    cudaMalloc(reinterpret_cast<void**>(&d_block_offsets_raw), num_tokens * sizeof(size_t));
    cudaMalloc(reinterpret_cast<void**>(&d_kcache_block_ptrs_raw), num_tokens * sizeof(float*));
    cudaMalloc(reinterpret_cast<void**>(&d_vcache_block_ptrs_raw), num_tokens * sizeof(float*));

    CudaUniquePtr<size_t> d_block_ids(d_block_ids_raw);
    CudaUniquePtr<size_t> d_block_offsets(d_block_offsets_raw);
    CudaUniquePtr<float*> d_kcache_block_ptrs(d_kcache_block_ptrs_raw);
    CudaUniquePtr<float*> d_vcache_block_ptrs(d_vcache_block_ptrs_raw);

    cudaMemcpy(d_block_ids.get(), h_block_ids.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets.get(), h_block_offsets.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kcache_block_ptrs.get(), h_kcache_block_ptrs.data(), num_tokens * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vcache_block_ptrs.get(), h_vcache_block_ptrs.data(), num_tokens * sizeof(float*), cudaMemcpyHostToDevice);

    Tensor attn_output;
    attn_output.data = context.workspace->get_attn_context_workspace();
    size_t layer_id = context.layer_id;
    launch_attention_qk_softmax_pv_kernel(
        (float*)q.data,
        d_kcache_block_ptrs.get(),
        d_vcache_block_ptrs.get(),
        d_block_ids.get(),
        d_block_offsets.get(),
        (float*)attn_output.data,
        batch_seq_len,
        context.config->num_hidden_layers,
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim,
        BLOCK_SIZE,
        context.config->max_seq_len,
        layer_id
    );

    output_projection(attn_output, layer_layout.o_proj_weight, output);

};

void Attention::write_cache(
    ForwardContext& context, 
    const Tensor& key, 
    const Tensor& value
) {

    size_t num_tokens = context.batch->num_tokens;
    std::vector<size_t> h_block_ids(num_tokens);
    std::vector<size_t> h_block_offsets(num_tokens);
    std::vector<void*> h_kcache_block_ptrs(num_tokens);
    std::vector<void*> h_vcache_block_ptrs(num_tokens);

    for(size_t i = 0; i < num_tokens; ++i) {
        Batch* batch = context.batch;
        auto seq = batch->sequences[i];
        size_t pos = batch->token_positions[i];

        size_t block_idx = pos / BLOCK_SIZE;
        size_t offset = pos % BLOCK_SIZE;

        h_block_ids[i] = seq->blocks[block_idx]->block_id;
        h_block_offsets[i] = offset;
        h_kcache_block_ptrs[i] = seq->blocks[block_idx]->key_cache_ptr;
        h_vcache_block_ptrs[i] = seq->blocks[block_idx]->value_cache_ptr;
    }

    size_t* d_block_ids_raw = nullptr;
    size_t* d_block_offsets_raw = nullptr;
    float** d_kcache_block_ptrs_raw = nullptr;
    float** d_vcache_block_ptrs_raw = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_block_ids_raw), num_tokens * sizeof(size_t));
    cudaMalloc(reinterpret_cast<void**>(&d_block_offsets_raw), num_tokens * sizeof(size_t));
    cudaMalloc(reinterpret_cast<void**>(&d_kcache_block_ptrs_raw), num_tokens * sizeof(void*));
    cudaMalloc(reinterpret_cast<void**>(&d_vcache_block_ptrs_raw), num_tokens * sizeof(void*));

    CudaUniquePtr<size_t> d_block_ids(d_block_ids_raw);
    CudaUniquePtr<size_t> d_block_offsets(d_block_offsets_raw);
    CudaUniquePtr<float*> d_kcache_block_ptrs(d_kcache_block_ptrs_raw);
    CudaUniquePtr<float*> d_vcache_block_ptrs(d_vcache_block_ptrs_raw);

    cudaMemcpy(d_block_ids.get(), h_block_ids.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets.get(), h_block_offsets.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kcache_block_ptrs.get(), h_kcache_block_ptrs.data(), num_tokens * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vcache_block_ptrs.get(), h_vcache_block_ptrs.data(), num_tokens * sizeof(void*), cudaMemcpyHostToDevice);


    launch_write_kvcache_kernel(
        d_kcache_block_ptrs.get(), 
        d_vcache_block_ptrs.get(), 
        d_block_ids.get(),
        d_block_offsets.get(), 
        key.data, 
        value.data,
        num_tokens,
        context.config->num_hidden_layers,
        context.config->num_kv_heads,
        context.config->head_dim,
        BLOCK_SIZE,
        context.layer_id
    );

}

void Attention::build_read_cache(
    ForwardContext& context, 
    size_t* block_ids, 
    size_t* block_offsets,
    void** kcache_block_ptrs,
    void** vcache_block_ptrs

) {
      
    for(size_t i = 0; i < context.batch->num_tokens; ++i) {
        Batch* batch = context.batch;
        auto seq = batch->sequences[i];
        size_t pos = batch->token_positions[i];

        size_t block_idx = pos / BLOCK_SIZE;
        size_t offset = pos % BLOCK_SIZE;

        block_ids[i] = seq->blocks[block_idx]->block_id;
        block_offsets[i] = offset;
        kcache_block_ptrs[i] = seq->blocks[block_idx]->key_cache_ptr;
        vcache_block_ptrs[i] = seq->blocks[block_idx]->value_cache_ptr;
    }



}
// split_qkv
void Attention::split_qkv(
    const Tensor& qkv, 
    Tensor& q, 
    Tensor& k, 
    Tensor& v,
    size_t batch_seq_len, 
    size_t num_heads, 
    size_t num_kv_heads,
    size_t head_dim
) {
    size_t q_total = batch_seq_len * num_heads * head_dim;
    size_t k_total = batch_seq_len * num_kv_heads * head_dim;
    size_t v_total = batch_seq_len * num_kv_heads * head_dim;

    if (qkv.dtype == DataType::FLOAT32) {
        float* base = static_cast<float*>(qkv.data);
        q.data = base;
        k.data = base + q_total;
        v.data = base + q_total + k_total;
    } else if (qkv.dtype == DataType::FLOAT16) {
        uint16_t* base = static_cast<uint16_t*>(qkv.data);
        q.data = base;
        k.data = base + q_total;
        v.data = base + q_total + k_total;
    } else {
        q.data = nullptr;
        k.data = nullptr;
        v.data = nullptr;
    }
    const size_t elem_bytes = Tensor::element_size_bytes(qkv.dtype);
    q.size = q_total * elem_bytes;
    k.size = k_total * elem_bytes;
    v.size = v_total * elem_bytes;
    q.shape = {batch_seq_len, num_heads, head_dim};
    k.shape = {batch_seq_len, num_kv_heads, head_dim};
    v.shape = {batch_seq_len, num_kv_heads, head_dim};
    q.dtype = qkv.dtype;
    k.dtype = qkv.dtype;
    v.dtype = qkv.dtype;
}
void Attention::qkv_projection(
    const Tensor& input, 
    const Tensor& weight, 
    Tensor& qkv, 
    size_t batch_seq_len, 
    size_t num_heads,
    size_t num_kv_heads, //GHA
    size_t head_dim
) {
    size_t qkv_hidden = num_heads * head_dim + 2 * num_kv_heads * head_dim;
    qkv.size = batch_seq_len * qkv_hidden * Tensor::element_size_bytes(input.dtype);
    qkv.shape = {batch_seq_len, qkv_hidden};
    qkv.dtype = input.dtype;

    launch_projection_kernel(
        input.data, 
        weight.data, 
        nullptr, 
        qkv.data,
        batch_seq_len, 
        num_heads,
        num_kv_heads, //GHA 
        head_dim
    );
    

}

void Attention::output_projection(
    const Tensor& input, 
    const Tensor& weight, 
    Tensor& output
) {
    size_t batch_seq_len = input.shape[0];
    size_t num_heads = attention_config.num_attention_heads;
    size_t head_dim = attention_config.head_dim;
    launch_output_projection_kernel(
        static_cast<const float*>(input.data),
        static_cast<const float*>(weight.data),
        static_cast<float*>(output.data),
        batch_seq_len,
        num_heads,
        head_dim
    );
}