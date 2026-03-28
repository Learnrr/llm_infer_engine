#include "layer/Attention.h"
#include "kernel/attention_kernel.h"
#include "kernel/projection_kernel.h"
#include "kernel/output_projection_kernel.h"
#include "kernel/write_kvcache_kernel.h"
#include "utils/cuda_deleter.h"
#include <cuda_runtime.h>
#include <vector>
#include "utils/logger.h"

void Attention::prefill_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) {

    ErrorCode err;
    cudaError_t cuda_err;

    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();
    if(qkv.data == nullptr) {
        LOG_ERROR("Failed to get workspace for qkv projection");
        return;
    }

    size_t batch_seq_len = input.shape[0];
    err = qkv_projection(
        input, 
        layer_layout.qkv_proj_weight, 
        qkv, 
        batch_seq_len, 
        context.config->num_heads, 
        context.config->num_kv_heads,
        context.config->head_dim
    );
    if (err != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to project qkv");
        return;
    }
    LOG_DEBUG("qkv_projection input shape: [" + std::to_string(input.shape[0]) + ", " + std::to_string(input.shape[1]) + "]"
        + ", weight shape: [" + std::to_string(layer_layout.qkv_proj_weight.shape[0]) + ", " + std::to_string(layer_layout.qkv_proj_weight.shape[1]) + "]"
        + ", qkv shape: [" + std::to_string(qkv.shape[0]) + ", " + std::to_string(qkv.shape[1]) + "]"
        + ", batch_seq_len: " + std::to_string(batch_seq_len)
        + ", num_heads: " + std::to_string(context.config->num_heads)
        + ", head_dim: " + std::to_string(context.config->head_dim)
        + ", num_kv_heads: " + std::to_string(context.config->num_kv_heads)
    );

    Tensor q;
    Tensor k;
    Tensor v;

    err = split_qkv(
        qkv, q, k, v, 
        batch_seq_len, 
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim
    );
    if (err != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to split qkv");
        return;
    }
    LOG_DEBUG("split_qkv q shape: [" + std::to_string(q.shape[0]) + ", " + std::to_string(q.shape[1]) + ", " + std::to_string(q.shape[2]) + "]"
        + ", k shape: [" + std::to_string(k.shape[0]) + ", " + std::to_string(k.shape[1]) + ", " + std::to_string(k.shape[2]) + "]"
        + ", v shape: [" + std::to_string(v.shape[0]) + ", " + std::to_string(v.shape[1]) + ", " + std::to_string(v.shape[2]) + "]"
        + ", batch_seq_len: " + std::to_string(batch_seq_len)
        + ", num_heads: " + std::to_string(context.config->num_heads)
        + ", head_dim: " + std::to_string(context.config->head_dim)
        + ", num_kv_heads: " + std::to_string(context.config->num_kv_heads)
    );
    if (!rope) {
        LOG_ERROR("rope is nullptr");
        return;
    }
    rope->apply(
        q,
        k,
        context,
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim
    );
    LOG_DEBUG("RoPE applied to q and k");
    //write k and v to blocked cache
    err = write_cache(context, k, v);
    if (err != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to write to cache");
        return;
    }
    LOG_DEBUG("write_cache completed");

    //build attention output tensor
    Tensor attn_output;
    attn_output.data = context.workspace->get_attn_context_workspace();
    attn_output.shape = {batch_seq_len, context.config->num_heads, context.config->head_dim};
    attn_output.dtype = input.dtype;
    attn_output.device = "gpu";
    attn_output.num_elements = batch_seq_len * context.config->num_heads * context.config->head_dim;
    attn_output.size = attn_output.num_elements * Tensor::element_size_bytes(attn_output.dtype);
    
    // block_ids， block_offsets
    size_t num_tokens = context.batch->num_tokens;
    std::vector<size_t> h_block_ids(num_tokens);
    std::vector<size_t> h_block_offsets(num_tokens);
    std::vector<void*> h_kcache_block_ptrs_void(num_tokens);
    std::vector<void*> h_vcache_block_ptrs_void(num_tokens);
    err = build_read_cache(
        context, 
        h_block_ids, 
        h_block_offsets, 
        h_kcache_block_ptrs_void, 
        h_vcache_block_ptrs_void
    );
    if (err != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build read cache");
        return;
    }
    std::vector<void*> h_kcache_block_ptrs(num_tokens);
    std::vector<void*> h_vcache_block_ptrs(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        h_kcache_block_ptrs[i] = h_kcache_block_ptrs_void[i];
        h_vcache_block_ptrs[i] = h_vcache_block_ptrs_void[i];
    }
    LOG_DEBUG("build_read_cache completed with num_tokens: " + std::to_string(num_tokens));

    size_t* d_block_ids = nullptr;
    size_t* d_block_offsets = nullptr;
    void** d_kcache_block_ptrs = nullptr;
    void** d_vcache_block_ptrs = nullptr;
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_block_ids), num_tokens * sizeof(size_t));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate device memory for block IDs");
        return;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_block_offsets), num_tokens * sizeof(size_t));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate device memory for block offsets");
        return;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_kcache_block_ptrs), num_tokens * sizeof(void*));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate device memory for K-cache block pointers");
        return;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_vcache_block_ptrs), num_tokens * sizeof(void*));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate device memory for V-cache block pointers");
        return;
    }
    cuda_err = cudaMemcpy(d_block_ids, h_block_ids.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy block IDs to device");
        return;
    }
    cuda_err = cudaMemcpy(d_block_offsets, h_block_offsets.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy block offsets to device");
        return;
    }
    cuda_err = cudaMemcpy(d_kcache_block_ptrs, h_kcache_block_ptrs.data(), num_tokens * sizeof(void*), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy K-cache block pointers to device");
        return;
    }
    cuda_err = cudaMemcpy(d_vcache_block_ptrs, h_vcache_block_ptrs.data(), num_tokens * sizeof(void*), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy V-cache block pointers to device");
        return;
    }
    CudaUniquePtr<void*> d_kcache_block_ptrs_dev(d_kcache_block_ptrs);
    CudaUniquePtr<void*> d_vcache_block_ptrs_dev(d_vcache_block_ptrs);
    CudaUniquePtr<size_t> d_block_ids_dev(d_block_ids);
    CudaUniquePtr<size_t> d_block_offsets_dev(d_block_offsets);
    LOG_DEBUG("Copied block IDs, block offsets, K-cache block pointers, and V-cache block pointers to device");
    size_t layer_id = context.layer_id;
    launch_attention_qk_softmax_pv_kernel(
        q.data,
        d_kcache_block_ptrs_dev.get(),
        d_vcache_block_ptrs_dev.get(),
        d_block_ids_dev.get(),
        d_block_offsets_dev.get(),
        attn_output.data,
        batch_seq_len,
        context.config->num_hidden_layers,
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim,
        BLOCK_SIZE,
        context.config->max_seq_len,
        layer_id,
        q.dtype
    );
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Prefill attention kernel launch failed");
        return;
    }
    LOG_DEBUG("Launched attention QK softmax PV kernel for layer " + std::to_string(layer_id));
    err = output_projection(attn_output, layer_layout.o_proj_weight, output);
    if (err != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to project attention output");
        return;
    }
    LOG_DEBUG("Output projection completed for layer " + std::to_string(layer_id));
};

void Attention::decode_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
){

    ErrorCode err;
    cudaError_t cuda_err;

    if (output.data != nullptr && output.size > 0) {
        cudaMemset(output.data, 0, output.size);
    }

    Tensor qkv;
    qkv.data = context.workspace->get_qkv_workspace();
    
    size_t batch_seq_len = input.shape[0];

    err = qkv_projection(
        input,                          //input tensor
        layer_layout.qkv_proj_weight,  //projection weight
        qkv,                            //output qkv tensor           
        batch_seq_len,                  // batch_seq_len
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim
    );

    if (err != ErrorCode::SUCCESS) {
        LOG_ERROR("QKV projection failed with error code: " + std::to_string(static_cast<int>(err)));
        return;
    }
    LOG_DEBUG("qkv_projection input shape: [" + std::to_string(input.shape[0]) + ", " + std::to_string(input.shape[1]) + "]"
        + ", weight shape: [" + std::to_string(layer_layout.qkv_proj_weight.shape[0]) + ", " + std::to_string(layer_layout.qkv_proj_weight.shape[1]) + "]"
        + ", qkv shape: [" + std::to_string(qkv.shape[0]) + ", " + std::to_string(qkv.shape[1]) + "]"
        + ", batch_seq_len: " + std::to_string(batch_seq_len)
        + ", num_heads: " + std::to_string(context.config->num_heads)
        + ", head_dim: " + std::to_string(context.config->head_dim)
        + ", num_kv_heads: " + std::to_string(context.config->num_kv_heads)
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
    LOG_DEBUG("split_qkv q shape: [" + std::to_string(q.shape[0]) + ", " + std::to_string(q.shape[1]) + ", " + std::to_string(q.shape[2]) + "]"
        + ", k shape: [" + std::to_string(k.shape[0]) + ", " + std::to_string(k.shape[1]) + ", " + std::to_string(k.shape[2]) + "]"
        + ", v shape: [" + std::to_string(v.shape[0]) + ", " + std::to_string(v.shape[1]) + ", " + std::to_string(v.shape[2]) + "]"
        + ", batch_seq_len: " + std::to_string(batch_seq_len)
        + ", num_heads: " + std::to_string(context.config->num_heads)
        + ", head_dim: " + std::to_string(context.config->head_dim)
        + ", num_kv_heads: " + std::to_string(context.config->num_kv_heads)
    );    

    rope->apply(
        q,
        k,
        context,
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim
    );
    LOG_DEBUG("RoPE applied to q and k");

    //write k and v to blocked cache
    err = write_cache(context, k, v);
    if (err != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to write cache with error code: " + std::to_string(static_cast<int>(err)));
        return;
    }
    LOG_DEBUG("Cache written successfully");

    // Build segmented history cache metadata for each decode query.
    size_t num_queries = context.batch->num_tokens;
    // history token block offsets
    std::vector<size_t> h_history_block_offsets;
    // history token starts position in all tokens for each query
    std::vector<size_t> h_query_hist_start;
    // history token length for each query
    std::vector<size_t> h_query_hist_len;
    // history K-cache block pointers
    std::vector<void*> h_history_kcache_block_ptrs_void;
    // history V-cache block pointers
    std::vector<void*> h_history_vcache_block_ptrs_void;
    
    err = build_decode_read_cache(
        context, 
        h_history_block_offsets,
        h_query_hist_start,
        h_query_hist_len,
        h_history_kcache_block_ptrs_void,
        h_history_vcache_block_ptrs_void
    );
    
    if (err != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build read cache with error code: " + std::to_string(static_cast<int>(err)));
        return;
    }
    //change void* cache block pointers to float* pointers for attention kernel
    size_t total_history_tokens = h_history_block_offsets.size();
    std::vector<void*> h_history_kcache_block_ptrs(total_history_tokens);
    std::vector<void*> h_history_vcache_block_ptrs(total_history_tokens);
    for (size_t i = 0; i < total_history_tokens; ++i) {
        h_history_kcache_block_ptrs[i] = h_history_kcache_block_ptrs_void[i];
        h_history_vcache_block_ptrs[i] = h_history_vcache_block_ptrs_void[i];
    }
    LOG_DEBUG("Read cache built successfully, total_history_tokens: " + std::to_string(total_history_tokens));

    //allocate these metadata memory on devices
    size_t* d_history_block_offsets = nullptr;
    size_t* d_query_hist_start = nullptr;
    size_t* d_query_hist_len = nullptr;
    void** d_history_kcache_block_ptrs = nullptr;
    void** d_history_vcache_block_ptrs = nullptr;
    
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_history_block_offsets), total_history_tokens * sizeof(size_t));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate device memory for history block offsets");
        return;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_query_hist_start), num_queries * sizeof(size_t));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate device memory for query history starts");
        return;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_query_hist_len), num_queries * sizeof(size_t));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate device memory for query history lengths");
        return;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_history_kcache_block_ptrs), total_history_tokens * sizeof(void*));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate device memory for history K-cache block pointers");
        return;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_history_vcache_block_ptrs), total_history_tokens * sizeof(void*));
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate device memory for history V-cache block pointers");
        return;
    }
    cuda_err = cudaMemcpy(d_history_block_offsets, h_history_block_offsets.data(), total_history_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy history block offsets to device");
        return;
    }
    cuda_err = cudaMemcpy(d_query_hist_start, h_query_hist_start.data(), num_queries * sizeof(size_t), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy query history starts to device");
        return;
    }
    cuda_err = cudaMemcpy(d_query_hist_len, h_query_hist_len.data(), num_queries * sizeof(size_t), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy query history lengths to device");
        return;
    }
    cuda_err = cudaMemcpy(d_history_kcache_block_ptrs, h_history_kcache_block_ptrs.data(), total_history_tokens * sizeof(void*), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy history K-cache block pointers to device");
        return;
    }
    cuda_err = cudaMemcpy(d_history_vcache_block_ptrs, h_history_vcache_block_ptrs.data(), total_history_tokens * sizeof(void*), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to copy history V-cache block pointers to device");
        return;
    }    
    //for automatic memory release using CudaUniquePtr
    CudaUniquePtr<size_t> d_history_block_offsets_dev(d_history_block_offsets);
    CudaUniquePtr<size_t> d_query_hist_start_dev(d_query_hist_start);
    CudaUniquePtr<size_t> d_query_hist_len_dev(d_query_hist_len);
    CudaUniquePtr<void*> d_history_kcache_block_ptrs_dev(d_history_kcache_block_ptrs);
    CudaUniquePtr<void*> d_history_vcache_block_ptrs_dev(d_history_vcache_block_ptrs);
    //copy metadata to device

    LOG_DEBUG("Copied block IDs, block offsets, K-cache block pointers, and V-cache block pointers to device");


    Tensor attn_output;
    attn_output.data = context.workspace->get_attn_context_workspace();
    if(attn_output.data == nullptr) {
        LOG_ERROR("Failed to get workspace for attention output");
        return;
    }
    attn_output.shape = {batch_seq_len, context.config->num_heads, context.config->head_dim};
    attn_output.dtype = input.dtype;
    attn_output.device = "gpu";
    attn_output.num_elements = batch_seq_len * context.config->num_heads * context.config->head_dim;
    attn_output.size = attn_output.num_elements * Tensor::element_size_bytes(attn_output.dtype);
    size_t layer_id = context.layer_id;
    launch_attention_qk_softmax_pv_kernel_decode(
        q.data,
        d_history_kcache_block_ptrs_dev.get(),
        d_history_vcache_block_ptrs_dev.get(),
        d_history_block_offsets_dev.get(),
        d_query_hist_start_dev.get(),
        d_query_hist_len_dev.get(),
        attn_output.data,
        batch_seq_len,
        total_history_tokens,
        context.config->num_hidden_layers,
        context.config->num_heads,
        context.config->num_kv_heads,
        context.config->head_dim,
        BLOCK_SIZE,
        context.config->max_seq_len,
        layer_id,
        q.dtype
    );
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Decode attention kernel launch failed");
        return;
    }
    LOG_DEBUG("Launched attention QK softmax PV kernel for layer " + std::to_string(layer_id));

    err = output_projection(attn_output, layer_layout.o_proj_weight, output);
    if (err != ErrorCode::SUCCESS) {
        LOG_ERROR("Output projection failed with error code: " + std::to_string(static_cast<int>(err)));
        return;
    }
    LOG_DEBUG("Output projection completed for layer " + std::to_string(layer_id));
};

ErrorCode Attention::write_cache(
    ForwardContext& context, 
    const Tensor& key, 
    const Tensor& value
) {
    if(key.data == nullptr || value.data == nullptr) {
        return ErrorCode::INVALID_INPUT;
    }
    if (context.batch == nullptr || context.batch->sequences.empty()) {
        return ErrorCode::INVALID_INPUT;
    }
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
    void** d_kcache_block_ptrs_raw = nullptr;
    void** d_vcache_block_ptrs_raw = nullptr;
    cudaError_t cuda_err;
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_block_ids_raw), num_tokens * sizeof(size_t));
    if (cuda_err != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_block_offsets_raw), num_tokens * sizeof(size_t));
    if (cuda_err != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_kcache_block_ptrs_raw), num_tokens * sizeof(void*));
    if (cuda_err != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_vcache_block_ptrs_raw), num_tokens * sizeof(void*));
    if (cuda_err != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    cuda_err = cudaMemcpy(d_block_ids_raw, h_block_ids.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    cuda_err = cudaMemcpy(d_block_offsets_raw, h_block_offsets.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    cuda_err = cudaMemcpy(d_kcache_block_ptrs_raw, h_kcache_block_ptrs.data(), num_tokens * sizeof(void*), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    cuda_err = cudaMemcpy(d_vcache_block_ptrs_raw, h_vcache_block_ptrs.data(), num_tokens * sizeof(void*), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    CudaUniquePtr<size_t> d_block_ids(d_block_ids_raw);
    CudaUniquePtr<size_t> d_block_offsets(d_block_offsets_raw);
    CudaUniquePtr<void*> d_kcache_block_ptrs(d_kcache_block_ptrs_raw);
    CudaUniquePtr<void*> d_vcache_block_ptrs(d_vcache_block_ptrs_raw);

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
        context.layer_id,
        key.dtype
    );
    return ErrorCode::SUCCESS;
}

ErrorCode Attention::build_read_cache(
    ForwardContext& context, 
    std::vector<size_t>& block_ids, 
    std::vector<size_t>& block_offsets,
    std::vector<void*>& kcache_block_ptrs,
    std::vector<void*>& vcache_block_ptrs

) {
    if (context.batch == nullptr) {
        return ErrorCode::INVALID_INPUT;
    }
    Batch* batch = context.batch;
    size_t num_tokens = batch->num_tokens;
    if (num_tokens == 0) {
        return ErrorCode::INVALID_INPUT;
    }
    if (batch->sequences.size() < num_tokens 
        || batch->token_positions.size() < num_tokens) {
        return ErrorCode::INVALID_INPUT;
    }
    if (block_ids.size() < num_tokens 
        || block_offsets.size() < num_tokens 
        || kcache_block_ptrs.size() < num_tokens 
        || vcache_block_ptrs.size() < num_tokens) {
        return ErrorCode::INVALID_INPUT;
    }
    for(size_t i = 0; i < num_tokens; ++i) {
        auto seq = batch->sequences[i];
        size_t pos = batch->token_positions[i];

        size_t block_idx = pos / BLOCK_SIZE;
        size_t offset = pos % BLOCK_SIZE;

        if (seq == nullptr || block_idx >= seq->blocks.size()) {
            return ErrorCode::INVALID_INPUT;
        }

        block_ids[i] = seq->blocks[block_idx]->block_id;
        block_offsets[i] = offset;
        kcache_block_ptrs[i] = seq->blocks[block_idx]->key_cache_ptr;
        vcache_block_ptrs[i] = seq->blocks[block_idx]->value_cache_ptr;
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Attention::build_decode_read_cache(
    ForwardContext& context, 
    std::vector<size_t>& history_block_offsets,
    std::vector<size_t>& query_hist_start,
    std::vector<size_t>& query_hist_len,
    std::vector<void*>& history_kcache_block_ptrs,
    std::vector<void*>& history_vcache_block_ptrs
){
    if (context.batch == nullptr) {
        return ErrorCode::INVALID_INPUT;
    }
    Batch* batch = context.batch;
    size_t num_queries = batch->num_tokens;
    if (num_queries == 0 
        || batch->token_positions.size() < num_queries 
        || batch->sequences.size() < num_queries) {
        return ErrorCode::INVALID_INPUT;
    }

    query_hist_start.clear();
    query_hist_len.clear();
    history_block_offsets.clear();
    history_kcache_block_ptrs.clear();
    history_vcache_block_ptrs.clear();

    query_hist_start.reserve(num_queries);
    query_hist_len.reserve(num_queries);

    size_t cursor = 0;
    for (size_t q = 0; q < num_queries; ++q) {
        auto seq = batch->sequences[q];
        size_t qpos = batch->token_positions[q];
        size_t len = qpos + 1;

        query_hist_start.push_back(cursor);
        query_hist_len.push_back(len);

        for (size_t t = 0; t < len; ++t) {
            //block idx in the sequence
            size_t block_idx = t / BLOCK_SIZE;
            size_t offset = t % BLOCK_SIZE;
            if (block_idx >= seq->blocks.size()) {
                return ErrorCode::INVALID_INPUT;
            }
            history_block_offsets.push_back(offset);
            history_kcache_block_ptrs.push_back(seq->blocks[block_idx]->key_cache_ptr);
            history_vcache_block_ptrs.push_back(seq->blocks[block_idx]->value_cache_ptr);
            ++cursor;
        }
    }
    return ErrorCode::SUCCESS;

}
// split_qkv
ErrorCode Attention::split_qkv(
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
    } else if (qkv.dtype == DataType::FLOAT16 || qkv.dtype == DataType::BF16) {
        uint16_t* base = static_cast<uint16_t*>(qkv.data);
        q.data = base;
        k.data = base + q_total;
        v.data = base + q_total + k_total;
    } else { //undefined data type
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
    return ErrorCode::SUCCESS;
}

ErrorCode Attention::qkv_projection(
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
        head_dim,
        input.dtype
    );
    return ErrorCode::SUCCESS;

}

ErrorCode Attention::output_projection(
    const Tensor& input, 
    const Tensor& weight, 
    Tensor& output
) {
    if (input.data == nullptr || weight.data == nullptr || output.data == nullptr) {
        return ErrorCode::INVALID_INPUT;
    }
    if (input.shape.empty()) {
        return ErrorCode::INVALID_INPUT;
    }
    size_t batch_seq_len = input.shape[0];
    size_t num_heads = attention_config.num_attention_heads;
    size_t head_dim = attention_config.head_dim;
    launch_output_projection_kernel(
        input.data,
        weight.data,
        output.data,
        batch_seq_len,
        num_heads,
        head_dim,
        input.dtype
    );
    return ErrorCode::SUCCESS;
}