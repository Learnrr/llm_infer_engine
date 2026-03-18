#include "cuda_runtime.h"
#include "write_kvcache_kernel.h"

__global__ void write_kvcache_kernel(
    float** kcache_block_ptrs,
    float** vcache_block_ptrs,
    const size_t* block_ids,
    const size_t* block_offsets,
    const float* key_data,
    const float* value_data,
    int num_tokens,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int layer_id
){
    int token = blockIdx.x;
    int head = blockIdx.y;
    int dim = threadIdx.x;
    if (token >= num_tokens || head >= num_kv_heads || dim >= head_dim) return;

    int kv_offset = token * num_kv_heads * head_dim + head * head_dim + dim;
    float k_val = key_data[kv_offset];
    float v_val = value_data[kv_offset];

    size_t off = block_offsets[token];

    // Pointer arrays are sized by num_tokens and populated per token.
    float* k_block = kcache_block_ptrs[token];
    float* v_block = vcache_block_ptrs[token];

    int cache_offset = off * num_layers * num_kv_heads * head_dim + layer_id * num_kv_heads * head_dim + head * head_dim + dim;
    k_block[cache_offset] = k_val;
    v_block[cache_offset] = v_val;
}

void launch_write_kvcache_kernel(
    float** kcache_block_ptrs,
    float** vcache_block_ptrs,
    const size_t* block_ids,
    const size_t* block_offsets,
    const float* key_data,
    const float* value_data,
    int num_tokens,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int layer_id
) {
    dim3 grid(num_tokens, num_kv_heads);
    dim3 block(head_dim);
    write_kvcache_kernel<<<grid, block>>>(
        kcache_block_ptrs,
        vcache_block_ptrs,
        block_ids,
        block_offsets,
        key_data,
        value_data,
        num_tokens,
        num_layers,
        num_kv_heads,
        head_dim,
        block_size,
        layer_id
    );
}