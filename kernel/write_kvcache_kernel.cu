#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "kernel/write_kvcache_kernel.h"

template <typename T>
__global__ void write_kvcache_kernel(
    void** kcache_block_ptrs,
    void** vcache_block_ptrs,
    const size_t* block_ids,
    const size_t* block_offsets,
    const T* key_data,
    const T* value_data,
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
    T k_val = key_data[kv_offset];
    T v_val = value_data[kv_offset];

    size_t off = block_offsets[token];

    // Pointer arrays are sized by num_tokens and populated per token.
    T* k_block = static_cast<T*>(kcache_block_ptrs[token]);
    T* v_block = static_cast<T*>(vcache_block_ptrs[token]);

    int cache_offset = off * num_layers * num_kv_heads * head_dim + layer_id * num_kv_heads * head_dim + head * head_dim + dim;
    k_block[cache_offset] = k_val;
    v_block[cache_offset] = v_val;
}

void launch_write_kvcache_kernel(
    void** kcache_block_ptrs,
    void** vcache_block_ptrs,
    const size_t* block_ids,
    const size_t* block_offsets,
    const void* key_data,
    const void* value_data,
    int num_tokens,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int layer_id,
    DataType dtype
) {
    dim3 grid(num_tokens, num_kv_heads);
    dim3 block(head_dim);
    if (dtype == DataType::FLOAT32) {
        write_kvcache_kernel<float><<<grid, block>>>(
            kcache_block_ptrs,
            vcache_block_ptrs,
            block_ids,
            block_offsets,
            static_cast<const float*>(key_data),
            static_cast<const float*>(value_data),
            num_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            layer_id
        );
    } else if (dtype == DataType::FLOAT16) {
        write_kvcache_kernel<__half><<<grid, block>>>(
            kcache_block_ptrs,
            vcache_block_ptrs,
            block_ids,
            block_offsets,
            static_cast<const __half*>(key_data),
            static_cast<const __half*>(value_data),
            num_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            layer_id
        );
    } else {
        write_kvcache_kernel<__nv_bfloat16><<<grid, block>>>(
            kcache_block_ptrs,
            vcache_block_ptrs,
            block_ids,
            block_offsets,
            static_cast<const __nv_bfloat16*>(key_data),
            static_cast<const __nv_bfloat16*>(value_data),
            num_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            layer_id
        );
    }
}