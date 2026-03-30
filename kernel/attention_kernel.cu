#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "kernel/attention_kernel.h"

template <typename T>
__device__ inline float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ inline T from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ inline __half from_float<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__global__ void attention_qk_softmax_Pv_kernel(
    const T* q,
    void** kcache_block_ptrs,
    void** vcache_block_ptrs,
    const size_t* block_offsets,
    const size_t* query_hist_start,
    const size_t* query_hist_len,
    T* attn_output,
    int num_tokens,
    int num_sequences,
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    int layer_id
) {
    int token = blockIdx.x;
    int head = blockIdx.y;
    int dim = threadIdx.x;
    if (token >= num_tokens || head >= num_q_heads || dim >= head_dim) return;

    const int safe_kv_heads = num_kv_heads > 0 ? num_kv_heads : 1;
    const int group_size = num_q_heads > 0 ? (num_q_heads / safe_kv_heads) : 1;
    const int safe_group_size = group_size > 0 ? group_size : 1;
    const int kv_head = (head / safe_group_size) < num_kv_heads ? (head / safe_group_size) : (num_kv_heads - 1);
    const float inv_sqrt_head_dim = rsqrtf(static_cast<float>(head_dim));

    int seq_idx = -1;
    int seq_start = 0;
    int seq_len = 0;
    for (int s = 0; s < num_sequences; ++s) {
        int start = static_cast<int>(query_hist_start[s]);
        int len = static_cast<int>(query_hist_len[s]);
        if (token >= start && token < start + len) {
            seq_idx = s;
            seq_start = start;
            seq_len = len;
            break;
        }
    }
    if (seq_idx < 0 || seq_len <= 0) return;

    int local_pos = token - seq_start;
    int hist_len = local_pos + 1;
    if (hist_len > max_seq_len) hist_len = max_seq_len;

    extern __shared__ float scores[];

    if (dim == 0) {
        for (int j = 0; j < hist_len; ++j) {
            int hist_idx = seq_start + j;
            if (hist_idx >= num_tokens) {
                scores[j] = -1e30f;
                continue;
            }
            size_t off = block_offsets[hist_idx];
            T* k_block = static_cast<T*>(kcache_block_ptrs[hist_idx]);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int q_offset = token * num_q_heads * head_dim + head * head_dim + d;
                int k_offset = static_cast<int>(off) * num_layers * num_kv_heads * head_dim
                    + layer_id * num_kv_heads * head_dim
                    + kv_head * head_dim
                    + d;
                dot += to_float<T>(q[q_offset]) * to_float<T>(k_block[k_offset]);
            }
            scores[j] = dot * inv_sqrt_head_dim;
        }

        float max_score = -1e30f;
        for (int j = 0; j < hist_len; ++j) {
            if (scores[j] > max_score) max_score = scores[j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < hist_len; ++j) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }

        for (int j = 0; j < hist_len; ++j) {
            scores[j] /= sum_exp;
        }
    }
    __syncthreads();

    float out = 0.0f;
    for (int j = 0; j < hist_len; ++j) {
        int hist_idx = seq_start + j;
        if (hist_idx >= num_tokens) break;
        size_t off = block_offsets[hist_idx];
        T* v_block = static_cast<T*>(vcache_block_ptrs[hist_idx]);
        int v_offset = static_cast<int>(off) * num_layers * num_kv_heads * head_dim
            + layer_id * num_kv_heads * head_dim
            + kv_head * head_dim
            + dim;
        float v_val = to_float<T>(v_block[v_offset]);
        out += scores[j] * v_val;
    }
    int out_offset = token * num_q_heads * head_dim + head * head_dim + dim;
    attn_output[out_offset] = from_float<T>(out);
}

void launch_attention_qk_softmax_pv_kernel(
    const void* q,
    void** kcache_block_ptrs,
    void** vcache_block_ptrs,
    const size_t* block_offsets,
    const size_t* query_hist_start,
    const size_t* query_hist_len,
    void* attn_output,
    int num_tokens,
    int num_sequences,
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    int layer_id,
    DataType dtype
) {
    dim3 grid(num_tokens, num_q_heads);
    dim3 block(head_dim);
    size_t shared_mem_bytes = static_cast<size_t>(max_seq_len > 0 ? max_seq_len : 1) * sizeof(float);
    if (dtype == DataType::FLOAT32) {
        attention_qk_softmax_Pv_kernel<float><<<grid, block, shared_mem_bytes>>>(
            static_cast<const float*>(q),
            kcache_block_ptrs,
            vcache_block_ptrs,
            block_offsets,
            query_hist_start,
            query_hist_len,
            static_cast<float*>(attn_output),
            num_tokens,
            num_sequences,
            num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_seq_len,
            layer_id
        );
    } else if (dtype == DataType::FLOAT16) {
        attention_qk_softmax_Pv_kernel<__half><<<grid, block, shared_mem_bytes>>>(
            static_cast<const __half*>(q),
            kcache_block_ptrs,
            vcache_block_ptrs,
            block_offsets,
            query_hist_start,
            query_hist_len,
            static_cast<__half*>(attn_output),
            num_tokens,
            num_sequences,
            num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_seq_len,
            layer_id
        );
    } else {
        attention_qk_softmax_Pv_kernel<__nv_bfloat16><<<grid, block, shared_mem_bytes>>>(
            static_cast<const __nv_bfloat16*>(q),
            kcache_block_ptrs,
            vcache_block_ptrs,
            block_offsets,
            query_hist_start,
            query_hist_len,
            static_cast<__nv_bfloat16*>(attn_output),
            num_tokens,
            num_sequences,
            num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_seq_len,
            layer_id
        );
    }
}

template <typename T>
__global__ void attention_qk_softmax_Pv_decode_kernel(
    const T* q,
    void** history_kcache_block_ptrs,
    void** history_vcache_block_ptrs,
    const size_t* history_block_offsets,
    const size_t* query_hist_start,
    const size_t* query_hist_len,
    T* attn_output,
    int num_queries,
    int total_history_tokens,
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    int layer_id
) {
    int query_idx = blockIdx.x;
    int head = blockIdx.y;
    int dim = threadIdx.x;
    if (query_idx >= num_queries || head >= num_q_heads || dim >= head_dim) return;

    const int safe_kv_heads = num_kv_heads > 0 ? num_kv_heads : 1;
    const int group_size = num_q_heads > 0 ? (num_q_heads / safe_kv_heads) : 1;
    const int safe_group_size = group_size > 0 ? group_size : 1;
    const int kv_head = (head / safe_group_size) < num_kv_heads ? (head / safe_group_size) : (num_kv_heads - 1);
    const float inv_sqrt_head_dim = rsqrtf(static_cast<float>(head_dim));

    int hist_start = static_cast<int>(query_hist_start[query_idx]);
    int hist_len = static_cast<int>(query_hist_len[query_idx]);
    if (hist_len <= 0) return;
    hist_len = hist_len < max_seq_len ? hist_len : max_seq_len;

    extern __shared__ float scores[];

    if (dim == 0) {
        for (int j = 0; j < hist_len; ++j) {
            int hist_idx = hist_start + j;
            if (hist_idx >= total_history_tokens) {
                scores[j] = -1e30f;
                continue;
            }
            size_t off = history_block_offsets[hist_idx];
            T* k_block = static_cast<T*>(history_kcache_block_ptrs[hist_idx]);

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int q_offset = query_idx * num_q_heads * head_dim + head * head_dim + d;
                int k_offset = static_cast<int>(off) * num_layers * num_kv_heads * head_dim
                    + layer_id * num_kv_heads * head_dim
                    + kv_head * head_dim
                    + d;
                dot += to_float<T>(q[q_offset]) * to_float<T>(k_block[k_offset]);
            }
            scores[j] = dot * inv_sqrt_head_dim;
        }

        float max_score = -1e30f;
        for (int j = 0; j < hist_len; ++j) {
            if (scores[j] > max_score) max_score = scores[j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < hist_len; ++j) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }

        for (int j = 0; j < hist_len; ++j) {
            scores[j] /= sum_exp;
        }
    }
    __syncthreads();

    float out = 0.0f;
    for (int j = 0; j < hist_len; ++j) {
        int hist_idx = hist_start + j;
        if (hist_idx >= total_history_tokens) break;
        size_t off = history_block_offsets[hist_idx];
        T* v_block = static_cast<T*>(history_vcache_block_ptrs[hist_idx]);
        int v_offset = static_cast<int>(off) * num_layers * num_kv_heads * head_dim
            + layer_id * num_kv_heads * head_dim
            + kv_head * head_dim
            + dim;
        out += scores[j] * to_float<T>(v_block[v_offset]);
    }

    int out_offset = query_idx * num_q_heads * head_dim + head * head_dim + dim;
    attn_output[out_offset] = from_float<T>(out);
}

void launch_attention_qk_softmax_pv_kernel_decode(
    const void* q,
    void** history_kcache_block_ptrs,
    void** history_vcache_block_ptrs,
    const size_t* history_block_offsets,
    const size_t* query_hist_start,
    const size_t* query_hist_len,
    void* attn_output,
    int num_queries,
    int total_history_tokens,
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    int layer_id,
    DataType dtype
) {
    dim3 grid(num_queries, num_q_heads);
    dim3 block(head_dim);
    size_t shared_tokens = static_cast<size_t>(max_seq_len > 0 ? max_seq_len : 1);
    size_t shared_mem_bytes = shared_tokens * sizeof(float);
    if (dtype == DataType::FLOAT32) {
        attention_qk_softmax_Pv_decode_kernel<float><<<grid, block, shared_mem_bytes>>>(
            static_cast<const float*>(q),
            history_kcache_block_ptrs,
            history_vcache_block_ptrs,
            history_block_offsets,
            query_hist_start,
            query_hist_len,
            static_cast<float*>(attn_output),
            num_queries,
            total_history_tokens,
            num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_seq_len,
            layer_id
        );
    } else if (dtype == DataType::FLOAT16) {
        attention_qk_softmax_Pv_decode_kernel<__half><<<grid, block, shared_mem_bytes>>>(
            static_cast<const __half*>(q),
            history_kcache_block_ptrs,
            history_vcache_block_ptrs,
            history_block_offsets,
            query_hist_start,
            query_hist_len,
            static_cast<__half*>(attn_output),
            num_queries,
            total_history_tokens,
            num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_seq_len,
            layer_id
        );
    } else {
        attention_qk_softmax_Pv_decode_kernel<__nv_bfloat16><<<grid, block, shared_mem_bytes>>>(
            static_cast<const __nv_bfloat16*>(q),
            history_kcache_block_ptrs,
            history_vcache_block_ptrs,
            history_block_offsets,
            query_hist_start,
            query_hist_len,
            static_cast<__nv_bfloat16*>(attn_output),
            num_queries,
            total_history_tokens,
            num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_seq_len,
            layer_id
        );
    }
}
