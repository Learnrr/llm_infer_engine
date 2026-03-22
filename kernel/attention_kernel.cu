
#include "cuda_runtime.h"
#include "attention_kernel.h"

__global__ void attention_qk_softmax_Pv_kernel(
    const float* q,                    // [num_tokens, num_q_heads, head_dim]
    float** kcache_block_ptrs,         // [num_blocks][block_size, layers, heads, head_dim]
    float** vcache_block_ptrs,         // [num_blocks][block_size, layers, heads, head_dim]
    const size_t* block_ids,           // [num_tokens]
    const size_t* block_offsets,       // [num_tokens]
    float* attn_output,                // [num_tokens, num_q_heads, head_dim]
    int num_tokens,
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

    const int group_size = max(1, num_q_heads / max(1, num_kv_heads));
    const int kv_head = min(num_kv_heads - 1, head / group_size);

    extern __shared__ float scores[]; // size: max_seq_len

    // Compute scalar attention scores by reducing dot(q, k_t) over head_dim.
    if (dim == 0) {
        for (int t = 0; t <= token && t < max_seq_len; ++t) {
            size_t off = block_offsets[t];
            float* k_block = kcache_block_ptrs[t];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int q_offset = token * num_q_heads * head_dim + head * head_dim + d;
                int k_offset = off * num_layers * num_kv_heads * head_dim
                    + layer_id * num_kv_heads * head_dim
                    + kv_head * head_dim
                    + d;
                dot += q[q_offset] * k_block[k_offset];
            }
            scores[t] = dot;
        }

        float max_score = -1e30f;
        for (int t = 0; t <= token && t < max_seq_len; ++t) {
            if (scores[t] > max_score) max_score = scores[t];
        }

        float sum_exp = 0.0f;
        for (int t = 0; t <= token && t < max_seq_len; ++t) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }

        for (int t = 0; t <= token && t < max_seq_len; ++t) {
            scores[t] /= sum_exp;
        }
    }
    __syncthreads();

    float out = 0.0f;
    for (int t = 0; t <= token && t < max_seq_len; ++t) {
        size_t off = block_offsets[t];
        float* v_block = vcache_block_ptrs[t];
        int v_offset = off * num_layers * num_kv_heads * head_dim
            + layer_id * num_kv_heads * head_dim
            + kv_head * head_dim
            + dim;
        float v_val = v_block[v_offset];
        out += scores[t] * v_val;
    }
    int out_offset = token * num_q_heads * head_dim + head * head_dim + dim;
    attn_output[out_offset] = out;
}

void launch_attention_qk_softmax_pv_kernel(
    const float* q,
    float** kcache_block_ptrs,
    float** vcache_block_ptrs,
    const size_t* block_ids,
    const size_t* block_offsets,
    float* attn_output,
    int num_tokens,
    int num_layers,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    int layer_id
) {
    dim3 grid(num_tokens, num_q_heads);
    dim3 block(head_dim);
    attention_qk_softmax_Pv_kernel<<<grid, block, max_seq_len * sizeof(float)>>>(
        q,
        kcache_block_ptrs,
        vcache_block_ptrs,
        block_ids,
        block_offsets,
        attn_output,
        num_tokens,
        num_layers,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_seq_len,
        layer_id
    );
}

__global__ void attention_qk_softmax_Pv_decode_kernel(
    const float* q,
    float** history_kcache_block_ptrs,
    float** history_vcache_block_ptrs,
    const size_t* history_block_offsets,
    const size_t* query_hist_start,
    const size_t* query_hist_len,
    float* attn_output,
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

    const int group_size = max(1, num_q_heads / max(1, num_kv_heads));
    const int kv_head = min(num_kv_heads - 1, head / group_size);

    int hist_start = static_cast<int>(query_hist_start[query_idx]);
    int hist_len = static_cast<int>(query_hist_len[query_idx]);
    if (hist_len <= 0) return;
    hist_len = min(hist_len, max_seq_len);

    extern __shared__ float scores[];

    if (dim == 0) {
        for (int j = 0; j < hist_len; ++j) {
            int hist_idx = hist_start + j;
            if (hist_idx >= total_history_tokens) {
                scores[j] = -1e30f;
                continue;
            }
            size_t off = history_block_offsets[hist_idx];
            float* k_block = history_kcache_block_ptrs[hist_idx];

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int q_offset = query_idx * num_q_heads * head_dim + head * head_dim + d;
                int k_offset = off * num_layers * num_kv_heads * head_dim
                    + layer_id * num_kv_heads * head_dim
                    + kv_head * head_dim
                    + d;
                dot += q[q_offset] * k_block[k_offset];
            }
            scores[j] = dot;
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
        float* v_block = history_vcache_block_ptrs[hist_idx];
        int v_offset = off * num_layers * num_kv_heads * head_dim
            + layer_id * num_kv_heads * head_dim
            + kv_head * head_dim
            + dim;
        out += scores[j] * v_block[v_offset];
    }

    int out_offset = query_idx * num_q_heads * head_dim + head * head_dim + dim;
    attn_output[out_offset] = out;
}

void launch_attention_qk_softmax_pv_kernel_decode(
    const float* q,
    float** history_kcache_block_ptrs,
    float** history_vcache_block_ptrs,
    const size_t* history_block_offsets,
    const size_t* query_hist_start,
    const size_t* query_hist_len,
    float* attn_output,
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
    dim3 grid(num_queries, num_q_heads);
    dim3 block(head_dim);
    attention_qk_softmax_Pv_decode_kernel<<<grid, block, max_seq_len * sizeof(float)>>>(
        q,
        history_kcache_block_ptrs,
        history_vcache_block_ptrs,
        history_block_offsets,
        query_hist_start,
        query_hist_len,
        attn_output,
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