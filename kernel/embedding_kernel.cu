#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "kernel/embedding_kernel.h"

template <typename T>
__global__ void embedding_kernel(
    const size_t* input, 
    const T* embedding_table,
    T* output,
    size_t batch_seq_len,
    size_t hidden_size
){
    int token = blockIdx.x;
    int h = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if(token < batch_seq_len && h < hidden_size) {
        size_t token_id = input[token];
        T val = embedding_table[token_id * hidden_size + h];
        output[token * hidden_size + h] = val;
    }

}

void launch_embedding_kernel(
    const size_t* input, 
    const void* embedding_table,
    void* output,
    size_t batch_seq_len,
    size_t hidden_size,
    DataType dtype
) {
    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(
        static_cast<unsigned int>(batch_seq_len),
        static_cast<unsigned int>((hidden_size + threads - 1) / threads)
    );
    if (dtype == DataType::FLOAT32) {
        embedding_kernel<float><<<grid, block>>>(
            input,
            static_cast<const float*>(embedding_table),
            static_cast<float*>(output),
            batch_seq_len,
            hidden_size
        );
    } else if (dtype == DataType::FLOAT16) {
        embedding_kernel<__half><<<grid, block>>>(
            input,
            static_cast<const __half*>(embedding_table),
            static_cast<__half*>(output),
            batch_seq_len,
            hidden_size
        );
    } else {
        embedding_kernel<__nv_bfloat16><<<grid, block>>>(
            input,
            static_cast<const __nv_bfloat16*>(embedding_table),
            static_cast<__nv_bfloat16*>(output),
            batch_seq_len,
            hidden_size
        );
    }
}