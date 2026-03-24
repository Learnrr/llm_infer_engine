#include "Embedding.h"
#include "embedding_kernel.h"
#include "cuda_runtime.h"
#include "utils/cuda_deleter.h"

void Embedding::forward(const std::vector<size_t>& token_ids, Tensor& output, size_t num_tokens) {
    //put token_ids to gpu
    size_t* d_token_ids_raw = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_token_ids_raw), num_tokens * sizeof(size_t));
    CudaUniquePtr<size_t> d_token_ids(d_token_ids_raw);

    cudaMemcpy(d_token_ids.get(), token_ids.data(), num_tokens * sizeof(size_t), cudaMemcpyHostToDevice);
    launch_embedding_kernel(
        d_token_ids.get(), 
        static_cast<float*>(embedding_weight_gpu->data),
        static_cast<float*>(output.data), 
        num_tokens, 
        embedding_dim
    );
}
