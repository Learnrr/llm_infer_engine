#include <algorithm>
#include"ModelConfig.h"
#include "cuda_runtime.h"
#include "Batch.h"
#include "Tensor.h"
class Embedding{
    public:

        Embedding(ModelConfig config, void*, Tensor& embedding_weight_gpu) {
            this->vocab_size = config.vocab_size;
            this->embedding_dim = config.hidden_size;
            this->embedding_weight_gpu = &embedding_weight_gpu;
        }

        ~Embedding() {}

        void forward(size_t* token_ids, Tensor& output, size_t num_tokens){
            // Prefill logic for the given token_ids and seq_len
            for(size_t i = 0; i < num_tokens; ++i) {
                size_t token_id = token_ids[i];
                // Use token_id to compute the embedding output
                float* embedding_vector = (float*)embedding_weight_gpu->data + token_id * embedding_dim;
                cudaMemcpy(
                    (float*)output.data + i * embedding_dim, 
                    embedding_vector, 
                    embedding_dim * sizeof(float), 
                    cudaMemcpyDeviceToDevice
                );
            }
        }
    
    private:
        size_t vocab_size;
        size_t embedding_dim;
        Tensor* embedding_weight_gpu;

}