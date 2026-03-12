#include "Embedding.h"

void Embedding::forward(size_t* token_ids, Tensor& output, size_t num_tokens){
    // Prefill logic for the given token_ids and seq_len
    embedding_kernel(
        token_ids, 
        layer_layout->embedding_weights.data, 
        output.data, 
        num_tokens
    );
    
}