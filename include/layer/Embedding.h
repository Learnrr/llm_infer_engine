#pragma once
#include <algorithm>
#include <vector>
#include"model/ModelConfig.h"
#include "cuda_runtime.h"
#include "Batch.h"
#include "Tensor.h"

class Embedding{
    public:

        Embedding(const ModelConfig& config, Tensor& embedding_weight_gpu) {
            this->vocab_size = config.vocab_size;
            this->embedding_dim = config.hidden_size;
            this->embedding_weight_gpu = &embedding_weight_gpu;
        }

        ~Embedding() {}

        void forward(const std::vector<size_t>& token_ids, Tensor& output, size_t num_tokens);


    
    private:
        size_t vocab_size;
        size_t embedding_dim;
        Tensor* embedding_weight_gpu;

    };