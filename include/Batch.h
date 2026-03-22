#pragma once
#include "Sequence.h"
#include <vector>
struct Batch {
    std::vector<std::shared_ptr<Sequence>> sequences;
    std::vector<size_t> token_ids; // Flattened token ids for all sequences in the batch
    std::vector<size_t> sampled_token_ids; // One sampled token per sequence after decode
    size_t num_tokens; // Total number of tokens in the batch
    // Add any additional information needed for processing the batch
    std::vector<size_t> token_positions; // Positions of tokens in the original sequences
    size_t batch_size;
};