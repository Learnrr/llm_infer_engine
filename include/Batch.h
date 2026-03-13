#pragma once
#include "Sequence.h"
#include <vector>
struct Batch {
    vector<shared_ptr<Sequence>> sequences;
    vector<size_t> token_ids; // Flattened token ids for all sequences in the batch
    size_t num_tokens; // Total number of tokens in the batch
    // Add any additional information needed for processing the batch
    vector<size_t> token_positions; // Positions of tokens in the original sequences
    size_t batch_size;
};