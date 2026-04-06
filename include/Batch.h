#pragma once
#include <vector>
#include <cstddef>
struct Batch {
    std::size_t batch_id = 0;
    std::vector<std::size_t> token_ids; // Flattened token ids for all sequences in the batch
    std::vector<std::size_t> sampled_token_ids; // One sampled token per sequence after decode
    std::size_t num_tokens = 0; // Total number of tokens in the batch
    // Add any additional information needed for processing the batch
    std::vector<std::size_t> token_positions; // Positions of tokens in the original sequences
    std::size_t batch_size = 0;
    std::vector<std::size_t> max_token_positions; // Max token position for each sequence in the batch

    std::vector<std::size_t> sequence_ids;

    std::vector<size_t> prefix_hit_tokens_per_seq;

};