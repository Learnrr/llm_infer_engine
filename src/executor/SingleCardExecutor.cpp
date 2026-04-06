#include "executor/SingleCardExecutor.h"
#include "utils/logger.h"

#include <algorithm>

ErrorCode SingleCardExecutor::run_prefill(Batch& batch, ModelForwardContext& context) {
    if (context.workspace == nullptr) {
        return ErrorCode::INVALID_INPUT;
    }
    model->prefill_forward(batch, context);

    ErrorCode prefix_cache_result = write_prefix_to_cache(batch);
    if (prefix_cache_result != ErrorCode::SUCCESS) {
        return prefix_cache_result;
    }

    return ErrorCode::SUCCESS;
}

ErrorCode SingleCardExecutor::run_decode(Batch& batch, ModelForwardContext& context) {
    if (context.workspace == nullptr) {
        return ErrorCode::INVALID_INPUT;
    }
    model->decode_forward(batch, context);
    return ErrorCode::SUCCESS;
}

ErrorCode SingleCardExecutor::run_free(Batch& batch) {
    if (seq_pool == nullptr || cache_manager == nullptr) {
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    for (size_t seq_id : batch.sequence_ids) {
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }

        for (const auto& block : seq->blocks) {
            if (block) {
                cache_manager->free_cache_block(block->block_id);
            }
        }
        seq->blocks.clear();
        seq_pool->erase(seq_id);
    }

    return ErrorCode::SUCCESS;
}

ErrorCode SingleCardExecutor::run_release_events(Batch& batch) {
    if (retained_outgoing_events == nullptr) {
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    auto it = retained_outgoing_events->find(batch.batch_id);
    if (it == retained_outgoing_events->end()) {
        return ErrorCode::SUCCESS;
    }

    cudaEvent_t event_to_release = it->second;
    retained_outgoing_events->erase(it);
    cudaError_t destroy_err = cudaEventDestroy(event_to_release);
    if (destroy_err != cudaSuccess) {
        LOG_ERROR("SingleCardExecutor failed to destroy retained outgoing event: " + std::string(cudaGetErrorString(destroy_err)));
        return ErrorCode::CUDA_FAILURE;
    }

    return ErrorCode::SUCCESS;
}

ErrorCode SingleCardExecutor::run_stop() {
    return ErrorCode::SUCCESS;
}

void SingleCardExecutor::run_prefix_probe(Batch& batch) {
    if(prefix_cache_manager == nullptr){
        return;
    }
    prefix_cache_manager->get_longest_prefix(batch);

    size_t batch_size = batch.sequence_ids.size();
    for(size_t i = 0; i < batch_size; ++i){
        size_t seq_id = batch.sequence_ids[i];
        size_t prefix_hit_tokens = batch.prefix_hit_tokens_per_seq[i];
        INFO("SingleCardExecutor prefix probe result for sequence " + std::to_string(seq_id) + ": " + std::to_string(prefix_hit_tokens) + " prefix tokens hit in cache.");
    }
}

ErrorCode SingleCardExecutor::write_prefix_to_cache(const Batch& batch) {

    if (prefix_cache_manager == nullptr) {
        return ErrorCode::SUCCESS;
    }

    

    const size_t n = std::min(batch.token_ids.size(), std::min(batch.sequence_ids.size(), batch.token_positions.size()));
    size_t cursor = 0;
    while (cursor < n) {
        const size_t seq_id = batch.sequence_ids[cursor];
        size_t end = cursor + 1;
        while (end < n && batch.sequence_ids[end] == seq_id) {
            ++end;
        }

        // Only cache prefixes that start from position 0 and are contiguous.
        bool contiguous_prefix = (batch.token_positions[cursor] == 0);
        for (size_t i = cursor; i < end && contiguous_prefix; ++i) {
            if (batch.token_positions[i] != (i - cursor)) {
                contiguous_prefix = false;
            }
        }

        if (contiguous_prefix) {
            auto seq = seq_pool != nullptr ? seq_pool->get(seq_id) : nullptr;
            if (seq && !seq->blocks.empty()) {
                std::vector<size_t> seq_tokens(batch.token_ids.begin() + cursor, batch.token_ids.begin() + end);

                std::vector<size_t> block_ids;
                block_ids.reserve(seq->blocks.size());
                for (const auto& block : seq->blocks) {
                    if (!block) {
                        break;
                    }
                    block_ids.push_back(block->block_id);
                }

                if (!block_ids.empty()) {
                    ErrorCode upsert_error = prefix_cache_manager->upsert_prefix_entry(seq_tokens, block_ids);
                    if (upsert_error != ErrorCode::SUCCESS) {
                        LOG_ERROR("SingleCardExecutor failed to upsert prefix cache entry for sequence " + std::to_string(seq_id));
                    }
                }
            }
        }

        cursor = end;
    }
    return ErrorCode::SUCCESS;
}