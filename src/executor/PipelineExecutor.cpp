#include "executor/PipelineExecutor.h"
#include "model/ModelForwardContext.h"
#include "utils/logger.h"

ErrorCode PipelineExecutor::run_prefill(Batch& batch, ModelForwardContext& context) {
    model->stage_prefill_forward(batch, context);
    return ErrorCode::SUCCESS;
}

ErrorCode PipelineExecutor::run_decode(Batch& batch, ModelForwardContext& context) {
    model->stage_decode_forward(batch, context);
    return ErrorCode::SUCCESS;
}

ErrorCode PipelineExecutor::run_release_events(Batch& batch) {
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
        LOG_ERROR("PipelineExecutor failed to destroy retained outgoing event: " + std::string(cudaGetErrorString(destroy_err)));
        return ErrorCode::CUDA_FAILURE;
    }

    return ErrorCode::SUCCESS;
}

ErrorCode PipelineExecutor::run_stop() {
    // pipeline executor does not have a receive thread to stop, just return
    return ErrorCode::SUCCESS;
}

ErrorCode PipelineExecutor::run_free(Batch& batch) {
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
                ErrorCode release_err = cache_manager->release_block_ref(block->block_id);
                if (release_err != ErrorCode::SUCCESS) {
                    LOG_ERROR("PipelineExecutor failed to release KV block ref for block_id=" + std::to_string(block->block_id));
                }
            }
        }
        seq->blocks.clear();
        seq_pool->erase(seq_id);
    }

    return ErrorCode::SUCCESS;
}

ErrorCode PipelineExecutor::run_prefix_probe(Batch& batch) {
    if(prefix_cache_manager == nullptr){
        return ErrorCode::SUCCESS;
    }
    prefix_cache_manager->get_longest_prefix(batch);

    size_t cursor = 0;
    for (size_t seq_idx = 0; seq_idx < batch.batch_size; ++seq_idx) {
        const size_t seq_len = batch.max_token_positions[seq_idx] + 1;
        if (cursor >= batch.sequence_ids.size()) {
            break;
        }
        const size_t seq_id = batch.sequence_ids[cursor];
        const size_t prefix_hit_tokens = batch.prefix_hit_tokens_per_seq[seq_idx];
        LOG_INFO(
            "SingleCardExecutor prefix probe result for sequence " +
            std::to_string(seq_id) + ": " +
            std::to_string(prefix_hit_tokens) +
            " prefix tokens hit in cache."
        );
        cursor += seq_len;
    }
    return ErrorCode::SUCCESS;
}


ErrorCode PipelineExecutor::write_prefix_to_cache(const Batch& batch) {

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
                    bool inserted = false;
                    size_t cached_blocks = 0;
                    ErrorCode upsert_error = prefix_cache_manager->upsert_prefix_entry(
                        seq_tokens,
                        block_ids,
                        &inserted,
                        &cached_blocks
                    );
                    if (upsert_error != ErrorCode::SUCCESS) {
                        LOG_ERROR("SingleCardExecutor failed to upsert prefix cache entry for sequence " + std::to_string(seq_id));
                        cursor = end;
                        continue;
                    }

                    //only pin blocks for newly inserted cache entry to avoid 
                    //duplicate refs for the same blocks for the same prefix

                    // If inserted is false, it means the same prefix entry already exists in the cache
                    if (inserted) {
                        const size_t blocks_to_pin = std::min(cached_blocks, block_ids.size());
                        for (size_t b = 0; b < blocks_to_pin; ++b) {
                            ErrorCode pin_err = cache_manager->add_block_ref(block_ids[b]);
                            if (pin_err != ErrorCode::SUCCESS) {
                                LOG_ERROR("SingleCardExecutor failed to pin KV block ref for block_id=" + std::to_string(block_ids[b]));
                            }
                        }
                    }
                }
            }
        }

        cursor = end;
    }
    return ErrorCode::SUCCESS;
}