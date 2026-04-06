#include "PrefixCacheManager.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

ErrorCode PrefixCacheManager::upsert_prefix_entry(
    const std::vector<size_t>& token_ids,
    const std::vector<size_t>& block_ids
) {

    const size_t full_blocks = std::min(token_ids.size() / engine_config.block_size, block_ids.size());
    if (full_blocks == 0) {
        return ErrorCode::SUCCESS;
    }

    const size_t cached_tokens = full_blocks * engine_config.block_size;
    std::vector<size_t> cached_prefix_tokens(token_ids.begin(), token_ids.begin() + cached_tokens);
    std::vector<uint64_t> block_hashes = build_block_hashes(cached_prefix_tokens, engine_config.block_size);
    if (block_hashes.empty()) {
        return ErrorCode::SUCCESS;
    }

    const uint64_t final_hash = block_hashes.back();
    auto existing_it = prefix_cache.find(final_hash);
    if (existing_it != prefix_cache.end()) {
        for (size_t idx : existing_it->second) {
            if (idx >= prefix_entries.size()) {
                continue;
            }
            const PrefixEntry& existing = prefix_entries[idx];
            if (existing.cached_tokens != cached_tokens) {
                continue;
            }
            if (existing.prefix_tokens == cached_prefix_tokens) {
                return ErrorCode::SUCCESS;
            }
        }
    }

    PrefixEntry entry;
    entry.cached_tokens = cached_tokens;
    entry.prefix_tokens = std::move(cached_prefix_tokens);
    entry.block_hashes = std::move(block_hashes);
    entry.block_ids.assign(block_ids.begin(), block_ids.begin() + full_blocks);

    const size_t new_idx = prefix_entries.size();
    prefix_entries.push_back(std::move(entry));

    const std::vector<uint64_t>& hashes = prefix_entries.back().block_hashes;
    for (uint64_t h : hashes) {
        prefix_cache[h].push_back(new_idx);
    }

    return ErrorCode::SUCCESS;
}

ErrorCode PrefixCacheManager::get_longest_prefix(Batch& batch) {
    batch.prefix_hit_tokens_per_seq.clear();

    const size_t n = std::min(batch.sequence_ids.size(), std::min(batch.token_positions.size(), batch.token_ids.size()));

    std::vector<size_t> sequence_order;
    std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> seq_to_pos_tokens;
    seq_to_pos_tokens.reserve(batch.batch_size > 0 ? batch.batch_size : n);

    for (size_t i = 0; i < n; ++i) {
        const size_t seq_id = batch.sequence_ids[i];
        auto it = seq_to_pos_tokens.find(seq_id);
        if (it == seq_to_pos_tokens.end()) {
            sequence_order.push_back(seq_id);
        }
        seq_to_pos_tokens[seq_id].push_back({batch.token_positions[i], batch.token_ids[i]});
    }

    batch.prefix_hit_tokens_per_seq.reserve(sequence_order.size());

    for (size_t seq_id : sequence_order) {
        auto& pos_tokens = seq_to_pos_tokens[seq_id];
        std::sort(
            pos_tokens.begin(),
            pos_tokens.end(),
            [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
                return a.first < b.first;
            }
        );

        std::vector<size_t> seq_token_ids;
        seq_token_ids.reserve(pos_tokens.size());
        for (const auto& pt : pos_tokens) {
            seq_token_ids.push_back(pt.second);
        }

        std::vector<uint64_t> block_hashes = build_block_hashes(seq_token_ids, engine_config.block_size);
        size_t longest_hit = 0;

        for (size_t blk_idx = block_hashes.size(); blk_idx > 0; --blk_idx) {
            const size_t hit_tokens = blk_idx * engine_config.block_size;
            const uint64_t prefix_hash = block_hashes[blk_idx - 1];
            auto cache_it = prefix_cache.find(prefix_hash);
            if (cache_it == prefix_cache.end()) {
                continue;
            }

            const std::vector<size_t>& candidate_entry_idxs = cache_it->second;
            bool found = false;
            for (size_t entry_idx : candidate_entry_idxs) {
                if (entry_idx >= prefix_entries.size()) {
                    continue;
                }
                const PrefixEntry& entry = prefix_entries[entry_idx];
                if (entry.cached_tokens < hit_tokens || entry.prefix_tokens.size() < hit_tokens) {
                    continue;
                }

                if (std::equal(seq_token_ids.begin(), seq_token_ids.begin() + hit_tokens, entry.prefix_tokens.begin())) {
                    longest_hit = hit_tokens;
                    found = true;
                    break;
                }
            }

            if (found) {
                break;
            }
        }

        batch.prefix_hit_tokens_per_seq.push_back(longest_hit);
    }

    return ErrorCode::SUCCESS;
}

