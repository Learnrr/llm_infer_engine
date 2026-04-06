#pragma once
#include "error.h"
#include <unordered_map>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "Batch.h"
#include "llm_engine_config.h"
static inline uint64_t fnv1a_append(uint64_t h, size_t tok) {
    constexpr uint64_t kPrime = 1099511628211ull;
    h ^= static_cast<uint64_t>(tok + 0x9e3779b97f4a7c15ull); // mix
    h *= kPrime;
    return h;
}

static inline std::vector<uint64_t> build_block_hashes(
    const std::vector<size_t>& tokens,
    size_t block_size
) {
    std::vector<uint64_t> out;
    uint64_t h = 1469598103934665603ull; // FNV offset basis

    const size_t full_blocks = tokens.size() / block_size;
    out.reserve(full_blocks);

    for (size_t i = 0; i < full_blocks * block_size; ++i) {
        h = fnv1a_append(h, tokens[i]);
        if ((i + 1) % block_size == 0) {
            out.push_back(h); // hash snapshot at block boundary
        }
    }
    return out;
}

struct PrefixEntry {
    std::vector<size_t> prefix_tokens;      // exact check
    std::vector<uint64_t> block_hashes;     // hash after each full block
    std::vector<size_t> block_ids;          // cached KV blocks
    size_t cached_tokens = 0;               // multiple of block_size
};


class PrefixCacheManager {
    public:
        PrefixCacheManager(LLMEngineConfig engine_config): engine_config(engine_config) {

        }
        ~PrefixCacheManager() = default;
        ErrorCode get_longest_prefix(Batch& batch);
        ErrorCode upsert_prefix_entry(const std::vector<size_t>& token_ids, const std::vector<size_t>& block_ids);

    private:
    LLMEngineConfig engine_config;
    // prefix hash -> prefix entry idx in prefix_entries
    std::unordered_map<uint64_t, std::vector<size_t>> prefix_cache;
    std::vector<PrefixEntry> prefix_entries;// prevent hash collision

};