#pragma once

#include <cstddef>
#include <memory>
#include <variant>
#include <vector>
#include <cuda_runtime.h>

#include "Cacheblock.h"
#include "error.h"
#include "llm_engine_config.h"

class KVCacheManager {
public:
    KVCacheManager() = default;
    ~KVCacheManager() {
        if (key_cache != nullptr) {
            cudaFree(key_cache);
            key_cache = nullptr;
        }
        if (value_cache != nullptr) {
            cudaFree(value_cache);
            value_cache = nullptr;
        }
    }

    ErrorCode init(const LLMEngineConfig& config);

    std::variant<std::shared_ptr<CacheBlock>, ErrorCode> get_cache_block(size_t block_id);
    std::variant<std::shared_ptr<CacheBlock>, ErrorCode> allocate_cache_block();
    ErrorCode free_cache_block(size_t block_id);

    ErrorCode add_block_ref(size_t block_id);
    ErrorCode release_block_ref(size_t block_id);

private:
    void* key_cache = nullptr;
    void* value_cache = nullptr;
    size_t num_blocks = 0;
    size_t block_size = 0;

    std::vector<std::shared_ptr<CacheBlock>> free_blocks;
    std::vector<std::shared_ptr<CacheBlock>> used_blocks;
};