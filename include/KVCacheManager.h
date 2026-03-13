#pragma once
#include "CacheBlock.h"
#include "define.h"
#include <cuda_runtime.h>
#include "Sequence.h"
#include "utils/include/error.h"
#include <variant>

class KVCacheManager {
    private:
        ErrorCode init();
        void* key_cache;
        void* value_cache;

        vector<CacheBlock> free_blocks;
        vector<CacheBlock> used_blocks;
    public:
        KVCacheManager(){};
        
        variant<CacheBlock*, ErrorCode> get_cache_block(size_t block_id);

        variant<CacheBlock*, ErrorCode> allocate_cache_block();

        ErrorCode free_cache_block(size_t block_id);

        ~KVCacheManager() {
            cudaFree(key_cache);
            cudaFree(value_cache);

            for (auto& block : used_blocks) {
                used_blocks.erase(std::remove_if(used_blocks.begin(), used_blocks.end(),
                    [&block](const CacheBlock& b) { return b.block_id == block.block_id; }), used_blocks.end());
            }

            for(auto& block : free_blocks) {
                free_blocks.erase(std::remove_if(free_blocks.begin(), free_blocks.end(),
                    [&block](const CacheBlock& b) { return b.block_id == block.block_id; }), free_blocks.end());
            }
        }
    private:


        size_t num_blocks;
        size_t block_size;
};