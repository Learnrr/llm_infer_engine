#pragma once
#include "define.h"
#include <vector>
//[block_size, layers, kv_heads, head_dim]
class CacheBlock{
    public:
        size_t block_id;
        std::vector<size_t> token_ids;
        void* key_cache_ptr;
        void* value_cache_ptr;
        bool is_valid;

        CacheBlock(size_t block_id, void* key_cache_ptr, void* value_cache_ptr) : 
        block_id(block_id), 
        key_cache_ptr(key_cache_ptr),
        value_cache_ptr(value_cache_ptr),
        is_valid(true) {}
        
        CacheBlock(const CacheBlock&) = delete;
        CacheBlock& operator=(const CacheBlock&) = delete;        
        
};