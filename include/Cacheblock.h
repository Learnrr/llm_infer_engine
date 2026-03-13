#pragma once
#include "define.h"
#include <vector>
//[block_size, layers, heads, head_dim]
class CacheBlock{
    public:
        size_t block_id;
        std::vector<size_t> token_ids;
        void* key_cache_ptr;
        void* value_cache_ptr;
        bool is_valid;

        CacheBlock(size_t block_id) : 
        block_id(block_id), 
        key_cache_ptr(nullptr),
        value_cache_ptr(nullptr),
        is_valid(true) {}
        
        CacheBlock(const CacheBlock&) = delete;
        CacheBlock& operator=(const CacheBlock&) = delete;        
        
};