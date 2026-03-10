#pragma once
#include "CacheBlock.h"
#include "define.h"
#include <cuda_runtime.h>
#include "Sequence.h"

class KVCacheManager {
    private:
        init(){
            
            size_t total_cache_size = 
                BLOCK_SIZE * NUM_LAYERS * HEAD_DIM * NUM_HEADS * DTYPE;

            num_blocks = total_cache_size / BLOCK_SIZE;

            cudaError_t error = cudaMalloc(&key_cache, total_cache_size);
            if (error != cudaSuccess) {
                // Handle error
            }
            error = cudaMalloc(&value_cache, total_cache_size);
            if (error != cudaSuccess) {
                // Handle error
            }
            for (size_t i = 0; i < num_blocks; ++i) {
                CacheBlock block(
                    i, 
                    key_cache + i * (BLOCK_SIZE*NUM_LAYERS*HEAD_DIM*NUM_HEADS*DTYPE),
                    value_cache + i * (BLOCK_SIZE*NUM_LAYERS*HEAD_DIM*NUM_HEADS*DTYPE)
                );
                free_blocks.push_back(block);
            }
        }
        void* key_cache;
        void* value_cache;

        vector<CacheBlock> free_blocks;
        vector<CacheBlock> used_blocks;
    public:
        KVCacheManager(){
            init();
        }
        
        CacheBlock* get_cache_block(size_t block_id) {
            if (block_id >= num_blocks) return nullptr;
            return &used_blocks[block_id];
        }


        CacheBlock* allocate_cache_block() {
            if (free_blocks.empty()) {
                return nullptr;
            }
            CacheBlock block = free_blocks.back();
            free_blocks.pop_back();
            used_blocks.push_back(block);
            return &used_blocks.back();
        }

        void free_cache_block(size_t block_id) {
            if (block_id >= num_blocks) return;
            auto it = std::find_if(used_blocks.begin(), used_blocks.end(), 
                [block_id](const CacheBlock& block) { return block.block_id == block_id; });
            if (it != used_blocks.end()) {
                free_blocks.push_back(*it);
                used_blocks.erase(it);
            }
        }

        void write_kvcache(Sequence& seq, size_t layer_id, size_t token_pos, void* key_data, void* value_data) {
            size_t block_id = token_pos / BLOCK_SIZE;
            CacheBlock* block = get_cache_block(block_id);
            if (block) {
                // Write key_data and value_data to the corresponding positions in the cache
                cudaMemcpy((char*)block->key_cache_ptr + layer_id * HEAD_DIM * NUM_HEADS * DTYPE, key_data, HEAD_DIM * NUM_HEADS * DTYPE, cudaMemcpyHostToDevice);
                cudaMemcpy((char*)block->value_cache_ptr + layer_id * HEAD_DIM * NUM_HEADS * DTYPE, value_data, HEAD_DIM * NUM_HEADS * DTYPE, cudaMemcpyHostToDevice);
            }
        }

        ~KVCacheManager() {
            cudaFree(key_cache);
            cudaFree(value_cache);
        }
    private:


        size_t num_blocks;
        size_t block_size;
};