#include "KVCacheManager.h"

ErrorCode KVCacheManager::init(const LLMEngineConfig& config) {

    size_t bytes_per_block = 
    config.block_size 
    * config.model_config.num_hidden_layers 
    * config.model_config.head_dim 
    * config.model_config.num_kv_heads 
    * config.model_config.dtype;
    
    if (bytes_per_block == 0) {
        return ErrorCode::COMPUTE_ERROR;
    }
    num_blocks = config.total_cache_size / bytes_per_block;

    cudaError_t error = cudaMalloc(&key_cache, config.total_cache_size);
    if (error != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    error = cudaMalloc(&value_cache, config.total_cache_size);
    if (error != cudaSuccess) {
        return ErrorCode::CUDA_FAILURE;
    }
    for (size_t i = 0; i < num_blocks; ++i) {
        auto block = std::make_shared<CacheBlock>(
            i, 
            (char*)key_cache + i * bytes_per_block,
            (char*)value_cache + i * bytes_per_block
        );
        free_blocks.push_back(block);
    }
    return ErrorCode::SUCCESS;
}

variant<shared_ptr<CacheBlock>, ErrorCode> KVCacheManager::get_cache_block(size_t block_id) {
    if (block_id >= num_blocks) {
        return ErrorCode::INVALID_INPUT;
    }
    for(const auto& block : used_blocks) {
        if (block->block_id == block_id) {
            return block;
        }
    }
    return ErrorCode::INVALID_INPUT;
}


variant<shared_ptr<CacheBlock>, ErrorCode> KVCacheManager::allocate_cache_block() {
    if (free_blocks.empty()) {
        return ErrorCode::MEMORY_FAILURE;
    }
    auto block = free_blocks.back();
    free_blocks.pop_back();
    used_blocks.push_back(block);
    return used_blocks.back();
}

ErrorCode KVCacheManager::free_cache_block(size_t block_id) {
    if (block_id >= num_blocks) {
        return ErrorCode::INVALID_INPUT;
    }
    auto it = std::find_if(used_blocks.begin(), used_blocks.end(), 
        [block_id](const shared_ptr<CacheBlock>& block) { return block->block_id == block_id; });
    if (it != used_blocks.end()) {
        free_blocks.push_back(*it);
        used_blocks.erase(it);
        return ErrorCode::SUCCESS;
    }
    return ErrorCode::INVALID_INPUT;
}