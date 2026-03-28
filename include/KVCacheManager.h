#include "KVCacheManager.h"
#include "utils/logger.h"

ErrorCode KVCacheManager::init(const LLMEngineConfig& config) {

    size_t bytes_per_block = 
    config.block_size 
    * config.model_config.num_hidden_layers 
    * config.model_config.head_dim 
    * config.model_config.num_kv_heads 
    * DataTypeBytes(config.model_config.data_type);
    
    {
        std::ostringstream oss;
        oss << "Initializing KVCacheManager with total size: " << config.total_cache_size
            << "Computed bytes per block: " << bytes_per_block 
            << " (block_size: " << config.block_size 
            << ", num_hidden_layers: " << config.model_config.num_hidden_layers
            << ", head_dim: " << config.model_config.head_dim
            << ", num_kv_heads: " << config.model_config.num_kv_heads
            << ", data_type: " << DataTypeName(config.model_config.data_type)
            << ")";
        LOG_INFO(oss.str());
    }
    
    if (bytes_per_block == 0) {
        
        std::ostringstream oss;
        oss << "Invalid block size or model config resulting in zero bytes per block";
        LOG_ERROR(oss.str());
    
        return ErrorCode::COMPUTE_ERROR;
    }
    num_blocks = config.total_cache_size / bytes_per_block;
    {
        std::ostringstream oss;
        oss << "Initializing KVCacheManager with " << num_blocks << " blocks, each block has size " << bytes_per_block << " bytes";
        LOG_INFO(oss.str());
    }

    cudaError_t error = cudaMalloc(&key_cache, config.total_cache_size);
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "Failed to allocate CUDA memory for key cache: " << cudaGetErrorString(error);
        LOG_ERROR(oss.str());
        return ErrorCode::CUDA_FAILURE;
    }
    error = cudaMalloc(&value_cache, config.total_cache_size);
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "Failed to allocate CUDA memory for value cache: " << cudaGetErrorString(error);
        LOG_ERROR(oss.str());
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

std::variant<std::shared_ptr<CacheBlock>, ErrorCode> KVCacheManager::get_cache_block(size_t block_id) {
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


std::variant<std::shared_ptr<CacheBlock>, ErrorCode> KVCacheManager::allocate_cache_block() {
    if (free_blocks.empty()) {
        std::ostringstream oss;
        oss << "No free cache blocks available";
        LOG_ERROR(oss.str());

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
        [block_id](const std::shared_ptr<CacheBlock>& block) { return block->block_id == block_id; });
    if (it != used_blocks.end()) {
        free_blocks.push_back(*it);
        used_blocks.erase(it);
        return ErrorCode::SUCCESS;
    }
    std::ostringstream oss;
    oss << "Attempted to free invalid cache block: " << block_id;
    LOG_ERROR(oss.str());
    return ErrorCode::INVALID_INPUT;
}