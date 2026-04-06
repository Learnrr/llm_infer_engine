#include "KVCacheManager.h"
#include "utils/logger.h"
#include <new>

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
            << " "
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
    block_size = config.block_size;
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
        cudaFree(key_cache);
        key_cache = nullptr;
        return ErrorCode::CUDA_FAILURE;
    }

    free_blocks.clear();
    used_blocks.clear();

    try {
        free_blocks.reserve(num_blocks);
        used_blocks.reserve(num_blocks);
        for (size_t i = 0; i < num_blocks; ++i) {
            auto block = std::make_shared<CacheBlock>(
                i,
                static_cast<char*>(key_cache) + i * bytes_per_block,
                static_cast<char*>(value_cache) + i * bytes_per_block
            );
            free_blocks.push_back(block);
        }
    } catch (const std::bad_alloc&) {
        LOG_ERROR("Host allocation failed while building cache block metadata (bad_alloc)");
        free_blocks.clear();
        used_blocks.clear();
        cudaFree(key_cache);
        cudaFree(value_cache);
        key_cache = nullptr;
        value_cache = nullptr;
        return ErrorCode::MEMORY_FAILURE;
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

ErrorCode KVCacheManager::add_block_ref(size_t block_id) {
    if (block_id >= num_blocks) {
        return ErrorCode::INVALID_INPUT;
    }
    auto it = std::find_if(used_blocks.begin(), used_blocks.end(), 
        [block_id](const std::shared_ptr<CacheBlock>& block) { return block->block_id == block_id; });
    if (it != used_blocks.end()) {
        (*it)->ref_count.fetch_add(1);
        return ErrorCode::SUCCESS;
    }
    return ErrorCode::INVALID_INPUT;
}

ErrorCode KVCacheManager::release_block_ref(size_t block_id) {
    if (block_id >= num_blocks) {
        return ErrorCode::INVALID_INPUT;
    }
    auto it = std::find_if(used_blocks.begin(), used_blocks.end(), 
        [block_id](const std::shared_ptr<CacheBlock>& block) { return block->block_id == block_id; });
    if (it != used_blocks.end()) {
        int new_ref_count = (*it)->ref_count.fetch_sub(1) - 1;
        if (new_ref_count < 0) {
            std::ostringstream oss;
            oss << "Cache block " << block_id << " reference count dropped below zero";
            LOG_ERROR(oss.str());
            (*it)->ref_count.store(0);
            return ErrorCode::UNKNOWN_ERROR;
        }
        return ErrorCode::SUCCESS;
    }
    return ErrorCode::INVALID_INPUT;
}