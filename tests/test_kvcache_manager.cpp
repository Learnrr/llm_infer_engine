/*
cd tests
nvcc -std=c++17 -O2 -I../include -I../include/model test_kvcache_manager.cpp \
    ../src/KVCacheManager.cpp -o ../build/tests/test_kvcache_manager.exe
../build/tests/test_kvcache_manager.exe [llm_engine_config.json]
*/

#include "KVCacheManager.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <variant>

namespace {

size_t DTypeBytes(DataType dtype) {
    return dtype == DataType::FLOAT16 ? 2 : 4;
}

size_t ComputeBytesPerBlock(const LLMEngineConfig& config) {
    return config.block_size
        * config.model_config.num_hidden_layers
        * config.model_config.head_dim
        * config.model_config.num_kv_heads
        * DTypeBytes(config.model_config.data_type);
}

LLMEngineConfig BuildDefaultEngineConfig() {
    LLMEngineConfig cfg(
        1,                              // max_batch_size
        32768,                          // max_sequence_length
        2ULL * 1024ULL * 1024ULL * 1024ULL, // total_cache_size
        16                              // block_size
    );

    cfg.model_config.max_seq_len = 32768;
    cfg.model_config.hidden_size = 3584;
    cfg.model_config.num_hidden_layers = 28;
    cfg.model_config.vocab_size = 152064;
    cfg.model_config.num_heads = 28;
    cfg.model_config.num_kv_heads = 4;
    cfg.model_config.head_dim = 128;
    cfg.model_config.mlp_intermediate_size = 18944;
    cfg.model_config.data_type = DataType::FLOAT16;
    return cfg;
}

void AssertError(const std::variant<std::shared_ptr<CacheBlock>, ErrorCode>& value, ErrorCode expected) {
    assert(std::holds_alternative<ErrorCode>(value));
    assert(std::get<ErrorCode>(value) == expected);
}

std::shared_ptr<CacheBlock> AssertBlock(const std::variant<std::shared_ptr<CacheBlock>, ErrorCode>& value) {
    assert(std::holds_alternative<std::shared_ptr<CacheBlock>>(value));
    auto block = std::get<std::shared_ptr<CacheBlock>>(value);
    assert(block != nullptr);
    return block;
}

} // namespace

int main(int argc, char** argv) {
    LLMEngineConfig config = BuildDefaultEngineConfig();

    if (argc >= 2) {
        const std::string config_path = argv[1];
        const ErrorCode cfg_err = config.build_from_file(config_path.c_str());
        if (cfg_err != ErrorCode::SUCCESS) {
            std::cerr << "Failed to load engine config: " << config_path
                      << ", error code: " << static_cast<int>(cfg_err) << "\n";
            return 1;
        }
        std::cout << "Loaded engine config from: " << config_path << "\n";
    } else {
        std::cout << "Using built-in default engine config for KVCacheManager test\n";
    }

    const size_t bytes_per_block = ComputeBytesPerBlock(config);
    assert(bytes_per_block > 0);
    const size_t kExpectedBlocks = config.total_cache_size / bytes_per_block;
    assert(kExpectedBlocks > 0);

    KVCacheManager manager;
    const ErrorCode init_err = manager.init(config);

    assert(init_err == ErrorCode::SUCCESS && "KVCacheManager::init should succeed");

    std::set<size_t> allocated_ids;
    for (size_t i = 0; i < kExpectedBlocks; ++i) {
        auto alloc_result = manager.allocate_cache_block();
        auto block = AssertBlock(alloc_result);

        assert(block->is_valid);
        assert(block->key_cache_ptr != nullptr);
        assert(block->value_cache_ptr != nullptr);
        allocated_ids.insert(block->block_id);

        auto get_result = manager.get_cache_block(block->block_id);
        auto got_block = AssertBlock(get_result);
        assert(got_block->block_id == block->block_id);
    }
    assert(allocated_ids.size() == kExpectedBlocks);

    auto exhausted = manager.allocate_cache_block();
    AssertError(exhausted, ErrorCode::MEMORY_FAILURE);

    const size_t recycled_id = *allocated_ids.begin();
    const ErrorCode free_err = manager.free_cache_block(recycled_id);
    assert(free_err == ErrorCode::SUCCESS);

    auto after_free_get = manager.get_cache_block(recycled_id);
    AssertError(after_free_get, ErrorCode::INVALID_INPUT);

    auto reused = manager.allocate_cache_block();
    auto reused_block = AssertBlock(reused);
    assert(reused_block->block_id == recycled_id);

    const ErrorCode free_invalid = manager.free_cache_block(kExpectedBlocks + 7);
    assert(free_invalid == ErrorCode::INVALID_INPUT);

    const ErrorCode free_once = manager.free_cache_block(reused_block->block_id);
    assert(free_once == ErrorCode::SUCCESS);

    const ErrorCode free_twice = manager.free_cache_block(reused_block->block_id);
    assert(free_twice == ErrorCode::INVALID_INPUT);

    std::cout << "test_kvcache_manager passed\n";
    return 0;
}
