#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include "ModelConfig.h"

class LLMEngineConfig {
public:
    size_t max_batch_size;
    size_t max_sequence_length;
    size_t total_cache_size;
    size_t block_size;
    ModelConfig model_config;

    LLMEngineConfig(
        size_t max_batch_size, 
        size_t max_sequence_length,
        size_t total_cache_size,
        size_t block_size
    )
        : max_batch_size(max_batch_size), 
        max_sequence_length(max_sequence_length), 
        total_cache_size(total_cache_size), 
        block_size(block_size) {}

    void build_from_file(const char* config_path) {
        std::ifstream file(config_path);
        nlohmann::json config;
        file >> config;

        max_batch_size = config.value("max_batch_size", 16);
        max_sequence_length = config.value("max_sequence_length", 1024);
        total_cache_size = config.value("total_cache_size", 1024 * 1024 * 1024); // 1GB default
        block_size = config.value("block_size", 16); // 1KB default

        char* model_config_path = const_cast<char*>(config.value("model_config_path", "").c_str());
        model_config.build_from_file(model_config_path);
    }
};