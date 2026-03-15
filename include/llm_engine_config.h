#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include "ModelConfig.h"

class LLMEngineConfig {
public:
    size_t max_batch_size;
    size_t max_sequence_length;
    ModelConfig model_config;

    LLMEngineConfig(size_t max_batch_size, size_t max_sequence_length)
        : max_batch_size(max_batch_size), max_sequence_length(max_sequence_length){}

    void build_from_file(const char* config_path) {
        std::ifstream file(config_path);
        nlohmann::json config;
        file >> config;

        max_batch_size = config.value("max_batch_size", 16);
        max_sequence_length = config.value("max_sequence_length", 1024);

        char* model_config_path = const_cast<char*>(config.value("model_config_path", "").c_str());
        model_config.build_from_file(model_config_path);
    }
};