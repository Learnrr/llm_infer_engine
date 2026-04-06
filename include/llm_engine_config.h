#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include "ModelConfig.h"
#include "utils/logger.h"
#include "error.h"


class LLMEngineConfig {
public:
    size_t max_decode_batch_size = 0;
    size_t max_prefill_batch_size = 0;
    size_t max_sequence_length = 0;
    size_t total_cache_size = 0;
    size_t block_size = 0;
    float temperature = 1.0f;
    float top_p = 1.0f;
    size_t top_k = 50;
    ModelConfig model_config;
    std::string model_config_path;
    bool greedy_decode = false;

    bool enable_pipeline_parallel = false;
    std::string role;
    int world_size = 1;
    int pipeline_rank = 0;
    int local_device_id = 0;
    size_t stage_start_layer = 0;
    size_t stage_end_layer = 0;

    size_t max_decode_batch_flight = 1;
    size_t max_prefill_batch_flight = 1;

    bool enable_prefix_cache = false;

    LLMEngineConfig() = default;

    ErrorCode build_from_file(const char* config_path) {
        std::ifstream file(config_path);
        if(!file.is_open()) {
            {
                std::ostringstream oss;
                oss << "Failed to open engine config file: " << config_path;
                LOG_ERROR(oss.str());
            }
            return ErrorCode::FAILED_TO_OPEN_CONFIG_FILE;
        }
        nlohmann::json config;
        file >> config;

        max_decode_batch_size = config.value("max_decode_batch_size", static_cast<size_t>(16));
        max_prefill_batch_size = config.value("max_prefill_batch_size", static_cast<size_t>(16));
        max_sequence_length = config.value("max_sequence_length", static_cast<size_t>(1024));
        total_cache_size = config.value("total_cache_size", static_cast<size_t>(1024ULL * 1024ULL * 1024ULL)); // 1GB default
        block_size = config.value("block_size", static_cast<size_t>(16)); // 1KB default
        temperature = config.value("temperature", temperature);
        top_p = config.value("top_p", top_p);
        top_k = config.value("top_k", top_k);
        model_config_path = config.value("model_config_path", "");
        greedy_decode = config.value("greedy_decode", false);
        role = config.value("role", "worker");
        enable_pipeline_parallel = config.value("enable_pipeline_parallel", false);
        world_size = config.value("world_size", 1);
        pipeline_rank = config.value("pipeline_rank", 0);
        local_device_id = config.value("local_device_id", 0);
        max_decode_batch_flight = config.value("max_decode_batch_flight", 1);
        max_prefill_batch_flight = config.value("max_prefill_batch_flight", 1);
        enable_prefix_cache = config.value("enable_prefix_cache", false);

        // Load model config from the specified path
        if(model_config_path.empty()) {
            {
                std::ostringstream oss;
                oss << "model_config_path is required in engine config JSON";
                LOG_ERROR(oss.str());
            }
            return ErrorCode::FAILED_TO_OPEN_CONFIG_FILE;
        }
        ErrorCode model_config_result = model_config.build_from_file(model_config_path.c_str());
        if (model_config_result != ErrorCode::SUCCESS) {
            return model_config_result;
        }

        if (world_size <= 0) {
            LOG_ERROR("world_size must be >= 1");
            return ErrorCode::INVALID_INPUT;
        }
        if (pipeline_rank < 0 || pipeline_rank >= world_size) {
            LOG_ERROR("pipeline_rank must be in [0, world_size)");
            return ErrorCode::INVALID_INPUT;
        }

        const size_t num_layers = model_config.num_hidden_layers;
        if (enable_pipeline_parallel) {
            if (!config.contains("stage_start_layer") || !config.contains("stage_end_layer")) {
                LOG_ERROR("stage_start_layer and stage_end_layer are required; automatic partition is disabled");
                return ErrorCode::INVALID_INPUT;
            }
            stage_start_layer = config.value("stage_start_layer", static_cast<size_t>(0));
            stage_end_layer = config.value("stage_end_layer", num_layers);
            if (stage_start_layer > stage_end_layer || stage_end_layer > num_layers) {
                std::ostringstream oss;
                oss << "Invalid stage layer range [" << stage_start_layer << ", " << stage_end_layer
                    << ") for num_hidden_layers=" << num_layers;
                LOG_ERROR(oss.str());
                return ErrorCode::INVALID_INPUT;
            }            
        } else {
            stage_start_layer = 0;
            stage_end_layer = num_layers;
        }
        return ErrorCode::SUCCESS;
    }

    bool is_first_stage() const {
        return pipeline_rank == 0;
    }

    bool is_last_stage() const {
        return pipeline_rank == world_size - 1;
    }
};