#pragma once

#include <cstddef>
#include <cctype>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "nlohmann/json.hpp"
#include <fstream>
#include "define.h" 
#include "error.h"
#include "utils/logger.h"

using json = nlohmann::json;

inline DataType ParseDataTypeFromJson(const json& config_json) {
    const json* dtype_node = nullptr;
    if (config_json.contains("data_type")) {
        dtype_node = &config_json["data_type"];
    } else if (config_json.contains("dtype")) {
        dtype_node = &config_json["dtype"];
    }

    if (dtype_node == nullptr) {
        return DataType::FLOAT16;
    }

    if (!dtype_node->is_string()) {
        throw std::invalid_argument("dtype/data_type must be a string");
    }

    std::string dtype_text = dtype_node->get<std::string>();
    for (char& ch : dtype_text) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (dtype_text == "float32" || dtype_text == "fp32" || dtype_text == "f32") {
        return DataType::FLOAT32;
    }
    if (dtype_text == "float16" || dtype_text == "fp16" || dtype_text == "f16" || dtype_text == "half") {
        return DataType::FLOAT16;
    }

    throw std::invalid_argument("unsupported dtype/data_type string: " + dtype_text);

}


struct LayerConfig {
    virtual ~LayerConfig() = default;
};

struct LinearConfig {
    size_t in_features = 0;
    size_t out_features = 0;
};

struct AttentionLayerConfig : public LayerConfig {
    size_t num_attention_heads = 0;
    size_t head_dim = 0;
    size_t num_kv_heads = 0; //For multi-query attention
};

struct MLPLayerConfig : public LayerConfig {
    enum class MLPType {
        GELU,
        SwiGLU,
        GLU
    };
    MLPType mlp_type = MLPType::SwiGLU;
    bool has_bias = false;
    size_t intermediate_size = 0;
    // Activate after this linear index, e.g. 0 means after first linear.
    size_t activation_after_linear_idx = 0;
    std::vector<LinearConfig> mlp_linears;
};
struct LayerNormLayerConfig : public LayerConfig {
    size_t norm_size = 0;
};

struct TransformerLayerConfig : public LayerConfig {
    AttentionLayerConfig attention_config;
    std::vector<LayerNormLayerConfig> norm_configs;
    MLPLayerConfig mlp_config;
};

struct LinearLayerConfig : public LayerConfig {
    LinearConfig linear_config;
};



class ModelConfig {
public:
    ModelConfig()
        : max_seq_len(512),
          hidden_size(768),
          num_hidden_layers(12),
          vocab_size(30522) {}

    size_t max_seq_len;
    size_t hidden_size;
    size_t num_hidden_layers;
    size_t vocab_size;
    float temperature = 1.0f;
    float top_p = 1.0f;
    size_t top_k = 50; 
    std::string model_path; //path to the model weights, e.g. safetensors file
    size_t eos_token_id;
    std::string weight_names_path; //path to the weight names txt
    //path to the json file that maps weight names to safetensors files
    std::string model_safetensors_index_json; 
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    DataType data_type;
    size_t mlp_intermediate_size;
    

    // Store per-layer configs polymorphically.
    std::vector<std::shared_ptr<LayerConfig>> layer_configs;

    template <typename T, typename... Args>
    std::shared_ptr<T> add_layer_config(Args&&... args) {
        static_assert(std::is_base_of_v<LayerConfig, T>, "T must derive from LayerConfig");
        auto cfg = std::make_shared<T>(std::forward<Args>(args)...);
        layer_configs.push_back(cfg);
        return cfg;
    }

    template <typename T>
    std::shared_ptr<T> get_layer_config(size_t idx) const {
        static_assert(std::is_base_of_v<LayerConfig, T>, "T must derive from LayerConfig");
        if (idx >= layer_configs.size()) {
            return nullptr;
        }
        return std::dynamic_pointer_cast<T>(layer_configs[idx]);
    }

    ErrorCode build_from_file(const char* config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            {
                std::ostringstream oss;
                oss << "Failed to open model config file: " << config_path;
                LOG_ERROR(oss.str());
            }
            return ErrorCode::FAILED_TO_OPEN_CONFIG_FILE;
        }
        nlohmann::json config_json;
        file >> config_json;
        max_seq_len = config_json.value("max_seq_len", max_seq_len);
        hidden_size = config_json.value("hidden_size", hidden_size);
        num_hidden_layers = config_json.value("num_hidden_layers", num_hidden_layers);
        vocab_size = config_json.value("vocab_size", vocab_size);
        model_path = config_json.value("model_path", model_path);
        temperature = config_json.value("temperature", temperature);
        top_p = config_json.value("top_p", top_p);
        top_k = config_json.value("top_k", top_k);
        eos_token_id = config_json.value("eos_token_id", eos_token_id);
        num_heads = config_json.value("num_heads", num_heads);
        num_kv_heads = config_json.value("num_kv_heads", num_kv_heads);
        head_dim = config_json.value("head_dim", head_dim);
        data_type = ParseDataTypeFromJson(config_json);
        weight_names_path = config_json.value("weight_names_path", weight_names_path);
        model_safetensors_index_json = config_json.value("model_safetensors_index_json", model_safetensors_index_json);
        mlp_intermediate_size = config_json.value("mlp_intermediate_size", mlp_intermediate_size);
        


        // Load layer-specific configs
        if (config_json.contains("layer_configs")) {
            for (const auto& layer_cfg_json : config_json["layer_configs"]) {
                std::string type = layer_cfg_json["type"];
                if (type == "TransformerLayer") {
                    auto transformer_cfg = add_layer_config<TransformerLayerConfig>();
                    transformer_cfg->attention_config.num_attention_heads = layer_cfg_json["attention_config"]["num_attention_heads"];
                    transformer_cfg->attention_config.head_dim = layer_cfg_json["attention_config"]["head_dim"];
                    transformer_cfg->attention_config.num_kv_heads = layer_cfg_json["attention_config"]["num_kv_heads"];
                    transformer_cfg->mlp_config.mlp_type = MLPLayerConfig::MLPType::SwiGLU;
                    transformer_cfg->mlp_config.has_bias = layer_cfg_json["mlp_config"]["has_bias"];
                    transformer_cfg->mlp_config.intermediate_size = layer_cfg_json["mlp_config"]["intermediate_size"];
                    transformer_cfg->mlp_config.activation_after_linear_idx =
                        layer_cfg_json["mlp_config"].value("activation_after_linear_idx", static_cast<size_t>(0));
                    for(const auto& norm_cfg_json : layer_cfg_json["norm_configs"]) {
                        LayerNormLayerConfig norm_cfg;
                        norm_cfg.norm_size = norm_cfg_json["norm_size"];
                        transformer_cfg->norm_configs.push_back(norm_cfg);
                    }
                    for (const auto& linear_json : layer_cfg_json["mlp_config"]["mlp_linears"]) {
                        LinearConfig linear_cfg;
                        linear_cfg.in_features = linear_json["in_features"];
                        linear_cfg.out_features = linear_json["out_features"];

                        transformer_cfg->mlp_config.mlp_linears.push_back(linear_cfg);
                    }
                } else if (type == "LinearLayer") {
                    auto linear_cfg = add_layer_config<LinearLayerConfig>();
                    linear_cfg->linear_config.in_features = layer_cfg_json["linear_config"]["in_features"];
                    linear_cfg->linear_config.out_features = layer_cfg_json["linear_config"]["out_features"];
                } else if (type == "LayerNormLayer") {
                    auto norm_cfg = add_layer_config<LayerNormLayerConfig>();
                    norm_cfg->norm_size = layer_cfg_json["norm_size"];
                }
            }

        }
        return ErrorCode::SUCCESS;
    }
};