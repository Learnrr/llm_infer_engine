#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;


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

struct TransformerLayerConfig : public LayerConfig {
    AttentionLayerConfig attention_config;
    MLPLayerConfig mlp_config;
};

struct LinearLayerConfig : public LayerConfig {
    LinearConfig linear_config;
};

struct LayerNormLayerConfig : public LayerConfig {
    size_t norm_size = 0;
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
    std::string model_path;
    size_t eos_token_id;
    std::string weight_names_path;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    

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

    void build_from_file(const char* config_path) {
        std::ifstream file(config_path);
        json config_json;
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
    }
};