#pragma once
#include "ModelConfig.h"
#include "cuda_runtime.h"
#include "define.h"
#include "Tensor.h"
#include <string>
#include <fstream>
#include <memory>
#include <type_traits>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>
#include <variant>
#include "error.h"
#include "utils/logger.h"

using json = nlohmann::json;

struct LayerWeightLayout {
    virtual ~LayerWeightLayout() = default;
};

struct LinearLayerWeightLayout : public LayerWeightLayout {
    Tensor linear_weight;
    Tensor linear_bias;
};

struct AttentionLayerWeightLayout{
    //attention
    Tensor qkv_proj_weight;
    Tensor o_proj_weight;
    Tensor qkv_proj_bias; 
};

struct LayerNormLayerWeightLayout : public LayerWeightLayout {
    Tensor norm_weight;
    void* gamma;
};

struct MLPLayerWeightLayout{
    //mlp
    std::vector<LinearLayerWeightLayout> mlp_linears_weight; 
};

struct TransformerLayerWeightLayout : public LayerWeightLayout {
    //attention
    AttentionLayerWeightLayout attention_weights;
    //layernorm
    std::vector<LayerNormLayerWeightLayout> norm_weights;
    //mlp
    MLPLayerWeightLayout mlp_weights;

};

struct WeightLayout{
    WeightLayout(void* weights_ptr = nullptr): weights(weights_ptr) {}
    Tensor embedding_weights;

    std::vector<std::shared_ptr<LayerWeightLayout>> layer_weights;

    size_t total_size;

    void* weights;

    ErrorCode build_weight_layout(const ModelConfig& config);


    template <typename T>
    std::shared_ptr<T> get_layer_layout(size_t layer_id) const {
        static_assert(std::is_base_of_v<LayerWeightLayout, T>, "T must derive from LayerWeightLayout");
        if (layer_id >= layer_weights.size()) {
            return nullptr;
        }
        return std::dynamic_pointer_cast<T>(layer_weights[layer_id]);
    }
    
};

struct WeightHeader {
    size_t layer_idx;
    std::vector<int> shape;
    std::string name;
    std::string shard_file;
    size_t offset_start;
    size_t offset_end;
    DataType dtype;

};

class ModelWeights {
    public: 
        ModelWeights(){};
        ~ModelWeights(){};

        ErrorCode init(const ModelConfig& config);
        
        ErrorCode parse_header(const char* file_name);

        // Build ordered weight-name list from a text file, one weight name per line.
        ErrorCode build_weight_names(const char* file_name);

        //load to cpu
        Tensor load_layer(std::ifstream& file, const std::string& name);
        //concat qkv on cpu
        Tensor concat_qkv(const Tensor& Wq, const Tensor& Wk, const Tensor& Wv);

        //copy from cpu to gpu
        ErrorCode load_weights(const char* weight_path);

        std::variant<ErrorCode, size_t> read_total_size(const char* model_safetensors_index_json);

        void* weights;
        std::unordered_map<std::string, WeightHeader> headers;
        std::vector<std::string> weight_names;
        WeightLayout layout;

    };