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

using json = nlohmann::json;

struct LayerWeightLayout {
    virtual ~LayerWeightLayout() = default;
};

struct AttentionLayerWeightLayout{
    //attention
    Tensor qkv_proj_weight;
    Tensor o_proj_weight;
};

struct MLPLayerWeightLayout{
    //mlp
    Tensor gate_proj_weight;
    Tensor up_proj_weight;
    Tensor down_proj_weight;
};

struct TransformerLayerWeightLayout : public LayerWeightLayout {
    //attention
    AttentionLayerWeightLayout attention_weights;

    //mlp
    MLPLayerWeightLayout mlp_weights;

    //layer norm
    Tensor attn_norm_weight;
    Tensor ffn_norm_weight;

};

struct LinearLayerWeightLayout : public LayerWeightLayout {
    Tensor linear_weight;
};

struct LayerNormLayerWeightLayout : public LayerWeightLayout {
    Tensor norm_weight;
};

struct WeightLayout{
    WeightLayout(void* weights_ptr = nullptr): weights(weights_ptr) {}
    Tensor embedding_weights;

    std::vector<std::shared_ptr<LayerWeightLayout>> layer_weights;

    size_t total_size;

    void* weights;

    void build_config(const ModelConfig& config);

    void build();

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
    size_t offset_start;
    size_t offset_end;
    DataType dtype;

};

class ModelWeights {
    public: 
        ModelWeights(){};
        ~ModelWeights(){};

        void init(const ModelConfig& config);
        
        void parse_header(const char* file_name);

        // Build ordered weight-name list from a text file, one weight name per line.
        ErrorCode build_weight_names(const char* file_name);

        //load to cpu
        Tensor load_layer(std::ifstream& file, const std::string& name);
        //concat qkv on cpu
        Tensor concat_qkv(const Tensor& Wq, const Tensor& Wk, const Tensor& Wv);

        //copy from cpu to gpu
        void load_weights(const char* weight_path);

        void* weights;
        std::unordered_map<std::string, WeightHeader> headers;
        std::vector<std::string> weight_names;
        WeightLayout layout;

    };